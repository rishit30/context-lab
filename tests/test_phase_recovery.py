"""Shared public-only phase recovery; all inference and Docker calls are fake."""
import json
import sys
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from contextlab.adapters.langchain_mw import ContextLabMiddleware, MeteringMiddleware
from contextlab.core.models import CompactionConfig, Pricing
from harness.config import ExperimentConfig
from harness.runner import MAX_PHASE_RECOVERIES, run_live
from harness.tasks.file_audit import FILE_AUDIT_TASKS, OUTPUT_NAMES


def run_recovery_fixture(monkeypatch, tmp_path, mode, *, max_calls=20):
    import langchain_anthropic

    import harness.agent

    task = FILE_AUDIT_TASKS["file_audit_short"]
    references = task.reference_outputs()
    observed = {"prompts": {}, "boundaries": [], "private_checks": 0, "counts": 0}

    class FakeModel:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            observed["counts"] += 1
            return 100

    class FakeExecutor:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            return {"exit_code": 0, "stdout": "Public tests passed"}

        def run(self, *args, **kwargs):
            observed["private_checks"] += 1
            return {"exit_code": 1, "stdout": "PRIVATE_GRADER_SENTINEL"}

    class FakeHeadroom:
        def __init__(self, **kwargs):
            self.telemetry, self.failures = [], []

    def build(**kwargs):
        meter = MeteringMiddleware(model_name="fake-recovery-model",
                                   before_call=kwargs["before_call"], on_usage=kwargs["on_usage"],
                                   max_model_calls=kwargs["max_model_calls"])
        observed["prompts"][kwargs["arm"]] = []

        class Agent:
            phase = 0
            attempt = 0

            def invoke(self, state, config):
                observed["prompts"][kwargs["arm"]].append(state["messages"][-1].content)
                failed = mode == "always_missing" or (self.phase == 0 and self.attempt == 0)
                truncated = failed and mode == "truncated_once"
                request = SimpleNamespace(system_message=None, messages=state["messages"], tools=[])
                response = meter.wrap_model_call(request, lambda _: ModelResponse(result=[AIMessage(
                    content="phase work", response_metadata={"stop_reason": "max_tokens" if truncated else "end_turn"},
                    usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105},
                )]))
                if not failed or truncated:
                    (kwargs["workspace"].root / OUTPUT_NAMES[self.phase]).write_text(
                        references[OUTPUT_NAMES[self.phase]])
                elif mode == "invalid_schema_once":
                    (kwargs["workspace"].root / OUTPUT_NAMES[self.phase]).write_text('{"unrelated":true}')
                if failed:
                    self.attempt += 1
                else:
                    self.phase += 1
                    self.attempt = 0
                return {"messages": [*state["messages"], *response.result]}

        return Agent()

    config = ExperimentConfig(
        arms=("baseline", "contextlab", "headroom"), tasks=(task.id,), repeats=1,
        session_cap_usd="1", experiment_cap_usd="10", max_model_calls=max_calls,
        max_input_tokens=10000, max_output_tokens=100, threshold_tokens=2000,
        threshold_keep=2, docker_image="fake-image", phase_boundaries=True,
    )
    pricing = Pricing(model="fake-recovery-model", input_per_million=1, output_per_million=1,
                      cache_write_5m_per_million=1, cache_write_1h_per_million=1,
                      cache_read_per_million=1, source="test-only", verified_at="2026-10-10")
    plan = {"approval_digest": "frozen-fixture", "compaction": CompactionConfig().model_dump(mode="json"),
            "matrix": [{"arm": arm, "task_id": task.id, "repeat": 0} for arm in config.arms]}
    monkeypatch.setattr("harness.runner.experiment_plan", lambda *args: plan)
    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr("harness.runner.DockerExecutor", FakeExecutor)
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: FakeModel())
    monkeypatch.setattr(harness.agent, "build_agent", build)
    monkeypatch.setattr(ContextLabMiddleware, "mark_boundary",
                        lambda self, name: observed["boundaries"].append(name))
    monkeypatch.setitem(sys.modules, "contextlab.adapters.headroom_mw",
                        SimpleNamespace(HeadroomMiddleware=FakeHeadroom))
    output = tmp_path / "experiment"
    results = run_live(output, config, pricing, pricing.model, "frozen-fixture")
    traces = {result.arm: json.loads((output / result.session_id / "trace.json").read_text())
              for result in results}
    return observed, results, traces, output


@pytest.mark.parametrize("mode", ["missing_once", "invalid_schema_once", "truncated_once"])
def test_shared_recovery_waits_for_complete_public_deliverable_and_normal_stop(monkeypatch, tmp_path, mode):
    observed, results, traces, _ = run_recovery_fixture(monkeypatch, tmp_path, mode)
    prompts = list(observed["prompts"].values())
    assert prompts[0] == prompts[1] == prompts[2]
    assert len(prompts[0]) == 5 and "current workflow phase is incomplete" in prompts[0][1]
    assert all(result.turns == 5 and result.cost_usd > 0 for result in results)
    assert observed["counts"] == 15
    for trace in traces.values():
        first = trace["phases"][0]
        assert first["completed"] and len(first["attempts"]) == 2
        assert first["attempts"][0]["completed"] is False
        assert first["attempts"][1]["completed"] is True
        assert [request["phase_index"] for request in trace["requests"]] == [0, 0, 1, 2, 3]
        if mode == "truncated_once":
            assert first["attempts"][0]["public_check"]["ok"]
            assert first["attempts"][0]["response_truncated"]
    assert len(observed["boundaries"]) == 4
    # A failed private verdict is never used to guide continuation or repair.
    assert "PRIVATE_GRADER_SENTINEL" not in json.dumps(prompts)
    assert all(not result.solved for result in results)


def test_exhausted_recovery_fails_before_boundary_or_snapshot_transition(monkeypatch, tmp_path):
    observed, results, traces, output = run_recovery_fixture(monkeypatch, tmp_path, "always_missing")
    expected_attempts = MAX_PHASE_RECOVERIES + 1
    assert all(result.turns == expected_attempts and result.error == "PhaseIncompleteError"
               for result in results)
    assert not observed["boundaries"] and observed["private_checks"] == 0
    for result in results:
        trace = traces[result.arm]
        assert len(trace["phases"]) == 1 and trace["phases"][0]["completed"] is False
        assert len(trace["phases"][0]["attempts"]) == expected_attempts
        assert trace["failure"]["stage"] == "phase_recovery_exhausted"
        workspace = output / result.session_id / "workspace"
        assert all((workspace / path).read_text() == text
                   for path, text in FILE_AUDIT_TASKS["file_audit_short"].snapshot(1).items())


def test_recovery_cannot_reset_existing_model_call_limit(monkeypatch, tmp_path):
    observed, results, traces, _ = run_recovery_fixture(monkeypatch, tmp_path, "always_missing", max_calls=1)
    assert all(result.turns == 1 and result.error == "RuntimeError" for result in results)
    assert observed["counts"] == 3 and not observed["boundaries"]
    assert all(trace["phases"][0]["completed"] is False for trace in traces.values())


def test_public_schema_never_uses_or_discloses_reference_answers(monkeypatch, tmp_path):
    task = FILE_AUDIT_TASKS["file_audit_short"]
    references = task.reference_outputs()
    workspace = tmp_path / "workspace"
    task.create(workspace)
    for name, content in references.items():
        (workspace / name).write_text(content)
    history = json.loads((workspace / "history.json").read_text())
    history["requests"][0]["initial_rollback_ticket"] = "wrong-but-structurally-valid"
    (workspace / "history.json").write_text(json.dumps(history))
    monkeypatch.setattr(type(task), "reference_outputs", lambda *args: pytest.fail("Private oracle used"))
    assert task.public_phase_status(workspace, 3)["ok"]
    (workspace / "audit-current.json").write_text('{"snapshot":"2026-10-10","findings":[')
    status = task.public_phase_status(workspace, 3)
    assert not status["ok"] and status["issues"][0]["file"] == "audit-current.json"
    assert "wrong-but-structurally-valid" not in json.dumps(status)
