"""Call-time public completion provenance; all provider/Docker calls are fake."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage

from contextlab.core.models import CompactionConfig, Pricing, Usage
from contextlab.core.view_builder import canonical_bytes
from harness.config import ExperimentConfig
from harness.runner import PUBLIC_PHASE_BOUNDARY_SOURCE, run_live


def run_fixture(monkeypatch, tmp_path, *, mode="complete", enabled=True, public_contract=True):
    import langchain_anthropic

    import harness.agent
    import harness.runner

    def create(workspace):
        workspace.mkdir()

    def public_status(workspace, index):
        ok = (workspace / f"phase-{index}.txt").exists()
        return {"ok": ok, "basis": "public_artifact", "issues": [] if ok else [{"reason": "missing"}]}

    task = SimpleNamespace(id="config_parser", create=create, phases=("Complete phase zero", "Complete phase one"),
                           phase_check_source=lambda _: "private checker", snapshot=lambda _: {},
                           checker_source=lambda: "private checker")
    if public_contract:
        task.public_phase_status = public_status

    class Model:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            return 100

    class Executor:
        def __init__(self, *args):
            pass

        def run(self, *args, **kwargs):
            return {"exit_code": 1, "stdout": "PRIVATE_VERDICT_SENTINEL"}

    class Headroom:
        def __init__(self, **kwargs):
            self.telemetry, self.failures = [], []

    def build(**kwargs):
        class Agent:
            phase = 0
            attempts = 0

            def invoke(self, state, config):
                kwargs["before_call"](SimpleNamespace(system_message=None, messages=state["messages"], tools=[]))
                kwargs["on_usage"]("fixture", Usage(input_tokens=100, output_tokens=5), 1.0, "ok", None)
                incomplete = self.phase == 0 and (mode == "always_missing" or (mode == "truncated_once" and not self.attempts))
                if mode == "raise_first":
                    raise RuntimeError("offline invocation failure")
                if not incomplete or mode == "truncated_once":
                    (kwargs["workspace"].root / f"phase-{self.phase}.txt").write_text("public artifact")
                stop_reason = "max_tokens" if incomplete and mode == "truncated_once" else "end_turn"
                self.attempts += 1
                if not incomplete:
                    self.phase += 1
                return {"messages": [*state["messages"], AIMessage(content="phase response", response_metadata={"stop_reason": stop_reason})]}
        return Agent()

    config = ExperimentConfig(arms=("baseline", "contextlab", "headroom"), tasks=(task.id,), repeats=1,
                              session_cap_usd="1", experiment_cap_usd="10", max_model_calls=10,
                              max_input_tokens=10000, max_output_tokens=100, threshold_tokens=2000,
                              threshold_keep=2, docker_image="fake", phase_boundaries=enabled)
    pricing = Pricing(model="fixture", input_per_million=1, output_per_million=1,
                      cache_write_5m_per_million=1, cache_write_1h_per_million=1,
                      cache_read_per_million=1, source="offline test fixture", verified_at="2026-10-10")
    plan = {"approval_digest": "offline", "compaction": CompactionConfig().model_dump(mode="json"),
            "matrix": [{"arm": arm, "task_id": task.id, "repeat": 0} for arm in config.arms]}
    monkeypatch.setitem(harness.runner.LIVE_TASKS, task.id, task)
    monkeypatch.setattr("harness.runner.experiment_plan", lambda *args: plan)
    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr("harness.runner.DockerExecutor", Executor)
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: Model())
    monkeypatch.setattr(harness.agent, "build_agent", build)
    monkeypatch.setitem(sys.modules, "contextlab.adapters.headroom_mw", SimpleNamespace(HeadroomMiddleware=Headroom))
    output = tmp_path / "run"
    results = run_live(output, config, pricing, pricing.model, "offline")
    return [json.loads((output / result.session_id / "trace.json").read_text()) for result in results]


@pytest.mark.parametrize("mode", ["complete", "truncated_once"])
def test_boundaries_are_call_time_prefix_evidence_shared_across_arms(monkeypatch, tmp_path, mode):
    traces = run_fixture(monkeypatch, tmp_path, mode=mode)
    requests = [trace["requests"] for trace in traces]
    assert requests[0] == requests[1] == requests[2]
    for trace in traces:
        assert not trace["requests"][0]["completed_phase_boundaries"]
        if mode == "truncated_once":
            assert not trace["requests"][1]["completed_phase_boundaries"]
        last = trace["requests"][-1]
        assert len(last["completed_phase_boundaries"]) == 1
        boundary = last["completed_phase_boundaries"][0]
        assert boundary["source"] == PUBLIC_PHASE_BOUNDARY_SOURCE
        assert boundary["phase_index"] == 0 and boundary["label"] == "config_parser/phase-1"
        prefix = last["actual_rendered_messages"][:boundary["normalized_message_count"]]
        assert hashlib.sha256(canonical_bytes(prefix)).hexdigest() == boundary["message_prefix_sha256"]
        assert "PRIVATE_VERDICT_SENTINEL" not in json.dumps(last)
        assert all(phase["independent_check"]["exit_code"] == 1 for phase in trace["phases"])


@pytest.mark.parametrize("mode", ["always_missing", "raise_first"])
def test_unfinished_phases_never_create_trusted_boundaries(monkeypatch, tmp_path, mode):
    for trace in run_fixture(monkeypatch, tmp_path, mode=mode):
        assert all(not request["completed_phase_boundaries"] for request in trace["requests"])
        assert not trace["phases"][0]["completed"]


@pytest.mark.parametrize("options", [{"enabled": False}, {"public_contract": False}])
def test_disabled_policy_or_no_public_contract_has_no_trusted_boundaries(monkeypatch, tmp_path, options):
    for trace in run_fixture(monkeypatch, tmp_path, **options):
        assert len(trace["requests"]) == 2
        assert all(not request["completed_phase_boundaries"] for request in trace["requests"])
