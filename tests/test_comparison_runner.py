"""Comparison wiring and audit grading; all providers/executors are replaced."""
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from contextlab.adapters.langchain_mw import ContextLabMiddleware, MeteringMiddleware
from contextlab.core.models import Pricing, Usage
from contextlab.ledger import LedgerStore
from harness.agent import SYSTEM_PROMPT, build_agent, invariant_hash
from harness.config import ExperimentConfig, experiment_plan
from harness.runner import run_live
from harness.tasks.file_audit import FILE_AUDIT_TASKS
from harness.workspace import Workspace


def comparison_config(**updates):
    values = dict(arms=("baseline", "headroom", "contextlab"), tasks=("file_audit_short",),
                  repeats=1, session_cap_usd="2", experiment_cap_usd="10",
                  max_model_calls=12, max_output_tokens=100, max_input_tokens=10000,
                  threshold_tokens=2000, threshold_keep=2, docker_image="fake-local",
                  cache_mode="disabled", phase_boundaries=True)
    return ExperimentConfig(**(values | updates))


def prices():
    return Pricing(model="fake-comparison", input_per_million=3,
                   cache_write_5m_per_million="3.75", cache_write_1h_per_million=6,
                   cache_read_per_million="0.3", output_per_million=15,
                   source="test fixture only", verified_at="2026-10-10")


def wire_comparison(monkeypatch, *, tamper_initial=False, diagnostic_failure=False,
                    final_failure=False, unknown_usage=False):
    import langchain_anthropic

    import contextlab.adapters.headroom_mw as headroom_module
    import contextlab.adapters.langchain_mw as context_module
    import harness.agent as agent_module

    task = FILE_AUDIT_TASKS["file_audit_short"]
    reference_outputs = task.reference_outputs()
    captured = {"builds": [], "model_kwargs": [], "counts": 0, "inferences": 0,
                "graders": [], "phase_prompts": [], "context_instances": [], "headroom_instances": []}

    class FakeModel:
        def get_num_tokens_from_messages(self, messages, **kwargs):
            captured["counts"] += 1
            return 100

    def make_model(**kwargs):
        captured["model_kwargs"].append(kwargs)
        return FakeModel()

    class FakeHeadroom:
        def __init__(self, **kwargs):
            self.failures, self.telemetry = [], []
            captured["headroom_instances"].append(self)

    def make_context(**kwargs):
        instance = ContextLabMiddleware(**kwargs)
        captured["context_instances"].append(instance)
        return instance

    class FakeExecutor:
        def __init__(self, workspace, image):
            self.workspace, self.phase_calls = workspace, 0

        def __call__(self, argv, cwd):
            assert argv == ["python", "-m", "pytest", "-q", "--", "tests"]
            return {"exit_code": 0, "stdout": "public tests passed", "stderr": ""}

        def run(self, argv, *, stdin=None):
            captured["graders"].append((self.workspace.parent.name, argv, stdin))
            self.phase_calls += 1
            failed = ((diagnostic_failure and self.phase_calls == 1)
                      or (final_failure and self.phase_calls == len(task.phases) + 1))
            return {"exit_code": int(failed),
                    "stdout": "PRIVATE_GRADER_DIAGNOSTIC" if failed else "CHECKS_PASSED", "stderr": ""}

    def make_agent(**kwargs):
        captured["builds"].append(kwargs)
        class FakeAgent:
            phase = 0

            def invoke(self, state, config):
                expected = task.snapshot(1 if self.phase == 0 else 2)
                root = kwargs["workspace"].root
                for relative, content in expected.items():
                    assert (root / relative).read_text() == content
                prompts = [message.content for message in state["messages"] if isinstance(message, HumanMessage)]
                assert prompts == list(task.phases[:self.phase + 1])
                captured["phase_prompts"].append((kwargs["arm"], self.phase, prompts))
                request = SimpleNamespace(system_message=None, messages=state["messages"], tools=[])
                kwargs["before_call"](request)
                captured["inferences"] += 1
                usage = None if unknown_usage else Usage(input_tokens=100, output_tokens=5)
                kwargs["on_usage"]("fake-comparison", usage, 1.0,
                                   "usage_missing" if unknown_usage else "ok", None,
                                   {"messages": [{"provider_usage": {"input_tokens": 100, "output_tokens": 5}}]})
                # These offline wiring fixtures must satisfy the real public
                # completion check; private grader outcomes remain independent.
                output_path = tuple(reference_outputs)[self.phase]
                (root / output_path).write_text(reference_outputs[output_path])
                if tamper_initial and self.phase == 0:
                    (root / task.snapshot_paths[0]).write_text("modified initial snapshot")
                self.phase += 1
                return {"messages": [*state["messages"], AIMessage(content="phase completed")]}
        return FakeAgent()

    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr("harness.runner.DockerExecutor", FakeExecutor)
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", make_model)
    monkeypatch.setattr(agent_module, "build_agent", make_agent)
    monkeypatch.setattr(context_module, "ContextLabMiddleware", make_context)
    monkeypatch.setattr(headroom_module, "configuration", lambda: {"profile": "offline test fixture"})
    monkeypatch.setattr(headroom_module, "HeadroomMiddleware", FakeHeadroom)
    return captured


def execute(directory, config):
    rates = prices()
    plan = experiment_plan(config, rates, rates.model)
    return run_live(directory, config, rates, rates.model, plan["approval_digest"])


def test_three_arms_run_all_phases_with_isolated_inputs_and_identical_invariants(monkeypatch, tmp_path):
    captured = wire_comparison(monkeypatch)
    root = tmp_path / "comparison"
    results = execute(root, comparison_config())
    assert len(results) == 3 and all(result.solved for result in results)
    assert captured["counts"] == captured["inferences"] == 12
    assert len(captured["context_instances"]) == len(captured["headroom_instances"]) == 1
    builds = {item["arm"]: item for item in captured["builds"]}
    assert builds["baseline"]["context_middleware"] is None
    assert builds["baseline"]["headroom_middleware"] is None
    assert builds["headroom"]["context_middleware"] is None
    assert builds["headroom"]["headroom_middleware"] is captured["headroom_instances"][0]
    assert builds["contextlab"]["context_middleware"] is captured["context_instances"][0]
    assert builds["contextlab"]["headroom_middleware"] is None
    assert captured["context_instances"][0].cache_ttl_seconds == 0
    assert all(item["cache_mode"] == "disabled" for item in captured["builds"])
    assert all(item["max_retries"] == 0 and item["model_name"] == "fake-comparison"
               for item in captured["model_kwargs"])
    manifests = [json.loads((root / result.session_id / "input-manifest.json").read_text()) for result in results]
    assert manifests[0] == manifests[1] == manifests[2]
    assert len(captured["graders"]) == 15
    assert all(argv == ["python", "-I", "-"] for _, argv, _ in captured["graders"])
    with LedgerStore(root / "ledger.sqlite") as store:
        hashes = [row[0] for row in store.connection.execute("SELECT config_hash FROM sessions")]
        assert len(set(hashes)) == 1
        for result in results:
            boundaries = [event for event in store.list_events(result.session_id) if event.kind == "phase_boundary"]
            assert len(boundaries) == (4 if result.arm == "contextlab" else 0)
            trace = json.loads((root / result.session_id / "trace.json").read_text())
            assert [row["phase_index"] for row in trace["requests"]] == [0, 1, 2, 3]
            assert all(phase["input_integrity"] is True for phase in trace["phases"])
            assert all(phase["independent_check"]["exit_code"] == 0 for phase in trace["phases"])


def test_phase_output_errors_are_diagnostic_and_do_not_leak_to_next_prompt(monkeypatch, tmp_path):
    captured = wire_comparison(monkeypatch, diagnostic_failure=True)
    root = tmp_path / "comparison"
    results = execute(root, comparison_config(arms=("baseline",)))
    assert results[0].solved
    trace = json.loads((root / results[0].session_id / "trace.json").read_text())
    assert trace["phases"][0]["independent_check"]["exit_code"] == 1
    assert len(trace["phases"]) == 4
    assert "PRIVATE_GRADER_DIAGNOSTIC" not in json.dumps(captured["phase_prompts"])


def test_initial_snapshot_tampering_cannot_be_hidden_by_phase_two_overwrite(monkeypatch, tmp_path):
    wire_comparison(monkeypatch, tamper_initial=True)
    root = tmp_path / "comparison"
    results = execute(root, comparison_config(arms=("baseline",)))
    assert not results[0].solved
    trace = json.loads((root / results[0].session_id / "trace.json").read_text())
    assert trace["phases"][0]["input_integrity"] is False
    assert trace["phases"][0]["input_integrity_failures"]
    assert all(phase["input_integrity"] is True for phase in trace["phases"][1:])
    assert trace["checks"]["independent"]["exit_code"] == 0


def test_final_independent_failure_overrides_model_phase_completion(monkeypatch, tmp_path):
    wire_comparison(monkeypatch, final_failure=True)
    results = execute(tmp_path / "comparison", comparison_config(arms=("baseline",)))
    assert not results[0].solved


def test_unknown_usage_stops_three_arm_comparison_without_more_counts(monkeypatch, tmp_path):
    captured = wire_comparison(monkeypatch, unknown_usage=True)
    results = execute(tmp_path / "comparison", comparison_config())
    assert len(results) == 1 and not results[0].solved
    assert captured["counts"] == captured["inferences"] == 1
    assert results[0].error == "BudgetExceeded" and results[0].cost_usd > 0
    assert not captured["graders"]


@pytest.mark.parametrize("cache_mode", ["enabled", "disabled"])
def test_agent_wiring_has_identical_tools_and_explicit_cache_policy(monkeypatch, tmp_path, cache_mode):
    import harness.agent as module
    captured = []
    monkeypatch.setattr(module, "create_agent", lambda **kwargs: captured.append(kwargs) or object())
    model = FakeMessagesListChatModel(responses=[AIMessage(content="done")])
    context, headroom = AgentMiddleware(), AgentMiddleware()
    hashes = []
    for arm in ("baseline", "headroom", "contextlab"):
        directory = tmp_path / arm
        directory.mkdir()
        workspace = Workspace(directory)
        hashes.append(invariant_hash(workspace, model_name="fixed", cache_mode=cache_mode))
        build_agent(model=model, workspace=workspace, arm=arm, cache_mode=cache_mode,
                    context_middleware=context if arm == "contextlab" else None,
                    headroom_middleware=headroom if arm == "headroom" else None)
    assert len(set(hashes)) == 1
    for index, arm in enumerate(("baseline", "headroom", "contextlab")):
        kwargs = captured[index]
        stack = kwargs["middleware"]
        assert kwargs["system_prompt"] == SYSTEM_PROMPT
        assert kwargs["model"] is model
        assert isinstance(stack[-1], MeteringMiddleware)
        assert sum(isinstance(item, AnthropicPromptCachingMiddleware) for item in stack) == (cache_mode == "enabled")
        assert (context in stack) == (arm == "contextlab")
        assert (headroom in stack) == (arm == "headroom")
    assert [[tool.name for tool in row["tools"]] for row in captured].count(
        [tool.name for tool in captured[0]["tools"]]) == 3


def test_comparison_approval_binds_cache_and_phase_policy(monkeypatch):
    import contextlab.adapters.headroom_mw as headroom_module
    monkeypatch.setattr(headroom_module, "configuration", lambda: {"profile": "offline test fixture"})
    config, rates = comparison_config(), prices()
    original = experiment_plan(config, rates, rates.model)
    assert len(original["matrix"]) == 3
    assert Decimal(original["maximum_authorized_cost_usd"]) == Decimal("6")
    for update in ({"cache_mode": "enabled"}, {"phase_boundaries": False}):
        changed = experiment_plan(config.model_copy(update=update), rates, rates.model)
        assert changed["approval_digest"] != original["approval_digest"]
