"""Bounded response and failure diagnostics; no API, Docker, or local-key reads."""
import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage

from contextlab.adapters.langchain_mw import (
    RESPONSE_SUMMARY_LIMIT_BYTES,
    ContextLabMiddleware,
    MeteringMiddleware,
    response_summary,
)
from contextlab.core.models import Pricing
from contextlab.ledger.store import LedgerStore
from harness.config import ExperimentConfig, experiment_plan
from harness.runner import run_live

USAGE = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}


def test_response_summary_retains_tools_and_stop_reason_without_headers_secrets_or_thinking(monkeypatch):
    secret = "fake-sensitive-api-credential"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    message = AIMessage(
        content=[{"type": "text", "text": f"Visible text {secret}"},
                 {"type": "thinking", "thinking": "private reasoning marker", "signature": "opaque"},
                 {"type": "image", "data": "opaque image data"}],
        tool_calls=[{"name": "recall", "id": "call-123", "type": "tool_call",
                     "args": {"chunk_id": "invented_chunk", "api_key": "another-secret",
                              "nonfinite": float("nan"),
                              "nested": {"Authorization": "Bearer other-secret"}}}],
        usage_metadata=USAGE,
        response_metadata={"stop_reason": "tool_use", "usage": {"input_tokens": 100},
                           "headers": {"x-private": "private header marker"}},
    )
    receipts = []
    meter = MeteringMiddleware(model_name="fake", on_usage=lambda *args: receipts.append(args[-1]))
    response = ModelResponse(result=[message])
    assert meter.wrap_model_call(SimpleNamespace(), lambda _: response) is response
    receipt = receipts[0]
    assert receipt["messages"][0]["usage_metadata"] == USAGE
    assert receipt["messages"][0]["provider_usage"] == {"input_tokens": 100}
    summary = receipt["response_summary"]
    assert summary["messages"][0]["stop_reason"] == "tool_use"
    call = summary["messages"][0]["tool_calls"][0]
    assert call["name"] == "recall" and call["id"] == "call-123"
    assert call["args"]["chunk_id"] == "invented_chunk"
    assert call["args"]["nonfinite"] == "[NON_FINITE]"
    json.dumps(summary, allow_nan=False)
    serialized = json.dumps(receipt)
    for excluded in (secret, "another-secret", "other-secret", "private reasoning marker",
                     "opaque image data", "private header marker"):
        assert excluded not in serialized
    assert message.content[0]["text"].endswith(secret)  # No model-response mutation.


def test_response_summary_is_bounded_for_large_content_and_arguments():
    calls = [{"name": "write_file", "id": f"call-{index}", "type": "tool_call",
              "args": {"path": "file.json", "content": "\u2603" * 10000}}
             for index in range(32)]
    message = AIMessage(content="\u2603" * 20000, tool_calls=calls,
                        response_metadata={"stop_reason": "max_tokens"})
    summary = response_summary([message] * 20)
    assert len(json.dumps(summary, ensure_ascii=False).encode()) <= RESPONSE_SUMMARY_LIMIT_BYTES
    assert summary["truncated"]
    assert summary["messages"][0]["stop_reason"] == "max_tokens"
    assert summary["messages"][0]["tool_calls"]


def test_diagnostic_extraction_failure_never_prevents_usage_callback(monkeypatch):
    def fail(_):
        raise ValueError("sensitive diagnostic exception")
    monkeypatch.setattr("contextlab.adapters.langchain_mw.response_summary", fail)
    calls = []
    meter = MeteringMiddleware(model_name="fake", on_usage=lambda *args: calls.append(args))
    meter.wrap_model_call(SimpleNamespace(), lambda _: ModelResponse(
        result=[AIMessage(content="done", usage_metadata=USAGE)]))
    assert calls[0][1].input_tokens == 100
    assert calls[0][-1]["response_summary"] == {"error_type": "ValueError"}
    assert "sensitive diagnostic exception" not in json.dumps(calls[0][-1])


def wire_runner(monkeypatch, mode):
    import langchain_anthropic

    import harness.agent

    observed = {"invocations": 0, "counts": 0, "responses": 0, "boundaries": []}

    class FakeModel:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            observed["counts"] += 1
            if mode == "count_failure":
                raise ValueError("sensitive exception must not be persisted")
            return 10001 if mode == "input_rejected" else 100

    class FakeExecutor:
        def __init__(self, *args):
            self.public_calls = 0

        def __call__(self, *args):
            self.public_calls += 1
            if mode == "check_failure" and self.public_calls == 3:
                raise OSError("sensitive exception must not be persisted")
            return {"exit_code": 0, "stdout": "CHECKS_PASSED"}

        def run(self, *args, **kwargs):
            return self()

    def build(**kwargs):
        meter = MeteringMiddleware(model_name="fake-diagnostic-model",
                                   before_call=kwargs["before_call"], on_usage=kwargs["on_usage"])

        class Agent:
            def invoke(self, state, config):
                phase = observed["invocations"]
                observed["invocations"] += 1
                request = SimpleNamespace(system_message=None, messages=state["messages"], tools=[])

                def respond(_):
                    if mode == "inference_failure":
                        raise TimeoutError("sensitive exception must not be persisted")
                    observed["responses"] += 1
                    message = AIMessage(
                        content="done", usage_metadata=None if mode == "missing_usage" else USAGE,
                        response_metadata={"stop_reason": "tool_use" if phase else "end_turn"},
                        tool_calls=[{"id": "call-recall", "name": "recall", "type": "tool_call",
                                     "args": {"chunk_id": "invented-id"}}] if phase else [],
                    )
                    return ModelResponse(result=[message])

                response = meter.wrap_model_call(request, respond)
                if phase == 1 and mode == "invocation_failure":
                    raise ValueError("sensitive exception must not be persisted")
                return {"messages": [*state["messages"], *response.result]}

        return Agent()

    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr("harness.runner.DockerExecutor", FakeExecutor)
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: FakeModel())
    monkeypatch.setattr(harness.agent, "build_agent", build)
    monkeypatch.setattr(ContextLabMiddleware, "mark_boundary",
                        lambda self, name: observed["boundaries"].append(name))
    return observed


def run_fixture(monkeypatch, tmp_path, mode):
    observed = wire_runner(monkeypatch, mode)
    config = ExperimentConfig(arms=("contextlab",), tasks=("inventory",), repeats=1,
                              session_cap_usd="1", experiment_cap_usd="3", max_model_calls=5,
                              max_input_tokens=10000, max_output_tokens=100, threshold_tokens=2000,
                              threshold_keep=2, docker_image="fake-image", phase_boundaries=True)
    pricing = Pricing(model="fake-diagnostic-model", input_per_million=1, output_per_million=1,
                      cache_write_5m_per_million=1, cache_write_1h_per_million=1,
                      cache_read_per_million=1, source="test-only", verified_at="2026-10-10")
    plan = experiment_plan(config, pricing, pricing.model)
    monkeypatch.setattr("harness.runner.experiment_plan", lambda *args: plan)
    output = tmp_path / "experiment"
    results = run_live(output, config, pricing, pricing.model, plan["approval_digest"])
    trace = json.loads((output / results[0].session_id / "trace.json").read_text())
    assert "sensitive exception must not be persisted" not in json.dumps(trace)
    return observed, results[0], trace, output


def test_failed_invocation_preserves_completed_phase_and_last_tool_response(monkeypatch, tmp_path):
    observed, result, trace, _ = run_fixture(monkeypatch, tmp_path, "invocation_failure")
    assert result.error == "ValueError"
    assert [phase["completed"] for phase in trace["phases"]] == [True, False]
    assert observed["boundaries"] == ["inventory/phase-1"]
    assert trace["failure"]["stage"] == "agent_invocation"
    assert trace["failure"]["phase_index"] == 1
    assert trace["failure"]["stack"][-1]["function"] == "invoke"
    assert all(set(frame) == {"file", "function", "line"} for frame in trace["failure"]["stack"])
    last = trace["response_summaries"][-1]
    assert last["phase_index"] == 1 and last["model_call_seq"] == 1
    assert last["summary"]["messages"][0]["tool_calls"][0]["args"]["chunk_id"] == "invented-id"


@pytest.mark.parametrize("mode,stage", [("count_failure", "token_count"),
                                       ("input_rejected", "input_admission")])
def test_count_and_admission_failures_have_distinct_stages_without_inference(monkeypatch, tmp_path, mode, stage):
    observed, result, trace, _ = run_fixture(monkeypatch, tmp_path, mode)
    assert trace["failure"]["stage"] == stage and trace["failure"]["phase_index"] == 0
    assert trace["phases"][0]["completed"] is False
    assert observed["responses"] == 0 and not observed["boundaries"]
    assert result.cost_usd == 0 and trace["response_summaries"] == []


def test_provider_failure_reports_inference_origin_without_raw_exception(monkeypatch, tmp_path):
    _, result, trace, _ = run_fixture(monkeypatch, tmp_path, "inference_failure")
    assert trace["failure"]["stage"] == "usage_settlement"
    assert trace["failure"]["upstream_stage"] == "model_inference"
    assert trace["failure"]["provider_error_type"] == "TimeoutError"
    assert result.cost_usd > 0 and result.error == "BudgetExceeded"


def test_missing_usage_keeps_response_diagnostics_and_conservative_charge(monkeypatch, tmp_path):
    _, result, trace, output = run_fixture(monkeypatch, tmp_path, "missing_usage")
    assert trace["failure"]["upstream_stage"] == "usage_validation"
    assert trace["response_summaries"][0]["summary"]["messages"][0]["content"] == "done"
    with LedgerStore(output / "ledger.sqlite") as store:
        call = store.list_model_calls(result.session_id)[0]
        assert call.status == "unknown_usage" and call.cost_basis == "reservation"
        assert call.raw_usage["messages"][0]["usage_metadata"] is None
        assert call.cost_usd == result.cost_usd > 0


def test_final_check_failure_does_not_invent_another_incomplete_phase(monkeypatch, tmp_path):
    observed, _, trace, _ = run_fixture(monkeypatch, tmp_path, "check_failure")
    assert len(trace["phases"]) == 2 and all(phase["completed"] for phase in trace["phases"])
    assert len(observed["boundaries"]) == 2
    assert trace["failure"]["stage"] == "final_public_checks"
