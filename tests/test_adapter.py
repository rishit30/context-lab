from copy import deepcopy
from decimal import Decimal

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contextlab.adapters.langchain_mw import (
    ContextLabMiddleware,
    MeteringMiddleware,
    apply_projection,
    langchain_usage,
    normalize_messages,
)
from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing
from contextlab.core.validator import validate_messages
from contextlab.ledger.store import LedgerStore
from harness.agent import build_agent, invariant_payload
from harness.workspace import Workspace


def model():
    # Constructing clients and request payloads does not call the network.
    return ChatAnthropic(model="claude-sonnet-4-6", api_key="offline-test-placeholder", max_retries=0)


def history():
    return [HumanMessage(content="Fix the code"), AIMessage(
        content=[{"type": "thinking", "thinking": "opaque", "signature": "retain-verbatim"}],
        tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "t1", "type": "tool_call"}],
    ), ToolMessage(content="original", tool_call_id="t1", artifact={"keep": True})]


def test_roundtrip_only_tool_content_changes_and_opaque_blocks_survive():
    original = history()
    canonical = deepcopy(original)
    normalized = normalize_messages(original)
    validate_messages(normalized)
    normalized[-1]["content"][0]["content"] = "[archived c1]"
    projected = apply_projection(original, normalized)
    assert projected[-1].content == "[archived c1]"
    assert projected[-1].artifact == {"keep": True}
    assert projected[1].content == canonical[1].content
    assert original == canonical


def test_projection_cannot_change_assistant_instructions():
    original = history()
    normalized = normalize_messages(original)
    normalized[0]["content"] = "different task"
    with pytest.raises(ValueError, match="outside tool-result"):
        apply_projection(original, normalized)


def test_parallel_tool_results_group_for_anthropic_validation():
    messages = [HumanMessage(content="inspect"), AIMessage(content="", tool_calls=[
        {"name": "read_file", "args": {}, "id": "a", "type": "tool_call"},
        {"name": "read_file", "args": {}, "id": "b", "type": "tool_call"},
    ]), ToolMessage(content="a", tool_call_id="a"), ToolMessage(content="b", tool_call_id="b")]
    validate_messages(normalize_messages(messages))
    assert len(normalize_messages(messages)[-1]["content"]) == 2


def test_usage_subtracts_cache_from_langchain_total_without_double_counting():
    message = AIMessage(content="done", usage_metadata={
        "input_tokens": 650, "output_tokens": 30, "total_tokens": 680,
        "input_token_details": {"cache_read": 400, "cache_creation": 0,
                                "ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 50},
    })
    result = langchain_usage(message)
    assert result.input_tokens == 100
    assert result.cache_write_5m_tokens == 100
    assert result.cache_write_1h_tokens == 50
    assert result.total_input_tokens == 650


def test_meter_records_provider_failure_once_and_does_not_retry():
    calls = []
    meter = MeteringMiddleware(model_name="test", on_usage=lambda *args: calls.append(args))
    request = ModelRequest(model=model(), messages=[HumanMessage(content="hello")])

    def failing(_request):
        raise RuntimeError("secret must not reach logging callback")

    with pytest.raises(RuntimeError):
        meter.wrap_model_call(request, failing)
    assert len(calls) == 1 and calls[0][1] is None
    assert calls[0][3:] == ("failed", "RuntimeError", {})


def test_budget_rejection_happens_before_model_invocation():
    called = []

    def deny(_request):
        raise RuntimeError("cap reached")

    meter = MeteringMiddleware(model_name="test", before_call=deny)
    with pytest.raises(RuntimeError, match="cap"):
        meter.wrap_model_call(ModelRequest(model=model(), messages=[]), lambda request: called.append(request))
    assert not called and meter.calls == 0


def test_caching_tags_identical_system_tools_settings_without_network(tmp_path):
    from langchain_anthropic.chat_models import convert_to_anthropic_tool

    from harness.agent import SYSTEM_PROMPT, build_tools

    cached_requests = []
    for _arm in ("baseline", "threshold_edit", "contextlab"):
        request = ModelRequest(model=model(), messages=history(),
                               system_message=SystemMessage(content=SYSTEM_PROMPT),
                               tools=build_tools(Workspace(tmp_path)))
        AnthropicPromptCachingMiddleware(ttl="5m").wrap_model_call(
            request, lambda r: cached_requests.append(r) or ModelResponse(result=[AIMessage(content="ok")]))
    assert all(r.system_message == cached_requests[0].system_message for r in cached_requests)
    schemas = [[convert_to_anthropic_tool(t) for t in r.tools] for r in cached_requests]
    assert all(schema == schemas[0] for schema in schemas)
    assert all(r.model_settings == {"cache_control": {"type": "ephemeral", "ttl": "5m"}} for r in cached_requests)
    assert cached_requests[0].tools[-1].extras["cache_control"]["ttl"] == "5m"
    assert cached_requests[0].system_message.content[-1]["cache_control"]["ttl"] == "5m"


def test_arm_invariants_do_not_depend_on_workspace_or_recall(tmp_path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    baseline = Workspace(first)
    optimized = Workspace(second, recall_reader=lambda *_: "content")
    assert invariant_payload(baseline, model_name="fixed") == invariant_payload(optimized, model_name="fixed")


def test_unsupported_competitor_arm_is_not_faked(tmp_path):
    with pytest.raises(NotImplementedError):
        build_agent(model=model(), workspace=Workspace(tmp_path), arm="deepagents")


def test_adapter_falls_back_on_ingestion_failure_without_retrying_provider(tmp_path):
    store = LedgerStore(tmp_path / "ledger.db")
    rates = Pricing(model="test", input_per_million=Decimal("3"),
                    cache_write_5m_per_million=Decimal("3.75"),
                    cache_write_1h_per_million=Decimal("6"),
                    cache_read_per_million=Decimal("0.3"), output_per_million=Decimal("15"))
    # Session intentionally absent: immutable ledger ingestion fails.
    middleware = ContextLabMiddleware(controller=ContextController(), store=store,
                                      session_id="missing", pricing=rates, config=CompactionConfig())
    request = ModelRequest(model=model(), messages=history())
    captured = []
    middleware.wrap_model_call(request, lambda r: captured.append(r) or ModelResponse(result=[AIMessage(content="ok")]))
    assert captured == [request]
    assert middleware.failures


def test_ingestion_preserves_original_and_line_metadata(tmp_path):
    import json

    store = LedgerStore(tmp_path / "ledger.db")
    store.create_session("run", "contextlab", "task")
    rates = Pricing(model="test", input_per_million=1, cache_write_5m_per_million=1,
                    cache_write_1h_per_million=1, cache_read_per_million=1, output_per_million=1)
    middleware = ContextLabMiddleware(controller=ContextController(), store=store,
                                      session_id="run", pricing=rates, config=CompactionConfig())
    messages = history()
    text = json.dumps({"path": "a.py", "content": "text", "offset": 4, "limit": 2,
                       "total_lines": 10, "full_read": False, "sha256": "version"})
    messages[-1] = ToolMessage(content=text, tool_call_id="t1")
    middleware._ingest(messages)
    middleware._ingest(messages)
    assert len(middleware.events) == 3
    assert middleware.chunks[0].content == text
    assert middleware.chunks[0].metadata["start_line"] == 5
    assert middleware.chunks[0].metadata["end_line"] == 6
    recalled = json.loads(middleware.recall(middleware.chunks[0].id))
    assert recalled["text"] == text
    assert middleware.chunks[0].id in middleware.state.pinned_ids


def test_restart_restores_events_seen_and_recall_pins(tmp_path):
    from contextlab.core.models import ProjectionResult

    store = LedgerStore(tmp_path / "ledger.db")
    store.create_session("run", "contextlab", "task")
    rates = Pricing(model="test", input_per_million=1, cache_write_5m_per_million=1,
                    cache_write_1h_per_million=1, cache_read_per_million=1, output_per_million=1)
    options = dict(controller=ContextController(), store=store, session_id="run", pricing=rates, config=CompactionConfig())
    first = ContextLabMiddleware(**options)
    first._ingest(history())
    first.recall(first.chunks[0].id)
    store.record_projection("run", len(first.events), ProjectionResult(messages=normalize_messages(history()), state=first.state))
    second = ContextLabMiddleware(**options)
    second._ingest(history())
    assert len(second.events) == 3 and len(second.chunks) == 1
    assert second.chunks[0].id in second.state.pinned_ids


@pytest.mark.parametrize("arm", ["baseline", "threshold_edit", "contextlab"])
def test_real_langchain_graph_runs_tools_with_fake_model_offline(tmp_path, arm):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

    class ToolCapableFake(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    (tmp_path / "a.py").write_text("print('hello')\n")
    store = LedgerStore(tmp_path / "ledger.db")
    store.create_session("run", arm, "task")
    rates = Pricing(model="test", input_per_million=1, cache_write_5m_per_million=1,
                    cache_write_1h_per_million=1, cache_read_per_million=1, output_per_million=1)
    middleware = ContextLabMiddleware(controller=ContextController(), store=store, session_id="run",
                                      pricing=rates, config=CompactionConfig(), observe_only=arm != "contextlab")
    usage = {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}
    fake = ToolCapableFake(responses=[
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "read1", "type": "tool_call"}], usage_metadata=usage),
        AIMessage(content="Inspected the file.", usage_metadata=usage),
    ])
    calls = []
    agent = build_agent(model=fake, workspace=Workspace(tmp_path), arm=arm,
                        context_middleware=middleware, on_usage=lambda *args: calls.append(args))
    response = agent.invoke({"messages": [HumanMessage(content="Read a.py")]})
    assert response["messages"][-1].content == "Inspected the file."
    assert len(calls) == 2 and all(call[1].input_tokens == 20 for call in calls)
    assert calls[0][5]["messages"][0]["usage_metadata"] == usage
    assert len(middleware.chunks) == 1
    assert any(event.kind == "file_read" for event in middleware.events)
    assert not middleware.failures
