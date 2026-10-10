"""Recoverable tool inputs through the real LangChain graph, without API calls."""
import json
import sqlite3

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from contextlab.adapters.langchain_mw import ContextLabMiddleware, normalize_messages
from contextlab.core.controller import ContextController
from contextlab.core.models import Chunk, CompactionConfig, Event, Pricing
from contextlab.core.signals import derive_signals
from contextlab.core.validator import validate_messages
from contextlab.ledger.budget import BudgetExceeded
from contextlab.ledger.store import LedgerConflict, LedgerStore
from harness.agent import build_agent, build_tools
from harness.workspace import Workspace


class ToolCapableFake(FakeMessagesListChatModel):
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_response(name, args, call_id):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id, "type": "tool_call",
    }])


def invoke_tool(workspace, name, args):
    selected = next(item for item in build_tools(workspace) if item.name == name)
    return selected.invoke({"name": name, "args": args, "id": "call", "type": "tool_call"})


def observer(store, *, observe_only=False):
    return ContextLabMiddleware(
        controller=ContextController(), store=store, session_id="run",
        pricing=Pricing(model="fake", input_per_million=1, cache_write_5m_per_million=1,
                        cache_write_1h_per_million=1, cache_read_per_million=1, output_per_million=1),
        config=CompactionConfig(), cache_ttl_seconds=0, observe_only=observe_only,
    )


@pytest.mark.parametrize("arm", ["baseline", "threshold_edit", "contextlab"])
def test_real_graph_recovers_from_actual_invented_recall_id_and_writes_result(tmp_path, arm):
    root = tmp_path / "workspace"
    root.mkdir()
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("run", arm, "task")
        middleware = observer(store) if arm == "contextlab" else None
        workspace = Workspace(root, recall_reader=middleware.recall if middleware else None)
        fake = ToolCapableFake(responses=[
            tool_response("recall", {"chunk_id": "read_file|deployments/shard-a.json|0"}, "bad-recall"),
            tool_response("write_file", {"path": "result.json", "content": '{"recovered": true}'}, "write"),
            AIMessage(content="Recovered using the original conversation output."),
        ])
        agent = build_agent(model=fake, workspace=workspace, arm=arm,
                            context_middleware=middleware, cache_mode="disabled", max_model_calls=3)
        result = agent.invoke({"messages": [HumanMessage(content="Finish the audit.")]})
        outputs = [message for message in result["messages"] if isinstance(message, ToolMessage)]
        assert [message.status for message in outputs] == ["error", "success"]
        assert outputs[0].tool_call_id == "bad-recall"
        assert "do not invent IDs" in outputs[0].content
        assert "conversation history" in outputs[0].content
        assert "deployments/shard-a.json" not in outputs[0].content
        assert any(isinstance(message, ToolMessage) and message.status == "error"
                   for message in fake.requests[1])
        assert json.loads((root / "result.json").read_text()) == {"recovered": True}
        assert len(fake.requests) == 3
        validate_messages(normalize_messages(result["messages"]))


@pytest.mark.parametrize("name,args", [
    ("recall", {"chunk_id": "nonexistent-valid-id"}),
    ("read_file", {"path": "/context/nonexistent-valid-id.txt"}),
])
def test_unknown_well_formed_archive_id_is_a_sanitized_error(tmp_path, name, args):
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("run", "contextlab", "task")
        middleware = observer(store)
        output = invoke_tool(Workspace(tmp_path, recall_reader=middleware.recall), name, args)
        assert output.status == "error"
        assert "supplied /context/<id>.txt pointer" in output.content
        assert "nonexistent-valid-id" not in output.content
        assert store.session_summary("run")["recalls"] == 0


def test_correct_recall_succeeds_and_persists_pin_and_continuation(tmp_path):
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("run", "contextlab", "task")
        event = Event(id="e0", session_id="run", seq=0, kind="file_read", content="original\nnext\n")
        store.append_event(event)
        store.add_chunk(Chunk(id="c0", session_id="run", event_id="e0", tool_call_id="initial",
                              kind="file_read", seq=0, content=event.content))
        middleware = observer(store)
        workspace = Workspace(tmp_path, recall_reader=middleware.recall)
        first = invoke_tool(workspace, "recall", {"chunk_id": "c0", "limit": 1})
        assert first.status == "success"
        page = json.loads(first.content)
        assert page["text"] == "original\n" and page["next_offset"] == 1
        second = invoke_tool(workspace, "read_file", {"path": "/context/c0.txt", "offset": 1})
        assert second.status == "success" and json.loads(second.content)["text"] == "next\n"
        assert store.get_chunk("run", "c0").pinned
        assert store.session_summary("run")["recalls"] == 2
        beyond = invoke_tool(workspace, "recall", {"chunk_id": "c0", "offset": 50})
        assert beyond.status == "error"
        assert store.session_summary("run")["recalls"] == 2


def test_missing_read_recovers_and_error_is_not_superseded_read_evidence(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("run", "contextlab", "task")
        middleware = observer(store)
        fake = ToolCapableFake(responses=[
            tool_response("read_file", {"path": "result.txt"}, "missing-read"),
            tool_response("write_file", {"path": "result.txt", "content": "created"}, "write"),
            tool_response("read_file", {"path": "result.txt"}, "valid-read"),
            AIMessage(content="Created and checked the result."),
        ])
        agent = build_agent(model=fake, workspace=Workspace(root), arm="contextlab",
                            context_middleware=middleware, cache_mode="disabled", max_model_calls=4)
        result = agent.invoke({"messages": [HumanMessage(content="Read or create result.txt.")]})
        error = next(message for message in result["messages"]
                     if isinstance(message, ToolMessage) and message.tool_call_id == "missing-read")
        assert error.status == "error" and str(root) not in error.content
        event = next(event for event in middleware.events if event.tool_call_id == "missing-read")
        chunk = next(chunk for chunk in middleware.chunks if chunk.tool_call_id == "missing-read")
        assert event.kind == chunk.kind == "other"
        assert event.metadata["tool_error"] is True
        assert not any(signal.chunk_id == chunk.id
                       for signal in derive_signals(middleware.events, middleware.chunks))
        assert not middleware.failures
        assert (root / "result.txt").read_text() == "created"


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "sensitive-invalid-path", "offset": -1}),
    ("read_file", {"path": "sensitive-invalid-path", "limit": 2001}),
    ("read_file", {"path": "sensitive-invalid-path", "offset": True}),
    ("recall", {"chunk_id": "sensitive-invalid-id", "offset": -1}),
    ("recall", {"chunk_id": "sensitive-invalid-id", "limit": 0}),
    ("recall", {"chunk_id": "sensitive-invalid-id", "limit": 201}),
    ("recall", {"chunk_id": "sensitive-invalid-id", "char_offset": -1}),
    ("recall", {"chunk_id": "sensitive-invalid-id", "limit": "200"}),
])
def test_schema_validation_is_recoverable_without_echoing_arguments(tmp_path, monkeypatch, name, args):
    workspace = Workspace(tmp_path)
    monkeypatch.setattr(workspace, name, lambda *args: pytest.fail("Invalid bounds reached workspace"))
    output = invoke_tool(workspace, name, args)
    assert output.status == "error"
    assert "Invalid tool arguments" in output.content
    assert "sensitive-invalid" not in output.content


def test_provider_tool_schema_exposes_integer_read_and_recall_bounds(tmp_path):
    schemas = {item.name: convert_to_openai_tool(item)["function"]["parameters"]["properties"]
               for item in build_tools(Workspace(tmp_path))}
    for name, maximum in (("read_file", 2000), ("recall", 200)):
        assert schemas[name]["offset"]["type"] == "integer"
        assert schemas[name]["offset"]["minimum"] == 0
        assert schemas[name]["limit"]["minimum"] == 1
        assert schemas[name]["limit"]["maximum"] == maximum
    assert schemas["recall"]["char_offset"]["minimum"] == 0


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "../outside"}),
    ("write_file", {"path": ".env", "content": "unused"}),
    ("edit_file", {"path": "missing.txt", "old_string": "old", "new_string": "new"}),
    ("search", {"query": ""}),
    ("shell", {"operation": "invalid-sensitive-command"}),
])
def test_expected_workspace_input_errors_are_recoverable(tmp_path, name, args):
    output = invoke_tool(Workspace(tmp_path), name, args)
    assert output.status == "error"
    assert "invalid-sensitive-command" not in output.content


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "binary.txt"}),
    ("edit_file", {"path": "binary.txt", "old_string": "a", "new_string": "b"}),
])
def test_utf8_decode_errors_return_safe_input_guidance(tmp_path, name, args):
    (tmp_path / "binary.txt").write_bytes(b"\xffsensitive-binary-content")
    output = invoke_tool(Workspace(tmp_path), name, args)
    assert output.status == "error" and "UTF-8" in output.content
    assert "sensitive-binary-content" not in output.content


@pytest.mark.parametrize("failure", [
    LedgerConflict("Invalid chunk ID"), BudgetExceeded("budget exhausted"),
    sqlite3.OperationalError("ledger unavailable"), PermissionError("access denied"),
    FileNotFoundError("ledger storage missing"), ValueError("unexpected ledger failure"),
    KeyError("Unknown session"), ToolException("unexpected external tool failure"),
])
def test_real_graph_does_not_mask_infrastructure_or_unknown_recall_errors(tmp_path, failure):
    def broken_reader(*args):
        raise failure

    fake = ToolCapableFake(responses=[
        tool_response("recall", {"chunk_id": "c0"}, "recall"), AIMessage(content="must not run"),
    ])
    agent = build_agent(model=fake, workspace=Workspace(tmp_path, recall_reader=broken_reader),
                        arm="baseline", cache_mode="disabled", max_model_calls=2)
    with pytest.raises(type(failure)) as caught:
        agent.invoke({"messages": [HumanMessage(content="Recall the supplied archive.")]})
    assert caught.value is failure
    assert len(fake.requests) == 1


@pytest.mark.parametrize("failure", [
    ValueError("Invalid chunk ID"), FileNotFoundError("docker missing"),
    PermissionError("docker denied"), RuntimeError("sandbox unavailable"),
    KeyError("exit_code"), BudgetExceeded("budget exhausted"),
])
def test_executor_errors_propagate_even_if_type_or_message_resembles_input_error(tmp_path, failure):
    (tmp_path / "tests").mkdir()
    def executor(*args):
        raise failure

    with pytest.raises(type(failure)) as caught:
        invoke_tool(Workspace(tmp_path, executor=executor), "run_tests", {"selection": "tests"})
    assert caught.value is failure


def test_invalid_test_selection_is_recoverable_before_executor(tmp_path):
    def executor(*args):
        pytest.fail("Bad test selection reached the executor")

    output = invoke_tool(Workspace(tmp_path, executor=executor), "run_tests", {"selection": "missing"})
    assert output.status == "error" and "existing relative test" in output.content
