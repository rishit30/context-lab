"""Real archive/adapter projection needs no fake provider or synthesized response."""

import hashlib
from copy import deepcopy

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from contextlab.adapters.langchain_mw import ContextLabMiddleware, normalize_messages
from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing
from contextlab.core.validator import validate_messages
from contextlab.ledger.store import LedgerStore
from harness.workspace import Workspace


def options(store):
    return dict(
        controller=ContextController(),
        store=store,
        session_id="s",
        pricing=Pricing(
            model="offline",
            input_per_million=3,
            cache_write_5m_per_million=3,
            cache_write_1h_per_million=3,
            cache_read_per_million=3,
            output_per_million=15,
        ),
        config=CompactionConfig(recent_outputs=1),
        cache_ttl_seconds=0,
    )


def repeated_history(tmp_path):
    workspace = Workspace(tmp_path)
    (tmp_path / "a.txt").write_text("same file line\n" * 1600)
    output = workspace.read_file("a.txt")
    messages = [HumanMessage(content="Inspect the same file twice.")]
    for call_id in ("first", "second"):
        messages.extend(
            [
                AIMessage(
                    content=[{"type": "thinking", "thinking": "opaque", "signature": "keep"}],
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "a.txt"},
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(content=output, tool_call_id=call_id, artifact={"local": "preserve"}),
            ]
        )
    return messages


def test_public_projection_compacts_real_workspace_output_and_survives_restart(tmp_path):
    store = LedgerStore(tmp_path / "ledger.sqlite")
    store.create_session("s", "contextlab", "offline")
    middleware = ContextLabMiddleware(**options(store))
    messages = repeated_history(tmp_path)
    original = deepcopy(messages)
    middleware.observe(messages)
    middleware.observe(messages)
    assert len(middleware.events) == 5 and len(middleware.chunks) == 2
    assert middleware.telemetry == [] and store.list_model_calls("s") == []
    boundary = middleware.mark_boundary("inspection-complete")
    result = middleware.project_messages(messages)
    first, second = middleware.chunks
    assert result[2].content != original[2].content
    assert result[4] == original[4] and result[1] == original[1]
    assert result[2].artifact == original[2].artifact
    assert middleware.state.duplicate_sources == {first.id: second.id}
    assert middleware.telemetry[-1]["exact_duplicate_retirements"] == {first.id: second.id}
    assert middleware.telemetry[-1]["rebuilt"]
    assert store.get_projection_state("s").duplicate_sources == {first.id: second.id}
    assert store.list_model_calls("s") == [] and messages == original
    assert store.list_chunks("s")[0].content == original[2].content
    validate_messages(normalize_messages(result))
    # A separate adapter restores the archive and proof from SQLite, without a
    # provider response or another semantic/model invocation.
    restarted = ContextLabMiddleware(**options(LedgerStore(tmp_path / "ledger.sqlite")))
    again = restarted.project_messages(messages)
    assert again == result and restarted.state.last_boundary_id == boundary.id
    assert len(restarted.events) == 6 and len(restarted.chunks) == 2
    assert restarted.telemetry[-1]["no_op_reason"] == "boundary_already_considered"
    assert not restarted.telemetry[-1]["rebuilt"]
    # Recall pins are immediate: a repeated boundary cannot keep the original hidden.
    restarted.recall(first.id)
    restored = restarted.project_messages(messages)
    assert restored == original and restarted.state.duplicate_sources == {}
    assert restarted.store.get_projection_state("s").levels[first.id] == "full"
    assert (
        hashlib.sha256(restarted.store.list_chunks("s")[0].content.encode()).digest()
        == hashlib.sha256(original[2].content.encode()).digest()
    )


def test_public_projection_failure_returns_original_and_never_synthesizes_response(tmp_path):
    store = LedgerStore(tmp_path / "missing-session.sqlite")
    middleware = ContextLabMiddleware(**options(store))
    messages = [HumanMessage(content="actual request")]
    projected = middleware.project_messages(messages)
    assert projected == messages and len(projected) == 1
    assert middleware.telemetry[-1]["fallback"]
    assert middleware.telemetry[-1]["error_type"] == "IntegrityError"
    assert store.list_model_calls("s") == []
