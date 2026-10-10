import sqlite3
from decimal import Decimal

import pytest

from contextlab.core.models import Chunk, Event, ProjectionResult, Signal, Usage, ViewState
from contextlab.ledger import LedgerConflict, LedgerStore, ModelCall


def populate(store, session="s1", content="one\ntwo\nthree\n"):
    store.create_session(session, "baseline", "task")
    event = Event(id="e1", session_id=session, seq=1, kind="file_read", content=content)
    chunk = Chunk(id="c1", session_id=session, event_id=event.id, tool_call_id="tool1",
                  kind="file_read", content=content, seq=1, refs=("path:src/a.py",))
    store.append_event(event)
    store.add_chunk(chunk)
    return event, chunk


def test_immutable_scoped_originals_survive_restart(tmp_path):
    path = tmp_path / "ledger.sqlite"
    with LedgerStore(path) as store:
        event, chunk = populate(store)
        populate(store, "s2", "other session\n")
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert store.append_event(event) is False
        assert store.add_chunk(chunk) is False
        assert store.get_chunk("s2", "c1").content == "other session\n"
        with pytest.raises(KeyError):
            store.get_chunk("nonexistent", "c1")
        with pytest.raises(LedgerConflict):
            store.append_event(event.model_copy(update={"content": "changed"}))
        with pytest.raises(LedgerConflict):
            store.append_event(event.model_copy(update={"id": "different-id"}))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store.connection.execute("UPDATE events SET content='changed'")
        result = store.recall("s1", "c1", 4, offset=1, limit=1)
        assert result.text == "two\n"
        assert result.next_offset == 2
        assert result.truncated
    with LedgerStore(path) as store:
        assert store.get_chunk("s1", "c1").content == "one\ntwo\nthree\n"
        assert store.get_chunk("s1", "c1").pinned
        assert not store.get_chunk("s2", "c1").pinned
        assert store.session_summary("s1")["recalls"] == 1


def test_recall_bounds_long_lines_and_session_binding(tmp_path):
    import json

    from contextlab.tools.recall import make_recall

    with LedgerStore(tmp_path / "db") as store:
        populate(store, content="x" * 24001 + "\nlast\n")
        first = store.recall("s1", "c1", 2)
        assert len(first.text) == 12000
        assert first.next_offset == 0
        second = store.recall("s1", "c1", 3, offset=first.next_offset,
                              char_offset=first.next_char_offset)
        third = store.recall("s1", "c1", 4, offset=second.next_offset,
                             char_offset=second.next_char_offset)
        assert first.text + second.text + third.text == store.get_chunk("s1", "c1").content
        bound = make_recall(store, "s1", lambda: 5)
        assert json.loads(bound("c1", offset=1))["text"] == "last\n"
        for kwargs in ({"limit": 201}, {"limit": 0}, {"offset": -1}, {"offset": True}, {"offset": 90}):
            with pytest.raises(ValueError):
                store.recall("s1", "c1", 9, **kwargs)
        with pytest.raises(KeyError):
            bound("/etc/passwd")


def test_recall_write_failure_rolls_back_pin_and_audit(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        populate(store)
        store.connection.executescript("""
            CREATE TRIGGER fail_pin BEFORE UPDATE OF pinned ON chunks BEGIN
            SELECT RAISE(ABORT, 'injected write failure'); END;
        """)
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            store.recall("s1", "c1", 2)
        assert store.session_summary("s1")["recalls"] == 0
        assert not store.get_chunk("s1", "c1").pinned
        assert not store.connection.in_transaction


def test_archive_write_failure_does_not_commit_partial_chunk(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        populate(store)
        chunk = store.get_chunk("s1", "c1").model_copy(update={"id": "new"})
        store.connection.executescript("""
            CREATE TRIGGER fail_ref BEFORE INSERT ON chunk_refs BEGIN
            SELECT RAISE(ABORT, 'injected ref failure'); END;
        """)
        with pytest.raises(sqlite3.IntegrityError):
            store.add_chunk(chunk)
        with pytest.raises(KeyError):
            store.get_chunk("s1", "new")


def test_projection_and_all_purpose_costs(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        populate(store)
        projection = ProjectionResult(messages=[{"role": "user", "content": "x"}],
            state=ViewState(epoch=1, levels={"c1": "hidden"}), rebuilt=True,
            signals=[Signal(chunk_id="c1", signal_type="superseded_read",
                            triggered_by_event_id="e1", reason="new read")])
        store.record_projection("s1", 2, projection)
        store.record_projection("s1", 2, projection)
        assert store.session_summary("s1")["rebuilds"] == 1
        for index, purpose in enumerate(("main", "decision", "delegate", "triage", "summary")):
            call = ModelCall(id=f"call-{index}", session_id="s1", seq=index, model="test",
                             purpose=purpose, usage=Usage(input_tokens=100), cost_usd=Decimal("0.1"),
                             status="failed" if purpose == "delegate" else "ok", cost_basis="simulation")
            assert store.record_model_call(call)
            assert not store.record_model_call(call)
        store.finish_session("s1", False)
        store.create_session("s2", "baseline", "task2")
        store.finish_session("s2", True)
        summary = store.experiment_summary()
        assert summary["cost_usd"] == Decimal("0.5")
        assert summary["cost_per_solved_task"] == Decimal("0.5")
        assert summary["pass_rate"] == 0.5
        assert store.session_summary("s1")["failed_calls"] == 1
        assert store.session_summary("s1")["usage"].input_tokens == 500


def test_no_solutions_is_undefined_and_not_zero(tmp_path):
    with LedgerStore(tmp_path / "db") as store:
        populate(store)
        store.finish_session("s1", False)
        assert store.experiment_summary()["cost_per_solved_task"] is None


def test_projection_state_persists_without_rebuild_and_includes_recalled_pins(tmp_path):
    path = tmp_path / "db"
    with LedgerStore(path) as store:
        populate(store)
        result = ProjectionResult(messages=[], state=ViewState(last_boundary_id="boundary-7"))
        store.record_projection("s1", 5, result)
        store.recall("s1", "c1", 6)
    with LedgerStore(path) as store:
        state = store.get_projection_state("s1")
        assert state.last_boundary_id == "boundary-7"
        assert state.pinned_ids == ("c1",)
        assert store.get_projection_state("s2") == ViewState()
        with pytest.raises(LedgerConflict):
            store.record_projection("s1", 4, result)
