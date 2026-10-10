"""Durable, session-scoped originals and an auditable usage ledger.

Snapshots live inside SQLite so an archived original and its metadata commit or
roll back together. The sqlite: references are internal identifiers, not paths.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Literal

from pydantic import Field

from contextlab.core.models import Chunk, Event, Pricing, ProjectionResult, Record, Usage, ViewState
from contextlab.ledger.pricing import money, price_usage


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


class LedgerConflict(ValueError):
    """An immutable identifier was reused for different content."""


class ModelCall(Record):
    id: str
    session_id: str
    seq: int = Field(ge=0)
    model: str
    purpose: Literal["main", "delegate", "decision", "triage", "summary"] = "main"
    usage: Usage = Field(default_factory=Usage)
    cost_usd: Decimal = Field(ge=0)
    latency_ms: float = Field(default=0, ge=0)
    cache_ttl: str = "5m"
    raw_usage: dict[str, Any] = Field(default_factory=dict)
    pricing: Pricing | None = None
    status: Literal["ok", "failed", "unknown_usage"] = "ok"
    cost_basis: Literal["usage", "reservation", "simulation"] = "usage"
    error: str | None = None


class RecallResult(Record):
    chunk_id: str
    text: str
    offset: int
    next_offset: int
    next_char_offset: int = 0
    total_lines: int
    truncated: bool
    tokens_returned: int


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
 id TEXT PRIMARY KEY, arm TEXT NOT NULL, task_id TEXT NOT NULL, run_idx INTEGER NOT NULL,
 model_main TEXT NOT NULL, config_hash TEXT NOT NULL, started_at TEXT NOT NULL,
 ended_at TEXT, solved INTEGER, notes TEXT NOT NULL DEFAULT '',
 experiment_id TEXT NOT NULL DEFAULT 'default', mode TEXT NOT NULL DEFAULT 'simulation'
);
CREATE TABLE IF NOT EXISTS events (
 id TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id), seq INTEGER NOT NULL,
 role TEXT NOT NULL, kind TEXT NOT NULL, tool_name TEXT NOT NULL, tool_call_id TEXT NOT NULL,
 args_json TEXT NOT NULL, content_ref TEXT NOT NULL, size_tokens INTEGER NOT NULL,
 created_at TEXT NOT NULL, content TEXT NOT NULL, record_json TEXT NOT NULL,
 PRIMARY KEY(session_id,id), UNIQUE(session_id,seq)
);
CREATE TABLE IF NOT EXISTS chunks (
 id TEXT NOT NULL, session_id TEXT NOT NULL, event_id TEXT NOT NULL, kind TEXT NOT NULL,
 size_tokens INTEGER NOT NULL, full_text_ref TEXT NOT NULL, long_text_ref TEXT NOT NULL,
 short_text_ref TEXT NOT NULL, signpost_text TEXT NOT NULL, record_json TEXT NOT NULL,
 content_sha256 TEXT NOT NULL, pinned INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(session_id,id),
 FOREIGN KEY(session_id,event_id) REFERENCES events(session_id,id)
);
CREATE TABLE IF NOT EXISTS chunk_refs (
 chunk_id TEXT NOT NULL, ref_type TEXT NOT NULL, ref_value TEXT NOT NULL, session_id TEXT NOT NULL,
 PRIMARY KEY(session_id,chunk_id,ref_type,ref_value),
 FOREIGN KEY(session_id,chunk_id) REFERENCES chunks(session_id,id)
);
CREATE TABLE IF NOT EXISTS signals (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), chunk_id TEXT NOT NULL,
 signal_type TEXT NOT NULL, triggered_by_event_id TEXT NOT NULL, created_at TEXT NOT NULL,
 reason TEXT NOT NULL, target TEXT NOT NULL,
 FOREIGN KEY(session_id,chunk_id) REFERENCES chunks(session_id,id)
);
CREATE TABLE IF NOT EXISTS epochs (
 id INTEGER NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id), started_at_seq INTEGER NOT NULL,
 prefix_hash TEXT NOT NULL, prefix_tokens INTEGER NOT NULL, reason TEXT NOT NULL,
 PRIMARY KEY(session_id,id)
);
CREATE TABLE IF NOT EXISTS visibility (
 chunk_id TEXT NOT NULL, epoch_id INTEGER NOT NULL, level TEXT NOT NULL, session_id TEXT NOT NULL,
 PRIMARY KEY(session_id,chunk_id,epoch_id),
 FOREIGN KEY(session_id,chunk_id) REFERENCES chunks(session_id,id),
 FOREIGN KEY(session_id,epoch_id) REFERENCES epochs(session_id,id)
);
CREATE TABLE IF NOT EXISTS decisions (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), seq INTEGER NOT NULL,
 decision_type TEXT NOT NULL, engine TEXT NOT NULL, inputs_json TEXT NOT NULL, output_json TEXT NOT NULL,
 confidence REAL, latency_ms REAL NOT NULL, cost_usd TEXT NOT NULL, accepted INTEGER NOT NULL,
 fallback_used INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recalls (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL, chunk_id TEXT NOT NULL, requested_at_seq INTEGER NOT NULL,
 hidden_since_epoch INTEGER, tokens_returned INTEGER NOT NULL, offset_lines INTEGER NOT NULL,
 limit_lines INTEGER NOT NULL, char_offset INTEGER NOT NULL, created_at TEXT NOT NULL,
 FOREIGN KEY(session_id,chunk_id) REFERENCES chunks(session_id,id)
);
CREATE TABLE IF NOT EXISTS model_calls (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), seq INTEGER NOT NULL,
 model TEXT NOT NULL, purpose TEXT NOT NULL, input_tokens INTEGER NOT NULL,
 cache_write_tokens INTEGER NOT NULL, cache_read_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
 cost_usd TEXT NOT NULL, latency_ms REAL NOT NULL, cache_ttl TEXT NOT NULL,
 cache_write_5m_tokens INTEGER NOT NULL, cache_write_1h_tokens INTEGER NOT NULL,
 raw_usage_json TEXT NOT NULL, pricing_json TEXT, status TEXT NOT NULL, cost_basis TEXT NOT NULL,
 error TEXT, record_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS delegations (
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), parent_seq INTEGER NOT NULL,
 subtask_kind TEXT NOT NULL, model TEXT NOT NULL, brief_tokens INTEGER NOT NULL,
 result_chunk_id TEXT, status TEXT NOT NULL, escalated_to TEXT, cost_usd TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
 id TEXT PRIMARY KEY, scope TEXT NOT NULL, entity TEXT NOT NULL, property TEXT NOT NULL, value TEXT NOT NULL,
 unit TEXT, record_type TEXT NOT NULL, source_event_ids TEXT NOT NULL, effective_from TEXT,
 supersedes_fact_id TEXT, version INTEGER NOT NULL, status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budgets (
 scope TEXT NOT NULL, scope_id TEXT NOT NULL, cap_usd TEXT NOT NULL, blocked INTEGER NOT NULL DEFAULT 0,
 PRIMARY KEY(scope,scope_id)
);
CREATE TABLE IF NOT EXISTS reservations (
 call_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), experiment_id TEXT NOT NULL,
 model TEXT NOT NULL, purpose TEXT NOT NULL, seq INTEGER NOT NULL, amount_usd TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'reserved', created_at TEXT NOT NULL, settled_at TEXT
);
CREATE TABLE IF NOT EXISTS projection_states (
 session_id TEXT PRIMARY KEY REFERENCES sessions(id), seq INTEGER NOT NULL, record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_order ON events(session_id,seq);
CREATE INDEX IF NOT EXISTS calls_session ON model_calls(session_id);
CREATE INDEX IF NOT EXISTS reservations_experiment ON reservations(experiment_id,status);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN
 SELECT RAISE(ABORT, 'original events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN
 SELECT RAISE(ABORT, 'original events are immutable'); END;
CREATE TRIGGER IF NOT EXISTS chunks_no_content_update BEFORE UPDATE OF
 record_json,content_sha256,full_text_ref,long_text_ref,short_text_ref,signpost_text,event_id ON chunks BEGIN
 SELECT RAISE(ABORT, 'original chunks are immutable'); END;
CREATE TRIGGER IF NOT EXISTS chunks_no_delete BEFORE DELETE ON chunks BEGIN
 SELECT RAISE(ABORT, 'original chunks are immutable'); END;
"""


class LedgerStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(_SCHEMA)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> LedgerStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise

    def create_session(self, session_id: str, arm: str, task_id: str, run_idx: int = 0,
                       model_main: str = "", config_hash: str = "", experiment_id: str = "default",
                       mode: str = "simulation", notes: str = "") -> None:
        if mode not in {"simulation", "live"}:
            raise ValueError("Session mode must be simulation or live")
        with self.transaction() as db:
            old = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            expected = (arm, task_id, run_idx, model_main, config_hash, experiment_id, mode)
            if old:
                actual = tuple(old[key] for key in ("arm", "task_id", "run_idx", "model_main", "config_hash", "experiment_id", "mode"))
                if actual != expected:
                    raise LedgerConflict("Session already exists with different configuration")
                return
            db.execute("INSERT INTO sessions(id,arm,task_id,run_idx,model_main,config_hash,started_at,notes,experiment_id,mode) VALUES(?,?,?,?,?,?,?,?,?,?)",
                       (session_id, arm, task_id, run_idx, model_main, config_hash, now(), notes, experiment_id, mode))

    def finish_session(self, session_id: str, solved: bool, notes: str = "") -> None:
        with self.transaction() as db:
            cursor = db.execute("UPDATE sessions SET ended_at=?,solved=?,notes=? WHERE id=?", (now(), int(solved), notes, session_id))
            if cursor.rowcount != 1:
                raise KeyError("Unknown session")

    def append_event(self, event: Event) -> bool:
        serialized = _json(event)
        with self.transaction() as db:
            old = db.execute("SELECT record_json FROM events WHERE session_id=? AND (id=? OR seq=?)", (event.session_id, event.id, event.seq)).fetchone()
            if old:
                if old[0] != serialized:
                    raise LedgerConflict("Event id or sequence conflicts with immutable history")
                return False
            db.execute("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                event.id, event.session_id, event.seq, event.role, event.kind, event.tool_name,
                event.tool_call_id, _json(event.args), f"sqlite:event/{event.session_id}/{event.id}",
                event.size_tokens, event.created_at or now(), event.content, serialized,
            ))
        return True

    def add_chunk(self, chunk: Chunk, full_text: str | None = None) -> bool:
        if full_text is not None and full_text != chunk.content:
            raise ValueError("full_text disagrees with chunk content")
        serialized = _json(chunk)
        with self.transaction() as db:
            old = db.execute("SELECT record_json FROM chunks WHERE session_id=? AND id=?", (chunk.session_id, chunk.id)).fetchone()
            if old:
                if old[0] != serialized:
                    raise LedgerConflict("Chunk id conflicts with immutable original")
                return False
            ref = f"sqlite:chunk/{chunk.session_id}/{chunk.id}"
            db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                chunk.id, chunk.session_id, chunk.event_id, chunk.kind, chunk.size_tokens,
                ref + "/full", ref + "/long", ref + "/short", chunk.signpost_text, serialized,
                hashlib.sha256(chunk.content.encode()).hexdigest(), int(chunk.pinned),
            ))
            for ref_value in chunk.refs:
                kind, _, value = ref_value.partition(":")
                if kind not in {"path", "symbol", "error_sig", "test_id", "search_query"} or not value:
                    kind, value = "path", ref_value
                db.execute("INSERT OR IGNORE INTO chunk_refs VALUES(?,?,?,?)", (chunk.id, kind, value, chunk.session_id))
        return True

    def list_events(self, session_id: str) -> list[Event]:
        rows = self.connection.execute("SELECT record_json FROM events WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
        return [Event.model_validate_json(row[0]) for row in rows]

    def get_chunk(self, session_id: str, chunk_id: str) -> Chunk:
        row = self.connection.execute("SELECT record_json,pinned FROM chunks WHERE session_id=? AND id=?", (session_id, chunk_id)).fetchone()
        if row is None:
            raise KeyError("Chunk not found in this session")
        return Chunk.model_validate_json(row[0]).model_copy(update={"pinned": bool(row[1])})

    def list_chunks(self, session_id: str) -> list[Chunk]:
        rows = self.connection.execute("SELECT id FROM chunks WHERE session_id=? ORDER BY rowid", (session_id,)).fetchall()
        return [self.get_chunk(session_id, row[0]) for row in rows]

    def recall(self, session_id: str, chunk_id: str, seq: int, offset: int = 0, limit: int = 80,
               char_offset: int = 0) -> RecallResult:
        for name, value in (("offset", offset), ("limit", limit), ("seq", seq), ("char_offset", char_offset)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not 1 <= limit <= 200:
            raise ValueError("limit must be between 1 and 200 lines")
        with self.transaction() as db:
            chunk = self.get_chunk(session_id, chunk_id)
            lines = chunk.content.splitlines(keepends=True)
            if offset > len(lines) or (char_offset and (offset == len(lines) or char_offset >= len(lines[offset]))):
                raise ValueError("Recall offset is outside the archived content")
            parts: list[str] = []
            remaining, next_line, next_char = 12000, offset, char_offset
            for index in range(offset, min(offset + limit, len(lines))):
                start = char_offset if index == offset else 0
                portion = lines[index][start:]
                if len(portion) > remaining:
                    parts.append(portion[:remaining])
                    next_line, next_char = index, start + remaining
                    break
                parts.append(portion)
                remaining -= len(portion)
                next_line, next_char = index + 1, 0
                if remaining == 0:
                    break
            text = "".join(parts)
            tokens = (len(text) + 3) // 4  # diagnostic estimate, never billed usage
            hidden = db.execute("SELECT MIN(epoch_id) FROM visibility WHERE session_id=? AND chunk_id=? AND level!='full'", (session_id, chunk_id)).fetchone()[0]
            db.execute("INSERT INTO recalls VALUES(?,?,?,?,?,?,?,?,?,?)", (
                str(uuid.uuid4()), session_id, chunk_id, seq, hidden, tokens, offset, limit, char_offset, now(),
            ))
            db.execute("UPDATE chunks SET pinned=1 WHERE session_id=? AND id=?", (session_id, chunk_id))
            return RecallResult(chunk_id=chunk_id, text=text, offset=offset, next_offset=next_line,
                                next_char_offset=next_char, total_lines=len(lines),
                                truncated=next_line < len(lines), tokens_returned=tokens)

    def record_projection(self, session_id: str, seq: int, result: ProjectionResult) -> None:
        with self.transaction() as db:
            previous = db.execute("SELECT seq FROM projection_states WHERE session_id=?", (session_id,)).fetchone()
            if previous is not None and previous[0] > seq:
                raise LedgerConflict("Cannot replace a newer projection with an older one")
            db.execute("INSERT INTO projection_states VALUES(?,?,?) ON CONFLICT(session_id) DO UPDATE SET seq=excluded.seq,record_json=excluded.record_json",
                       (session_id, seq, _json(result)))
            for signal in result.signals:
                signal_id = hashlib.sha256(_json([session_id, signal.model_dump()]).encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?)", (
                    signal_id, session_id, signal.chunk_id, signal.signal_type,
                    signal.triggered_by_event_id, now(), signal.reason, signal.target))
            if result.rebuilt:
                rendered = _json(result.messages)
                values = (result.state.epoch, session_id, seq, hashlib.sha256(rendered.encode()).hexdigest(),
                          (len(rendered) + 3) // 4, result.decision.reason if result.decision else "rebuild")
                old = db.execute("SELECT * FROM epochs WHERE session_id=? AND id=?", (session_id, result.state.epoch)).fetchone()
                if old and tuple(old) != values:
                    raise LedgerConflict("Epoch prefix cannot change after being recorded")
                db.execute("INSERT OR IGNORE INTO epochs VALUES(?,?,?,?,?,?)", values)
                for chunk_id, level in result.state.levels.items():
                    db.execute("INSERT OR IGNORE INTO visibility VALUES(?,?,?,?)", (chunk_id, result.state.epoch, level, session_id))
            if result.decision or result.fallback:
                identifier = hashlib.sha256(_json([session_id, seq, result.model_dump(mode="json")]).encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                    identifier, session_id, seq, "compaction", "rules", "{}",
                    _json({"decision": result.decision.model_dump(mode="json") if result.decision else None,
                           "error": result.error, "rebuilt": result.rebuilt,
                           "estimated_tokens_saved": result.estimated_tokens_saved}),
                    None, 0, "0", int(bool(result.decision and result.decision.accepted)), int(result.fallback)))
            if result.semantic_decision is not None:
                payload = result.semantic_decision
                identifier = hashlib.sha256(_json([
                    session_id, seq, "semantic_relevance", payload,
                ]).encode()).hexdigest()
                # This row describes the assessment. All financial charges live in
                # model_calls, including failed/unknown-usage decision attempts.
                db.execute("INSERT OR IGNORE INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                    identifier, session_id, seq, "semantic_relevance",
                    str(payload.get("engine", "jev")), "{}", _json(payload), None, 0, "0",
                    int(bool(payload.get("allowed_ids"))), int(bool(payload.get("fallback")))))

    def get_projection(self, session_id: str) -> ProjectionResult | None:
        row = self.connection.execute("SELECT record_json FROM projection_states WHERE session_id=?", (session_id,)).fetchone()
        return ProjectionResult.model_validate_json(row[0]) if row else None

    def get_projection_state(self, session_id: str) -> ViewState:
        projection = self.get_projection(session_id)
        state = projection.state if projection else ViewState()
        pinned = [row[0] for row in self.connection.execute("SELECT id FROM chunks WHERE session_id=? AND pinned=1 ORDER BY id", (session_id,))]
        return state.model_copy(update={"pinned_ids": tuple(sorted(set(state.pinned_ids).union(pinned)))})

    def _insert_call(self, db: sqlite3.Connection, call: ModelCall) -> bool:
        if call.pricing is not None and call.cost_basis in {"usage", "simulation"}:
            if call.model != call.pricing.model:
                raise ValueError("Call model and rate-card model differ")
            if money(call.cost_usd) != price_usage(call.usage, call.pricing):
                raise ValueError("Call cost disagrees with its usage and rate snapshot")
        serialized = _json(call)
        old = db.execute("SELECT record_json FROM model_calls WHERE id=?", (call.id,)).fetchone()
        if old:
            if old[0] != serialized:
                raise LedgerConflict("Model call already recorded with different data")
            return False
        u = call.usage
        db.execute("INSERT INTO model_calls VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            call.id, call.session_id, call.seq, call.model, call.purpose, u.input_tokens,
            u.cache_write_5m_tokens + u.cache_write_1h_tokens, u.cache_read_tokens, u.output_tokens,
            str(money(call.cost_usd)), call.latency_ms, call.cache_ttl, u.cache_write_5m_tokens,
            u.cache_write_1h_tokens, _json(call.raw_usage), _json(call.pricing) if call.pricing else None,
            call.status, call.cost_basis, call.error, serialized))
        return True

    def record_model_call(self, call: ModelCall) -> bool:
        with self.transaction() as db:
            return self._insert_call(db, call)

    def list_model_calls(self, session_id: str) -> list[ModelCall]:
        rows = self.connection.execute("SELECT record_json FROM model_calls WHERE session_id=? ORDER BY seq,rowid", (session_id,)).fetchall()
        return [ModelCall.model_validate_json(row[0]) for row in rows]

    def session_summary(self, session_id: str) -> dict[str, Any]:
        session = self.connection.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if session is None:
            raise KeyError("Unknown session")
        calls = self.list_model_calls(session_id)
        usage = {field: sum(getattr(call.usage, field) for call in calls) for field in Usage.model_fields}
        decisions = [call for call in calls if call.purpose == "decision"]
        decision_usage = {field: sum(getattr(call.usage, field) for call in decisions)
                          for field in Usage.model_fields}
        decision_reserved = self.connection.execute(
            "SELECT amount_usd FROM reservations WHERE session_id=? AND purpose='decision' AND status='reserved'",
            (session_id,),
        ).fetchall()
        return {**dict(session), "cost_usd": sum((call.cost_usd for call in calls), Decimal("0")),
                "usage": Usage(**usage), "model_calls": len(calls),
                "decision_calls": len(decisions),
                "decision_cost_usd": sum((call.cost_usd for call in decisions), Decimal("0")),
                "decision_usage": Usage(**decision_usage),
                "decision_reserved_usd": sum((Decimal(row[0]) for row in decision_reserved), Decimal("0")),
                "failed_calls": sum(call.status != "ok" for call in calls),
                "unknown_usage_calls": sum(call.status == "unknown_usage" for call in calls),
                "recalls": self.connection.execute("SELECT COUNT(*) FROM recalls WHERE session_id=?", (session_id,)).fetchone()[0],
                "rebuilds": self.connection.execute("SELECT COUNT(*) FROM epochs WHERE session_id=?", (session_id,)).fetchone()[0]}

    def experiment_summary(self, experiment_id: str = "default") -> dict[str, Any]:
        ids = [row[0] for row in self.connection.execute("SELECT id FROM sessions WHERE experiment_id=?", (experiment_id,))]
        sessions = [self.session_summary(session_id) for session_id in ids]
        total = sum((item["cost_usd"] for item in sessions), Decimal("0"))
        solved = sum(item["solved"] == 1 for item in sessions)
        return {"sessions": len(sessions), "solved": solved, "cost_usd": total,
                "cost_per_solved_task": total / solved if solved else None,
                "pass_rate": solved / len(sessions) if sessions else None}
