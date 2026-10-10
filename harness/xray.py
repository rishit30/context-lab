"""Offline causal candidate-context analysis, never a claim of waste or savings."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

from contextlab.core.duplicates import exact_read_copies, reserve_copy_anchors
from contextlab.core.models import Chunk, CompactionConfig, Event, Usage, ViewState
from contextlab.core.planner import plan_visibility
from contextlab.core.signals import derive_signals, recalled_chunk_ids, test_pass_boundary
from contextlab.core.validator import validate_messages
from contextlab.core.view_builder import canonical_bytes
from contextlab.ledger.pricing import normalize_anthropic_usage, price_usage
from contextlab.ledger.store import ModelCall

BOUNDARY_SOURCE = "contextlab.harness.public_phase_completion/v1"
INPUT_RATES = {"input_tokens": "input_per_million", "cache_write_5m_tokens": "cache_write_5m_per_million",
               "cache_write_1h_tokens": "cache_write_1h_per_million", "cache_read_tokens": "cache_read_per_million"}
LIMITATIONS = [
    "Current rule-candidate analysis, not an executed policy, cost-gate, protocol-rewrite or Jev decision replay.",
    "Candidate does not mean irrelevant: an old value may still be essential to a later historical comparison.",
    "The primary byte share covers normalized message views only; system instructions and tool schemas are absent.",
    "Candidate token counts use ceil(UTF-8 bytes / 4), not a provider tokenizer. Financial ranges are conditional on that estimate.",
    "Cache bucket assignment to particular chunks is unknown. Ranges allocate estimated tokens across observed buckets.",
    "No rebuild, signpost, recall, decision, future-turn, output or task-quality effects are estimated; this is not savings.",
    "Repeated candidate exposure is counted per baseline request. No hypothetical deletion changes later baseline histories.",
    "Final workspace contents, mutable pin columns, final graders and future events are not used as call-time evidence.",
    "Historical phase_index/completed fields alone never establish a trusted completion boundary.",
]


def _safe_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected object")
    return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _ledger(path: Path) -> dict[str, list[dict]]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        return {name: [dict(row) for row in db.execute(f"SELECT * FROM {name}")]
                for name in ("sessions", "events", "chunks", "model_calls")}


def _metadata(name: str, args: dict, content: str, is_error: bool) -> tuple[str, dict]:
    kind = {"read_file": "file_read", "write_file": "file_write", "edit_file": "file_edit",
            "run_tests": "test_run", "search": "search", "recall": "recall", "shell": "shell"}.get(name, "other")
    if is_error:
        return "other", {"tool_error": True}
    try:
        parsed = json.loads(content)
    except ValueError:
        parsed = None
    metadata = ({key: value for key, value in parsed.items() if key not in {"content", "output", "matches"}}
                if isinstance(parsed, dict) else {})
    if kind == "file_read" and str(args.get("path", "")).startswith("/context/"):
        return "recall", {**metadata, "chunk_id": str(args["path"]).removeprefix("/context/").removesuffix(".txt")}
    if kind in {"file_read", "test_run"} and not isinstance(parsed, dict):
        raise ValueError("Tool observation lacks structured provenance")
    if kind == "file_read":
        if (not isinstance(metadata.get("path"), str) or not metadata["path"]
                or not isinstance(metadata.get("full_read"), bool)
                or any(type(metadata.get(key)) is not int or metadata[key] < 0
                       for key in ("offset", "limit", "total_lines"))):
            raise ValueError("Read provenance is incomplete")
        metadata.update(version=metadata.get("sha256"), start_line=metadata["offset"] + 1,
                        end_line=min(metadata["offset"] + metadata["limit"], metadata["total_lines"]))
    if kind == "test_run":
        if type(metadata.get("exit_code")) is not int or not isinstance(metadata.get("test_selection"), str):
            raise ValueError("Test provenance is incomplete")
        metadata.update(environment="contextlab-docker-pytest", cwd=".")
    if kind == "recall":
        metadata.setdefault("chunk_id", args.get("chunk_id", ""))
    return kind, metadata


def _reconstruct(messages: list[dict], session: str, archived: dict[str, tuple[Event, Chunk] | None]) -> tuple[list[Event], list[Chunk]]:
    """Only tool results actually visible now may select immutable archive records."""
    validate_messages(messages)
    calls: dict[str, dict] = {}
    events, chunks = [], []
    for message_index, message in enumerate(messages):
        content = message["content"]
        blocks = content if isinstance(content, list) else []
        for block in blocks:
            if block.get("type") == "tool_use":
                calls[block["id"]] = block
        if not blocks or any(block.get("type") != "tool_result" for block in blocks):
            events.append(Event(id=f"xr-e{len(events)}", session_id=session, seq=len(events), role=message["role"],
                                kind="user_msg" if message["role"] == "user" else "assistant_msg",
                                content=content if isinstance(content, str) else json.dumps(content, sort_keys=True),
                                metadata={"_message_index": message_index}))
        for block in blocks:
            if block.get("type") != "tool_result" or not isinstance(block.get("content"), str):
                continue  # Opaque tool results are never candidates in the supported policy.
            call_id, text = block["tool_use_id"], block["content"]
            call = calls[call_id]
            old = archived.get(call_id)
            if call_id in archived and old is None:
                raise ValueError("Visible archive record is invalid")
            if old is not None:
                original, chunk = old
                if (original.content != text or chunk.content != text or original.tool_name != call["name"]
                        or original.args != call["input"]):
                    raise ValueError("Archive disagrees with visible tool result")
                kind, metadata, chunk_id = original.kind, dict(original.metadata), chunk.id
                # Read the immutable initial pinned bit, never final mutable recall pins.
                pinned = chunk.pinned
            else:
                kind, metadata = _metadata(call["name"], call["input"], text, bool(block.get("is_error")))
                chunk_id, pinned = f"xr-c{len(events)}", False
            if block.get("is_error"):
                kind, metadata = "other", {**metadata, "tool_error": True}
            metadata["_message_index"] = message_index
            event = Event(id=f"xr-e{len(events)}", session_id=session, seq=len(events), role="tool", kind=kind,
                          tool_name=call["name"], tool_call_id=call_id, args=call["input"], content=text, metadata=metadata)
            events.append(event)
            refs = (metadata["path"],) if isinstance(metadata.get("path"), str) else ()
            chunks.append(Chunk(id=chunk_id, session_id=session, event_id=event.id, tool_call_id=call_id,
                                kind=kind, content=text, seq=event.seq, refs=refs, metadata=metadata, pinned=pinned))
    return events, chunks


def _candidates(events: list[Event], chunks: list[Chunk], messages: list[dict], policy: CompactionConfig) -> tuple[list[Chunk], dict[str, str]]:
    proposed = plan_visibility(events, chunks, ViewState(), policy)
    protected = {chunk.id for chunk in chunks if chunk.pinned} | recalled_chunk_ids(events)
    if policy.recent_outputs:
        protected.update(chunk.id for chunk in chunks[-policy.recent_outputs:])
    copies = exact_read_copies(events, chunks, messages)
    exact = reserve_copy_anchors(proposed, ViewState(), chunks, copies, protected)
    return [chunk for chunk in chunks if proposed.get(chunk.id, "full") != "full"], exact


def _boundary(request: dict, messages: list[dict], events: list[Event]) -> tuple[int | None, str, bool]:
    passing = test_pass_boundary(events)
    cutoff = next((event.seq for event in events if event.id == passing), None)
    source = "test_fail_to_pass" if cutoff is not None else "unavailable"
    markers = request.get("completed_phase_boundaries")
    provenance_available = isinstance(markers, list)
    if markers is not None and not isinstance(markers, list):
        raise ValueError("Malformed boundary provenance")
    previous_phase, previous_count = -1, 0
    for marker in markers or []:
        if not isinstance(marker, dict) or marker.get("source") != BOUNDARY_SOURCE:
            raise ValueError("Unsupported boundary provenance")
        phase = marker.get("phase_index")
        if type(phase) is not int or phase <= previous_phase:
            raise ValueError("Invalid boundary phase")
        if not isinstance(marker.get("label"), str) or not marker["label"].strip():
            raise ValueError("Invalid boundary label")
        if "phase_index" in request and (type(request["phase_index"]) is not int or phase >= request["phase_index"]):
            raise ValueError("Boundary is not from an earlier phase")
        count = marker.get("normalized_message_count")
        if type(count) is not int or not previous_count < count <= len(messages):
            raise ValueError("Boundary prefix is unavailable")
        digest = hashlib.sha256(canonical_bytes(messages[:count])).hexdigest()
        if digest != marker.get("message_prefix_sha256"):
            raise ValueError("Boundary prefix differs from call-time history")
        previous_phase, previous_count = phase, count
        eligible = [event.seq for event in events if event.metadata["_message_index"] < count]
        if eligible and (cutoff is None or max(eligible) > cutoff):
            cutoff, source = max(eligible), "public_phase_completion"
    return cutoff, source, provenance_available


def _receipt(call: ModelCall) -> Usage | None:
    if call.cost_basis != "usage" or call.status == "unknown_usage" or call.pricing is None:
        return None
    raw = call.raw_usage
    if "messages" in raw:
        if not isinstance(raw["messages"], list) or not raw["messages"]:
            return None
        rows = [normalize_anthropic_usage(item["provider_usage"], call.cache_ttl) for item in raw["messages"]]
        usage = Usage(**{name: sum(getattr(row, name) for row in rows) for name in Usage.model_fields})
    elif "input_tokens" in raw and "output_tokens" in raw:
        usage = normalize_anthropic_usage(raw, call.cache_ttl)
    else:
        return None
    if usage != call.usage or price_usage(usage, call.pricing) != call.cost_usd or call.pricing.model != call.model:
        return None
    return usage


def _priced(call: ModelCall, candidate_tokens: int | None) -> dict[str, Any]:
    try:
        usage = _receipt(call)
    except (KeyError, TypeError, ValueError, AttributeError):
        usage = None
    result: dict[str, Any] = {"receipt_verified": usage is not None, "actual_input_cost_usd": None,
                              "total_input_tokens": None, "input_usage": None,
                              "estimated_candidate_provider_input_share": None,
                              "priced_candidate_input_estimate_usd": None, "pricing_status": "unknown"}
    if usage is None or call.pricing is None:
        return result
    result.update(total_input_tokens=usage.total_input_tokens,
                  input_usage={name: getattr(usage, name) for name in INPUT_RATES},
                  actual_input_cost_usd=price_usage(usage.model_copy(update={"output_tokens": 0}), call.pricing))
    if candidate_tokens is None or candidate_tokens > usage.total_input_tokens:
        return result
    buckets = [(getattr(call.pricing, rate), getattr(usage, token)) for token, rate in INPUT_RATES.items()]
    def allocate(reverse: bool) -> Decimal:
        remaining, amount = candidate_tokens, Decimal(0)
        for rate, tokens in sorted(buckets, reverse=reverse):
            chosen = min(remaining, tokens)
            amount += chosen * rate / Decimal(1000000)
            remaining -= chosen
        return amount
    result.update(pricing_status="conditional_estimate", priced_candidate_input_estimate_usd={
        "lower": allocate(False), "upper": allocate(True)},
        estimated_candidate_provider_input_share=(candidate_tokens / usage.total_input_tokens
                                                  if usage.total_input_tokens else None))
    return result


def _aggregate(rows: list[dict], **identity: Any) -> dict[str, Any]:
    known = [row for row in rows if row["status"] == "known"]
    byte_count = sum(row["message_view_bytes"] for row in known)
    candidates = sum(row["candidate_text_bytes"] for row in known)
    complete = len(known) == len(rows) and bool(rows)
    priced = [row for row in rows if row.get("priced_candidate_input_estimate_usd") is not None]
    receipts = [row for row in rows if row.get("receipt_verified")]
    boundary_rows = [row for row in known if row.get("verified_boundary_candidate_text_bytes") is not None]
    modes = {row["mode"] for row in rows}
    mixed = len(modes) > 1
    result = {**identity, "mode": "mixed" if mixed else next(iter(modes), identity.get("mode", "unknown")),
              "sessions": len({row["session_id"] for row in rows}), "baseline_calls": len(rows),
              "reconstructed_calls": len(known), "unknown_calls": len(rows) - len(known),
              "receipt_verified_calls": len(receipts), "priced_calls": len(priced), "coverage_complete": complete,
              "candidate_text_bytes": candidates if complete and not mixed else None,
              "message_view_bytes": byte_count if complete and not mixed else None,
              "estimated_candidate_tokens": sum(row["estimated_candidate_tokens"] for row in known) if complete and not mixed else None,
              "estimated_candidate_context_share": candidates / byte_count if complete and byte_count and not mixed else None,
              "covered_estimated_candidate_context_share": candidates / byte_count if byte_count and not mixed else None,
              "verified_boundary_calls": sum(row.get("boundary_source") in {"test_fail_to_pass", "public_phase_completion"} for row in known),
              "boundary_coverage_calls": len(boundary_rows),
              "verified_boundary_candidate_context_share": (sum(row["verified_boundary_candidate_text_bytes"] for row in boundary_rows) / byte_count
                                                             if complete and len(boundary_rows) == len(rows) and byte_count and not mixed else None),
              "total_input_tokens": sum(row["total_input_tokens"] for row in receipts) if len(receipts) == len(rows) and rows and not mixed else None,
              "actual_input_cost_usd": sum((row["actual_input_cost_usd"] for row in receipts), Decimal(0)) if len(receipts) == len(rows) and rows and not mixed else None,
              "priced_candidate_input_estimate_usd": ({key: sum((row["priced_candidate_input_estimate_usd"][key] for row in priced), Decimal(0)) for key in ("lower", "upper")}
                                                      if len(priced) == len(rows) and rows and not mixed else None)}
    return result


def _analyze_run(directory: Path) -> dict[str, Any]:
    """Analyze baseline requests without inference, writes or final-state replay."""
    root = Path(directory).resolve()
    errors, warnings, output = [], [], []
    report: dict[str, Any] = {"schema_version": 1, "ok": False, "analysis_kind": "current_policy_candidate_replay",
                             "metric_label": "Estimated candidate-context share", "read_only": True,
                             "requires_boundary_for_primary_metric": False, "errors": errors, "warnings": warnings,
                             "limitations": LIMITATIONS.copy(), "summary": {}, "per_mode": [],
                             "per_task": [], "per_session": [], "calls": output}
    try:
        plan_path = root / "plan.json"
        if not plan_path.resolve().is_relative_to(root):
            raise ValueError("Plan path escapes run directory")
        plan = _safe_json(plan_path) if plan_path.exists() else {}
        policy = CompactionConfig.model_validate(plan.get("compaction", {}))
        report["analysis_policy"] = policy.model_dump(mode="json")
        report["algorithm_version"] = "causal-signal-candidates/v1"
        report["analysis_policy_source"] = "saved_plan_compaction" if plan.get("compaction") else "current_defaults"
        report["execution_source_hash"] = plan.get("source_hash")
        report["execution_approval_digest"] = plan.get("approval_digest")
        project = Path(__file__).parents[1]
        source_files = [Path(__file__), *sorted((project / "contextlab/core").glob("*.py")),
                        project / "contextlab/ledger/pricing.py", project / "contextlab/ledger/store.py"]
        report["analysis_source_hash"] = hashlib.sha256(json.dumps([
            (str(path.relative_to(project)), hashlib.sha256(path.read_bytes()).hexdigest()) for path in source_files]).encode()).hexdigest()
    except (OSError, TypeError, ValueError):
        errors.append("Saved plan or analysis policy is unreadable")
        return report
    if not plan.get("compaction"):
        warnings.append("No saved compaction settings; analysis explicitly uses current default policy parameters")
    ledgers = [root / "ledger.sqlite"] if (root / "ledger.sqlite").is_file() else sorted(root.glob("*/ledger.sqlite"))
    if not ledgers:
        errors.append("No root or per-session ledger.sqlite was found")
    seen_sessions = set()
    for ledger_path in ledgers:
        if not ledger_path.resolve().is_relative_to(root):
            errors.append("Ledger path escapes run directory")
            continue
        try:
            tables = _ledger(ledger_path)
        except sqlite3.Error:
            errors.append("A ledger is unreadable or has an unsupported schema")
            continue
        for session in tables["sessions"]:
            if session["arm"] != "baseline":
                continue
            sid, mode = session["id"], session["mode"]
            if sid in seen_sessions:
                errors.append("Duplicate baseline session identity across ledgers")
                continue
            seen_sessions.add(sid)
            try:
                session_dir = root / sid if ledger_path.parent == root else ledger_path.parent
                if not session_dir.resolve().is_relative_to(root):
                    raise ValueError("Unsafe session path")
                trace_path = session_dir / "trace.json"
                if not trace_path.resolve().is_relative_to(root):
                    raise ValueError("Unsafe trace path")
                trace = _safe_json(trace_path)
                trace_digest = hashlib.sha256(trace_path.read_bytes()).hexdigest()
                requests = trace.get("requests", trace.get("timeline"))
                if not isinstance(requests, list) or trace.get("mode") != mode:
                    raise ValueError("Unsupported trace")
                request_map = {item["seq"]: item for item in requests}
                if len(request_map) != len(requests) or any(type(seq) is not int for seq in request_map):
                    raise ValueError("Ambiguous request sequence")
                archive_events = {row["id"]: row
                                  for row in tables["events"] if row["session_id"] == sid}
                archived = {}
                for row in tables["chunks"]:
                    if row["session_id"] != sid:
                        continue
                    event_row = archive_events.get(row["event_id"])
                    if event_row is None:
                        continue  # An unrelated future record cannot change earlier calls.
                    call_id = event_row["tool_call_id"]
                    duplicate = call_id in archived
                    try:
                        event = Event.model_validate_json(event_row["record_json"])
                        chunk = Chunk.model_validate_json(row["record_json"])
                        if (duplicate or chunk.session_id != sid or event.session_id != sid
                                or chunk.event_id != event.id or chunk.tool_call_id != event.tool_call_id
                                or hashlib.sha256(chunk.content.encode()).hexdigest() != row["content_sha256"]):
                            raise ValueError("Invalid archive")
                        archived[call_id] = event, chunk
                    except (KeyError, TypeError, ValueError):
                        archived[call_id] = None  # Fail only when this record is actually visible.
            except (OSError, KeyError, TypeError, ValueError, AttributeError):
                request_map, archived, trace_digest = {}, {}, None
                errors.append(f"Baseline session {len(seen_sessions)} has missing or unreconstructable trace/archive data")
            calls = [row for row in tables["model_calls"] if row["session_id"] == sid and row["purpose"] == "main"]
            if not calls and not request_map:
                errors.append(f"Baseline session {len(seen_sessions)} has no admitted main-call evidence")
            previous, history_broken = [], False
            seen_call_seqs = set()
            structural_call_seqs = {record["seq"] for record in calls}
            for record in sorted(calls, key=lambda row: row["seq"]):
                seq = record["seq"]
                row: dict[str, Any] = {"session_id": sid, "task_id": session["task_id"], "mode": mode,
                                       "seq": seq, "status": "unknown", "reasons": [], "recorded_solved": session["solved"],
                                       "trace_sha256": trace_digest,
                                       "candidate_text_bytes": None, "message_view_bytes": None,
                                       "estimated_candidate_tokens": None, "estimated_candidate_context_share": None,
                                       "boundary_eligibility": "unknown", "boundary_source": "unavailable"}
                output.append(row)
                try:
                    call = ModelCall.model_validate_json(record["record_json"], strict=True)
                    if (call.session_id != sid or call.seq != seq or call.purpose != "main" or call.id != record["id"]
                            or seq in seen_call_seqs or call.model != record["model"]
                            or any(getattr(call.usage, name) != record[name] for name in Usage.model_fields)
                            or call.raw_usage != json.loads(record["raw_usage_json"])):
                        raise ValueError("Ambiguous call sequence")
                    seen_call_seqs.add(seq)
                    request = request_map[seq]
                    messages = request["actual_rendered_messages"]
                    if not isinstance(messages, list) or history_broken or messages[:len(previous)] != previous:
                        history_broken = True
                        raise ValueError("Non-prefix or missing request history")
                    previous = messages
                    events, chunks = _reconstruct(messages, sid, archived)
                    signals = derive_signals(events, chunks)
                    candidates, exact = _candidates(events, chunks, messages, policy)
                    cutoff, boundary_source, boundary_complete = _boundary(request, messages, events)
                    boundary_candidates, _ = _candidates([event for event in events if cutoff is not None and event.seq <= cutoff],
                                                         [chunk for chunk in chunks if cutoff is not None and chunk.seq <= cutoff],
                                                         messages, policy)
                    boundary_ids = {chunk.id for chunk in boundary_candidates}
                    at_boundary = [chunk for chunk in candidates if chunk.id in boundary_ids]
                    candidate_bytes = sum(len(chunk.content.encode()) for chunk in candidates)
                    view_bytes = len(canonical_bytes(messages))
                    row.update(status="known", candidate_text_bytes=candidate_bytes, message_view_bytes=view_bytes,
                               estimated_candidate_tokens=(candidate_bytes + 3) // 4,
                               estimated_candidate_context_share=candidate_bytes / view_bytes if view_bytes else None,
                               candidate_count=len(candidates), exact_duplicate_count=len(exact),
                               signal_counts=dict(Counter(signal.signal_type for signal in signals if signal.chunk_id in {chunk.id for chunk in candidates})),
                               boundary_source=boundary_source, boundary_eligibility="observed" if cutoff is not None else "not_observed" if boundary_complete else "unknown",
                               verified_boundary_candidate_text_bytes=sum(len(chunk.content.encode()) for chunk in at_boundary) if boundary_complete or cutoff is not None else None,
                               unverified_phase_index=request.get("phase_index"))
                    row.update(_priced(call, row["estimated_candidate_tokens"]))
                    if not row["receipt_verified"]:
                        row["reasons"].append("Provider input attribution unavailable; simulation, unknown usage or incomplete receipt")
                    elif row["pricing_status"] == "unknown":
                        row["reasons"].append("Heuristic candidate token estimate exceeds provider input; price/share left unknown")
                except (KeyError, TypeError, ValueError, AttributeError, StopIteration):
                    row["reasons"].append("Call-time trace, immutable observation or provider record cannot be reconstructed")
                    errors.append(f"Baseline session {len(seen_sessions)} call {seq}: reconstruction unavailable")
            for seq in request_map.keys() - structural_call_seqs:
                errors.append(f"Baseline session {len(seen_sessions)} has a request without a matching main-call record")
                output.append({"session_id": sid, "task_id": session["task_id"], "mode": mode, "seq": seq,
                               "status": "unknown", "reasons": ["Missing matching ledger call"],
                               "boundary_eligibility": "unknown", "boundary_source": "unavailable"})
    if not seen_sessions:
        errors.append("No baseline sessions available for analysis")
    if any(row["mode"] == "simulation" for row in output):
        warnings.append("Simulation traces demonstrate mechanics; no provider spend or performance is measured")
    if any(row.get("boundary_eligibility") == "unknown" for row in output):
        warnings.append("Some request-time phase boundary provenance is unavailable; primary signal-candidate share remains separate")
    if any(not row.get("receipt_verified") for row in output):
        warnings.append("Some provider input prices cannot be attributed; financial aggregates are null when coverage is incomplete")
    report["summary"] = _aggregate(output)
    for fields, key in ((("mode",), "per_mode"), (("mode", "task_id"), "per_task"), (("mode", "session_id", "task_id"), "per_session")):
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in output:
            groups[tuple(row[field] for field in fields)].append(row)
        report[key] = [_aggregate(rows, **dict(zip(fields, values, strict=True))) for values, rows in sorted(groups.items())]
    report.update(ok=not errors, readiness="complete" if not errors else "partial" if any(row["status"] == "known" for row in output) else "unavailable")
    return _jsonable(report)


def analyze_run(directory: Path) -> dict[str, Any]:
    """Return sanitized derived metrics; unavailable evidence remains unknown."""
    try:
        return _analyze_run(Path(directory))
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError, AttributeError):
        return {"schema_version": 1, "ok": False, "read_only": True, "readiness": "unavailable",
                "analysis_kind": "current_policy_candidate_replay", "metric_label": "Estimated candidate-context share",
                "errors": ["Run evidence has an unsupported or malformed schema"], "warnings": [],
                "limitations": LIMITATIONS.copy(), "summary": _aggregate([]),
                "per_mode": [], "per_task": [], "per_session": [], "calls": []}
