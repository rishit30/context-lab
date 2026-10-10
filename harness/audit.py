"""Read-only accounting reconciliation for saved runs; never contacts a provider."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

from contextlab.core.models import Chunk, Event, Pricing, Usage
from contextlab.ledger.pricing import money, normalize_anthropic_usage, price_usage
from contextlab.ledger.store import ModelCall
from contextlab.providers.jev import JevError, normalize_jev_usage

TOKENS = tuple(Usage.model_fields)
ARMS = {"baseline", "threshold_edit", "contextlab", "headroom"}
PURPOSES = {"main", "decision", "delegate", "triage", "summary"}


def _stats() -> dict[str, Any]:
    return {"calls": 0, "cost_usd": Decimal(0), "usage": dict.fromkeys(TOKENS, 0),
            "statuses": Counter(), "cost_bases": Counter()}


def _add(stats: dict[str, Any], call: ModelCall) -> None:
    stats["calls"] += 1
    stats["cost_usd"] += call.cost_usd
    stats["statuses"][call.status] += 1
    stats["cost_bases"][call.cost_basis] += 1
    for name in TOKENS:
        stats["usage"][name] += getattr(call.usage, name)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


class _Audit:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.receipts = 0
        self.price_checks = 0
        self.plan: dict[str, Any] = {}

    def check(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def read_json(self, path: Path, label: str) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                raise ValueError("Expected object")
            return data
        except (OSError, UnicodeError, ValueError):
            self.errors.append(f"{label}: unreadable or malformed JSON")
            return None

    def load_plan(self) -> None:
        path = self.directory / "plan.json"
        if not path.exists():
            self.warnings.append("No saved plan; only ledger price snapshots and configured ledger caps can be checked")
            return
        self.plan = self.read_json(path, "Saved plan") or {}
        if "approval_digest" in self.plan:
            payload = {key: value for key, value in self.plan.items() if key != "approval_digest"}
            digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            self.check(digest == self.plan["approval_digest"], "Saved plan digest does not match its contents")
        else:
            self.warnings.append("Saved plan has no approval digest; authorization integrity was not verified")

    def receipt_usage(self, call: ModelCall, label: str) -> None:
        if call.cost_basis == "simulation":
            return
        raw = call.raw_usage
        try:
            if raw.get("provider") == "typesafe":
                receipt = raw["receipt"]
                usage = normalize_jev_usage(receipt)
                self.check(receipt.get("model") == call.model, f"{label}: Jev receipt model mismatch")
            elif "messages" in raw:
                messages = raw["messages"]
                if not isinstance(messages, list) or not messages:
                    raise ValueError("Missing messages")
                totals = dict.fromkeys(TOKENS, 0)
                for item in messages:
                    original = item["provider_usage"]
                    item_usage = normalize_anthropic_usage(original, call.cache_ttl)
                    metadata = item.get("usage_metadata")
                    if metadata:
                        self.check(metadata.get("input_tokens") == item_usage.total_input_tokens
                                   and metadata.get("output_tokens") == item_usage.output_tokens,
                                   f"{label}: framework metadata disagrees with provider usage")
                    for name in TOKENS:
                        totals[name] += getattr(item_usage, name)
                    if original.get("server_tool_use"):
                        self.warnings.append(f"{label}: server-tool fees are outside token-price reconciliation")
                usage = Usage(**totals)
            elif "input_tokens" in raw and "output_tokens" in raw:
                # Early integrations may save Anthropic's usage mapping directly.
                usage = normalize_anthropic_usage(raw, call.cache_ttl)
            else:
                self.warnings.append(f"{label}: original provider receipt is absent or unrecognized; token counts unverified")
                return
        except (KeyError, TypeError, ValueError, AttributeError, JevError):
            self.errors.append(f"{label}: original provider usage is missing or malformed")
            return
        self.check(usage == call.usage, f"{label}: original provider usage differs from ledger")
        if usage == call.usage:
            self.receipts += 1

    def price(self, call: ModelCall, label: str) -> None:
        if call.cost_basis == "reservation":
            self.check(call.status == "unknown_usage", f"{label}: reservation charge lacks unknown-usage status")
            return
        if call.pricing is None:
            self.errors.append(f"{label}: missing saved price snapshot")
            return
        self.check(call.pricing.model == call.model, f"{label}: price snapshot model mismatch")
        self.check(price_usage(call.usage, call.pricing) == call.cost_usd,
                   f"{label}: Decimal token-price cost mismatch")
        self.price_checks += 1
        if call.purpose == "main" and self.plan.get("pricing"):
            expected = Pricing.model_validate(self.plan["pricing"])
            self.check(call.pricing == expected, f"{label}: saved pricing differs from plan")
            self.check(call.model == self.plan.get("model"), f"{label}: main model differs from plan")
        config = self.plan.get("config", {})
        jev = config.get("jev", {})
        if call.raw_usage.get("provider") == "typesafe" and jev:
            rates = call.pricing
            self.check(call.model == jev.get("model"), f"{label}: Jev model differs from plan")
            self.check(all(getattr(rates, key) == money(jev["input_per_million"]) for key in (
                "input_per_million", "cache_write_5m_per_million", "cache_write_1h_per_million", "cache_read_per_million"))
                and rates.output_per_million == money(jev["output_per_million"])
                and rates.source == jev.get("pricing_source") and rates.verified_at == jev.get("verified_at"),
                f"{label}: Jev price snapshot differs from plan")
            self.check(call.usage.input_tokens <= jev["reservation_input_tokens"],
                       f"{label}: Jev input exceeds planned reservation bound")
        if config.get("cache_mode") == "disabled":
            self.check(call.usage.cache_read_tokens == call.usage.cache_write_5m_tokens
                       == call.usage.cache_write_1h_tokens == 0, f"{label}: nonzero cache usage with caching disabled")
        if call.purpose == "main":
            for name, count in (("max_input_tokens", call.usage.total_input_tokens),
                                ("max_output_tokens", call.usage.output_tokens)):
                if name in config:
                    self.check(count <= config[name], f"{label}: usage exceeds planned {name}")

    def result(self, session: dict, calls: list[ModelCall], reserved: Decimal, label: str) -> None:
        path = (self.directory / session["id"] / "result.json").resolve()
        if not path.is_relative_to(self.directory):
            self.errors.append(f"{label}: result path escapes run directory")
            return
        if not path.exists():
            self.warnings.append(f"{label}: no result.json; result totals not checked")
            return
        result = self.read_json(path, f"{label} result")
        if result is None:
            return
        total = _stats()
        for call in calls:
            _add(total, call)
        try:
            self.check(money(result["cost_usd"]) == total["cost_usd"] + reserved,
                       f"{label}: result cost differs from ledger and outstanding reservations")
            self.check(type(result["turns"]) is int and result["turns"] == len(calls),
                       f"{label}: result call count differs from ledger")
            self.check(Usage.model_validate(result["usage"], strict=True).model_dump() == total["usage"],
                       f"{label}: result token totals differ from ledger")
            for name, expected in (("session_id", session["id"]), ("arm", session["arm"]),
                                   ("task_id", session["task_id"]), ("repeat", session["run_idx"]),
                                   ("mode", session["mode"])):
                if name in result:
                    self.check(result[name] == expected, f"{label}: result identity differs from session")
            if session["solved"] is not None and "solved" in result:
                self.check(type(result["solved"]) is bool and result["solved"] == bool(session["solved"]),
                           f"{label}: result outcome differs from session record")
            decisions = [call for call in calls if call.purpose == "decision"]
            if "decision_calls" in result:
                self.check(result["decision_calls"] == len(decisions), f"{label}: result decision count differs")
            if "decision_usage" in result:
                expected_usage = {name: sum(getattr(call.usage, name) for call in decisions) for name in TOKENS}
                self.check(Usage.model_validate(result["decision_usage"], strict=True).model_dump() == expected_usage,
                           f"{label}: result decision usage differs")
            if "decision_cost_usd" in result and reserved == 0:
                self.check(money(result["decision_cost_usd"]) == sum((call.cost_usd for call in decisions), Decimal(0)),
                           f"{label}: result decision cost differs")
        except (KeyError, TypeError, ValueError):
            self.errors.append(f"{label}: malformed result accounting fields")

    def run(self) -> dict[str, Any]:
        self.load_plan()
        stats, arms, purposes = _stats(), {}, {}
        report: dict[str, Any] = {"schema_version": 1, "read_only": True, "ok": False,
                                 "total_cost_usd": "0", "total_calls": 0, "per_arm": arms,
                                 "per_purpose": purposes, "errors": self.errors, "warnings": self.warnings,
                                 "limitations": ["Reconciles saved evidence, not provider invoices or task quality.",
                                                 "Historical source is not compared with the current working tree.",
                                                 "Final exposure caps are checked; provider-side charges may require reconciliation."]}
        path = self.directory / "ledger.sqlite"
        if not path.is_file():
            self.errors.append("ledger.sqlite is missing")
            return report
        try:
            with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                db.execute("BEGIN")
                self.check([row[0] for row in db.execute("PRAGMA integrity_check")] == ["ok"],
                           "SQLite integrity check failed")
                self.check(not db.execute("PRAGMA foreign_key_check").fetchall(), "SQLite foreign-key check failed")
                names = ("sessions", "events", "chunks", "model_calls", "reservations", "budgets")
                tables = {name: [dict(row) for row in db.execute(f"SELECT * FROM {name}")] for name in names}
        except sqlite3.Error:
            self.errors.append("Cannot read the ledger or its expected schema")
            return report
        sessions = {row["id"]: row for row in tables["sessions"]}
        reservations = {row["call_id"]: row for row in tables["reservations"]}
        budgets = {(row["scope"], row["scope_id"]): row for row in tables["budgets"]}
        valid_calls: dict[str, ModelCall] = {}
        for index, row in enumerate(tables["events"], 1):
            try:
                event = Event.model_validate_json(row["record_json"])
                fields = ("id", "session_id", "seq", "role", "kind", "tool_name", "tool_call_id", "size_tokens", "content")
                self.check(all(getattr(event, key) == row[key] for key in fields)
                           and event.args == json.loads(row["args_json"]), f"Event #{index}: table and original record differ")
            except (TypeError, ValueError, KeyError):
                self.errors.append(f"Event #{index}: malformed original record")
        for index, row in enumerate(tables["chunks"], 1):
            try:
                chunk = Chunk.model_validate_json(row["record_json"])
                fields = ("id", "session_id", "event_id", "kind", "size_tokens", "signpost_text")
                self.check(all(getattr(chunk, key) == row[key] for key in fields),
                           f"Chunk #{index}: table and original record differ")
                self.check(hashlib.sha256(chunk.content.encode()).hexdigest() == row["content_sha256"],
                           f"Chunk #{index}: archived content hash mismatch")
            except (TypeError, ValueError, KeyError):
                self.errors.append(f"Chunk #{index}: malformed original record")
        for index, row in enumerate(tables["model_calls"], 1):
            label = f"Call #{index}"
            try:
                call = ModelCall.model_validate_json(row["record_json"], strict=True)
                valid_calls[call.id] = call
                fields = ("id", "session_id", "seq", "model", "purpose", "status", "cost_basis", "cache_ttl", "error", "latency_ms")
                self.check(all(getattr(call, key) == row[key] for key in fields), f"{label}: table and call record differ")
                self.check(call.cost_usd == money(row["cost_usd"]), f"{label}: table and recorded cost differ")
                self.check(all(type(row[name]) is int and row[name] == getattr(call.usage, name) for name in TOKENS),
                           f"{label}: table and recorded usage differ")
                self.check(row["cache_write_tokens"] == call.usage.cache_write_5m_tokens + call.usage.cache_write_1h_tokens,
                           f"{label}: aggregate cache creation differs from TTL buckets")
                self.check(json.loads(row["raw_usage_json"]) == call.raw_usage, f"{label}: table and raw receipt differ")
                table_pricing = Pricing.model_validate_json(row["pricing_json"]) if row["pricing_json"] else None
                self.check(table_pricing == call.pricing, f"{label}: table and recorded pricing differ")
                session = sessions[call.session_id]
                if call.purpose == "main":
                    self.check(call.model == session["model_main"], f"{label}: main model differs from session")
                _add(stats, call)
                arm = session["arm"] if session["arm"] in ARMS else "other"
                _add(arms.setdefault(arm, _stats()), call)
                _add(purposes.setdefault(call.purpose, _stats()), call)
                reservation = reservations.get(call.id)
                if reservation:
                    self.check(all(reservation[key] == getattr(call, key) for key in ("session_id", "seq", "model", "purpose")),
                               f"{label}: reservation identity mismatch")
                    self.check(reservation["status"] == "settled", f"{label}: charged call reservation is not settled")
                    amount = money(reservation["amount_usd"])
                    self.check(call.cost_usd <= amount, f"{label}: charge exceeds reservation")
                    if call.cost_basis == "reservation":
                        self.check(call.cost_usd == amount, f"{label}: conservative charge differs from reservation")
                elif session["mode"] == "live":
                    self.errors.append(f"{label}: live call has no reservation")
                if call.status == "unknown_usage":
                    self.errors.append(f"{label}: unknown usage requires provider reconciliation")
                    self.check(call.cost_basis == "reservation", f"{label}: unknown usage lacks conservative cost basis")
                    self.check(all(budgets.get(key, {}).get("blocked") == 1 for key in (
                        ("session", call.session_id), ("experiment", session["experiment_id"]))),
                        f"{label}: unknown usage did not block both budgets")
                else:
                    self.receipt_usage(call, label)
                self.price(call, label)
            except (KeyError, TypeError, ValueError, AttributeError):
                self.errors.append(f"{label}: malformed ledger accounting or plan fields")
        outstanding, reservation_states = Decimal(0), Counter()
        reservation_amounts = {}
        for index, row in enumerate(tables["reservations"], 1):
            label = f"Reservation #{index}"
            try:
                amount = money(row["amount_usd"])
                reservation_amounts[row["call_id"]] = amount
                status = row["status"]
                self.check(status in {"settled", "reserved", "cancelled"}, f"{label}: invalid status")
                reservation_states[status if status in {"settled", "reserved", "cancelled"} else "invalid"] += 1
                self.check(row["experiment_id"] == sessions[row["session_id"]]["experiment_id"],
                           f"{label}: experiment identity differs from session")
                self.check((row["call_id"] in valid_calls) == (status == "settled"),
                           f"{label}: status and recorded call presence disagree")
                if status == "reserved":
                    outstanding += amount
                    self.errors.append(f"{label}: unresolved reservation")
                else:
                    self.check(bool(row["settled_at"]) and row["settled_at"] >= row["created_at"],
                               f"{label}: invalid settlement timestamp")
            except (KeyError, TypeError, ValueError):
                self.errors.append(f"{label}: malformed reservation")
        cap_rows = []
        for index, row in enumerate(tables["budgets"], 1):
            try:
                scope = row["scope"]
                self.check(scope in {"session", "experiment"}, f"Budget #{index}: invalid scope")
                self.check(row["blocked"] in (0, 1), f"Budget #{index}: invalid blocked flag")
                matched = {sid for sid, session in sessions.items()
                           if (sid if scope == "session" else session["experiment_id"]) == row["scope_id"]}
                spent = sum((call.cost_usd for call in valid_calls.values() if call.session_id in matched), Decimal(0))
                pending = sum((reservation_amounts.get(item["call_id"], Decimal(0)) for item in reservations.values()
                               if item["session_id"] in matched and item["status"] == "reserved"), Decimal(0))
                cap = money(row["cap_usd"])
                self.check(spent + pending <= cap, f"Budget #{index}: spend plus outstanding reservations exceeds cap")
                config = self.plan.get("config", {})
                key = "session_cap_usd" if scope == "session" else "experiment_cap_usd"
                if key in config:
                    self.check(cap == money(config[key]), f"Budget #{index}: cap differs from saved plan")
                cap_rows.append({"scope": scope if scope in {"session", "experiment"} else "invalid",
                                 "cap_usd": cap, "spent_usd": spent, "reserved_usd": pending,
                                 "blocked": bool(row["blocked"])})
            except (KeyError, TypeError, ValueError):
                self.errors.append(f"Budget #{index}: malformed cap record")
        outcomes = Counter()
        for index, session in enumerate(sessions.values(), 1):
            label = f"Session #{index}"
            calls = [call for call in valid_calls.values() if call.session_id == session["id"]]
            pending = sum((reservation_amounts.get(row["call_id"], Decimal(0)) for row in reservations.values()
                           if row["session_id"] == session["id"] and row["status"] == "reserved"), Decimal(0))
            self.check(sorted(call.seq for call in calls) == list(range(len(calls))), f"{label}: call sequence is not contiguous")
            if session["mode"] == "live" and calls:
                self.check(("session", session["id"]) in budgets
                           and ("experiment", session["experiment_id"]) in budgets, f"{label}: live calls lack budget caps")
            if not session["ended_at"]:
                self.warnings.append(f"{label}: session is unfinished")
            outcome = "passed" if session["solved"] == 1 else "failed" if session["solved"] == 0 else "unfinished"
            outcomes[outcome] += 1
            arm = session["arm"] if session["arm"] in ARMS else "other"
            arms.setdefault(arm, _stats()).setdefault("recorded_outcomes", Counter())[outcome] += 1
            self.result(session, calls, pending, label)
        if stats["cost_bases"]["simulation"] or any(row["mode"] == "simulation" for row in sessions.values()):
            self.warnings.append("Simulation evidence is not measured provider spend")
        if stats["statuses"]["failed"]:
            self.warnings.append("Failed provider calls are included in all spend totals")
        if stats["statuses"]["unknown_usage"]:
            self.warnings.append("Unknown usage is included at its conservative reservation charge, not verified actual cost")
        if any(row["blocked"] for row in cap_rows):
            self.warnings.append("One or more ledger budgets are blocked")
        self.check(stats["calls"] == len(tables["model_calls"]), "Not all model-call rows could be reconciled; totals may be incomplete")
        report.update(ok=not self.errors, total_cost_usd=stats["cost_usd"], total_calls=stats["calls"],
                      usage=stats["usage"], statuses=stats["statuses"], cost_bases=stats["cost_bases"],
                      outstanding_reserved_usd=outstanding, committed_exposure_usd=stats["cost_usd"] + outstanding,
                      reservation_statuses=reservation_states, caps=cap_rows, recorded_outcomes=outcomes,
                      receipt_verified_calls=self.receipts, priced_calls=self.price_checks,
                      all_live_receipts_verified=self.receipts == stats["calls"] - stats["cost_bases"]["simulation"],
                      archive_chunks_checked=len(tables["chunks"]), sessions=len(sessions))
        return _jsonable(report)


def audit_run(directory: Path) -> dict[str, Any]:
    """Return sanitized accounting findings without writing any run artifact.

    ``ok`` confirms the checks performed, not task quality or an invoice match.
    Unknown receipt formats are explicit warnings; receipt coverage is separate.
    """
    auditor = _Audit(Path(directory))
    try:
        return auditor.run()
    except (KeyError, TypeError, ValueError, AttributeError, OSError, sqlite3.Error):
        # Historical files are input data. Do not print exception messages that
        # could quote a prompt, a provider body, or other private stored fields.
        return {"ok": False, "read_only": True, "total_cost_usd": None,
                "errors": auditor.errors + ["Malformed or unsupported accounting schema; totals unavailable"],
                "warnings": auditor.warnings}
