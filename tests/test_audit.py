from __future__ import annotations

import hashlib
import json
import sqlite3
from decimal import Decimal

import pytest

from contextlab.core.models import Chunk, Event, Pricing, Usage
from contextlab.ledger.budget import BudgetManager
from contextlab.ledger.pricing import normalize_anthropic_usage, price_usage
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.audit import audit_run


def example_run(path, *, unknown=False, failed=False, simulation=False, jev=False):
    path.mkdir(exist_ok=True)
    model = "jev-1.13.0" if jev else "claude-test"
    pricing = Pricing(model=model, input_per_million=Decimal(".042") if jev else 3,
                      cache_write_5m_per_million=Decimal(".042") if jev else Decimal("3.75"),
                      cache_write_1h_per_million=Decimal(".042") if jev else 6,
                      cache_read_per_million=Decimal(".042") if jev else Decimal(".3"),
                      output_per_million=0 if jev else 15, source="fixture")
    original = {"input_tokens": 100, "output_tokens": 20, "cache_creation_input_tokens": 75,
                "cache_creation": {"ephemeral_5m_input_tokens": 50, "ephemeral_1h_input_tokens": 25},
                "cache_read_input_tokens": 400}
    usage = Usage(input_tokens=1000, output_tokens=8) if jev else normalize_anthropic_usage(original)
    raw = ({"provider": "typesafe", "receipt": {"model": model, "usage": {"input_tokens": 1000, "output_tokens": 8}}}
           if jev else {"messages": [{"provider_usage": original,
                                     "usage_metadata": {"input_tokens": 575, "output_tokens": 20}}]})
    with LedgerStore(path / "ledger.sqlite") as store:
        store.create_session("s", "contextlab" if jev else "baseline", "test", model_main=model,
                             mode="simulation" if simulation else "live")
        event = Event(id="e", session_id="s", seq=0, role="tool", kind="tool_result", content="secret prompt")
        store.append_event(event)
        store.add_chunk(Chunk(id="c", session_id="s", event_id="e", tool_call_id="t",
                              seq=0, kind="file_read", content=event.content))
        budgets = BudgetManager(store, "1", "1")
        purpose = "decision" if jev else "main"
        budgets.reserve("s", "call", model, purpose, ".1", 0)
        if unknown:
            call = budgets.settle("call")
        else:
            call = ModelCall(id="call", session_id="s", seq=0, model=model, purpose=purpose,
                             usage=usage, cost_usd=price_usage(usage, pricing), pricing=pricing,
                             raw_usage={} if simulation else raw, status="failed" if failed else "ok",
                             cost_basis="simulation" if simulation else "usage",
                             error="secret provider error" if failed else None)
            budgets.settle("call", call)
        store.finish_session("s", not (failed or unknown))
    result = {"session_id": "s", "arm": "contextlab" if jev else "baseline", "task_id": "test", "repeat": 0,
              "mode": "simulation" if simulation else "live", "solved": not (failed or unknown),
              "cost_usd": str(call.cost_usd), "turns": 1, "usage": call.usage.model_dump(),
              "decision_calls": int(jev), "decision_cost_usd": str(call.cost_usd if jev else 0),
              "decision_usage": call.usage.model_dump() if jev else Usage().model_dump()}
    (path / "s").mkdir()
    (path / "s" / "result.json").write_text(json.dumps(result))
    return call


def mutate(path, sql, args=()):
    with sqlite3.connect(path / "ledger.sqlite") as db:
        db.execute(sql, args)


def rewrite_call(path, callback):
    with sqlite3.connect(path / "ledger.sqlite") as db:
        data = json.loads(db.execute("SELECT record_json FROM model_calls").fetchone()[0])
        callback(data)
        db.execute("UPDATE model_calls SET record_json=?", (json.dumps(data),))


@pytest.mark.parametrize("jev", [False, True])
def test_receipts_prices_results_and_caps_reconcile_without_writes(tmp_path, jev):
    call = example_run(tmp_path, jev=jev)
    existing = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.rglob("*") if path.is_file()}
    (tmp_path / "audit.json").write_text("preserve prior independent audit")
    result = audit_run(tmp_path)
    assert result["ok"] and result["errors"] == []
    assert Decimal(result["total_cost_usd"]) == call.cost_usd
    assert result["receipt_verified_calls"] == result["priced_calls"] == result["total_calls"] == 1
    assert result["all_live_receipts_verified"]
    assert result["reservation_statuses"] == {"settled": 1}
    assert result["outstanding_reserved_usd"] == "0"
    assert result["per_purpose"][call.purpose]["usage"] == call.usage.model_dump()
    assert result["per_arm"]["contextlab" if jev else "baseline"]["recorded_outcomes"] == {"passed": 1}
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in existing.items())
    assert (tmp_path / "audit.json").read_text() == "preserve prior independent audit"
    assert "secret" not in json.dumps(result)


def test_missing_database_does_not_create_one(tmp_path):
    result = audit_run(tmp_path)
    assert not result["ok"] and "ledger.sqlite is missing" in result["errors"]
    assert list(tmp_path.iterdir()) == []


def test_unresolved_reservation_and_failed_calls_are_not_omitted(tmp_path):
    call = example_run(tmp_path, failed=True)
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        BudgetManager(store, "1", "1").reserve("s", "outstanding", "claude-test", "main", ".2", 1)
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert any("unresolved reservation" in error for error in result["errors"])
    assert result["total_cost_usd"] == str(call.cost_usd)
    assert result["outstanding_reserved_usd"] == "0.2"
    assert result["committed_exposure_usd"] == str(call.cost_usd + Decimal(".2"))
    assert result["statuses"] == {"failed": 1}
    assert any("Failed provider calls" in warning for warning in result["warnings"])
    assert "secret" not in json.dumps(result)


def test_unknown_usage_retains_conservative_charge_and_requires_reconciliation(tmp_path):
    example_run(tmp_path, unknown=True)
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert result["total_cost_usd"] == "0.1"
    assert result["cost_bases"] == {"reservation": 1}
    assert result["reservation_statuses"] == {"settled": 1}
    assert all(cap["blocked"] for cap in result["caps"])
    assert not result["all_live_receipts_verified"]
    assert any("unknown usage requires" in error for error in result["errors"])


def test_simulation_is_explicit_not_mislabeled_as_provider_spend(tmp_path):
    example_run(tmp_path, simulation=True)
    result = audit_run(tmp_path)
    assert result["ok"]
    assert result["receipt_verified_calls"] == 0 and result["cost_bases"] == {"simulation": 1}
    assert "Simulation evidence is not measured provider spend" in result["warnings"]


@pytest.mark.parametrize("sql,needle", [
    ("UPDATE model_calls SET input_tokens=999", "recorded usage differ"),
    ("UPDATE model_calls SET cache_write_tokens=150", "aggregate cache creation"),
    ("UPDATE model_calls SET cost_usd='0.009'", "recorded cost differ"),
    ("UPDATE reservations SET status='reserved'", "not settled"),
    ("DELETE FROM reservations", "no reservation"),
    ("DELETE FROM model_calls", "status and recorded call presence"),
    ("UPDATE budgets SET cap_usd='0'", "exceeds cap"),
    ("UPDATE model_calls SET session_id='missing'", "foreign-key"),
    ("UPDATE model_calls SET pricing_json='{}'", "malformed ledger accounting"),
    ("UPDATE reservations SET amount_usd='secret-invalid-amount'", "malformed reservation"),
])
def test_corruption_is_detected_and_sanitized(tmp_path, sql, needle):
    example_run(tmp_path)
    mutate(tmp_path, sql)
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert any(needle in error for error in result["errors"]), result["errors"]
    assert "secret" not in json.dumps(result)


def test_changed_archived_content_hash_is_detected(tmp_path):
    example_run(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        db.execute("DROP TRIGGER chunks_no_content_update")
        db.execute("UPDATE chunks SET content_sha256='0'")
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert "Chunk #1: archived content hash mismatch" in result["errors"]


@pytest.mark.parametrize("jev", [False, True])
def test_malformed_provider_usage_is_safe_and_not_marked_verified(tmp_path, jev):
    example_run(tmp_path, jev=jev)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        data = json.loads(db.execute("SELECT record_json FROM model_calls").fetchone()[0])
        raw = data["raw_usage"]
        if jev:
            raw["receipt"]["usage"]["input_tokens"] = "secret-malformed"
        else:
            raw["messages"][0]["provider_usage"]["input_tokens"] = "secret-malformed"
        db.execute("UPDATE model_calls SET record_json=?, raw_usage_json=?", (json.dumps(data), json.dumps(raw)))
    result = audit_run(tmp_path)
    assert not result["ok"] and result["receipt_verified_calls"] == 0
    assert "Call #1: original provider usage is missing or malformed" in result["errors"]
    assert "secret" not in json.dumps(result)


def test_unrecognized_receipt_has_explicit_coverage_warning(tmp_path):
    example_run(tmp_path)
    rewrite_call(tmp_path, lambda row: row.update(raw_usage={"different": "secret"}))
    mutate(tmp_path, "UPDATE model_calls SET raw_usage_json=?", (json.dumps({"different": "secret"}),))
    result = audit_run(tmp_path)
    assert result["ok"] and not result["all_live_receipts_verified"]
    assert any("unrecognized" in warning for warning in result["warnings"])
    assert "secret" not in json.dumps(result)


def test_result_cost_and_call_count_mismatches_are_detected(tmp_path):
    example_run(tmp_path)
    path = tmp_path / "s" / "result.json"
    data = json.loads(path.read_text())
    data.update(cost_usd="9", turns=10)
    path.write_text(json.dumps(data))
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert any("result cost differs" in error for error in result["errors"])
    assert any("result call count differs" in error for error in result["errors"])


def test_historical_plan_validates_saved_digest_without_current_source_comparison(tmp_path):
    example_run(tmp_path)
    plan = {"source_hash": "historical-version-no-longer-current", "config": {"session_cap_usd": "1", "experiment_cap_usd": "1"}}
    plan["approval_digest"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    assert audit_run(tmp_path)["ok"]
    plan["config"]["session_cap_usd"] = "2"
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert "Saved plan digest does not match its contents" in result["errors"]
    assert any("cap differs from saved plan" in error for error in result["errors"])


def test_matching_table_and_record_cost_must_still_equal_saved_token_prices(tmp_path):
    example_run(tmp_path)
    rewrite_call(tmp_path, lambda row: row.update(cost_usd="0.01"))
    mutate(tmp_path, "UPDATE model_calls SET cost_usd='0.01'")
    result = audit_run(tmp_path)
    assert not result["ok"]
    assert "Call #1: Decimal token-price cost mismatch" in result["errors"]


def test_per_arm_and_purpose_aggregates_include_failed_decision_cost(tmp_path):
    main = example_run(tmp_path)
    pricing = Pricing(model="jev-1.13.0", input_per_million=Decimal(".042"),
                      cache_write_5m_per_million=Decimal(".042"), cache_write_1h_per_million=Decimal(".042"),
                      cache_read_per_million=Decimal(".042"), output_per_million=0)
    usage = Usage(input_tokens=1000, output_tokens=8)
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("second", "contextlab", "test", model_main="claude-test", mode="live")
        budgets = BudgetManager(store, "1", "1")
        budgets.reserve("second", "decision", pricing.model, "decision", ".1", 0)
        decision = ModelCall(id="decision", session_id="second", seq=0, purpose="decision", model=pricing.model,
                             pricing=pricing, usage=usage, cost_usd=price_usage(usage, pricing), status="failed",
                             raw_usage={"provider": "typesafe", "receipt": {"model": pricing.model,
                                        "usage": {"input_tokens": 1000, "output_tokens": 8}}})
        budgets.settle("decision", decision)
        store.finish_session("second", False)
    result = audit_run(tmp_path)
    assert result["ok"] and result["total_calls"] == result["receipt_verified_calls"] == 2
    assert Decimal(result["total_cost_usd"]) == main.cost_usd + decision.cost_usd
    assert result["per_arm"]["contextlab"]["cost_usd"] == str(decision.cost_usd)
    assert result["per_purpose"]["decision"]["statuses"] == {"failed": 1}
    assert result["recorded_outcomes"] == {"passed": 1, "failed": 1}


def test_unsupported_schema_cannot_expose_stored_fields(tmp_path):
    example_run(tmp_path)
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        db.execute("ALTER TABLE sessions RENAME COLUMN id TO secret_unsupported_column")
    result = audit_run(tmp_path)
    assert not result["ok"] and result["total_cost_usd"] is None
    assert "secret" not in json.dumps(result)
