"""Offline five-arm integration and approval guards, never provider-quality evidence."""
import json
import socket
import sqlite3
from decimal import Decimal

import pytest

from contextlab.ledger.store import ModelCall
from examples import routing_agent
from harness.workspace import Workspace


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline test attempted network access")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    files = tmp_path / "files"
    files.mkdir()
    (files / "service.txt").write_bytes(b"owner: ops\r\nrollback: errors > 5%\r\n")
    paths = []
    for name, price in (("main", "3"), ("worker", "1")):
        path = tmp_path / f"{name}.yaml"
        path.write_text(f"model: fixture-{name}\ninput_per_million: '{price}'\noutput_per_million: '{price}'\n"
                        f"cache_write_5m_per_million: '{price}'\ncache_write_1h_per_million: '{price}'\n"
                        f"cache_read_per_million: '{price}'\nsource: unit-fixture\nverified_at: '2026-10-10'\n")
        paths.append(path)
    monkeypatch.setattr(routing_agent, "source_manifest", lambda: [("test-fixture", "unchanged")])
    loaded = []
    monkeypatch.setattr(routing_agent, "load_anthropic_key", lambda path: loaded.append("anthropic"))
    monkeypatch.setattr(routing_agent, "load_typesafe_key", lambda path: loaded.append("typesafe"))
    return files, paths[0], paths[1], loaded


def make_plan(inputs, **kwargs):
    files, main, worker, _ = inputs
    return routing_agent.plan(files=files, pricing_path=main, worker_pricing_path=worker, **kwargs)


def test_five_arm_plan_binds_models_caps_and_equal_agent_contract_without_loading_keys(inputs):
    result = make_plan(inputs)
    assert tuple(result["settings"]) == routing_agent.ARMS
    assert result["approval_digest"] == routing_agent.approval_digest(result)
    assert result["maximum_authorized_cost_usd"] == "5"
    assert {row["pricing"]["model"] for row in result["settings"].values()} == {"fixture-main"}
    assert {row["routing"]["worker_pricing"]["model"] for row in result["settings"].values()} == {"fixture-worker"}
    for arm, row in result["settings"].items():
        assert row["routing"]["enabled"] == (arm in routing_agent.ROUTING_ARMS)
        assert row["mode"] == "live" and row["cache_mode"] == "disabled"
        assert Decimal(row["session_cap_usd"]) == 1
    assert inputs[3] == []


@pytest.mark.parametrize("change", ["approval", "main_price", "worker_price", "caps", "questions", "sources", "source_code", "mode"])
def test_invalid_plan_rejected_before_keys_output_or_session(inputs, tmp_path, monkeypatch, change):
    reviewed = make_plan(inputs)
    approval = reviewed["approval_digest"]
    if change == "approval":
        approval = "wrong"
    elif change in {"main_price", "worker_price"}:
        target = reviewed["settings"]["contextlab"]
        (target["pricing"] if change == "main_price" else target["routing"]["worker_pricing"])["input_per_million"] = "9"
    elif change == "caps":
        reviewed["settings"]["baseline"]["session_cap_usd"] = "9"
    elif change == "questions":
        reviewed["questions"] = ["unreviewed"]
    elif change == "sources":
        (inputs[0] / "service.txt").write_text("changed")
    elif change == "source_code":
        monkeypatch.setattr(routing_agent, "source_manifest", lambda: [("test-fixture", "changed")])
    else:
        reviewed["mode"] = "simulation"
    monkeypatch.setattr(routing_agent, "ContextLabSession", lambda *a, **k: pytest.fail("Session constructed"))
    output = tmp_path / "out"
    with pytest.raises(ValueError):
        routing_agent.run(output, reviewed, approval)
    assert inputs[3] == [] and not output.exists()


def test_source_evidence_preserves_original_line_endings(inputs):
    source = routing_agent.evidence_sources(Workspace(inputs[0]))[0]
    assert source.id == source.path == "service.txt"
    assert source.content.encode() == (inputs[0] / "service.txt").read_bytes()


@pytest.mark.parametrize("mutation", ["too_large", "too_many"])
def test_plan_rejects_unsupported_catalog_before_key_loading(inputs, mutation):
    if mutation == "too_large":
        (inputs[0] / "service.txt").write_text("x" * 32_001)
    else:
        for index in range(64):
            (inputs[0] / f"extra-{index}.txt").write_text("fixture")
    with pytest.raises(ValueError):
        make_plan(inputs)
    assert inputs[3] == []


@pytest.mark.parametrize("mutation", ["arm", "ceiling"])
def test_self_consistent_digest_cannot_bypass_live_plan_invariants(inputs, tmp_path, mutation):
    reviewed = make_plan(inputs)
    if mutation == "arm":
        reviewed["settings"]["cl_compact_only"]["arm"] = "baseline"
    else:
        reviewed["maximum_authorized_cost_usd"] = "0.01"
    reviewed["approval_digest"] = routing_agent.approval_digest(reviewed)
    with pytest.raises(ValueError):
        routing_agent.run(tmp_path / "out", reviewed, reviewed["approval_digest"])
    assert inputs[3] == [] and not (tmp_path / "out").exists()


@pytest.mark.parametrize("mutation", ["main_model", "worker_model", "limits", "cache", "routing", "shadow"])
def test_approved_plan_cannot_mislabel_unequal_comparison_settings(inputs, tmp_path, mutation):
    reviewed = make_plan(inputs)
    row = reviewed["settings"]["contextlab"]
    if mutation == "main_model":
        row["pricing"]["model"] = "different-main"
    elif mutation == "worker_model":
        row["routing"]["worker_pricing"]["model"] = "different-worker"
    elif mutation == "limits":
        row["max_model_calls"] += 1
    elif mutation == "cache":
        row["cache_mode"] = "enabled"
    elif mutation == "routing":
        row["routing"]["enabled"] = False
    else:
        row["shadow"] = True
    reviewed["approval_digest"] = routing_agent.approval_digest(reviewed)
    with pytest.raises(ValueError, match="comparison|Comparison"):
        routing_agent.run(tmp_path / "out", reviewed, reviewed["approval_digest"])
    assert inputs[3] == [] and not (tmp_path / "out").exists()


def test_demo_runs_real_five_arm_graphs_with_fallback_and_all_in_simulated_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(routing_agent, "load_anthropic_key", lambda *args: pytest.fail("Demo loaded a key"))
    monkeypatch.setattr(routing_agent, "load_typesafe_key", lambda *args: pytest.fail("Demo loaded a key"))
    monkeypatch.setattr(routing_agent, "source_manifest", lambda: [("test-fixture", "unchanged")])
    models = []
    model_type = routing_agent.SimulationMain

    def make_model():
        model = model_type()
        models.append(model)
        return model

    monkeypatch.setattr(routing_agent, "SimulationMain", make_model)
    output = tmp_path / "demo"
    result = routing_agent.demo(output)
    assert result["mode"] == "simulation" and result["provider_spend_usd"] == "0"
    assert result["all_arms_attempted"] and result["error_type"] is None
    assert tuple(row["configuration"] for row in result["sessions"]) == routing_agent.ARMS
    assert all(row["mechanical_fixture_passed"] and row["questions_completed"] == 3 for row in result["sessions"])
    assert len(models) == 5
    assert all(model._schemas == models[0]._schemas for model in models)
    assert {schema["function"]["name"] for schema in models[0]._schemas} == {
        "list_files", "read_file", "recall", "analyze_files",
    }
    db = sqlite3.connect(output / "ledger.sqlite")
    db.row_factory = sqlite3.Row
    calls = [ModelCall.model_validate_json(row["record_json"]) for row in db.execute("SELECT record_json FROM model_calls")]
    assert Decimal(result["total_cost_usd"]) == sum((call.cost_usd for call in calls), Decimal(0))
    assert all(call.cost_basis == "simulation" for call in calls)
    assert {row[0] for row in db.execute("SELECT mode FROM sessions")} == {"simulation"}
    assert db.execute("SELECT COUNT(*) FROM reservations WHERE status='reserved'").fetchone()[0] == 0
    for row in result["sessions"]:
        if row["configuration"] in {"cl_compact_only", "contextlab"}:
            assert row["modified_requests_this_process"] > 0
        elif row["configuration"] in {"baseline", "cl_route_only"}:
            assert row["modified_requests_this_process"] == 0
        delegation_calls = [call for call in calls if call.session_id == row["configuration"] and call.purpose == "delegate"]
        if row["configuration"] in routing_agent.ROUTING_ARMS:
            assert len(delegation_calls) == 2
            assert {outcome["status"] for outcome in row["tool_outcomes"]} == {"delegated", "fallback"}
        else:
            assert not delegation_calls
    db.close()
    assert json.loads((output / "summary.json").read_text()) == result
    html = (output / "report.html").read_text()
    assert "OFFLINE SIMULATION" in html and "Synthetic accounting units" in html
    assert "No LLM quality" in html and "no optimization savings" in html


def test_report_escapes_answers_and_does_not_call_simulation_real_dollars(tmp_path):
    from harness.report.routing import build_routing_report
    result = {"mode": "simulation", "total_cost_usd": "0.01", "quality_note": "No model-quality evidence",
              "cost_basis": "Synthetic accounting, no provider spend", "sessions": [{
                  "configuration": "baseline", "answers": ["<script>secret()</script>"], "tool_outcomes": []}]}
    html = build_routing_report(tmp_path, result).read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "Usage cost at reviewed list prices (USD)" not in html
