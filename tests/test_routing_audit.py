"""Saved five-arm results reconcile independently; tampering cannot lower cost."""
import json
import socket

import pytest

from examples import routing_agent
from harness.audit import audit_run


@pytest.fixture
def saved_run(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Audit fixtures must not contact a provider")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(routing_agent, "source_manifest", lambda: [("test-fixture", "unchanged")])
    output = tmp_path / "routing"
    result = routing_agent.demo(output)
    assert result["error_type"] is None
    assert all(row["mechanical_fixture_passed"] for row in result["sessions"])
    return output


def test_audit_reconciles_five_configurations_and_delegate_receipts(saved_run):
    report = audit_run(saved_run)
    assert report["ok"], report["errors"]
    assert set(report["per_arm"]) == set(routing_agent.ARMS)
    assert report["per_purpose"]["delegate"]["calls"] == 4
    assert any("Simulation" in warning for warning in report["warnings"])


@pytest.mark.parametrize("field,value", [("delegate_cost_usd", "0"), ("delegate_calls", 0),
                                         ("total_cost_usd", "0"), ("mode", "live")])
def test_audit_detects_changed_routing_report_totals(saved_run, field, value):
    path = saved_run / "contextlab.json"
    result = json.loads(path.read_text())
    result[field] = value
    path.write_text(json.dumps(result))
    report = audit_run(saved_run)
    assert not report["ok"]
    assert any("result" in error for error in report["errors"])


@pytest.mark.parametrize("mutation", ["worker_price", "delegate_disabled", "call_limit", "cap"])
def test_audit_compares_routing_receipts_with_the_saved_arm_plan(saved_run, mutation):
    path = saved_run / "plan.json"
    plan = json.loads(path.read_text())
    setting = plan["settings"]["contextlab"]
    if mutation == "worker_price":
        setting["routing"]["worker_pricing"]["input_per_million"] = "999"
    elif mutation == "delegate_disabled":
        setting["routing"]["enabled"] = False
    elif mutation == "call_limit":
        setting["routing"]["max_calls_per_session"] = 1
    else:
        setting["experiment_cap_usd"] = "999"
    path.write_text(json.dumps(plan))
    report = audit_run(saved_run)
    assert not report["ok"]
    assert any("plan" in error or "limit exceeded" in error for error in report["errors"])
