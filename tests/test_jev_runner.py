"""Jev wiring, shared budgets and overhead reporting; all providers are fake."""
import json
import sys
from decimal import Decimal
from types import SimpleNamespace

import pytest

from contextlab.core.models import Pricing, ProjectionResult, RunResult, Usage, ViewState
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.config import ExperimentConfig, experiment_plan
from harness.report import build_report
from harness.runner import preflight, run_live
from harness.scoreboard import scoreboard


def configuration(**updates):
    values = dict(
        arms=("contextlab", "baseline"), tasks=("inventory",), repeats=1,
        session_cap_usd=Decimal("1"), experiment_cap_usd=Decimal("3"),
        max_model_calls=5, max_input_tokens=10000, max_output_tokens=100,
        threshold_tokens=2000, threshold_keep=2, docker_image="fake-local-image",
        decision_engine="jev",
    )
    return ExperimentConfig(**(values | updates))


def pricing():
    return Pricing(model="fake-main-model", input_per_million=1, output_per_million=2,
                   cache_write_5m_per_million=1, cache_write_1h_per_million=1,
                   cache_read_per_million=1, source="test-only", verified_at="2026-10-10")


def wire_runner(monkeypatch, mode="ok"):
    import langchain_anthropic
    from langchain_core.messages import HumanMessage

    import harness.agent

    counters = {"main_calls": 0, "counts": 0, "clients": [], "gates": []}

    class FakeClient:
        @classmethod
        def from_environment(cls, *, timeout_seconds):
            counters["clients"].append(timeout_seconds)
            return cls()

    class FakeGate:
        def __init__(self, client, store, budgets, session_id, *, config):
            self.store, self.budgets, self.session = store, budgets, session_id
            self.client, self.config = client, config
            self.pending = {}
            self.reconciled = False
            counters["gates"].append(self)

        def assess(self):
            seq = len(self.store.list_model_calls(self.session))
            call_id = f"{self.session}-decision-{seq}"
            self.budgets.reserve(self.session, call_id, "fake-decision", "decision", "0.004", seq)
            if mode == "unknown":
                self.budgets.settle(call_id, None)
                return
            record = ModelCall(id=call_id, session_id=self.session, seq=seq,
                               model="fake-decision", purpose="decision",
                               usage=Usage(input_tokens=50), cost_usd=Decimal("0.002"))
            if mode in {"pending", "reconcile_failure"}:
                self.pending.update(id=call_id, record=record)
                raise RuntimeError("Fake settlement failure")
            self.budgets.settle(call_id, record)

        def reconcile_pending(self):
            self.reconciled = True
            if mode == "reconcile_failure":
                raise RuntimeError("Fake persistent ledger failure")
            if self.pending:
                self.budgets.settle(self.pending["id"], self.pending["record"])
                self.pending.clear()

    class FakeModel:
        def get_num_tokens_from_messages(self, messages, **kwargs):
            counters["counts"] += 1
            return 100

    class FakeExecutor:
        def __init__(self, *args):
            pass

        def __call__(self, *args):
            return {"exit_code": 0, "stdout": "CHECKS_PASSED"}

        def run(self, *args, **kwargs):
            return self()

    def build_agent(**kwargs):
        middleware = kwargs["context_middleware"]

        class FakeAgent:
            def invoke(self, state, config):
                if middleware is not None:
                    gate = middleware.controller._semantic_gate
                    if gate is not None:
                        try:
                            gate.assess()
                        except RuntimeError:
                            pass  # The real core retains context after a gate error.
                    middleware.telemetry.append({"changed": False})
                request = SimpleNamespace(system_message=None,
                                          messages=[HumanMessage(content="fake task")], tools=[])
                kwargs["before_call"](request)
                counters["main_calls"] += 1
                kwargs["on_usage"]("fake-main-model", Usage(input_tokens=100, output_tokens=10),
                                   1, "ok", None, {"test": True})
                return state

        return FakeAgent()

    monkeypatch.setitem(sys.modules, "contextlab.providers.jev", SimpleNamespace(JevClient=FakeClient))
    monkeypatch.setitem(sys.modules, "harness.jev_decision", SimpleNamespace(JevDecisionGate=FakeGate))
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: FakeModel())
    monkeypatch.setattr(harness.agent, "build_agent", build_agent)
    monkeypatch.setattr("harness.runner.DockerExecutor", FakeExecutor)
    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    return counters


def run_frozen(monkeypatch, output, config):
    rates = pricing()
    plan = experiment_plan(config, rates, rates.model)
    # Other build agents may edit sources concurrently; this test exercises the
    # runner with one frozen fixture plan, not approval hash races.
    monkeypatch.setattr("harness.runner.experiment_plan", lambda *args: plan)
    return run_live(output, config, rates, rates.model, plan["approval_digest"])


def test_gate_only_wires_contextlab_and_overhead_uses_same_ledger_and_sequence(monkeypatch, tmp_path):
    counters = wire_runner(monkeypatch)
    output = tmp_path / "experiment"
    results = run_frozen(monkeypatch, output, configuration())
    assert len(counters["gates"]) == 1 and counters["gates"][0].reconciled
    assert counters["clients"] == [10]
    contextlab, baseline = results
    assert contextlab.solved and baseline.solved
    assert contextlab.turns == 4 and contextlab.decision_calls == 2
    assert contextlab.decision_cost_usd == Decimal("0.004")
    assert contextlab.cost_usd == Decimal("0.00424")
    assert contextlab.usage.input_tokens == 300
    assert contextlab.decision_usage.input_tokens == 100
    assert baseline.turns == 2 and baseline.decision_calls == 0
    trace = json.loads((output / contextlab.session_id / "trace.json").read_text())
    assert [entry["model_call_seq"] for entry in trace["interventions"]] == [1, 3]
    with LedgerStore(output / "ledger.sqlite") as store:
        calls = store.list_model_calls(contextlab.session_id)
        assert [call.purpose for call in calls] == ["decision", "main", "decision", "main"]
        assert store.session_summary(contextlab.session_id)["cost_usd"] == contextlab.cost_usd


def test_unknown_decision_usage_blocks_main_and_remaining_matrix(monkeypatch, tmp_path):
    counters = wire_runner(monkeypatch, "unknown")
    results = run_frozen(monkeypatch, tmp_path / "experiment", configuration())
    assert len(results) == 1 and counters["main_calls"] == 0 and counters["counts"] == 0
    assert results[0].error == "BudgetExceeded"
    assert results[0].decision_calls == 1 and results[0].cost_usd == Decimal("0.004")
    assert results[0].decision_cost_usd == results[0].cost_usd


def test_pending_decision_blocks_main_before_token_count_then_reconciles(monkeypatch, tmp_path):
    counters = wire_runner(monkeypatch, "pending")
    results = run_frozen(monkeypatch, tmp_path / "experiment", configuration(arms=("contextlab",)))
    assert counters["counts"] == 0 and counters["main_calls"] == 0
    assert counters["gates"][0].reconciled and not counters["gates"][0].pending
    assert results[0].decision_cost_usd == Decimal("0.002")
    assert results[0].error == "BudgetExceeded"


def test_decision_settlement_failure_stops_matrix_and_includes_reserved_cost(monkeypatch, tmp_path):
    counters = wire_runner(monkeypatch, "reconcile_failure")
    results = run_frozen(monkeypatch, tmp_path / "experiment", configuration())
    assert len(results) == 1 and counters["main_calls"] == 0
    assert results[0].error == "LedgerSettlementError"
    assert results[0].cost_usd == results[0].decision_cost_usd == Decimal("0.004")
    assert "unsettled" in results[0].notes


@pytest.mark.parametrize("engine,expected", [("rules", False), ("jev", True)])
def test_preflight_requires_typesafe_key_only_for_jev(monkeypatch, engine, expected):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-test-key")
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("harness.runner.docker_status", lambda *args: (True, "fake ready"))
    issues = preflight(configuration(decision_engine=engine), pricing(), pricing().model)
    assert any("TYPESAFE_API_KEY" in issue for issue in issues) == expected


def test_semantic_assessment_persists_without_duplicate_financial_charge(tmp_path):
    payload = {"engine": "jev", "allowed_ids": ["old"], "needed_probabilities": {"old": 0.02},
               "cost_usd": "0.0001", "reason": "fixture", "fallback": False}
    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("session", "contextlab", "fixture", 0)
        projection = ProjectionResult(messages=[], state=ViewState(), semantic_decision=payload)
        store.record_projection("session", 1, projection)
        store.record_projection("session", 1, projection)
        rows = store.connection.execute("SELECT * FROM decisions").fetchall()
        assert len(rows) == 1 and rows[0]["decision_type"] == "semantic_relevance"
        assert json.loads(rows[0]["output_json"]) == payload
        assert Decimal(rows[0]["cost_usd"]) == 0
        assert store.get_projection("session").semantic_decision == payload
        assert store.session_summary("session")["cost_usd"] == 0


def test_reports_preserve_legacy_defaults_and_split_overhead_without_double_count(tmp_path):
    legacy = RunResult.model_validate({"session_id": "old", "arm": "baseline", "task_id": "task",
                                     "repeat": 0, "mode": "live", "solved": True, "cost_usd": "1"})
    newer = legacy.model_copy(update={"session_id": "new", "arm": "contextlab",
                                     "cost_usd": Decimal("2"), "decision_calls": 2,
                                     "decision_cost_usd": Decimal("0.01"),
                                     "decision_usage": Usage(input_tokens=100)})
    assert legacy.decision_calls == 0 and legacy.decision_cost_usd == 0
    rows = scoreboard([legacy, newer])
    row = next(item for item in rows if item["arm"] == "contextlab")
    assert row["total_cost_usd"] == "2" and row["decision_cost_usd"] == "0.01"
    for result in (legacy, newer):
        directory = tmp_path / result.session_id
        directory.mkdir()
        (directory / "result.json").write_text(result.model_dump_json())
    html = build_report(tmp_path).read_text()
    assert "All model calls" in html and "Decision calls / cost" in html
    assert "already included" in html
    assert "decision_input_tokens" in (tmp_path / "summary.csv").read_text()
