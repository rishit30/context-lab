"""Exercise live-run accounting with no provider or container calls."""
from decimal import Decimal
from types import SimpleNamespace

from contextlab.core.models import Pricing, Usage
from contextlab.ledger import BudgetManager, LedgerStore, ModelCall
from contextlab.ledger.pricing import price_usage, worst_case_cost
from harness.config import ExperimentConfig, experiment_plan
from harness.runner import run_live


def config(arms=("baseline",), **updates):
    fields = dict(arms=arms, tasks=("inventory",), repeats=1,
                  session_cap_usd="1", experiment_cap_usd="3", max_model_calls=3,
                  max_input_tokens=10000, max_output_tokens=100, threshold_tokens=2000,
                  threshold_keep=2, docker_image="local-test-image")
    return ExperimentConfig(**(fields | updates))


def pricing():
    return Pricing(model="fake-budget-model", input_per_million=3, cache_write_5m_per_million=4,
                   cache_write_1h_per_million=6, cache_read_per_million="0.3", output_per_million=15,
                   source="test-only fixture", verified_at="2026-10-10")


def fake_agent(monkeypatch, *, fail_normalization=False):
    import langchain_anthropic
    from langchain_core.messages import HumanMessage

    import harness.agent

    receipt = {"messages": [{"usage_metadata": {"input_tokens": 100, "output_tokens": 5},
                             "provider_usage": {"input_tokens": 100, "output_tokens": 5}}]}
    usage = Usage(input_tokens=100, output_tokens=5)
    calls = {"count": 0, "inference": 0}

    class FakeModel:
        def get_num_tokens_from_messages(self, *args, **kwargs):
            calls["count"] += 1
            return 100

    def build(**kwargs):
        class Agent:
            def invoke(self, state, config):
                request = SimpleNamespace(system_message=None, messages=[HumanMessage(content="fixture")], tools=[])
                kwargs["before_call"](request)
                calls["inference"] += 1
                kwargs["on_usage"]("fake-budget-model", usage, 5.0, "ok", None, receipt)
                raise RuntimeError("End fake task before any executor call")
        return Agent()

    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: FakeModel())
    monkeypatch.setattr(harness.agent, "build_agent", build)
    if fail_normalization:
        def fail(*args):
            raise ValueError("local rendering failure")
        monkeypatch.setattr("contextlab.adapters.langchain_mw.normalize_messages", fail)
    return calls, usage, receipt


def run(root, cfg, rates):
    plan = experiment_plan(cfg, rates, rates.model)
    return run_live(root, cfg, rates, rates.model, plan["approval_digest"])


def test_raw_receipt_and_full_input_ceiling_reservation_preserved(monkeypatch, tmp_path):
    calls, usage, receipt = fake_agent(monkeypatch)
    cfg, rates = config(), pricing()
    root = tmp_path / "experiment"
    results = run(root, cfg, rates)
    assert calls == {"count": 1, "inference": 1}
    assert results[0].cost_usd == price_usage(usage, rates)
    with LedgerStore(root / "ledger.sqlite") as store:
        charged = store.list_model_calls(results[0].session_id)[0]
        assert charged.raw_usage == receipt
        reserved = store.connection.execute("SELECT amount_usd FROM reservations").fetchone()[0]
        assert Decimal(reserved) == worst_case_cost(cfg.max_input_tokens, cfg.max_output_tokens, rates)


def test_transient_settlement_failure_retries_original_usage_receipt(monkeypatch, tmp_path):
    calls, usage, receipt = fake_agent(monkeypatch)
    original = BudgetManager.settle
    attempts = []

    def flaky(self, call_id, call=None, **kwargs):
        attempts.append(call)
        if len(attempts) == 1:
            raise RuntimeError("transient ledger write error")
        return original(self, call_id, call, **kwargs)

    monkeypatch.setattr(BudgetManager, "settle", flaky)
    rates = pricing()
    root = tmp_path / "experiment"
    results = run(root, config(), rates)
    assert calls["inference"] == 1
    assert len(attempts) == 2 and all(isinstance(item, ModelCall) for item in attempts)
    assert attempts[0] == attempts[1]
    assert results[0].cost_usd == price_usage(usage, rates)
    with LedgerStore(root / "ledger.sqlite") as store:
        charged = store.list_model_calls(results[0].session_id)[0]
        assert charged.raw_usage == receipt
        assert charged.cost_basis == "usage"


def test_persistent_settlement_failure_preserves_reservation_and_stops(monkeypatch, tmp_path):
    calls, _, _ = fake_agent(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError("persistent ledger write error")
    monkeypatch.setattr(BudgetManager, "settle", fail)
    cfg, rates = config(arms=("baseline", "contextlab")), pricing()
    root = tmp_path / "experiment"
    results = run(root, cfg, rates)
    assert calls == {"count": 1, "inference": 1}
    assert len(results) == 1
    assert results[0].error == "LedgerSettlementError"
    assert results[0].cost_usd == worst_case_cost(cfg.max_input_tokens, cfg.max_output_tokens, rates)
    assert "unsettled" in results[0].notes
    with LedgerStore(root / "ledger.sqlite") as store:
        assert store.list_model_calls(results[0].session_id) == []
        assert store.connection.execute("SELECT status FROM reservations").fetchone()[0] == "reserved"


def test_local_rendering_failure_does_not_reserve_or_charge(monkeypatch, tmp_path):
    calls, _, _ = fake_agent(monkeypatch, fail_normalization=True)
    root = tmp_path / "experiment"
    results = run(root, config(), pricing())
    assert calls == {"count": 1, "inference": 0}
    assert results[0].cost_usd == 0
    with LedgerStore(root / "ledger.sqlite") as store:
        assert store.connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


def test_insufficient_experiment_cap_stops_before_token_count_endpoint(monkeypatch, tmp_path):
    calls, _, _ = fake_agent(monkeypatch)
    root = tmp_path / "experiment"
    cfg = config(arms=("baseline", "contextlab"), experiment_cap_usd="0.01")
    results = run(root, cfg, pricing())
    assert calls == {"count": 0, "inference": 0}
    assert len(results) == 1
    assert results[0].error == "BudgetExceeded"
