"""Cross-component regression checks; every provider and executor is fake."""

from decimal import Decimal
from types import SimpleNamespace

import pytest

from contextlab.core.models import Pricing
from contextlab.ledger.store import LedgerStore
from harness.config import ExperimentConfig, experiment_plan
from harness.runner import run_live


def review_config(**updates):
    fields = dict(
        arms=("baseline", "threshold_edit", "contextlab"),
        tasks=("inventory",),
        repeats=1,
        session_cap_usd=Decimal("1"),
        experiment_cap_usd=Decimal("3"),
        max_model_calls=3,
        max_input_tokens=10000,
        max_output_tokens=100,
        threshold_tokens=2000,
        threshold_keep=2,
        docker_image="local-test-image",
    )
    return ExperimentConfig(**(fields | updates))


def review_pricing():
    return Pricing(
        model="fake-review-model",
        input_per_million=3,
        cache_write_5m_per_million=4,
        cache_write_1h_per_million=6,
        cache_read_per_million=0.3,
        output_per_million=15,
        source="test-only fixture",
        verified_at="2026-10-10",
    )


def wire_fake_runner(monkeypatch, *, counter_error=False):
    import langchain_anthropic
    from langchain_core.messages import HumanMessage

    import harness.agent

    counters = {"count": 0, "inference": 0}

    class FakeModel:
        def get_num_tokens_from_messages(self, messages, **kwargs):
            counters["count"] += 1
            if counter_error:
                raise RuntimeError("counter unavailable")
            return 100

    def fake_build_agent(**kwargs):
        class FakeAgent:
            def invoke(self, state, config):
                request = SimpleNamespace(
                    system_message=None, messages=[HumanMessage(content="fixture")], tools=[]
                )
                kwargs["before_call"](request)
                counters["inference"] += 1
                kwargs["on_usage"]("fake-review-model", None, 0, "usage_missing", None)
                raise AssertionError("unknown usage must abort")

        return FakeAgent()

    monkeypatch.setattr("harness.runner.preflight", lambda *args: [])
    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", lambda **kwargs: FakeModel())
    monkeypatch.setattr(harness.agent, "build_agent", fake_build_agent)
    return counters


def test_unknown_provider_usage_stops_remaining_experiment_matrix(monkeypatch, tmp_path):
    counters = wire_fake_runner(monkeypatch)
    config, pricing = review_config(), review_pricing()
    plan = experiment_plan(config, pricing, pricing.model)
    results = run_live(
        tmp_path / "experiment", config, pricing, pricing.model, plan["approval_digest"]
    )
    assert counters == {"count": 1, "inference": 1}
    assert len(results) == 1
    assert results[0].error == "BudgetExceeded"
    assert results[0].cost_usd > 0


def test_counter_failure_prevents_inference_and_does_not_invent_usage(monkeypatch, tmp_path):
    counters = wire_fake_runner(monkeypatch, counter_error=True)
    config, pricing = review_config(arms=("baseline",)), review_pricing()
    plan = experiment_plan(config, pricing, pricing.model)
    root = tmp_path / "experiment"
    results = run_live(root, config, pricing, pricing.model, plan["approval_digest"])
    assert counters == {"count": 1, "inference": 0}
    assert len(results) == 1 and results[0].error == "RuntimeError"
    assert results[0].cost_usd == 0 and results[0].turns == 0
    with LedgerStore(root / "ledger.sqlite") as store:
        assert store.list_model_calls(results[0].session_id) == []


def test_approval_digest_changes_with_caps_and_prices():
    config, pricing = review_config(), review_pricing()
    original = experiment_plan(config, pricing, pricing.model)
    changed_cap = experiment_plan(
        config.model_copy(update={"experiment_cap_usd": Decimal("2")}), pricing, pricing.model
    )
    changed_price = experiment_plan(
        config, pricing.model_copy(update={"output_per_million": Decimal("16")}), pricing.model
    )
    assert original["approval_digest"] != changed_cap["approval_digest"]
    assert original["approval_digest"] != changed_price["approval_digest"]
    assert Decimal(original["maximum_authorized_cost_usd"]) == Decimal("3")


def test_wrong_approval_digest_does_not_call_preflight(monkeypatch, tmp_path):
    def unexpected(*args):
        pytest.fail("Preflight must not run for a stale approval")

    monkeypatch.setattr("harness.runner.preflight", unexpected)
    with pytest.raises(ValueError, match="Approval does not match"):
        run_live(
            tmp_path / "experiment", review_config(), review_pricing(), "fake-review-model", "stale"
        )
    assert not (tmp_path / "experiment").exists()
