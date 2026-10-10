from __future__ import annotations

import json
import sqlite3
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage

import contextlab.routing as routing
from contextlab.core.models import Pricing
from contextlab.core.routing import (
    EvidenceSource,
    EvidenceTask,
    RoutingConfig,
    build_evidence_brief,
)
from contextlab.ledger.budget import BudgetExceeded, BudgetManager
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerConflict, LedgerStore
from contextlab.providers.jev_config import JevConfig
from contextlab.routing import DelegationRouter


def rates(model="worker", normal=".1", output=".5"):
    n = Decimal(normal)
    return Pricing(model=model, input_per_million=n, cache_write_5m_per_million=n * Decimal("1.25"),
                   cache_write_1h_per_million=n * 2, cache_read_per_million=n / 10,
                   output_per_million=Decimal(output), source="synthetic")


MAIN = rates("main", "3", "15")
WORKER = rates()
CONTENT = "timeout: 30\nowner: platform\n" + "#" + " filler" * 1800 + "\n"


def evidence(content=CONTENT):
    return EvidenceTask(goal="Find the timeout", questions={"timeout": "What is the timeout?"},
                        source_ids=("a",)), (EvidenceSource(id="a", path="config/a.yaml", content=content),)


def answer(quote="timeout: 30\n", **extra):
    return json.dumps({"quotes": [{"question_id": "timeout", "source_id": "a", "start_line": 1,
                                  "end_line": 1, "quote": quote, **extra}]})


class Worker:
    def __init__(self, text=None, usage=None, count=4000, fail=False, model="worker", stop="end_turn"):
        self.text = answer() if text is None else text
        self.usage = {"input_tokens": 100, "output_tokens": 40} if usage is None else usage
        self.count, self.fail, self.model, self.stop = count, fail, model, stop
        self.counts, self.invocations = [], []

    def get_num_tokens_from_messages(self, messages):
        self.counts.append(messages)
        if isinstance(self.count, Exception):
            raise self.count
        return self.count

    def invoke(self, messages):
        self.invocations.append(messages)
        if self.fail:
            raise RuntimeError("secret provider error")
        return AIMessage(content=self.text, response_metadata={"model_name": self.model,
                         "stop_reason": self.stop, "usage": self.usage})


class Suitability:
    def __init__(self, probability=.99, invalid=False, usage=None, model="jev-1.13.0", fail=False):
        self.probability, self.invalid, self.model, self.fail = probability, invalid, model, fail
        self.usage = {"input_tokens": 1000, "output_tokens": 4} if usage is None else usage
        self.payloads = []

    def infer(self, payload):
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("secret TypeSafe error")
        return {"model": self.model, "usage": self.usage,
                "answers": {} if self.invalid else {"suitable": {"type": "noul", "noul": self.probability}}}


def setup_router(tmp_path, monkeypatch, worker=None, *, mode="simulation", cfg=None, client=None,
                 cap="1", jev_config=None, store=None):
    worker = worker or Worker()
    store = store or LedgerStore(tmp_path / "ledger.sqlite")
    store.create_session("s", "routing", "audit", model_main="main", mode=mode)
    budgets = BudgetManager(store, cap, cap)
    config = cfg or RoutingConfig(enabled=True, worker_pricing=WORKER, max_output_tokens=256)
    kwargs = {}
    if mode == "simulation":
        kwargs["worker_factory"] = lambda _: worker
    else:
        monkeypatch.setattr(routing, "_default_worker", lambda _: worker)
    router = DelegationRouter(store, budgets, "s", main_pricing=MAIN, config=config,
                              jev_config=jev_config, suitability_client=client, **kwargs)
    return router, store, budgets, worker


def run(router, request_id="request-1", content=CONTENT):
    task, sources = evidence(content)
    return router.analyze(task, sources, request_id=request_id)


def test_success_is_single_read_only_completion_with_original_receipt_and_shared_charge(tmp_path, monkeypatch):
    usage = {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 300,
             "cache_creation_input_tokens": 70,
             "cache_creation": {"ephemeral_5m_input_tokens": 20, "ephemeral_1h_input_tokens": 50}}
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch, Worker(usage=usage))
    result = run(router)
    assert result.status == "delegated" and result.selection.quotes[0].quote == "timeout: 30\n"
    assert result.sources == () and result.validation == "exact_quote_provenance_only"
    assert len(result.source_provenance) == 1 and result.source_provenance[0].path == "config/a.yaml"
    assert len(worker.counts) == len(worker.invocations) == 1
    assert worker.counts[0] == worker.invocations[0]
    assert [msg.type for msg in worker.invocations[0]] == ["system", "human"]
    call, = store.list_model_calls("s")
    assert call.purpose == "delegate" and call.id == result.delegate_call_id
    assert call.raw_usage["messages"][0]["provider_usage"] == usage
    assert call.usage.total_input_tokens == 470
    assert call.cost_usd == price_usage(call.usage, WORKER) == result.cost_usd
    assert call.cost_basis == "simulation" and result.provider_spend_usd == 0
    assert result.cost_basis == "simulation" and result.mode == "simulation"
    assert budgets.status("s")["session"]["spent_usd"] == result.cost_usd
    assert budgets.status("s")["experiment"]["reserved_usd"] == 0
    assert store.connection.execute("SELECT status FROM delegations").fetchone()[0] == "delegated"
    decision = store.connection.execute("SELECT * FROM decisions WHERE decision_type='delegation'").fetchone()
    assert json.loads(decision["inputs_json"])["brief"]["sources"][0]["content"] == evidence()[1][0].content
    assert json.loads(decision["output_json"])["delegate_call_id"] == call.id
    assert store.connection.execute("SELECT model_main FROM sessions").fetchone()[0] == "main"


def test_idempotency_is_durable_and_changed_content_conflicts(tmp_path, monkeypatch):
    router, store, _, worker = setup_router(tmp_path, monkeypatch)
    result = run(router)
    assert run(router) == result
    store.close()
    restarted, _, _, replacement = setup_router(tmp_path, monkeypatch)
    assert run(restarted) == result and replacement.invocations == [] and replacement.counts == []
    with pytest.raises(LedgerConflict, match="different inputs"):
        run(restarted, content="timeout: 40\n")
    assert len(worker.invocations) == 1


@pytest.mark.parametrize("kwargs,reason", [
    ({"text": "not json"}, "delegate_result_rejected"),
    ({"text": answer("made up\n")}, "delegate_result_rejected"),
    ({"text": answer(extra="not allowed")}, "delegate_result_rejected"),
    ({"stop": "max_tokens"}, "delegate_result_rejected"),
])
def test_failed_output_still_charged_and_falls_back_to_all_originals(tmp_path, monkeypatch, kwargs, reason):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch, Worker(**kwargs))
    result = run(router)
    assert result.status == "fallback" and result.reason == reason
    assert result.selection is None and result.sources == evidence()[1]
    call, = store.list_model_calls("s")
    assert call.status == "failed" and call.cost_usd > 0 and call.cost_basis == "simulation"
    assert len(worker.invocations) == 1 and not budgets.status("s")["experiment"]["blocked"]
    assert all(call.purpose != "main" for call in store.list_model_calls("s"))


@pytest.mark.parametrize("kwargs", [{"fail": True}, {"usage": {}}, {"model": "unpriced-model"}])
def test_unknown_usage_charges_worst_case_blocks_main_and_no_hidden_retry(tmp_path, monkeypatch, kwargs):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch, Worker(**kwargs), mode="live")
    result = run(router)
    call, = store.list_model_calls("s")
    assert result.status == "fallback" and result.sources == evidence()[1]
    assert call.status == "unknown_usage" and call.cost_basis == "reservation"
    assert call.cost_usd == worst_case_cost(router.config.max_input_tokens, router.config.max_output_tokens, WORKER)
    assert result.provider_spend_usd is None
    assert result.unknown_usage_calls == 1 and result.conservative_reservation_cost_usd == result.cost_usd
    assert result.cost_basis == "includes_conservative_reservation"
    assert budgets.status("s")["experiment"]["blocked"]
    assert "secret" not in call.model_dump_json()
    with pytest.raises(BudgetExceeded):
        budgets.reserve("s", "main-next", "main", "main", ".01", 1)
    assert run(router, "request-2").reason == "shared_budget_blocked"
    assert len(worker.counts) == len(worker.invocations) == 1


@pytest.mark.parametrize("count,reason", [(None, "delegate_count_invalid"), (True, "delegate_count_invalid"),
                                          (-1, "delegate_count_invalid"), (40000, "delegate_input_limit"),
                                          (RuntimeError("secret"), "delegate_count_unavailable")])
def test_count_failure_never_dispatches_or_reserves(tmp_path, monkeypatch, count, reason):
    router, store, _, worker = setup_router(tmp_path, monkeypatch, Worker(count=count))
    result = run(router)
    assert result.reason == reason and result.sources == evidence()[1]
    assert not worker.invocations and not store.list_model_calls("s")
    assert store.connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


def test_caps_and_economic_rejection_are_code_enforced(tmp_path, monkeypatch):
    router, store, _, worker = setup_router(tmp_path, monkeypatch, cap=".001")
    assert run(router).reason == "delegate_budget_unavailable"
    assert not worker.invocations and not store.list_model_calls("s")
    expensive = WORKER.model_copy(update={"input_per_million": Decimal("5")})
    config = RoutingConfig(enabled=True, worker_pricing=expensive)
    other, _, _, worker = setup_router(tmp_path / "price", monkeypatch, cfg=config)
    assert run(other, "price-reject").reason == "economic_gate_rejected"
    assert not worker.invocations


def test_disabled_never_constructs_provider_but_returns_bounded_complete_evidence(tmp_path, monkeypatch):
    router, store, _, worker = setup_router(tmp_path, monkeypatch, cfg=RoutingConfig(enabled=False))
    router.worker_factory = lambda _: pytest.fail("disabled router constructed worker")
    result = run(router)
    assert result.reason == "routing_disabled" and result.cost_usd == 0
    assert result.sources == evidence()[1] and not store.list_model_calls("s")
    assert not worker.counts


def test_persistent_attempt_limit_includes_failed_delegate(tmp_path, monkeypatch):
    config = RoutingConfig(enabled=True, worker_pricing=WORKER, max_calls_per_session=1, max_output_tokens=256)
    router, store, _, worker = setup_router(tmp_path, monkeypatch, Worker(text="bad"), cfg=config)
    assert run(router).status == "fallback"
    restarted, _, _, next_worker = setup_router(tmp_path, monkeypatch, cfg=config, store=store)
    assert run(restarted, "second").reason == "delegate_call_limit"
    assert len(worker.invocations) == 1 and not next_worker.counts


def test_invalid_briefs_reject_before_claim_or_provider(tmp_path, monkeypatch):
    router, store, _, worker = setup_router(tmp_path, monkeypatch)
    task, sources = evidence()
    with pytest.raises(ValueError, match="missing"):
        router.analyze(task.model_copy(update={"source_ids": ("missing",)}), sources, request_id="x")
    with pytest.raises(ValueError, match="byte limit"):
        run(router, content="a" * 31999)
    assert not worker.counts and not store.list_model_calls("s")
    assert store.connection.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 0


def test_database_failure_keeps_exact_receipt_for_settlement_without_redispatch(tmp_path, monkeypatch):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch)
    settle = budgets.settle
    monkeypatch.setattr(budgets, "settle", lambda *a, **kw: (_ for _ in ()).throw(sqlite3.OperationalError("disk failure")))
    with pytest.raises(sqlite3.OperationalError):
        run(router)
    assert router.pending["record"].usage.input_tokens == 100
    assert store.list_model_calls("s") == []
    with pytest.raises(BudgetExceeded):
        run(router)
    monkeypatch.setattr(budgets, "settle", settle)
    router.reconcile_pending()
    result = run(router)
    assert result.reason == "interrupted_no_retry" and result.cost_usd > 0
    assert not router.pending and len(worker.invocations) == 1
    assert not budgets.status("s")["experiment"]["blocked"]


def test_process_restart_with_lost_receipt_charges_and_blocks(tmp_path, monkeypatch):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch)
    task, sources = evidence()
    brief = build_evidence_brief(task, sources, router.config)
    key = router._identity("request-1")
    router._claim(key, "request-1", brief)
    router._reserve(key, "delegate", WORKER, Decimal(".01"))
    restarted, _, _, _ = setup_router(tmp_path, monkeypatch, store=store)
    with pytest.raises(BudgetExceeded):
        run(restarted)
    restarted.reconcile_pending()
    result = run(restarted)
    assert result.reason == "interrupted_no_retry" and result.cost_usd == Decimal(".01")
    assert result.provider_spend_usd == 0 and result.cost_basis == "includes_conservative_reservation"
    assert budgets.status("s")["experiment"]["blocked"]
    assert not worker.invocations


def test_unresolved_main_call_prevents_worker_count_and_is_not_router_owned(tmp_path, monkeypatch):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch)
    budgets.reserve("s", "main-pending", "main", "main", ".01", 0)
    with pytest.raises(BudgetExceeded, match="unresolved"):
        run(router)
    router.reconcile_pending()
    assert not worker.counts and not store.list_model_calls("s")
    assert budgets.status("s")["experiment"]["reserved_usd"] == Decimal(".01")


def test_reported_token_limit_violation_blocks_even_when_charge_fits_reservation(tmp_path, monkeypatch):
    worker = Worker(usage={"input_tokens": 100, "output_tokens": 1100})
    router, store, budgets, _ = setup_router(tmp_path, monkeypatch, worker)
    result = run(router)
    assert result.reason == "delegate_token_bound_exceeded"
    assert store.list_model_calls("s")[0].status == "failed"
    assert budgets.status("s")["experiment"]["blocked"]


def test_live_and_simulation_do_not_silently_substitute_providers(tmp_path, monkeypatch):
    router, store, budgets, _ = setup_router(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="simulated worker"):
        DelegationRouter(store, budgets, "s", main_pricing=MAIN, config=router.config)
    with pytest.raises(ValueError, match="Jev"):
        DelegationRouter(store, budgets, "s", main_pricing=MAIN,
                         config=router.config.model_copy(update={"engine": "jev"}), worker_factory=lambda _: Worker())
    store.create_session("live", "routing", "audit", model_main="main", mode="live")
    with pytest.raises(ValueError, match="simulation"):
        DelegationRouter(store, budgets, "live", main_pricing=MAIN, config=router.config, worker_factory=lambda _: Worker())


def test_jev_metered_suitability_and_worker_receipts_both_in_total(tmp_path, monkeypatch):
    client = Suitability()
    config = RoutingConfig(enabled=True, engine="jev", worker_pricing=WORKER, max_output_tokens=256)
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch, mode="live", cfg=config, client=client)
    result = run(router)
    assert result.status == "delegated"
    calls = store.list_model_calls("s")
    assert [call.purpose for call in calls] == ["decision", "delegate"]
    assert calls[0].raw_usage["receipt"]["usage"] == client.usage
    assert calls[0].cost_usd == Decimal(".000042")
    assert result.cost_usd == sum(call.cost_usd for call in calls)
    assert result.provider_spend_usd is None and result.cost_basis == "usage_at_configured_prices"
    assert result.decision_call_id == calls[0].id and result.delegate_call_id == calls[1].id
    assert result.forecast.decision_cost_usd == calls[0].cost_usd
    assert len(worker.invocations) == len(client.payloads) == 1
    assert client.payloads[0]["state"]["sources"][0]["content"] == evidence()[1][0].content
    assert not budgets.status("s")["experiment"]["blocked"]


@pytest.mark.parametrize("client,reason,unknown", [
    (Suitability(probability=.5), "suitability_veto", False),
    (Suitability(invalid=True), "decision_failed", False),
    (Suitability(usage={}), "decision_failed", True),
    (Suitability(model="wrong"), "decision_failed", True),
    (Suitability(fail=True), "decision_failed", True),
])
def test_jev_invalid_uncertain_or_unpriced_answers_never_dispatch_worker(tmp_path, monkeypatch, client, reason, unknown):
    config = RoutingConfig(enabled=True, engine="jev", worker_pricing=WORKER, max_output_tokens=256)
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch, mode="live", cfg=config, client=client)
    result = run(router)
    assert result.status == "fallback" and result.reason == reason and result.sources == evidence()[1]
    call, = store.list_model_calls("s")
    assert call.purpose == "decision" and call.cost_usd > 0
    assert (call.status == "unknown_usage") == unknown
    assert budgets.status("s")["experiment"]["blocked"] == unknown
    assert not worker.invocations and len(client.payloads) == 1
    assert "secret" not in call.model_dump_json()


def test_jev_complete_evidence_bound_and_durable_shared_decision_limit(tmp_path, monkeypatch):
    config = RoutingConfig(enabled=True, engine="jev", worker_pricing=WORKER, max_output_tokens=256)
    client = Suitability(probability=.2)
    router, store, _, worker = setup_router(tmp_path, monkeypatch, mode="live", cfg=config, client=client,
                                          jev_config=JevConfig(max_calls_per_session=1))
    assert run(router).reason == "suitability_veto"
    assert run(router, "second").reason == "decision_call_limit"
    assert len(client.payloads) == 1 and not worker.invocations
    # Complete evidence that does not fit cannot be silently truncated for Jev.
    smaller = DelegationRouter(store, router.budgets, "s", main_pricing=MAIN, config=config,
                               jev_config=JevConfig(max_state_bytes=100), suitability_client=client)
    assert run(smaller, "third").reason == "decision_complete_evidence_too_large"
    assert len(client.payloads) == 1


def test_catalog_binding_is_session_scoped_immutable_and_durable(tmp_path):
    path = tmp_path / "ledger.sqlite"
    store = LedgerStore(path)
    store.create_session("s", "routing", "audit")
    store.bind_evidence_catalog("s", "a" * 64)
    store.bind_evidence_catalog("s", "a" * 64)
    store.close()
    with LedgerStore(path) as restarted:
        restarted.bind_evidence_catalog("s", "a" * 64)
        with pytest.raises(LedgerConflict):
            restarted.bind_evidence_catalog("s", "b" * 64)
        with pytest.raises(ValueError):
            restarted.bind_evidence_catalog("s", "invalid")
        with pytest.raises(sqlite3.IntegrityError):
            restarted.bind_evidence_catalog("missing", "a" * 64)
        restarted.create_session("other", "baseline", "audit")
        restarted.bind_evidence_catalog("other", "b" * 64)


def test_economic_forecast_does_not_credit_main_answer_or_worker_instructions(tmp_path, monkeypatch):
    router, _, _, worker = setup_router(tmp_path, monkeypatch)
    brief = build_evidence_brief(*evidence(), router.config)
    forecast = router._forecast(4000, brief)
    source_tokens = (len(CONTENT.encode()) + 3) // 4
    assert forecast.main_cost_usd == Decimal(source_tokens) * MAIN.input_per_million / 1_000_000
    router.main_pricing = MAIN.model_copy(update={"output_per_million": Decimal("1000000")})
    assert router._forecast(4000, brief) == forecast
    short = run(router, "tiny", content="timeout: 30\n")
    assert short.reason == "economic_gate_rejected" and not worker.invocations
    assert short.forecast.main_cost_usd == Decimal(3) * MAIN.input_per_million / 1_000_000


def test_tool_payload_contains_only_evidence_and_review_instructions(tmp_path, monkeypatch):
    router, _, _, _ = setup_router(tmp_path, monkeypatch)
    result = run(router)
    payload = json.loads(result.tool_payload_json())
    assert set(payload) == {"status", "selection", "validation", "review_required", "source_provenance"}
    assert payload["selection"]["quotes"][0]["quote"] == "timeout: 30\n"
    assert "cost_usd" in result.model_dump_json() and "cost_usd" not in result.tool_payload_json()
    assert "forecast" not in payload and "delegate_call_id" not in payload
    rejected = run(router, "small", content="timeout: 30\n")
    fallback = json.loads(rejected.tool_payload_json())
    assert set(fallback) == {"status", "reason", "sources"}
    assert fallback["sources"][0]["content"] == "timeout: 30\n"


def test_budget_preflight_prevents_even_token_count_and_jev_dispatch(tmp_path, monkeypatch):
    client = Suitability()
    config = RoutingConfig(enabled=True, engine="jev", worker_pricing=WORKER, max_output_tokens=256)
    router, store, _, worker = setup_router(tmp_path, monkeypatch, cfg=config, mode="live", client=client, cap=".001")
    assert run(router).reason == "delegate_budget_unavailable"
    assert not worker.counts and not worker.invocations and not client.payloads
    assert store.list_model_calls("s") == []


def test_two_sessions_share_budget_without_cross_session_outcome_reuse(tmp_path, monkeypatch):
    router, store, budgets, worker = setup_router(tmp_path, monkeypatch)
    first = run(router)
    store.create_session("second", "routing", "audit", model_main="main", mode="simulation")
    second = DelegationRouter(store, budgets, "second", main_pricing=MAIN, config=router.config,
                              worker_factory=lambda _: worker)
    result = run(second)
    assert result.delegate_call_id != first.delegate_call_id
    assert len(worker.invocations) == 2
    assert budgets.status("s")["experiment"]["spent_usd"] == first.cost_usd + result.cost_usd
    assert budgets.status("s")["session"]["spent_usd"] == first.cost_usd
    assert budgets.status("second")["session"]["spent_usd"] == result.cost_usd
