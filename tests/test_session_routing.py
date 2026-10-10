"""Session-level delegation simulations through real LangChain tool execution."""

import json
import socket
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, ValidationError

from contextlab.core.models import Pricing, Usage
from contextlab.core.routing import EvidenceSource, RoutingConfig
from contextlab.ledger.budget import BudgetExceeded
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerConflict
from contextlab.session import ContextLabSession, SessionConfig

ARMS = ("baseline", "cl_compact_only", "cl_route_only", "contextlab", "headroom")
RAW = {"input_tokens": 1_000, "output_tokens": 20}


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Session routing tests must not access a network")

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


def rates(*, worker=False, live=False):
    return Pricing(model="simulation-worker" if worker else "simulation-main",
                   input_per_million=1 if worker else 10,
                   output_per_million=2 if worker else 50,
                   cache_write_5m_per_million=1 if worker else 10,
                   cache_write_1h_per_million=1 if worker else 10,
                   cache_read_per_million=1 if worker else 10,
                   source="test-only verified fixture" if live else "synthetic",
                   verified_at="2026-10-10" if live else "")


def config(*, arm="contextlab", routing=False, live=False, **changes):
    return SessionConfig(
        pricing=rates(live=live), arm=arm, mode="live" if live else "simulation",
        max_input_tokens=10_000, max_output_tokens=128,
        routing=RoutingConfig(enabled=routing, worker_pricing=rates(worker=True, live=live)
                              if routing else None, max_input_tokens=2_048,
                              max_output_tokens=128, failure_probability=Decimal("0")),
        **changes,
    )


def source_catalog():
    return [EvidenceSource(id="config/settings.txt", path="config/settings.txt",
                           content="owner=Mira\nlimit=3\n" + "".join(
                               f"observation-{index}: shard={index % 16} load={index * 3}\n"
                               for index in range(450)))]


def args():
    return {"goal": "Find the configured limit without modifying files.",
            "questions": {"limit": "What is the configured limit?"},
            "source_ids": ["config/settings.txt"]}


def selection_text():
    return json.dumps({"quotes": [{"question_id": "limit", "source_id": "config/settings.txt",
                                  "start_line": 2, "end_line": 2, "quote": "limit=3\n"}]})


def response(content="done", *, tool_calls=None, worker=False, usage=True):
    return AIMessage(
        content=content, tool_calls=tool_calls or [],
        usage_metadata={**RAW, "total_tokens": 1_020} if usage else None,
        response_metadata={"stop_reason": "end_turn", "model_name": rates(worker=worker).model,
                           **({"usage": dict(RAW)} if usage else {})},
    )


class MainSimulation(FakeMessagesListChatModel):
    schemas: list = Field(default_factory=list)
    count_calls: int = 0
    inference_calls: int = 0

    def bind_tools(self, tools, **kwargs):
        self.schemas.append([convert_to_openai_tool(item) for item in tools])
        return self

    def get_num_tokens_from_messages(self, *args, **kwargs):
        self.count_calls += 1
        return 1_000

    def _generate(self, *args, **kwargs):
        self.inference_calls += 1
        return super()._generate(*args, **kwargs)


class WorkerSimulation:
    def __init__(self, *, usage=True):
        self.count_calls = 0
        self.inference_calls = 0
        self.usage = usage

    def get_num_tokens_from_messages(self, messages):
        self.count_calls += 1
        return 1_000

    def invoke(self, messages):
        self.inference_calls += 1
        return response(selection_text(), worker=True, usage=self.usage)


def open_simulation(path, *, settings=None, responses=None, worker=None):
    main = MainSimulation(responses=responses or [response()])
    delegate = worker or WorkerSimulation()
    session = ContextLabSession(path, "session", settings or config(routing=True),
                                model_factory=lambda _: main, worker_factory=lambda _: delegate)
    return session, main, delegate


def tool_call(call_id="evidence-request", values=None):
    return {"type": "tool_call", "name": "analyze_files", "id": call_id,
            "args": args() if values is None else values}


@pytest.mark.parametrize("inject_main,inject_worker", [(True, False), (False, True), (True, True)])
def test_live_mode_rejects_injected_providers_before_opening_ledger(tmp_path, inject_main, inject_worker):
    path = tmp_path / "should-not-exist.sqlite"
    with pytest.raises(ValueError, match="Injected providers"):
        ContextLabSession(path, "session", config(live=True),
                          model_factory=(lambda _: None) if inject_main else None,
                          worker_factory=(lambda _: None) if inject_worker else None)
    assert not path.exists()


def test_simulation_requires_all_factories_synthetic_pricing_and_no_jev(tmp_path):
    path = tmp_path / "simulation.sqlite"
    with pytest.raises(ValueError, match="explicit model factories"):
        ContextLabSession(path, "session", config())
    with pytest.raises(ValueError, match="explicit model factories"):
        ContextLabSession(path, "session", config(routing=True), model_factory=lambda _: None)
    assert not path.exists()
    with pytest.raises(ValidationError, match="synthetic pricing"):
        SessionConfig(pricing=rates(live=True), mode="simulation")
    with pytest.raises(ValidationError, match="Jev"):
        config(decision_engine="jev")
    with pytest.raises(ValidationError, match="Jev"):
        SessionConfig(pricing=rates(), mode="simulation",
                      routing=RoutingConfig(engine="jev"))


@pytest.mark.parametrize("arm", ["baseline", "cl_compact_only", "headroom"])
def test_routing_cannot_leak_into_unselected_arms(arm):
    with pytest.raises(ValidationError, match="Routing belongs"):
        config(arm=arm, routing=True)


def test_five_arms_keep_identical_analysis_schema_and_isolate_active_components(tmp_path):
    try:
        version("headroom-ai")
    except PackageNotFoundError:
        pytest.skip("Genuine optional Headroom dependency is unavailable")
    schemas = []
    for arm in ARMS:
        session, main, worker = open_simulation(
            tmp_path / f"{arm}.sqlite", settings=config(arm=arm, routing=arm in {"cl_route_only", "contextlab"}))
        with session:
            graph = session.build_agent(tools=[], system_prompt="Review exact evidence.",
                                         evidence_sources=source_catalog())
            assert (session.middleware is not None) == (arm in {"cl_compact_only", "contextlab"})
            assert (session.headroom is not None) == (arm == "headroom")
            graph.invoke({"messages": [HumanMessage(content="Begin review.")]})
            schemas.append(main.schemas[0])
            assert worker.count_calls == worker.inference_calls == 0
            assert session.summary()["provider_spend_usd"] == "0"
    assert all(schema == schemas[0] for schema in schemas)
    analysis = next(item["function"] for item in schemas[0] if item["function"]["name"] == "analyze_files")
    assert set(analysis["parameters"]["properties"]) == {"goal", "questions", "source_ids"}
    assert "tool_call_id" not in analysis["parameters"]["properties"]


def test_catalog_binding_is_order_independent_immutable_and_persistent(tmp_path):
    path = tmp_path / "catalog.sqlite"
    extra = EvidenceSource(id="a.txt", path="a.txt", content="second snapshot\n")
    catalog = [*source_catalog(), extra]
    session, _, _ = open_simulation(path)
    with session:
        first_tool = session.evidence_tool(catalog)
        session.evidence_tool(list(reversed(catalog)))
        # The closure owns its snapshots rather than the caller's mutable list.
        catalog.clear()
        outcome = first_tool.invoke(tool_call())
        assert json.loads(outcome.content)["selection"]["quotes"][0]["quote"] == "limit=3\n"
        changed = source_catalog()[0].model_copy(update={"path": "renamed.txt"})
        with pytest.raises(LedgerConflict):
            session.evidence_tool([changed, extra])
    restarted, _, _ = open_simulation(path)
    with restarted:
        restarted.evidence_tool([extra, *source_catalog()])
        changed = EvidenceSource(id="config/settings.txt", path="config/settings.txt", content="limit=9\n")
        with pytest.raises(LedgerConflict):
            restarted.evidence_tool([changed, extra])


@pytest.mark.parametrize("bad", [
    {"goal": "", "questions": {"q": "What?"}, "source_ids": ["config/settings.txt"]},
    {"goal": "Read", "questions": {}, "source_ids": ["config/settings.txt"]},
    {"goal": "Read", "questions": {"q": "What?"}, "source_ids": ["missing.txt"]},
    {"goal": "Read", "questions": {"q": "What?"}, "source_ids": "not-a-list"},
])
def test_invalid_agent_tasks_return_tool_errors_without_dispatch(tmp_path, bad):
    session, _, worker = open_simulation(tmp_path / "invalid.sqlite")
    with session:
        result = session.evidence_tool(source_catalog()).invoke(tool_call(values=bad))
        assert isinstance(result, ToolMessage) and result.status == "error"
        assert "Invalid" in result.content
        assert worker.count_calls == worker.inference_calls == 0
        assert session.summary()["provider_calls"] == 0


def test_analysis_storage_failures_propagate_instead_of_becoming_tool_validation_errors(tmp_path, monkeypatch):
    session, _, worker = open_simulation(tmp_path / "storage.sqlite")
    with session:
        analysis = session.evidence_tool(source_catalog())

        def fail(*args, **kwargs):
            raise LedgerConflict("fixture storage failure")

        monkeypatch.setattr(session.router, "analyze", fail)
        with pytest.raises(LedgerConflict, match="storage failure"):
            analysis.invoke(tool_call())
        assert worker.inference_calls == 0


def test_real_graph_injects_call_id_and_all_in_totals_include_delegate_once(tmp_path):
    session, main, worker = open_simulation(tmp_path / "graph.sqlite", responses=[
        response("", tool_calls=[tool_call("opaque-live-shaped-id")]), response("The configured limit is 3.")])
    with session:
        graph = session.build_agent(tools=[], system_prompt="Review quotes against the question.",
                                     evidence_sources=source_catalog())
        state = graph.invoke({"messages": [HumanMessage(content="Inspect the configured limit.")]})
        evidence = next(message for message in state["messages"] if isinstance(message, ToolMessage))
        outcome = json.loads(evidence.content)
        assert evidence.tool_call_id == "opaque-live-shaped-id"
        saved = [json.loads(row[0]) for row in session.store.connection.execute(
            "SELECT output_json FROM decisions WHERE session_id=? AND decision_type='delegation'", ("session",))]
        assert len(saved) == 1 and saved[0]["request_id"] == evidence.tool_call_id
        assert not {"request_id", "cost_usd", "provider_spend_usd", "forecast"}.intersection(outcome)
        assert outcome["status"] == "delegated"
        assert outcome["validation"] == "exact_quote_provenance_only"
        assert "sources" not in outcome
        assert outcome["selection"]["quotes"][0]["quote"] == "limit=3\n"
        assert state["messages"][-1].content == "The configured limit is 3."
        calls = session.store.list_model_calls("session")
        assert [call.purpose for call in calls] == ["main", "delegate", "main"]
        assert len({call.id for call in calls}) == 3
        expected_delegate = price_usage(Usage(**RAW), rates(worker=True))
        summary = session.summary()
        assert Decimal(summary["delegate_cost_usd"]) == expected_delegate
        assert Decimal(summary["total_cost_usd"]) == 2 * price_usage(Usage(**RAW), rates()) + expected_delegate
        assert summary["provider_spend_usd"] == "0" and summary["delegate_calls"] == 1
        assert all(call.cost_basis == "simulation" for call in calls)
        assert main.count_calls == main.inference_calls == 2
        assert worker.count_calls == worker.inference_calls == 1


def test_unknown_delegate_usage_blocks_main_continuation_and_further_worker_dispatch(tmp_path):
    worker = WorkerSimulation(usage=False)
    session, main, _ = open_simulation(tmp_path / "unknown.sqlite", worker=worker, responses=[
        response("", tool_calls=[tool_call()]), response("must not execute")])
    with session:
        analysis = session.evidence_tool(source_catalog())
        graph = session.build_agent(tools=[], system_prompt="Review quotes.", evidence_sources=source_catalog())
        with pytest.raises(BudgetExceeded):
            graph.invoke({"messages": [HumanMessage(content="Inspect.")]})
        assert main.inference_calls == main.count_calls == worker.inference_calls == worker.count_calls == 1
        summary = session.summary()
        assert summary["unknown_usage_calls"] == 1
        assert summary["budget"]["experiment"]["blocked"]
        reserve = worst_case_cost(2_048, 128, rates(worker=True))
        assert Decimal(summary["delegate_cost_usd"]) == reserve
        assert Decimal(summary["total_cost_usd"]) == price_usage(Usage(**RAW), rates()) + reserve
        with pytest.raises(BudgetExceeded):
            graph.invoke({"messages": [HumanMessage(content="Try again.")]})
        fallback = json.loads(analysis.invoke(tool_call("blocked-follow-up")).content)
        assert fallback["reason"] == "shared_budget_blocked"
        assert main.inference_calls == worker.inference_calls == 1


def test_rebinding_same_catalog_keeps_pending_known_receipt_for_exact_reconciliation(tmp_path, monkeypatch):
    session, _, worker = open_simulation(tmp_path / "receipt.sqlite")
    with session:
        analysis = session.evidence_tool(source_catalog())
        router = session.router
        actual_settle = session.budgets.settle

        def unavailable(*args, **kwargs):
            raise LedgerConflict("temporary settlement failure")

        monkeypatch.setattr(session.budgets, "settle", unavailable)
        with pytest.raises(LedgerConflict, match="settlement failure"):
            analysis.invoke(tool_call())
        assert worker.inference_calls == 1
        receipt = router.pending["record"]
        assert receipt.cost_usd == price_usage(Usage(**RAW), rates(worker=True))
        session.evidence_tool(source_catalog())
        assert session.router is router and router.pending["record"] is receipt
        monkeypatch.setattr(session.budgets, "settle", actual_settle)
        session.reconcile_pending()
        assert not router.pending
        summary = session.summary()
        assert summary["unknown_usage_calls"] == 0
        assert Decimal(summary["delegate_cost_usd"]) == receipt.cost_usd
        assert not summary["budget"]["experiment"]["blocked"]
        replay = json.loads(analysis.invoke(tool_call()).content)
        assert replay["status"] == "fallback"  # Interrupted result is never silently redispatched.
        assert worker.inference_calls == worker.count_calls == 1
