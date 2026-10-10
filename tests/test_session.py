"""Real LangChain graphs using fake local models; never provider inference."""
import json
import traceback
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from contextlab.core.models import Pricing, Usage
from contextlab.ledger.budget import BudgetExceeded, BudgetManager
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerConflict, LedgerStore, ModelCall
from contextlab.session import ContextLabSession, SessionConfig

SECRET = "fake-private-provider-key-do-not-echo"
RAW_USAGE = {
    "input_tokens": 200, "output_tokens": 10, "total_tokens": 210,
    "input_token_details": {"cache_read": 120, "cache_creation": 40},
}


class ToolCapableFake(FakeMessagesListChatModel):
    bound_schemas: list[Any] = Field(default_factory=list)
    count_calls: int = 0
    inference_calls: int = 0
    fail_count: bool = False
    fail_inference: bool = False

    def bind_tools(self, tools, **kwargs):
        self.bound_schemas.append([convert_to_openai_tool(item) for item in tools])
        return self

    def get_num_tokens_from_messages(self, *args, **kwargs):
        self.count_calls += 1
        if self.fail_count:
            raise RuntimeError(SECRET)
        return 200

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.inference_calls += 1
        if self.fail_inference:
            raise RuntimeError(SECRET)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def pricing():
    return Pricing(model="claude-sonnet-4-6", input_per_million=1, output_per_million=2,
                   cache_write_5m_per_million="1.25", cache_write_1h_per_million=2,
                   cache_read_per_million="0.1", source="unit-test rate fixture", verified_at="2026-10-10")


def config(**updates):
    return SessionConfig(pricing=pricing(), max_input_tokens=10000, max_output_tokens=100,
                         **updates)


def reply(text="done", *, usage=RAW_USAGE):
    return AIMessage(content=text, usage_metadata=usage,
                     response_metadata={"stop_reason": "end_turn", "usage": {"input_tokens": 40}})


def install_model(monkeypatch, responses, **fields):
    model = ToolCapableFake(responses=responses, **fields)
    constructed = []

    def construct(**kwargs):
        constructed.append(kwargs)
        return model

    monkeypatch.setattr("contextlab.session.ChatAnthropic", construct)
    return model, constructed


def read_tool():
    @tool
    def read_file(path: str) -> str:
        """Read the fixture's current file as structured text."""
        return json.dumps({"path": path, "content": "historical deployment evidence\n",
                           "offset": 0, "limit": 80, "total_lines": 1,
                           "full_read": True, "sha256": "fixture-version"})
    return read_file


@pytest.mark.parametrize("arm", ["baseline", "contextlab", "headroom"])
def test_actual_graph_tools_and_receipt_math_across_arms(monkeypatch, tmp_path, arm):
    if arm == "headroom":
        try:
            version("headroom-ai")
        except PackageNotFoundError:
            pytest.skip("Optional genuine Headroom dependency is not installed")
    call = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "deployment.json"},
                                             "id": "read-1", "type": "tool_call"}], usage_metadata=RAW_USAGE)
    model, constructed = install_model(monkeypatch, [call, reply()])
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config(arm=arm)) as session:
        graph = session.build_agent(tools=[read_tool()], system_prompt="Inspect the fixture.")
        assert model.count_calls == model.inference_calls == 0
        result = graph.invoke({"messages": [HumanMessage(content="Read deployment.json")]})
        assert any(isinstance(message, ToolMessage) for message in result["messages"])
        assert result["messages"][-1].content == "done"
        summary = session.summary()
        expected = Usage(input_tokens=40, cache_write_5m_tokens=40, cache_read_tokens=120, output_tokens=10)
        assert Decimal(summary["total_cost_usd"]) == 2 * price_usage(expected, pricing())
        assert summary["usage"]["input_tokens"] == 80
        assert summary["usage"]["cache_write_5m_tokens"] == 80
        assert summary["usage"]["cache_read_tokens"] == 240
        assert summary["provider_calls"] == 2
        assert model.count_calls == model.inference_calls == 2
        calls = session.store.list_model_calls("session")
        assert calls[0].raw_usage["messages"][0]["usage_metadata"] == RAW_USAGE
        assert all(entry["admitted"] for entry in summary["interventions_this_process"])
    assert constructed[0]["max_retries"] == 0


def test_tool_schemas_are_identical_between_arms(monkeypatch, tmp_path):
    try:
        version("headroom-ai")
    except PackageNotFoundError:
        pytest.skip("Optional genuine Headroom dependency is not installed")
    schemas = []
    for arm in ("baseline", "contextlab", "headroom"):
        model, _ = install_model(monkeypatch, [reply()])
        with ContextLabSession(tmp_path / f"{arm}.sqlite", arm, config(arm=arm)) as session:
            graph = session.build_agent(tools=[read_tool()], system_prompt="Same public instructions.")
            graph.invoke({"messages": [HumanMessage(content="Inspect")]})
            schemas.append(model.bound_schemas[0])
    assert schemas[0] == schemas[1] == schemas[2]
    assert [item["function"]["name"] for item in schemas[0]] == ["read_file", "recall"]


def test_recall_input_failures_are_recoverable_but_storage_failures_propagate(monkeypatch, tmp_path):
    call = AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "data.json"},
                                             "id": "read-1", "type": "tool_call"}], usage_metadata=RAW_USAGE)
    install_model(monkeypatch, [call, reply()])
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config()) as session:
        graph = session.build_agent(tools=[read_tool()], system_prompt="Read.")
        graph.invoke({"messages": [HumanMessage(content="Read data.json")]})
        chunk = session.store.list_chunks("session")[0]
        recall = session.recall_tool()
        assert "historical deployment evidence" in recall.invoke({"chunk_id": chunk.id})
        assert "outside" in recall.invoke({"chunk_id": chunk.id, "offset": 999})
        assert "not found" in recall.invoke({"chunk_id": "c999999"})
        assert SECRET not in recall.invoke({"chunk_id": SECRET})
        assert "Invalid recall" in recall.invoke({"chunk_id": chunk.id, "limit": 0})

        def storage_error(*args):
            raise KeyError("ledger infrastructure failure")
        monkeypatch.setattr(session.middleware, "recall", storage_error)
        with pytest.raises(KeyError, match="ledger infrastructure"):
            recall.invoke({"chunk_id": chunk.id})


def test_real_graph_recovers_from_invented_recall_id(monkeypatch, tmp_path):
    invalid = AIMessage(content="", tool_calls=[{"name": "recall", "args": {"chunk_id": "invented-id"},
                                                "id": "bad-1", "type": "tool_call"}], usage_metadata=RAW_USAGE)
    model, _ = install_model(monkeypatch, [invalid, reply("I will use actual supplied IDs.")])
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config()) as session:
        graph = session.build_agent(tools=[], system_prompt="Use only provided IDs.")
        state = graph.invoke({"messages": [HumanMessage(content="Inspect history")]})
        errors = [message for message in state["messages"] if isinstance(message, ToolMessage)]
        assert errors[0].status == "error"
        assert "invented-id" not in errors[0].content
        assert state["messages"][-1].content == "I will use actual supplied IDs."
        assert model.inference_calls == 2


@pytest.mark.parametrize("provider_failure", [False, True])
def test_unknown_usage_or_provider_failure_charges_reservation_and_blocks_more_calls(
    monkeypatch, tmp_path, provider_failure,
):
    model, _ = install_model(monkeypatch, [reply(usage=None)], fail_inference=provider_failure)
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config()) as session:
        graph = session.build_agent(tools=[], system_prompt="Inspect.")
        with pytest.raises(BudgetExceeded) as caught:
            graph.invoke({"messages": [HumanMessage(content="hello")]})
        assert SECRET not in "".join(traceback.format_exception(caught.value))
        summary = session.summary()
        assert Decimal(summary["total_cost_usd"]) == worst_case_cost(10000, 100, pricing())
        assert summary["unknown_usage_calls"] == 1 and summary["budget"]["experiment"]["blocked"]
        assert SECRET not in json.dumps(summary)
        with pytest.raises(BudgetExceeded):
            graph.invoke({"messages": [HumanMessage(content="try again")]})
        assert model.inference_calls == 1 and model.count_calls == 1


def test_count_failure_has_no_charge_and_unadmitted_projection_is_not_counted(monkeypatch, tmp_path):
    model, _ = install_model(monkeypatch, [reply()], fail_count=True)
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config()) as session:
        graph = session.build_agent(tools=[], system_prompt="Inspect.")
        with pytest.raises(RuntimeError):
            graph.invoke({"messages": [HumanMessage(content="hello")]})
        # Even a changed projection blocked by admission is not a model request.
        session.middleware.telemetry[-1]["changed"] = True
        summary = session.summary()
        assert summary["modified_requests_this_process"] == 0
        assert not summary["interventions_this_process"][-1]["admitted"]
        assert Decimal(summary["total_cost_usd"]) == 0 and model.inference_calls == 0


def test_restart_keeps_main_call_limit_before_counter_or_inference(monkeypatch, tmp_path):
    path = tmp_path / "ledger.sqlite"
    settings = config(max_model_calls=1)
    first, _ = install_model(monkeypatch, [reply()])
    with ContextLabSession(path, "session", settings) as session:
        session.build_agent(tools=[], system_prompt="Inspect.").invoke(
            {"messages": [HumanMessage(content="first")]})
    assert first.inference_calls == 1
    second, _ = install_model(monkeypatch, [reply()])
    with ContextLabSession(path, "session", settings) as session:
        graph = session.build_agent(tools=[], system_prompt="Inspect.")
        with pytest.raises(BudgetExceeded, match="Persistent main-model call limit"):
            graph.invoke({"messages": [HumanMessage(content="second")]})
        assert session.summary()["provider_calls"] == 1
    assert second.count_calls == second.inference_calls == 0


def test_restart_unresolved_reservation_is_visible_and_blocks_new_dispatch(monkeypatch, tmp_path):
    path = tmp_path / "ledger.sqlite"
    with ContextLabSession(path, "session", config()) as session:
        session.budgets.reserve("session", "orphan", pricing().model, "main", "0.02")
    model, _ = install_model(monkeypatch, [reply()])
    with ContextLabSession(path, "session", config()) as session:
        summary = session.summary()
        assert Decimal(summary["total_cost_usd"]) == Decimal("0.02")
        assert Decimal(summary["recorded_cost_usd"]) == 0
        assert Decimal(summary["unsettled_reservations_usd"]) == Decimal("0.02")
        assert "not verified provider charges" in summary["cost_basis"]
        graph = session.build_agent(tools=[], system_prompt="Inspect.")
        with pytest.raises(BudgetExceeded, match="durable reservation"):
            graph.invoke({"messages": [HumanMessage(content="continue")]})
    assert model.count_calls == model.inference_calls == 0


def test_reconcile_retries_exact_pending_receipt_before_final_summary(monkeypatch, tmp_path):
    model, _ = install_model(monkeypatch, [reply()])
    path = tmp_path / "ledger.sqlite"
    session = ContextLabSession(path, "session", config())
    original = BudgetManager.settle
    records = []

    def flaky(self, call_id, call=None, **kwargs):
        records.append(call)
        if len(records) == 1:
            raise RuntimeError("transient ledger failure")
        return original(self, call_id, call, **kwargs)

    monkeypatch.setattr(BudgetManager, "settle", flaky)
    graph = session.build_agent(tools=[], system_prompt="Inspect.")
    with pytest.raises(RuntimeError, match="transient ledger"):
        graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert Decimal(session.summary()["unsettled_reservations_usd"]) > 0
    session.reconcile_pending()
    assert len(records) == 2 and records[0] == records[1]
    assert model.inference_calls == 1
    summary = session.summary()
    assert Decimal(summary["unsettled_reservations_usd"]) == 0
    assert Decimal(summary["recorded_cost_usd"]) == records[0].cost_usd
    session.close()
    assert len(records) == 2
    with LedgerStore(path) as store:
        assert store.list_model_calls("session")[0] == records[0]


def test_close_attempts_main_settlement_even_if_decision_cleanup_fails(tmp_path):
    path = tmp_path / "ledger.sqlite"
    session = ContextLabSession(path, "session", config())
    session.budgets.reserve("session", "pending-main", pricing().model, "main", "0.02")
    record = ModelCall(id="pending-main", session_id="session", seq=0,
                       model=pricing().model, cost_usd=Decimal("0.001"))
    session._pending.update(id=record.id, record=record)

    def fail():
        raise RuntimeError("decision cleanup failed")

    session.gate = SimpleNamespace(reconcile_pending=fail)
    with pytest.raises(RuntimeError, match="decision cleanup failed"):
        session.close()
    session.close()  # Idempotent after failed cleanup.
    with LedgerStore(path) as store:
        assert store.list_model_calls("session") == [record]


def test_close_preserves_failed_reconciliation_summary(monkeypatch, tmp_path):
    path = tmp_path / "ledger.sqlite"
    session = ContextLabSession(path, "session", config())
    session.budgets.reserve("session", "pending-main", pricing().model, "main", "0.02")
    session._pending.update(id="pending-main")
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("persistent ledger failure")

    monkeypatch.setattr(BudgetManager, "settle", fail)
    with pytest.raises(RuntimeError, match="persistent ledger"):
        session.close()
    summary = session.summary()
    session.close()
    assert len(attempts) == 1
    with LedgerStore(path) as store:
        assert not store.list_model_calls("session")
        budget = BudgetManager(store, config().session_cap_usd, config().experiment_cap_usd, "default")
        assert str(budget.status("session")["session"]["reserved_usd"]) == summary["unsettled_reservations_usd"]
    summary["budget"]["session"]["reserved_usd"] = "mutated"
    assert session.summary()["budget"]["session"]["reserved_usd"] != "mutated"


@pytest.mark.parametrize("environment_name", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
def test_builder_rejects_enabled_external_tracing_before_model_creation(monkeypatch, tmp_path, environment_name):
    model, constructed = install_model(monkeypatch, [reply()])
    monkeypatch.setenv(environment_name, "TrUe")
    with ContextLabSession(tmp_path / "ledger.sqlite", "session", config()) as session:
        with pytest.raises(ValueError, match="Disable LangSmith/LangChain tracing"):
            session.build_agent(tools=[], system_prompt="Inspect.")
        assert not constructed and model.inference_calls == 0
        monkeypatch.setenv(environment_name, "false")
        session.build_agent(tools=[], system_prompt="Inspect.")


def test_configuration_and_lifecycle_guards(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="verified provider pricing"):
        SessionConfig(pricing=pricing().model_copy(update={"source": "synthetic"}))
    with pytest.raises(ValueError, match="only to the ContextLab arm"):
        config(arm="baseline", decision_engine="jev")
    with pytest.raises(ValueError, match="session ID"):
        ContextLabSession(tmp_path / "bad.sqlite", "../bad", config())
    install_model(monkeypatch, [reply()])
    path = tmp_path / "ledger.sqlite"
    session = ContextLabSession(path, "session", config())
    with pytest.raises(ValueError, match="duplicate recall"):
        session.build_agent(tools=[session.recall_tool()], system_prompt="Inspect.")
    session.build_agent(tools=[], system_prompt="Inspect.")
    with pytest.raises(RuntimeError, match="one agent"):
        session.build_agent(tools=[], system_prompt="Inspect.")
    session.close()
    assert Decimal(session.summary()["total_cost_usd"]) == 0
    with pytest.raises(RuntimeError, match="closed"):
        session.reconcile_pending()
    with pytest.raises(LedgerConflict):
        ContextLabSession(path, "session", config(max_model_calls=2))
