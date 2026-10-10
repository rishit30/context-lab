"""A bounded, persistent LangChain/Anthropic integration, independent of the harness.

Constructing a session makes no provider request. ``agent.invoke`` runs the real
model and is billable; review the configured prices and caps before invoking it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.tools import BaseTool, ToolException, tool
from pydantic import Field, model_validator

from contextlab.adapters.langchain_mw import ContextLabMiddleware, MeteringMiddleware
from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing, Record
from contextlab.ledger.budget import BudgetExceeded, BudgetManager
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerStore, ModelCall
from contextlab.providers.jev_config import JevConfig


class SessionConfig(Record):
    """Immutable limits and pricing for one application-owned agent session."""
    pricing: Pricing
    arm: Literal["baseline", "contextlab", "headroom"] = "contextlab"
    shadow: bool = False
    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    session_cap_usd: Decimal = Field(default=Decimal("3"), gt=0, allow_inf_nan=False)
    experiment_cap_usd: Decimal = Field(default=Decimal("9"), gt=0, allow_inf_nan=False)
    max_model_calls: int = Field(default=48, ge=1, le=200)
    max_input_tokens: int = Field(default=100000, ge=1, le=200000)
    max_output_tokens: int = Field(default=4096, ge=1, le=8192)
    cache_mode: Literal["enabled", "disabled"] = "disabled"
    decision_engine: Literal["rules", "jev"] = "rules"
    jev: JevConfig = Field(default_factory=JevConfig)

    @model_validator(mode="after")
    def consistent(self) -> SessionConfig:
        if self.pricing.source == "synthetic" or not self.pricing.verified_at:
            raise ValueError("Use dated, verified provider pricing for the live session API")
        if self.arm != "contextlab" and (self.shadow or self.decision_engine != "rules"):
            raise ValueError("Jev and shadow mode belong only to the ContextLab arm")
        reserve = worst_case_cost(self.max_input_tokens, self.max_output_tokens, self.pricing)
        if reserve > min(self.session_cap_usd, self.experiment_cap_usd):
            raise ValueError("Caps must cover one worst-case model call")
        return self


class ContextLabSession:
    """One synchronous agent session with exact archives and all-in cost accounting.

    Keep a single instance/agent active for a session. Persist the canonical
    LangChain conversation yourself; ContextLab persists its ledger/projection,
    not your graph's checkpoint. Custom tools are the application's responsibility.
    """
    def __init__(self, ledger_path: Path, session_id: str, config: SessionConfig, *,
                 experiment_id: str = "default") -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", session_id):
            raise ValueError("Use a 1–80 character alphanumeric session ID with - or _")
        self.config, self.session_id = config, session_id
        self.store = LedgerStore(ledger_path)
        self._pending: dict[str, Any] = {}
        self._admitted_projection_indices: set[int] = set()
        self._built = False
        self._closed = False
        self._final_summary: dict[str, Any] | None = None
        self.gate = None
        self.middleware = None
        self.headroom = None
        try:
            fingerprint = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
            self.store.create_session(session_id, config.arm, "application-agent",
                                      model_main=config.pricing.model, config_hash=fingerprint,
                                      mode="live", experiment_id=experiment_id)
            self.budgets = BudgetManager(self.store, config.session_cap_usd,
                                         config.experiment_cap_usd, experiment_id)
            if config.arm == "contextlab":
                if config.decision_engine == "jev":
                    from contextlab.providers.jev import JevClient
                    from contextlab.providers.jev_gate import JevDecisionGate
                    self.gate = JevDecisionGate(
                        JevClient.from_environment(timeout_seconds=config.jev.timeout_seconds),
                        self.store, self.budgets, session_id, config=config.jev,
                    )
                self.middleware = ContextLabMiddleware(
                    controller=ContextController(semantic_gate=self.gate),
                    store=self.store, session_id=session_id, pricing=config.pricing,
                    config=config.compaction, shadow=config.shadow,
                    cache_ttl_seconds=300 if config.cache_mode == "enabled" else 0,
                )
            elif config.arm == "headroom":
                from contextlab.adapters.headroom_mw import HeadroomMiddleware
                self.headroom = HeadroomMiddleware(model_name=config.pricing.model)
        except BaseException:
            self.store.close()
            raise

    def __enter__(self) -> ContextLabSession:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def reconcile_pending(self) -> None:
        """Settle this process's pending receipts without closing the ledger.

        Call before saving a final summary, including after a failed invocation.
        Orphaned reservations from earlier processes remain blocked for explicit
        operator reconciliation; no provider request is made here.
        """
        if self._closed:
            raise RuntimeError("Session is closed")
        failure: Exception | None = None
        if self.gate is not None:
            try:
                self.gate.reconcile_pending()
            except Exception as exc:
                failure = exc
        if self._pending:
            try:
                self.budgets.settle(self._pending["id"], self._pending.get("record"))
                self._pending.clear()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure

    def close(self) -> None:
        """Reconcile, snapshot final accounting, then close the ledger.

        ``summary()`` remains available after close, even when settlement failed
        and the snapshot includes conservative outstanding reservations.
        """
        if self._closed:
            return
        failure: Exception | None = None
        try:
            try:
                self.reconcile_pending()
            except Exception as exc:
                failure = exc
            try:
                self._final_summary = self.summary()
            except Exception as exc:
                failure = failure or exc
        finally:
            self.store.close()
            self._closed = True
        if failure is not None:
            raise failure

    def _admit(self, request: Any) -> None:
        if self._closed:
            raise RuntimeError("Session is closed")
        if self._pending or (self.gate is not None and self.gate.pending):
            raise BudgetExceeded("An earlier provider receipt is unresolved")
        unresolved = self.store.connection.execute(
            "SELECT 1 FROM reservations WHERE session_id=? AND status='reserved' LIMIT 1",
            (self.session_id,),
        ).fetchone()
        if unresolved:
            raise BudgetExceeded("A durable reservation is unresolved; reconcile before resuming")
        calls = self.store.list_model_calls(self.session_id)
        if sum(call.purpose == "main" for call in calls) >= self.config.max_model_calls:
            raise BudgetExceeded("Persistent main-model call limit reached")
        reserve = worst_case_cost(self.config.max_input_tokens, self.config.max_output_tokens,
                                  self.config.pricing)
        status = self.budgets.status(self.session_id)
        for scope, fallback_cap in (("session", self.config.session_cap_usd),
                                     ("experiment", self.config.experiment_cap_usd)):
            state = status.get(scope, {})
            if state.get("blocked") or state.get("remaining_usd", fallback_cap) < reserve:
                raise BudgetExceeded(f"{scope} budget cannot admit another model call")
        all_messages = ([request.system_message] if request.system_message else []) + request.messages
        input_tokens = request.model.get_num_tokens_from_messages(all_messages, tools=request.tools)
        if input_tokens > self.config.max_input_tokens:
            raise BudgetExceeded("Input token limit reached")
        call_id, seq = f"{self.session_id}-{uuid4().hex}", len(calls)
        self.budgets.reserve(self.session_id, call_id, self.config.pricing.model, "main", reserve, seq)
        self._pending.update(id=call_id, seq=seq, amount=reserve)
        active = self.middleware if self.middleware is not None else self.headroom
        if active is not None and active.telemetry:
            self._admitted_projection_indices.add(len(active.telemetry) - 1)

    def _settle(self, model: str, usage: Any, latency_ms: float, status: str,
                error_type: str | None, raw_usage: dict | None = None) -> None:
        pending = self._pending
        common = dict(id=pending["id"], session_id=self.session_id, seq=pending["seq"],
                      model=self.config.pricing.model, latency_ms=latency_ms,
                      raw_usage=raw_usage or {})
        if usage is None:
            record = ModelCall(**common, cost_usd=pending["amount"], status="unknown_usage",
                               cost_basis="reservation", error="ProviderUsageUnavailable")
        else:
            record = ModelCall(**common, usage=usage, pricing=self.config.pricing,
                               cost_usd=price_usage(usage, self.config.pricing),
                               status="ok" if status == "ok" else "failed", error=error_type)
        pending["record"] = record
        self.budgets.settle(pending["id"], record)
        pending.clear()
        if self.budgets.status(self.session_id)["experiment"]["blocked"]:
            raise BudgetExceeded("Unknown or excessive usage; experiment blocked") from None

    def recall_tool(self) -> BaseTool:
        session = self

        @tool
        def recall(chunk_id: str, offset: Annotated[int, Field(ge=0, strict=True)] = 0,
                   limit: Annotated[int, Field(ge=1, le=200, strict=True)] = 80,
                   char_offset: Annotated[int, Field(ge=0, strict=True)] = 0) -> str:
            """Read a supplied ContextLab archive ID; follow next_offset/next_char_offset for more."""
            if session.middleware is None:
                return "No ContextLab archive in this arm; original outputs remain in history."
            if not re.fullmatch(r"c[0-9]+", chunk_id):
                raise ToolException("Use an archive ID actually supplied in a context signpost.")
            try:
                return session.middleware.recall(chunk_id, offset, limit, char_offset)
            except KeyError as exc:
                if type(exc) is not KeyError or exc.args != ("Chunk not found in this session",):
                    raise
                raise ToolException("Archive ID not found in this session.") from None
            except ValueError as exc:
                if type(exc) is not ValueError or exc.args != ("Recall offset is outside the archived content",):
                    raise
                raise ToolException("Recall offset is outside this archive; use its returned continuation offsets.") from None

        recall.handle_tool_error = True
        recall.handle_validation_error = "Invalid recall range or ID; use the supplied signpost."
        return recall

    def build_agent(self, *, tools: list[BaseTool], system_prompt: str) -> Any:
        """Create the real Claude agent; no network until the returned graph is invoked."""
        if self._closed or self._built:
            raise RuntimeError("Build one agent per open ContextLabSession")
        if any(os.environ.get(name, "").lower() == "true"
               for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")):
            raise ValueError("Disable LangSmith/LangChain tracing for this local-only session")
        if any(item.name == "recall" for item in tools):
            raise ValueError("The session owns the recall tool; remove a duplicate recall definition")
        model = ChatAnthropic(model_name=self.config.pricing.model,
                              max_tokens=self.config.max_output_tokens,
                              max_retries=0, timeout=60, temperature=0)
        middlewares: list[Any] = []
        if self.middleware is not None:
            middlewares.append(self.middleware)
        if self.headroom is not None:
            middlewares.append(self.headroom)
        if self.config.cache_mode == "enabled":
            middlewares.append(AnthropicPromptCachingMiddleware(ttl="5m"))
        middlewares.append(MeteringMiddleware(
            model_name=self.config.pricing.model, before_call=self._admit, on_usage=self._settle,
            max_model_calls=self.config.max_model_calls,
        ))
        graph = create_agent(model=model, tools=[*tools, self.recall_tool()],
                             system_prompt=system_prompt, middleware=middlewares)
        self._built = True
        return graph

    def complete_phase(self, label: str) -> None:
        """Application-only: call after verifying your phase's public acceptance checks."""
        if self._closed:
            raise RuntimeError("Session is closed")
        if self.middleware is not None:
            self.middleware.mark_boundary(label)

    def summary(self) -> dict[str, Any]:
        if self._closed:
            if self._final_summary is None:
                raise RuntimeError("Final session summary is unavailable")
            return deepcopy(self._final_summary)
        calls = self.store.list_model_calls(self.session_id)
        recorded = sum((call.cost_usd for call in calls), Decimal("0"))
        decisions = [call for call in calls if call.purpose == "decision"]
        raw_telemetry = (self.middleware.telemetry if self.middleware is not None else
                         self.headroom.telemetry if self.headroom is not None else [])
        telemetry = [{**item, "admitted": index in self._admitted_projection_indices}
                     for index, item in enumerate(raw_telemetry)]
        budget = self.budgets.status(self.session_id)
        reserved = budget.get("session", {}).get("reserved_usd", Decimal("0"))
        decision_reserved = self.store.session_summary(self.session_id)["decision_reserved_usd"]
        unknown_usage_calls = sum(call.status == "unknown_usage" for call in calls)
        cost_basis = "Provider usage × configured list prices; includes decision fees, not an invoice."
        if reserved:
            cost_basis = ("Includes unresolved worst-case reservations, not verified provider charges; "
                          "reconcile before comparing costs. Decision fees are included.")
        elif unknown_usage_calls:
            cost_basis = ("Includes conservative reservation charges for unknown usage; "
                          "reconcile before comparing costs. Decision fees are included.")
        return {
            "session_id": self.session_id, "arm": self.config.arm,
            "total_cost_usd": str(recorded + reserved), "provider_calls": len(calls),
            "recorded_cost_usd": str(recorded), "unsettled_reservations_usd": str(reserved),
            "unknown_usage_calls": unknown_usage_calls,
            "decision_calls": len(decisions),
            "decision_cost_usd": str(sum((call.cost_usd for call in decisions), Decimal("0")) + decision_reserved),
            "usage": {field: sum(getattr(call.usage, field) for call in calls)
                      for field in ("input_tokens", "cache_write_5m_tokens", "cache_write_1h_tokens",
                                    "cache_read_tokens", "output_tokens")},
            "modified_requests_this_process": sum(bool(item.get("changed")) and item["admitted"]
                                                  for item in telemetry),
            "interventions_this_process": telemetry,
            "budget": json.loads(json.dumps(budget, default=str)),
            "cost_basis": cost_basis,
        }
