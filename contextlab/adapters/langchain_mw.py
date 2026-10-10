"""Verified synchronous LangChain v1 adapter; provider objects stay intact."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from contextlab.core.controller import ContextController
from contextlab.core.models import (
    Chunk,
    CompactionConfig,
    Event,
    Pricing,
    ProjectionResult,
    Usage,
    ViewState,
)
from contextlab.core.signals import (
    TRUSTED_BOUNDARY_SOURCE,
    is_trusted_phase_boundary,
    workflow_boundary,
)
from contextlab.core.validator import validate_messages
from contextlab.core.view_builder import canonical_bytes, estimate_tokens
from contextlab.ledger.store import LedgerStore

BeforeCall = Callable[[ModelRequest], None]
UsageCallback = Callable[[str, Usage | None, float, str, str | None, dict[str, Any]], None]

RESPONSE_SUMMARY_LIMIT_BYTES = 16000
_SECRET_FIELD = re.compile(r"api.?key|authorization|password|secret|access.?token|session.?token", re.I)
_SECRET_VALUES = ("ANTHROPIC_API_KEY", "TYPESAFE_API_KEY", "OPENAI_API_KEY",
                  "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")


def _diagnostic_text(value: str, limit: int = 2000) -> str:
    for name in _SECRET_VALUES:
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{12,}\b", "[REDACTED]", value)
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    encoded = value.encode("utf-8", errors="replace")
    return encoded[:limit].decode("utf-8", errors="ignore") + ("…" if len(encoded) > limit else "")


def _diagnostic_value(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return "[DEPTH_LIMIT]"
    if isinstance(value, str):
        return _diagnostic_text(value)
    if isinstance(value, dict):
        return {_diagnostic_text(str(key), 100): "[REDACTED]" if _SECRET_FIELD.search(str(key))
                else _diagnostic_value(item, depth + 1)
                for key, item in list(value.items())[:32]}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_value(item, depth + 1) for item in value[:32]]
    if isinstance(value, float):
        return value if math.isfinite(value) else "[NON_FINITE]"
    if value is None or isinstance(value, (bool, int)):
        return value
    return f"[{type(value).__name__}]"


def response_summary(messages: list[Any]) -> dict[str, Any]:
    """Bounded local diagnostics, excluding headers and opaque/reasoning blocks."""
    summary: dict[str, Any] = {"messages": [], "truncated": False}

    def fits() -> bool:
        return len(json.dumps(summary, ensure_ascii=False).encode()) <= RESPONSE_SUMMARY_LIMIT_BYTES

    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if len(summary["messages"]) >= 8:
            summary["truncated"] = True
            break
        content = message.content
        if isinstance(content, str):
            visible = content
        else:
            visible = "\n".join(block.get("text", "") for block in content
                                if isinstance(block, dict) and block.get("type") == "text"
                                and isinstance(block.get("text"), str))
        stop_reason = message.response_metadata.get(
            "stop_reason", message.response_metadata.get("finish_reason"))
        entry: dict[str, Any] = {
            "stop_reason": _diagnostic_text(stop_reason, 100) if isinstance(stop_reason, str) else None,
            "content": _diagnostic_text(visible, 4000), "tool_calls": [],
        }
        summary["messages"].append(entry)
        if not fits():
            summary["messages"].pop()
            summary["truncated"] = True
            break
        calls = [*message.tool_calls, *message.invalid_tool_calls]
        for call in calls[:16]:
            args = _diagnostic_value(call.get("args"))
            serialized = json.dumps(args, ensure_ascii=False)
            if len(serialized.encode()) > 2400:
                args = {"preview": _diagnostic_text(serialized, 1600), "truncated": True}
            entry["tool_calls"].append({
                "name": _diagnostic_text(str(call.get("name", "")), 100),
                "id": _diagnostic_text(str(call.get("id", "")), 200),
                "args": args, "invalid": call.get("type") == "invalid_tool_call",
            })
            if not fits():
                entry["tool_calls"].pop()
                summary["truncated"] = True
                break
        if len(calls) > len(entry["tool_calls"]) or len(visible.encode()) > 4000:
            summary["truncated"] = True
    return summary


def normalize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Create a validator/controller view, never a replacement provider history."""
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            block = {"type": "tool_result", "tool_use_id": message.tool_call_id,
                     "content": deepcopy(message.content)}
            if message.status == "error":
                block["is_error"] = True
            if normalized and normalized[-1]["role"] == "user" and isinstance(normalized[-1]["content"], list):
                normalized[-1]["content"].append(block)
            else:
                normalized.append({"role": "user", "content": [block]})
        elif isinstance(message, AIMessage):
            content = deepcopy(message.content)
            if message.tool_calls:
                blocks = [{"type": "text", "text": content}] if isinstance(content, str) and content else (content if isinstance(content, list) else [])
                existing = {b.get("id") for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"}
                for call in message.tool_calls:
                    if call["id"] not in existing:
                        blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": deepcopy(call["args"])})
                content = blocks
            normalized.append({"role": "assistant", "content": content})
        elif isinstance(message, HumanMessage):
            normalized.append({"role": "user", "content": deepcopy(message.content)})
        else:
            raise ValueError(f"Unsupported agent message type: {type(message).__name__}")
    return normalized


def apply_projection(original: list[BaseMessage], projected: list[dict[str, Any]]) -> list[BaseMessage]:
    """Copy only tool-result text; preserve IDs, opaque blocks and all metadata."""
    validate_messages(projected)
    replacements = {}
    for message in projected:
        if isinstance(message["content"], list):
            for block in message["content"]:
                if block.get("type") == "tool_result":
                    replacements[block["tool_use_id"]] = block["content"]
    result: list[BaseMessage] = []
    for message in original:
        if isinstance(message, ToolMessage) and message.tool_call_id in replacements:
            content = replacements[message.tool_call_id]
            if content != message.content:
                # Opaque/multimodal tool results are never compressed by this adapter.
                if not isinstance(message.content, str) or not isinstance(content, str):
                    raise ValueError("Projection attempted to change an opaque tool result")
                message = message.model_copy(update={"content": content}, deep=True)
        result.append(message)
    # No other model-visible mutations are permitted by the adapter contract.
    if normalize_messages(result) != projected:
        raise ValueError("Projection modified fields outside tool-result text")
    return result


def langchain_usage(message: AIMessage, ttl: str = "5m") -> Usage | None:
    """LC input_tokens includes caches, unlike Anthropic's raw input_tokens."""
    raw = message.usage_metadata
    if raw is None:
        return None
    detail = raw.get("input_token_details", {})
    read = detail.get("cache_read", 0)
    short = detail.get("ephemeral_5m_input_tokens", 0)
    long = detail.get("ephemeral_1h_input_tokens", 0)
    aggregate = detail.get("cache_creation", 0)
    if not short and not long:
        short, long = (aggregate, 0) if ttl == "5m" else (0, aggregate)
    base = raw["input_tokens"] - read - short - long
    return Usage(input_tokens=base, cache_write_5m_tokens=short, cache_write_1h_tokens=long,
                 cache_read_tokens=read, output_tokens=raw["output_tokens"])


class MeteringMiddleware(AgentMiddleware):
    """Exactly one ledger callback for each attempted model invocation."""
    def __init__(self, *, model_name: str, before_call: BeforeCall | None = None,
                 on_usage: UsageCallback | None = None, max_model_calls: int = 30,
                 cache_ttl: str = "5m") -> None:
        self.model_name = model_name
        self.before_call = before_call
        self.on_usage = on_usage
        self.max_model_calls = max_model_calls
        self.cache_ttl = cache_ttl
        self.calls = 0

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        if self.calls >= self.max_model_calls:
            raise RuntimeError("Model call limit reached")
        if self.before_call:
            self.before_call(request)
        self.calls += 1
        started = time.perf_counter()
        try:
            response = handler(request)
        except Exception as exc:
            if self.on_usage:
                self.on_usage(self.model_name, None, (time.perf_counter() - started) * 1000,
                              "failed", type(exc).__name__, {})
            raise
        receipts = {"messages": [
            {"usage_metadata": deepcopy(item.usage_metadata),
             "provider_usage": deepcopy(item.response_metadata.get("usage"))}
            for item in response.result if isinstance(item, AIMessage)
        ]}
        try:
            receipts["response_summary"] = response_summary(response.result)
        except Exception as exc:
            # Diagnostic extraction must never prevent receipt settlement.
            receipts["response_summary"] = {"error_type": type(exc).__name__}
        try:
            metadata = [langchain_usage(item, self.cache_ttl) for item in response.result if isinstance(item, AIMessage)]
        except (ValueError, TypeError, KeyError):
            metadata = []
        usage: Usage | None = None
        if metadata and all(item is not None for item in metadata):
            usage = Usage(**{field: sum(getattr(item, field) for item in metadata)
                             for field in Usage.model_fields})
        if self.on_usage:
            self.on_usage(self.model_name, usage, (time.perf_counter() - started) * 1000,
                          "ok" if usage is not None else "usage_missing", None, receipts)
        return response


class ContextLabMiddleware(AgentMiddleware):
    """Observe canonical history and render a separate model-facing view."""
    def __init__(self, *, controller: ContextController, store: LedgerStore,
                 session_id: str, pricing: Pricing, config: CompactionConfig,
                 shadow: bool = False, observe_only: bool = False,
                 cache_ttl_seconds: int = 300) -> None:
        if not isinstance(cache_ttl_seconds, int) or isinstance(cache_ttl_seconds, bool) or cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds must be a nonnegative integer; zero disables caching forecasts")
        self.controller, self.store = controller, store
        self.session_id, self.pricing, self.config = session_id, pricing, config
        self.shadow, self.cache_ttl_seconds = shadow, cache_ttl_seconds
        self.observe_only = observe_only
        self.events = self.store.list_events(session_id)
        self.chunks = self.store.list_chunks(session_id)
        self.state = self.store.get_projection_state(session_id)
        self._seen = {str(event.metadata["_message_signature"]) for event in self.events
                      if "_message_signature" in event.metadata}
        self._last_call: float | None = None
        self._lock = threading.RLock()
        self.failures: list[str] = []
        self.telemetry: list[dict[str, Any]] = []

    def mark_boundary(self, label: str) -> Event:
        """Persist a trusted completed harness phase, never a simulated test pass.

        Labels are unique phase identifiers: repeating one is idempotent across
        restarts. This API is not exposed as an agent tool and does not add text
        to the model conversation. The next request may evaluate eligible data.
        """
        if not isinstance(label, str) or not label.strip() or len(label) > 120:
            raise ValueError("Boundary label must contain 1–120 characters")
        with self._lock:
            for event in self.events:
                if is_trusted_phase_boundary(event) and event.metadata.get("label") == label:
                    return event
            seq = max((event.seq for event in self.events), default=-1) + 1
            event = Event(id=f"b{seq:06d}", session_id=self.session_id, seq=seq,
                          role="system", kind="phase_boundary", metadata={
                              "source": TRUSTED_BOUNDARY_SOURCE, "trusted": True, "label": label,
                          })
            self.store.append_event(event)
            self.events.append(event)
            return event

    def _ingest(self, messages: list[BaseMessage]) -> None:
        calls = {call["id"]: call for msg in messages if isinstance(msg, AIMessage) for call in msg.tool_calls}
        for index, message in enumerate(messages):
            signature = f"{index}:" + hashlib.sha256(message.model_dump_json().encode()).hexdigest()
            if signature in self._seen:
                continue
            seq = max((event.seq for event in self.events), default=-1) + 1
            name, call_id, args, metadata = "", "", {}, {}
            kind = "assistant_msg" if isinstance(message, AIMessage) else "user_msg"
            content = message.content if isinstance(message.content, str) else json.dumps(message.content, sort_keys=True)
            role = "assistant" if isinstance(message, AIMessage) else "user"
            if isinstance(message, ToolMessage):
                role, call_id = "tool", message.tool_call_id
                call = calls.get(call_id, {})
                name, args = call.get("name", message.name or ""), call.get("args", {})
                kind = {"read_file": "file_read", "write_file": "file_write", "edit_file": "file_edit", "run_tests": "test_run", "search": "search", "recall": "recall", "shell": "shell"}.get(name, "other")
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        metadata = {k: v for k, v in parsed.items() if k not in {"content", "output", "matches"}}
                except (ValueError, TypeError):
                    pass
                if kind == "file_read":
                    metadata.update(version=metadata.get("sha256"), start_line=metadata.get("offset", 0) + 1,
                                    end_line=min(metadata.get("offset", 0) + metadata.get("limit", 0), metadata.get("total_lines", 0)))
                    if str(args.get("path", "")).startswith("/context/"):
                        kind = "recall"
                        metadata["chunk_id"] = args["path"].removeprefix("/context/").removesuffix(".txt")
                if kind == "test_run":
                    metadata.update(environment="contextlab-docker-pytest", cwd=".")
                if kind == "recall":
                    metadata.setdefault("chunk_id", args.get("chunk_id", ""))
                if message.status == "error":
                    metadata["tool_error"] = True
                    kind = "other"
            event = Event(id=f"e{seq:06d}", session_id=self.session_id, seq=seq, role=role,
                          kind=kind, tool_name=name, tool_call_id=call_id, args=args,
                          content=content, metadata={**metadata, "_message_signature": signature},
                          size_tokens=estimate_tokens(content))
            self.store.append_event(event)
            self.events.append(event)
            if isinstance(message, ToolMessage) and isinstance(message.content, str):
                chunk = Chunk(id=f"c{seq:06d}", session_id=self.session_id, event_id=event.id,
                              tool_call_id=call_id, kind=kind, content=content, seq=seq,
                              size_tokens=event.size_tokens,
                              refs=(str(metadata.get("path", args.get("path"))),) if metadata.get("path", args.get("path")) else (),
                              metadata=metadata)
                self.store.add_chunk(chunk)
                self.chunks.append(chunk)
            self._seen.add(signature)

    def observe(self, messages: list[BaseMessage]) -> None:
        """Archive actual canonical messages without projection or model execution."""
        with self._lock:
            self._ingest(messages)

    def project_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Persist and project a canonical history without invoking any provider.

        A configured semantic gate may run its own budgeted assessment. With a
        rules-only controller this method makes no model or network calls.
        """
        replacement = messages
        with self._lock:
            started = time.perf_counter()
            observation: dict[str, Any] = {
                "request_index": len(self.telemetry), "changed": False, "changed_tool_results": 0,
                "estimated_tokens_saved": 0, "proposed_tokens_saved": 0, "rebuilt": False,
                "fallback": False, "error_type": None, "no_op_reason": None, "decision": None,
                "boundary_id": None, "boundary_kind": None, "boundary_status": "none",
                "cache_warm": False,
                "exact_duplicate_retirements": {},
                "cache_mode": "disabled" if self.cache_ttl_seconds == 0 else "enabled",
            }
            try:
                self._ingest(messages)
                now = time.monotonic()
                warm = (self.cache_ttl_seconds > 0 and self._last_call is not None
                        and now - self._last_call < self.cache_ttl_seconds)
                canonical = normalize_messages(messages)
                boundary = workflow_boundary(self.events)
                observation.update(
                    boundary_id=boundary,
                    boundary_kind=next((event.kind for event in self.events if event.id == boundary), None),
                    boundary_status=("none" if boundary is None else
                                     "already_considered" if boundary == self.state.last_boundary_id else "new"),
                    cache_warm=warm,
                )
                # With caching disabled, every future input token uses the base
                # price. This changes forecasts only; billing uses real receipts.
                forecast_pricing = (self.pricing.model_copy(update={
                    "cache_read_per_million": self.pricing.input_per_million,
                    "cache_write_5m_per_million": self.pricing.input_per_million,
                    "cache_write_1h_per_million": self.pricing.input_per_million,
                }) if self.cache_ttl_seconds == 0 else self.pricing)
                projection = (ProjectionResult(messages=canonical, state=self.state, no_op_reason="observe_only") if self.observe_only
                              else self.controller.project(canonical, self.events,
                                                           self.chunks, self.state, forecast_pricing, self.config,
                                                           cache_warm=warm, shadow=self.shadow))
                if projection.fallback:
                    error_type = (projection.error or "ProjectionError").split(":", 1)[0]
                    self.failures.append(error_type)
                    projection = projection.model_copy(update={"error": error_type})
                projected_messages = apply_projection(messages, projection.messages)
                self.store.record_projection(self.session_id, len(self.events), projection)
                self.state = projection.state
                replacement = projected_messages
                self._last_call = now
                changed = sum(isinstance(old, ToolMessage) and old.content != new.content
                              for old, new in zip(messages, projected_messages, strict=True))
                observation.update(
                    changed=bool(changed), changed_tool_results=changed,
                    estimated_tokens_saved=max(estimate_tokens(canonical_bytes(canonical))
                                               - estimate_tokens(canonical_bytes(projection.messages)), 0),
                    proposed_tokens_saved=projection.estimated_tokens_saved,
                    rebuilt=projection.rebuilt, fallback=projection.fallback,
                    error_type=projection.error, no_op_reason=projection.no_op_reason,
                    decision=projection.decision.model_dump(mode="json") if projection.decision else None,
                    exact_duplicate_retirements=dict(projection.exact_duplicate_retirements),
                )
            except Exception as exc:
                # Type-only logs avoid copying message contents or secret-like data.
                replacement = messages
                self.failures.append(type(exc).__name__)
                observation.update(fallback=True, error_type=type(exc).__name__, no_op_reason="adapter_failed")
                self.state = ViewState(epoch=self.state.epoch + 1, pinned_ids=self.state.pinned_ids)
                try:
                    self.store.record_projection(self.session_id, len(self.events), ProjectionResult(
                        messages=[], state=self.state, fallback=True, error=type(exc).__name__,
                        no_op_reason="adapter_failed"))
                except Exception:
                    pass  # Store unavailability is already visible in failures.
            observation.update(latency_ms=(time.perf_counter() - started) * 1000,
                               recall_events=sum(event.kind == "recall" for event in self.events))
            self.telemetry.append(observation)
        return replacement

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        # Provider errors stay outside projection recovery; never retry here.
        with self._lock:
            projected = self.project_messages(request.messages)
            observation = self.telemetry[-1]
        response = handler(request.override(messages=projected))
        with self._lock:
            try:
                self._ingest([*request.messages, *response.result])
            except Exception as exc:
                self.failures.append(type(exc).__name__)
                observation["response_ingestion_error_type"] = type(exc).__name__
        return response

    def recall(self, chunk_id: str, offset: int = 0, limit: int = 80, char_offset: int = 0) -> str:
        with self._lock:
            result = self.store.recall(self.session_id, chunk_id, len(self.events), offset=offset, limit=limit, char_offset=char_offset)
            # Pin immediately; ingestion records the tool evidence on the next request.
            self.state = self.state.model_copy(update={"pinned_ids": tuple(sorted(set(self.state.pinned_ids) | {chunk_id}))})
            return result.model_dump_json()
