"""Genuine Headroom lossless compression ahead of LangChain request metering.

This adapter uses the public ``headroom.compress`` API. It does not wrap the
provider client, run a second model, or replace upstream compression algorithms.
The selected live-agent profile disables CCR and ML compression. Token metrics
here are offline estimates; only provider receipts establish billed usage.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

from contextlab.adapters.langchain_mw import apply_projection, normalize_messages

HEADROOM_VERSION = "0.40.0"
_COMPRESS_CONFIG: dict[str, Any] = {
    "compress_user_messages": False,
    "compress_system_messages": False,
    "protect_recent": 0,
    "protect_analysis_context": True,
    "frozen_message_count": 0,
    "target_ratio": None,
    "min_tokens_to_compress": 250,
    "kompress_model": "disabled",
    "savings_profile": None,
    "diagnostics": True,
    "mode": "agent",
    "verify_lossless": True,
}
_OFFLINE_ENV = {
    "HEADROOM_OFFLINE": "1",
    "HEADROOM_TELEMETRY": "off",
    "HEADROOM_UPDATE_CHECK": "off",
    "HEADROOM_DISABLE_KOMPRESS": "1",
    "HEADROOM_DETECT_BACKEND": "python",
    "HEADROOM_CCR_BACKEND": "memory",
    "HEADROOM_TOIN_BACKEND": "none",
}
_COMPRESSION_LOCK = threading.RLock()


def configuration() -> dict[str, Any]:
    """Return the exact comparison profile, requiring the optional pinned SDK."""
    try:
        installed = version("headroom-ai")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "The Headroom arm requires the optional dependency: uv sync --extra headroom"
        ) from exc
    if installed != HEADROOM_VERSION:
        raise RuntimeError(
            f"Headroom comparison requires headroom-ai=={HEADROOM_VERSION}; found {installed}"
        )
    return {
        "distribution": "headroom-ai",
        "version": installed,
        "api": "headroom.compress",
        "profile": "agent_lossless_offline",
        "compress_config": deepcopy(_COMPRESS_CONFIG),
        "offline_environment": dict(_OFFLINE_ENV),
        "token_counter": "headroom.tokenizers.EstimatingTokenCounter",
        "token_counter_chars_per_token": 3.5,
        "token_counts_are_estimates": True,
        "ccr_enabled": False,
        "recall_applicable": False,
        "provider_calls_by_compressor": 0,
        "message_scope": "tool_result_text_only",
    }


class HeadroomMiddleware(AgentMiddleware):
    """Synchronous request projection with original-history fallback on errors.

    Install this BEFORE metering in ``create_agent(middleware=[...])`` so the
    counter and trace observe the transformed request. Process-wide Headroom
    offline switches are intentional. The upstream tokenizer registry is given
    its own character estimator to avoid implicit tiktoken vocabulary downloads.
    This does not change ChatAnthropic or its provider token-count endpoint.
    """

    def __init__(self, *, model_name: str = "claude-sonnet-4-6",
                 model_limit: int = 200000) -> None:
        if not model_name or model_limit <= 0:
            raise ValueError("model_name and a positive model_limit are required")
        self.configuration = configuration()
        self.model_name = model_name
        self.model_limit = model_limit
        self.failures: list[str] = []
        self.telemetry: list[dict[str, Any]] = []
        self._stats: dict[str, int | float] = {
            "calls": 0, "changed_calls": 0, "changed_tool_results": 0,
            "compression_ms": 0.0, "fallback_calls": 0, "recalls": 0,
        }
        # Set before importing the optional package or constructing a pipeline.
        with _COMPRESSION_LOCK:
            os.environ.update(_OFFLINE_ENV)
            from headroom import CompressConfig, compress
            from headroom.offline import OfflineEgressBlocked
            from headroom.tokenizers import EstimatingTokenCounter, register_tokenizer

            register_tokenizer(model_name, EstimatingTokenCounter(chars_per_token=3.5))
        self._compress = compress
        self._config = CompressConfig(**_COMPRESS_CONFIG)
        self._offline_error = OfflineEgressBlocked

    @property
    def stats(self) -> dict[str, int | float]:
        return dict(self._stats)

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        started = time.perf_counter()
        self._stats["calls"] += 1
        entry: dict[str, Any] = {
            "call_index": self._stats["calls"], "profile": "agent_lossless_offline",
            "changed": False, "changed_tool_results": 0, "fallback": False,
            "token_counts_are_estimates": True, "recall_applicable": False,
        }
        forwarded = request
        try:
            normalized = normalize_messages(request.messages)
            with _COMPRESSION_LOCK:
                # An accidental environment override must not enable downloads.
                os.environ.update(_OFFLINE_ENV)
                compressed = self._compress(
                    deepcopy(normalized), model=self.model_name,
                    model_limit=self.model_limit, config=deepcopy(self._config),
                )
            if compressed.lossless is not True:
                # Upstream's exception fallback returns unchecked, zero-count
                # results. Expose it as failure rather than successful execution.
                raise ValueError("Headroom returned an unverified compression result")
            projected = apply_projection(request.messages, compressed.messages)
            changes = sum(
                isinstance(before, ToolMessage) and before.content != after.content
                for before, after in zip(request.messages, projected, strict=True)
            )
            entry.update({
                "tokens_before_estimate": compressed.tokens_before,
                "tokens_after_estimate": compressed.tokens_after,
                "transforms_applied": list(compressed.transforms_applied),
                "lossless_verified": True,
                "reverted_messages": compressed.reverted_messages,
                "changed": bool(changes),
                "changed_tool_results": changes,
                "message_decisions": [
                    {"message_index": decision.message_index, "role": decision.role,
                     "action": decision.action,
                     "tokens_before_estimate": decision.tokens_before,
                     "tokens_after_estimate": decision.tokens_after}
                    for decision in (compressed.diagnostics or [])
                ],
            })
            if changes:
                forwarded = request.override(messages=projected)
                self._stats["changed_calls"] += 1
                self._stats["changed_tool_results"] += changes
        except (Exception, self._offline_error) as exc:
            # Never log raw exception messages: SDK exceptions can include input.
            failure = type(exc).__name__
            self.failures.append(failure)
            self._stats["fallback_calls"] += 1
            entry.update({"fallback": True, "error_type": failure})
        elapsed = (time.perf_counter() - started) * 1000
        entry["compression_ms"] = elapsed
        self._stats["compression_ms"] += elapsed
        self.telemetry.append(entry)
        # Deliberately outside the catch: provider failures must never retry.
        return handler(forwarded)
