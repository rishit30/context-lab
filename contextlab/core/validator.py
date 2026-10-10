"""Validate Anthropic-shaped message tool protocol without rewriting messages."""

from collections.abc import Mapping, Sequence
from typing import Any


class MessageValidationError(ValueError):
    """A request violates the supported message or tool-call protocol."""


def validate_messages(messages: Sequence[Mapping[str, Any]]) -> None:
    """Require each assistant tool call to have an immediate user result.

    Opaque content blocks are deliberately left alone. The middleware owns tool
    result *content*, never provider reasoning, image, signature, or cache fields.
    """
    pending: set[str] = set()
    seen_calls: set[str] = set()
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise MessageValidationError(f"message {index}: unsupported role {role!r}")
        content = message.get("content")
        if isinstance(content, str):
            blocks: list[Any] = []
        elif isinstance(content, list):
            blocks = content
        else:
            raise MessageValidationError(f"message {index}: content must be text or blocks")
        if any(not isinstance(block, dict) for block in blocks):
            raise MessageValidationError(f"message {index}: content blocks must be objects")
        if pending and role != "user":
            raise MessageValidationError(
                f"message {index}: tool results must immediately follow calls"
            )

        results: set[str] = set()
        calls: set[str] = set()
        non_result_seen = False
        for block in blocks:
            kind = block.get("type")
            if kind == "tool_use":
                if role != "assistant":
                    raise MessageValidationError(
                        f"message {index}: tool_use must have assistant role"
                    )
                call_id = block.get("id")
                if not isinstance(call_id, str) or not call_id:
                    raise MessageValidationError(f"message {index}: tool_use needs a nonempty id")
                if call_id in seen_calls or call_id in calls:
                    raise MessageValidationError(
                        f"message {index}: duplicate tool_use id {call_id}"
                    )
                if not isinstance(block.get("name"), str) or not isinstance(
                    block.get("input"), dict
                ):
                    raise MessageValidationError(f"message {index}: tool_use needs name and input")
                calls.add(call_id)
            if kind == "tool_result":
                if role != "user":
                    raise MessageValidationError(
                        f"message {index}: tool_result must have user role"
                    )
                if non_result_seen:
                    raise MessageValidationError(
                        f"message {index}: tool results must precede user text"
                    )
                result_id = block.get("tool_use_id")
                if not isinstance(result_id, str) or result_id not in pending:
                    raise MessageValidationError(
                        f"message {index}: tool_result has no immediate matching call"
                    )
                if result_id in results:
                    raise MessageValidationError(
                        f"message {index}: duplicate result for {result_id}"
                    )
                results.add(result_id)
            else:
                non_result_seen = True
        if pending != results:
            raise MessageValidationError(
                f"message {index}: missing tool results for {sorted(pending - results)}"
            )
        seen_calls.update(calls)
        pending = calls
    if pending:
        raise MessageValidationError(f"request ends before tool results for {sorted(pending)}")
