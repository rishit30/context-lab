"""Deterministic content-only projections preserving canonical message order."""

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import ceil

from contextlab.core.models import Chunk, Level, Message


def canonical_bytes(messages: Sequence[Message]) -> bytes:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def estimate_tokens(text: str | bytes) -> int:
    """Conservative byte heuristic for policy forecasts, not usage accounting."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return ceil(len(data) / 4)


def render_chunk(chunk: Chunk, level: Level) -> str:
    if level == "full":
        return chunk.content
    pointer = (
        chunk.signpost_text
        or f'[archived {chunk.id}: {chunk.kind} at event {chunk.event_id}; recall("{chunk.id}", offset=0, limit=80) for the original snapshot]'
    )
    if level == "hidden":
        return pointer
    if level == "short":
        excerpt = chunk.short_text or "\n".join(chunk.content.splitlines()[:3])[:400]
        return f"{excerpt}\n{pointer}"
    excerpt = chunk.long_text
    if not excerpt:
        lines = chunk.content.splitlines()
        excerpt = "\n".join(
            lines if len(lines) <= 24 else [*lines[:12], "[… archived middle …]", *lines[-12:]]
        )[:4000]
    return f"{excerpt}\n{pointer}"


def render_view(
    messages: Sequence[Message], chunks: Sequence[Chunk], levels: Mapping[str, Level],
    duplicate_sources: Mapping[str, str] | None = None,
) -> list[Message]:
    rendered = deepcopy(list(messages))
    by_call: dict[str, Chunk] = {}
    by_id = {chunk.id: chunk for chunk in chunks}
    for archived_chunk in chunks:
        if archived_chunk.tool_call_id in by_call:
            raise ValueError(f"ambiguous chunk mapping for tool call {archived_chunk.tool_call_id}")
        by_call[archived_chunk.tool_call_id] = archived_chunk
    replacements: set[str] = set()
    for message in rendered:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            chunk = by_call.get(block.get("tool_use_id", ""))
            if chunk is None or levels.get(chunk.id, "full") == "full":
                continue
            if not isinstance(block.get("content"), str):
                raise ValueError(
                    "only plain-text tool results may be compacted; opaque blocks are preserved"
                )
            if block["content"] != chunk.content:
                raise ValueError(f"archive does not match canonical tool result for {chunk.id}")
            source_id = (duplicate_sources or {}).get(chunk.id)
            if source_id:
                source = by_id.get(source_id)
                if (source is None or levels.get(source_id, "full") != "full"
                        or source.content != chunk.content or source.seq <= chunk.seq
                        or levels[chunk.id] != "hidden"):
                    raise ValueError("invalid exact-duplicate anchor")
                block["content"] = (
                    f"[Exact duplicate output: the identical original text remains fully visible "
                    f"in later tool result {source.tool_call_id}; same file path, version and read range. "
                    f'Original observation at /context/{chunk.id}.txt: recall("{chunk.id}", offset=0, limit=80).]'
                )
            else:
                block["content"] = render_chunk(chunk, levels[chunk.id])
            replacements.add(chunk.id)
    expected = {chunk_id for chunk_id, level in levels.items() if level != "full"}
    if expected - replacements:
        raise ValueError(
            f"compacted chunks lack a matching result: {sorted(expected - replacements)}"
        )
    return rendered
