"""Exact read-copy proofs and persistent full-copy anchors; no semantic guesses."""

from collections.abc import Mapping, Sequence

from contextlab.core.models import Chunk, Event, Level, Message, ViewState


def full_result_chunks(
    messages: Sequence[Message], chunks: Sequence[Chunk], levels: Mapping[str, Level]
) -> tuple[Chunk, ...]:
    """Only complete originals actually present and kept full in this request."""
    results: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and isinstance(block.get("content"), str)
            ):
                results[block.get("tool_use_id", "")] = block["content"]
    return tuple(
        chunk
        for chunk in chunks
        if levels.get(chunk.id, "full") == "full"
        and results.get(chunk.tool_call_id) == chunk.content
    )


def exact_read_copies(
    events: Sequence[Event], chunks: Sequence[Chunk], messages: Sequence[Message]
) -> dict[str, tuple[str, ...]]:
    """Match later identical outputs with equal, explicit read provenance.

    Equality is deliberately stricter than overlapping ranges or equal file
    hashes. The entire original tool-result text and exact range must agree.
    Both results must actually be present in this canonical request. Metadata
    comes from tool observations, never from agent-supplied call arguments.
    """
    results: dict[str, str] = {}
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_result"
                and not block.get("is_error")
                and isinstance(block.get("content"), str)
            ):
                results[block.get("tool_use_id", "")] = block["content"]
    by_event = {event.id: event for event in events}
    groups: dict[tuple[str, str, str, int, int, bool, str], list[Chunk]] = {}
    for chunk in sorted(chunks, key=lambda item: (item.seq, item.id)):
        event = by_event.get(chunk.event_id)
        if (
            event is None
            or chunk.kind != "file_read"
            or event.kind != "file_read"
            or event.role != "tool"
            or event.session_id != chunk.session_id
            or event.seq != chunk.seq
            or event.tool_call_id != chunk.tool_call_id
            or event.content != chunk.content
            or results.get(chunk.tool_call_id) != chunk.content
            or event.metadata.get("tool_error")
        ):
            continue
        path, version = event.metadata.get("path"), event.metadata.get("version")
        start, end = event.metadata.get("start_line"), event.metadata.get("end_line")
        full = event.metadata.get("full_read")
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(version, str)
            or not version
            or not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not 1 <= start <= end
            or not isinstance(full, bool)
            or (full and start != 1)
        ):
            continue
        groups.setdefault(
            (chunk.session_id, path, version, start, end, full, chunk.content), []
        ).append(chunk)
    copies: dict[str, tuple[str, ...]] = {}
    for group in groups.values():
        for original in group:
            later = tuple(other.id for other in group if other.seq > original.seq)
            if later:
                copies[original.id] = later
    return copies


def restore_copy_invariants(
    state: ViewState, copies: Mapping[str, tuple[str, ...]], protected: set[str]
) -> ViewState:
    """Restore content when a persisted proof, full anchor, or pin is invalid.

    Restoration is mandatory even without a new boundary. A restart must never
    trust an old pointer to an absent, modified, or now compacted replacement.
    """
    levels = dict(state.levels)
    for chunk_id in protected:
        if levels.get(chunk_id, "full") != "full":
            levels[chunk_id] = "full"
    valid: dict[str, str] = {}
    for original, retained in state.duplicate_sources.items():
        if (
            levels.get(original, "full") == "hidden"
            and original not in protected
            and retained in copies.get(original, ())
            and levels.get(retained, "full") == "full"
        ):
            valid[original] = retained
        elif levels.get(original, "full") != "full":
            levels[original] = "full"
    return state.model_copy(
        update={
            "levels": levels,
            "duplicate_sources": valid,
            "epoch": state.epoch + int(levels != state.levels),
        }
    )


def reserve_copy_anchors(
    proposed: dict[str, Level],
    state: ViewState,
    chunks: Sequence[Chunk],
    copies: Mapping[str, tuple[str, ...]],
    protected: set[str],
) -> dict[str, str]:
    """Reduce older copies while retaining a currently full, later exact copy.

    Anchor reservation is conservative: a semantic plan cannot subsequently
    remove an anchor used by an existing duplicate pointer. No chains arise.
    """
    anchors = set(state.duplicate_sources.values())
    for chunk_id in protected | anchors:
        proposed[chunk_id] = "full"
    new: dict[str, str] = {}
    for chunk in sorted(chunks, key=lambda item: (item.seq, item.id), reverse=True):
        if (
            chunk.id in protected
            or chunk.id in anchors
            or proposed.get(chunk.id, "full") != "hidden"
            or state.levels.get(chunk.id, "full") == "hidden"
        ):
            continue
        targets = [
            target
            for target in copies.get(chunk.id, ())
            if state.levels.get(target, "full") == "full"
        ]
        if not targets:
            continue
        target = targets[-1]
        proposed[target] = "full"
        anchors.add(target)
        new[chunk.id] = target
    return new
