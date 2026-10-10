"""Evidence-based lifecycle signals; absence of references is not evidence."""

from collections.abc import Sequence
from typing import Any

from contextlab.core.models import Chunk, Event, Signal

# This tag identifies an event emitted through the explicit adapter control API.
# Tool observations and user text are never parsed into this event kind or role.
TRUSTED_BOUNDARY_SOURCE = "contextlab.harness.mark_boundary/v1"


def _field(event: Event, name: str, default: Any = None) -> Any:
    return event.metadata.get(name, event.args.get(name, default))


def _path(event: Event) -> str:
    value = _field(event, "path", "")
    return value if isinstance(value, str) else ""


def _covers(new: Event, old: Event) -> bool:
    if _field(new, "full_read", False) is True:
        return True
    if _field(old, "full_read", False) is True:
        return False
    values = [_field(event, key) for event in (old, new) for key in ("start_line", "end_line")]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        return False
    old_start, old_end, new_start, new_end = values
    return bool(1 <= new_start <= old_start <= old_end <= new_end)


def _status(event: Event) -> str | None:
    exit_code = _field(event, "exit_code")
    if isinstance(exit_code, int) and not isinstance(exit_code, bool):
        return "passed" if exit_code == 0 else "failed"
    status = _field(event, "status")
    return status if isinstance(status, str) and status in {"passed", "failed"} else None


def _test_identity(event: Event) -> tuple[str, str, str] | None:
    selection = _field(event, "test_selection", _field(event, "command"))
    if not isinstance(selection, str) or not selection.strip():
        return None
    environment = _field(event, "environment", "")
    cwd = _field(event, "cwd", "")
    if not isinstance(environment, str) or not isinstance(cwd, str):
        return None
    return selection, environment, cwd


def _matching_pass(old: Event, new: Event) -> bool:
    # A matching command is evidence about the workflow, not proof that test
    # files or assertions were preserved. The harness must independently grade
    # completion. Enumerated identities, when supplied, strengthen this signal.
    identity = _test_identity(old)
    if identity is None or identity != _test_identity(new):
        return False
    if _status(old) != "failed" or _status(new) != "passed":
        return False
    # When both runs enumerate tests, a narrower successful selection is not a
    # resolution even if a wrapper reused the same command string.
    old_tests, new_tests = _field(old, "test_ids"), _field(new, "test_ids")
    if isinstance(old_tests, list):
        if not isinstance(new_tests, list):
            return False
        if not set(map(str, old_tests)).issubset(set(map(str, new_tests))):
            return False
    return True


def derive_signals(events: Sequence[Event], chunks: Sequence[Chunk]) -> list[Signal]:
    ordered = sorted(events, key=lambda event: event.seq)
    by_id = {event.id: event for event in ordered}
    signals: list[Signal] = []
    for chunk in sorted(chunks, key=lambda item: (item.seq, item.id)):
        original = by_id.get(chunk.event_id)
        if original is None:
            continue
        later = [
            event
            for event in ordered
            if event.seq > original.seq and event.session_id == original.session_id
        ]
        if original.kind == "file_read" and _path(original):
            replacements = [
                event
                for event in later
                if event.kind == "file_read"
                and _path(event) == _path(original)
                and _covers(event, original)
            ]
            if replacements:
                replacement = replacements[-1]
                signals.append(
                    Signal(
                        chunk_id=chunk.id,
                        signal_type="superseded_read",
                        triggered_by_event_id=replacement.id,
                        reason=f"Later read {replacement.id} covers the earlier read of {_path(original)}; historical content remains retrievable.",
                        target="hidden",
                    )
                )
        elif original.kind == "test_run" and _status(original) == "failed":
            replacements = [
                event
                for event in later
                if event.kind == "test_run" and _matching_pass(original, event)
            ]
            if replacements:
                replacement = replacements[-1]
                signals.append(
                    Signal(
                        chunk_id=chunk.id,
                        signal_type="resolved_test_log",
                        triggered_by_event_id=replacement.id,
                        reason=f"Matching test selection passed in {replacement.id}; historical failure remains retrievable.",
                        target="short",
                    )
                )
    return signals


def test_pass_boundary(events: Sequence[Event]) -> str | None:
    """Return the latest evidenced fail-to-pass event, never an unrelated pass."""
    ordered = sorted(
        (event for event in events if event.kind == "test_run"), key=lambda event: event.seq
    )
    boundary: str | None = None
    latest: dict[tuple[str, tuple[str, str, str]], Event] = {}
    for event in ordered:
        identity = _test_identity(event)
        if identity is None:
            continue
        key = event.session_id, identity
        previous = latest.get(key)
        if previous is not None and _matching_pass(previous, event):
            boundary = event.id
        latest[key] = event
    return boundary


def is_trusted_phase_boundary(event: Event) -> bool:
    return (
        event.kind == "phase_boundary"
        and event.role == "system"
        and not event.tool_name
        and not event.tool_call_id
        and event.metadata.get("source") == TRUSTED_BOUNDARY_SOURCE
        and event.metadata.get("trusted") is True
    )


def workflow_boundary(events: Sequence[Event]) -> str | None:
    """Latest real fail→pass transition or explicitly trusted harness boundary.

    A phase boundary permits evaluating a plan; it is never evidence that any
    particular output is stale. Ordinary message content cannot create one.
    """
    passing_id = test_pass_boundary(events)
    candidates = [event for event in events
                  if event.id == passing_id or is_trusted_phase_boundary(event)]
    return max(candidates, key=lambda event: event.seq).id if candidates else None


def recalled_chunk_ids(events: Sequence[Event]) -> set[str]:
    """Pin recalled originals for the remainder of this MVP session."""
    return {
        str(_field(event, "chunk_id"))
        for event in events
        if event.kind == "recall"
        and _field(event, "chunk_id")
        and _field(event, "success", True) is not False
    }
