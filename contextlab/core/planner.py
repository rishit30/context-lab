"""Boundary-limited plans with persistent recall pins and recent-output guards."""

from collections.abc import Sequence

from contextlab.core.models import Chunk, CompactionConfig, Event, Level, ViewState
from contextlab.core.signals import derive_signals, recalled_chunk_ids


def plan_visibility(
    events: Sequence[Event], chunks: Sequence[Chunk], state: ViewState, config: CompactionConfig
) -> dict[str, Level]:
    levels = dict(state.levels)
    pins = (
        set(state.pinned_ids)
        | recalled_chunk_ids(events)
        | {chunk.id for chunk in chunks if chunk.pinned}
    )
    recent = sorted(chunks, key=lambda chunk: chunk.seq)
    if config.recent_outputs:
        pins.update(chunk.id for chunk in recent[-config.recent_outputs :])
    for signal in derive_signals(events, chunks):
        if signal.chunk_id not in pins:
            levels[signal.chunk_id] = signal.target
    for chunk_id in pins:
        levels[chunk_id] = "full"
    return levels
