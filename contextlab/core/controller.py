"""Safe orchestration: failures skip intervention and return original history."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from decimal import Decimal

from contextlab.core.cost_model import decide_rebuild
from contextlab.core.models import (
    Chunk,
    CompactionConfig,
    CostInputs,
    Event,
    Level,
    Message,
    Pricing,
    ProjectionResult,
    ViewState,
)
from contextlab.core.planner import plan_visibility
from contextlab.core.signals import derive_signals, recalled_chunk_ids, test_pass_boundary
from contextlab.core.validator import validate_messages
from contextlab.core.view_builder import canonical_bytes, estimate_tokens, render_view

Planner = Callable[
    [Sequence[Event], Sequence[Chunk], ViewState, CompactionConfig], dict[str, Level]
]


class ContextController:
    def __init__(self, planner: Planner | None = None) -> None:
        self._planner = planner or plan_visibility

    def project(
        self,
        messages: list[Message],
        events: list[Event],
        chunks: list[Chunk],
        state: ViewState,
        pricing: Pricing,
        config: CompactionConfig,
        *,
        cache_warm: bool = True,
        shadow: bool = False,
        verified_stable_tokens: int = 0,
    ) -> ProjectionResult:
        """Project only historical messages; unchanged system/tools are excluded.

        ``verified_stable_tokens`` is reserved for an adapter that has verified
        an eligible cache breakpoint before the earliest edit. Identical bytes
        alone do not establish a provider cache hit; the conservative default
        gives no credit for an unchanged message prefix.
        """
        baseline = deepcopy(messages)
        try:
            validate_messages(baseline)
            sessions = {event.session_id for event in events} | {
                chunk.session_id for chunk in chunks
            }
            if len(sessions) > 1:
                raise ValueError("cannot project events or chunks from multiple sessions")
            signals = derive_signals(events, chunks)
            pins = tuple(
                sorted(
                    set(state.pinned_ids)
                    | recalled_chunk_ids(events)
                    | {chunk.id for chunk in chunks if chunk.pinned}
                )
            )
            state = state.model_copy(update={"pinned_ids": pins})
            current = render_view(messages, chunks, state.levels)
            boundary = test_pass_boundary(events)
            if boundary is None or boundary == state.last_boundary_id:
                return ProjectionResult(
                    messages=baseline if shadow else current, state=state, signals=signals
                )
            considered = state.model_copy(update={"last_boundary_id": boundary})
            proposed_levels = self._planner(events, chunks, state, config)
            proposed = render_view(messages, chunks, proposed_levels)
            validate_messages(proposed)
            old_bytes, new_bytes = canonical_bytes(current), canonical_bytes(proposed)
            old_tokens, new_tokens = estimate_tokens(old_bytes), estimate_tokens(new_bytes)
            saved = old_tokens - new_tokens
            if saved < config.min_saved_tokens or proposed_levels == state.levels:
                return ProjectionResult(
                    messages=baseline if shadow else current, state=considered, signals=signals
                )
            common_bytes = 0
            for old, new in zip(old_bytes, new_bytes, strict=False):
                if old != new:
                    break
                common_bytes += 1
            if verified_stable_tokens < 0 or verified_stable_tokens > common_bytes // 4:
                raise ValueError("verified stable prefix exceeds unchanged forecast prefix")
            newly_hidden = sum(
                1
                for chunk_id, level in proposed_levels.items()
                if level != "full" and state.levels.get(chunk_id, "full") == "full"
            )
            decision = decide_rebuild(
                CostInputs(
                    old_tokens=old_tokens,
                    new_tokens=new_tokens,
                    stable_tokens=verified_stable_tokens,
                    remaining_turns=config.remaining_turns,
                    cache_warm=cache_warm,
                    decision_cost_usd=config.policy_cost_usd,
                    expected_recall_cost_usd=Decimal(newly_hidden)
                    * config.recall_probability
                    * config.expected_recall_cost_usd,
                    safety_factor=config.safety_factor,
                ),
                pricing,
            )
            accepted = decision.accepted and not shadow
            next_state = (
                considered.model_copy(update={"levels": proposed_levels, "epoch": state.epoch + 1})
                if accepted
                else considered
            )
            return ProjectionResult(
                messages=proposed if accepted else (baseline if shadow else current),
                state=next_state,
                signals=signals,
                decision=decision,
                rebuilt=accepted,
                estimated_tokens_saved=max(saved, 0),
            )
        except Exception as exc:
            # A fallback starts a fresh baseline epoch; keeping a partially
            # applied projection state would misrepresent subsequent requests.
            reset = ViewState(epoch=state.epoch + 1, pinned_ids=state.pinned_ids)
            return ProjectionResult(
                messages=baseline, state=reset, fallback=True, error=f"{type(exc).__name__}: {exc}"
            )
