"""Safe orchestration: failures skip intervention and return original history."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from decimal import Decimal

from contextlab.core.cost_model import decide_rebuild
from contextlab.core.duplicates import (
    exact_read_copies,
    full_result_chunks,
    reserve_copy_anchors,
    restore_copy_invariants,
)
from contextlab.core.models import (
    Chunk,
    CompactionConfig,
    CostInputs,
    Event,
    Level,
    Message,
    Pricing,
    ProjectionResult,
    Signal,
    ViewState,
)
from contextlab.core.planner import plan_visibility
from contextlab.core.semantic import SemanticAssessment, SemanticGate
from contextlab.core.signals import derive_signals, recalled_chunk_ids, workflow_boundary
from contextlab.core.validator import validate_messages
from contextlab.core.view_builder import canonical_bytes, estimate_tokens, render_view

Planner = Callable[
    [Sequence[Event], Sequence[Chunk], ViewState, CompactionConfig], dict[str, Level]
]
_LEVEL_RANK: dict[Level, int] = {"full": 3, "long": 2, "short": 1, "hidden": 0}


class ContextController:
    def __init__(
        self, planner: Planner | None = None, semantic_gate: SemanticGate | None = None
    ) -> None:
        self._planner = planner or plan_visibility
        self._semantic_gate = semantic_gate

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
            protected = set(pins)
            if config.recent_outputs:
                protected.update(chunk.id for chunk in sorted(chunks, key=lambda chunk: chunk.seq)[-config.recent_outputs:])
            copies = exact_read_copies(events, chunks, messages)
            state = restore_copy_invariants(state, copies, protected)
            current = render_view(messages, chunks, state.levels, state.duplicate_sources)
            boundary = workflow_boundary(events)
            if boundary is None or boundary == state.last_boundary_id:
                return ProjectionResult(
                    messages=baseline if shadow else current, state=state, signals=signals,
                    no_op_reason="no_boundary" if boundary is None else "boundary_already_considered",
                )
            considered = state.model_copy(update={"last_boundary_id": boundary})
            proposed_levels = dict(self._planner(events, chunks, state, config))
            exact_reductions = reserve_copy_anchors(proposed_levels, state, chunks, copies, protected)
            duplicate_sources = {original: retained for original, retained in
                                 {**state.duplicate_sources, **exact_reductions}.items()
                                 if proposed_levels.get(original, "full") == "hidden"}
            chunk_by_id = {chunk.id: chunk for chunk in chunks}
            signals.extend(Signal(
                chunk_id=original, signal_type="exact_duplicate_read",
                triggered_by_event_id=chunk_by_id[retained].event_id,
                reason=f"Byte-identical read with equal path/version/range remains fully visible in {retained}.",
            ) for original, retained in exact_reductions.items())
            proposed = render_view(messages, chunks, proposed_levels, duplicate_sources)
            validate_messages(proposed)
            old_bytes, new_bytes = canonical_bytes(current), canonical_bytes(proposed)
            old_tokens, new_tokens = estimate_tokens(old_bytes), estimate_tokens(new_bytes)
            saved = old_tokens - new_tokens
            if old_bytes == new_bytes or proposed_levels == state.levels:
                return ProjectionResult(
                    messages=baseline if shadow else current, state=considered, signals=signals,
                    no_op_reason="no_eligible_retirements",
                )
            if saved < config.min_saved_tokens:
                return ProjectionResult(
                    messages=baseline if shadow else current, state=considered, signals=signals,
                    no_op_reason="insufficient_token_savings",
                )
            reductions = [
                chunk
                for chunk in chunks
                if _LEVEL_RANK[proposed_levels.get(chunk.id, "full")]
                < _LEVEL_RANK[state.levels.get(chunk.id, "full")]
            ]
            semantic_candidates = [chunk for chunk in reductions if chunk.id not in exact_reductions]
            assessment: SemanticAssessment | None = None
            if self._semantic_gate is not None and semantic_candidates:
                try:
                    # Record fields are frozen, but nested metadata is mutable.
                    # A plugin receives isolated evidence and cannot rewrite
                    # the canonical archive or change the planner's inputs.
                    raw_assessment = self._semantic_gate.assess(
                        events=deepcopy(tuple(events)),
                        candidates=deepcopy(tuple(semantic_candidates)),
                        signals=deepcopy(tuple(signals)),
                        retained_chunks=deepcopy(full_result_chunks(messages, chunks, proposed_levels)),
                    )
                    if not isinstance(raw_assessment, SemanticAssessment):
                        raise TypeError("semantic gate returned an invalid assessment")
                    # model_copy/model_construct bypass Pydantic validation;
                    # validate a fresh mapping before trusting plugin output.
                    assessment = SemanticAssessment.model_validate(
                        raw_assessment.model_dump(), strict=True
                    )
                except Exception:
                    return ProjectionResult(
                        messages=baseline if shadow else current,
                        state=considered,
                        signals=signals,
                        fallback=True,
                        error="SemanticGateError",
                        no_op_reason="semantic_gate_failed",
                    )
                semantic_decision = assessment.model_dump(mode="json")
                if assessment.fallback:
                    return ProjectionResult(
                        messages=baseline if shadow else current,
                        state=considered,
                        signals=signals,
                        fallback=True,
                        error="SemanticGateError",
                        no_op_reason="semantic_gate_failed",
                        semantic_decision=semantic_decision,
                    )
                allowed = set(assessment.allowed_ids) | set(exact_reductions)
                for chunk in semantic_candidates:
                    if chunk.id not in allowed:
                        if chunk.id in state.levels:
                            proposed_levels[chunk.id] = state.levels[chunk.id]
                        else:
                            proposed_levels.pop(chunk.id, None)
                reductions = [chunk for chunk in reductions if chunk.id in allowed]
                if not reductions:
                    return ProjectionResult(
                        messages=baseline if shadow else current,
                        state=considered,
                        signals=signals,
                        no_op_reason="semantic_veto",
                        semantic_decision=semantic_decision,
                    )
                proposed = render_view(messages, chunks, proposed_levels, duplicate_sources)
                validate_messages(proposed)
                new_bytes = canonical_bytes(proposed)
                new_tokens = estimate_tokens(new_bytes)
                saved = old_tokens - new_tokens
                if saved < config.min_saved_tokens or old_bytes == new_bytes:
                    return ProjectionResult(
                        messages=baseline if shadow else current,
                        state=considered,
                        signals=signals,
                        no_op_reason="insufficient_token_savings",
                        estimated_tokens_saved=max(saved, 0),
                        semantic_decision=semantic_decision,
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
            recall_risk = Decimal(newly_hidden) * config.recall_probability
            policy_cost = config.policy_cost_usd
            if assessment is not None:
                policy_cost += assessment.cost_usd
                recall_risk = sum(
                    (
                        max(
                            config.recall_probability,
                            Decimal(str(assessment.needed_probabilities[chunk.id]))
                            if chunk.id in assessment.needed_probabilities
                            else config.recall_probability,
                        )
                        for chunk in reductions
                    ),
                    Decimal("0"),
                )
            decision = decide_rebuild(
                CostInputs(
                    old_tokens=old_tokens,
                    new_tokens=new_tokens,
                    stable_tokens=verified_stable_tokens,
                    remaining_turns=config.remaining_turns,
                    cache_warm=cache_warm,
                    decision_cost_usd=policy_cost,
                    expected_recall_cost_usd=recall_risk * config.expected_recall_cost_usd,
                    safety_factor=config.safety_factor,
                ),
                pricing,
            )
            accepted = decision.accepted and not shadow
            next_state = (
                considered.model_copy(update={"levels": proposed_levels, "epoch": state.epoch + 1,
                                              "duplicate_sources": duplicate_sources})
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
                no_op_reason=None if accepted else ("shadow_only" if decision.accepted else "cost_gate_rejected"),
                semantic_decision=assessment.model_dump(mode="json") if assessment else None,
                exact_duplicate_retirements=exact_reductions if accepted else {},
            )
        except Exception as exc:
            # A fallback starts a fresh baseline epoch; keeping a partially
            # applied projection state would misrepresent subsequent requests.
            reset = ViewState(epoch=state.epoch + 1, pinned_ids=state.pinned_ids)
            return ProjectionResult(
                messages=baseline, state=reset, fallback=True, error=f"{type(exc).__name__}: {exc}",
                no_op_reason="projection_failed",
            )
