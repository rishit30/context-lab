"""Semantic decisions can veto reductions but cannot override lifecycle guards."""

from copy import deepcopy
from decimal import Decimal

import pytest
from pydantic import ValidationError

from contextlab.core.controller import ContextController
from contextlab.core.models import Chunk, CompactionConfig, Event, Pricing, ViewState
from contextlab.core.semantic import SemanticAssessment
from contextlab.core.signals import TRUSTED_BOUNDARY_SOURCE
from contextlab.core.view_builder import render_view


def fixture():
    events = [
        Event(id=f"e{i}", session_id="s", seq=i, kind="file_read", tool_call_id=f"t{i}",
              content=(f"snapshot {i}: historical evidence\n" * 800),
              metadata={"path": "a.json" if i % 2 else "b.json", "full_read": True})
        for i in range(1, 5)
    ]
    chunks = [Chunk(id=f"c{e.seq}", session_id="s", seq=e.seq, event_id=e.id,
                    tool_call_id=e.tool_call_id, kind=e.kind, content=e.content) for e in events]
    messages = [{"role": "user", "content": "Compare the original snapshots with today."}]
    for event in events:
        messages.extend([
            {"role": "assistant", "content": [{"type": "tool_use", "id": event.tool_call_id,
                                                  "name": "read_file", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": event.tool_call_id,
                                             "content": event.content}]},
        ])
    events.append(Event(id="question", session_id="s", seq=5, kind="message", role="user",
                        content="Compare the original snapshots with today."))
    events.append(Event(id="boundary", session_id="s", seq=6, kind="phase_boundary", role="system",
                        metadata={"source": TRUSTED_BOUNDARY_SOURCE, "trusted": True}))
    return messages, events, chunks


def prices():
    return Pricing(model="offline", input_per_million=3, cache_write_5m_per_million=3,
                   cache_write_1h_per_million=3, cache_read_per_million=3, output_per_million=15)


def config(**kwargs):
    return CompactionConfig(**{"recent_outputs": 0, "remaining_turns": 8,
                               "min_saved_tokens": 128, **kwargs})


class Gate:
    def __init__(self, assessment=None, error=None):
        self.assessment = assessment or SemanticAssessment(engine="mock", reason="uncertain")
        self.error = error
        self.calls = []

    def assess(self, *, events, candidates, signals, retained_chunks=()):
        self.calls.append((events, candidates, signals, retained_chunks))
        if self.error:
            raise self.error
        return self.assessment


def evaluate(gate, *, state=None, settings=None, data=None, shadow=False):
    messages, events, chunks = data or fixture()
    return ContextController(semantic_gate=gate).project(
        messages, events, chunks, state or ViewState(), prices(), settings or config(), shadow=shadow
    )


def test_historical_question_and_uncertainty_veto_preserve_current_view_once():
    gate = Gate(SemanticAssessment(engine="mock", reason="historical snapshots needed",
                                   needed_probabilities={"c1": 0.98, "c2": 0.5}))
    messages, events, chunks = fixture()
    result = evaluate(gate, data=(messages, events, chunks))
    assert result.messages == messages and result.state.levels == {}
    assert result.no_op_reason == "semantic_veto" and not result.rebuilt
    assert result.semantic_decision["needed_probabilities"] == {"c1": 0.98, "c2": 0.5}
    assert gate.calls[0][0][-2].content == "Compare the original snapshots with today."
    assert {chunk.id for chunk in gate.calls[0][1]} == {"c1", "c2"}
    again = evaluate(gate, state=result.state, data=(messages, events, chunks))
    assert again.no_op_reason == "boundary_already_considered"
    assert len(gate.calls) == 1


@pytest.mark.parametrize("case", ["no_boundary", "untrusted_boundary", "no_reductions", "too_small"])
def test_semantic_gate_only_runs_at_worthwhile_eligible_boundary(case):
    messages, events, chunks = fixture()
    settings = config()
    if case == "no_boundary":
        events.pop()
    elif case == "untrusted_boundary":
        events[-1] = events[-1].model_copy(update={"role": "user"})
    elif case == "no_reductions":
        events[2] = events[2].model_copy(update={"metadata": {"path": "other-a", "full_read": True}})
        events[3] = events[3].model_copy(update={"metadata": {"path": "other-b", "full_read": True}})
    else:
        settings = config(min_saved_tokens=1_000_000)
    gate = Gate()
    result = evaluate(gate, data=(messages, events, chunks), settings=settings)
    assert gate.calls == [] and result.messages == messages
    assert not result.rebuilt and result.semantic_decision is None


def test_allow_list_cannot_hide_ineligible_or_pinned_or_recent_chunks():
    messages, events, chunks = fixture()
    # c1 has a real supersession signal but is pinned. c2 is eligible; c3/c4
    # have no superseding read and are recent. Extra IDs are not instructions.
    gate = Gate(SemanticAssessment(allowed_ids=("c1", "c2", "c3", "c4", "nonexistent"),
                                   engine="mock", reason="attempt broad allow-list"))
    result = evaluate(gate, state=ViewState(pinned_ids=("c1",)), settings=config(recent_outputs=2),
                      data=(messages, events, chunks))
    assert result.rebuilt
    assert [chunk.id for chunk in gate.calls[0][1]] == ["c2"]
    assert result.state.levels == {"c1": "full", "c2": "hidden", "c3": "full", "c4": "full"}
    assert result.messages[2] == messages[2]
    assert result.messages[6:] == messages[6:]


def test_recent_eligible_chunk_is_not_sent_to_gate():
    gate = Gate(SemanticAssessment(allowed_ids=("c1", "c2"), engine="mock", reason="allow"))
    result = evaluate(gate, settings=config(recent_outputs=3))
    assert result.rebuilt
    assert [chunk.id for chunk in gate.calls[0][1]] == ["c1"]
    assert result.state.levels["c2"] == "full"


def test_veto_restores_prior_partial_level_and_allows_other_reduction():
    state = ViewState(epoch=2, levels={"c1": "long"})
    gate = Gate(SemanticAssessment(allowed_ids=("c2",), engine="mock", reason="retain excerpt"))
    result = evaluate(gate, state=state)
    assert result.rebuilt and result.state.epoch == 3
    assert result.state.levels == {"c1": "long", "c2": "hidden"}
    assert [chunk.id for chunk in gate.calls[0][1]] == ["c1", "c2"]
    messages, _, chunks = fixture()
    assert result.messages[2] == render_view(messages, chunks, state.levels)[2]
    assert state.levels == {"c1": "long"}


def test_savings_recomputed_after_partial_veto_before_cost_decision():
    gate = Gate(SemanticAssessment(allowed_ids=("c1",), engine="mock", reason="one needed"))
    result = evaluate(gate, settings=config(min_saved_tokens=8000))
    assert len(gate.calls) == 1
    assert 0 < result.estimated_tokens_saved < 8000
    assert result.no_op_reason == "insufficient_token_savings" and result.decision is None
    assert result.state.levels == {} and not result.rebuilt


def test_actual_policy_cost_and_each_risk_floor_are_in_cost_gate():
    baseline = evaluate(None)
    gate = Gate(SemanticAssessment(allowed_ids=("c1", "c2"), engine="mock", reason="allow",
                                   cost_usd=Decimal("0.005"),
                                   needed_probabilities={"c1": 0.9, "c2": 0.01}))
    result = evaluate(gate)
    assert result.rebuilt and baseline.rebuilt
    assert result.decision.stay_cost_usd == baseline.decision.stay_cost_usd
    # c1 raises recall risk .1→.9; c2 stays at the .1 prior. Actual gate cost
    # is counted in addition to the fixed configured policy cost.
    assert result.decision.move_cost_usd - baseline.decision.move_cost_usd == Decimal("0.013")
    gate.assessment = gate.assessment.model_copy(update={"cost_usd": Decimal("10")})
    expensive = evaluate(gate)
    assert expensive.no_op_reason == "cost_gate_rejected" and not expensive.rebuilt
    assert expensive.semantic_decision["cost_usd"] == "10"


def test_absent_needed_probabilities_keep_prior_and_fixed_policy_cost():
    settings = config(policy_cost_usd=Decimal("0.0007"))
    baseline = evaluate(None, settings=settings)
    gate = Gate(SemanticAssessment(allowed_ids=("c1", "c2"), engine="mock", reason="allow",
                                   cost_usd=Decimal("0.0003")))
    result = evaluate(gate, settings=settings)
    assert result.decision.move_cost_usd - baseline.decision.move_cost_usd == Decimal("0.0003")


@pytest.mark.parametrize("failure", ["exception", "wrong_type", "invalid_probability", "fallback"])
def test_gate_failure_preserves_current_epoch_marks_boundary_and_sanitizes(failure):
    state = ViewState(epoch=3, levels={"c1": "long"})
    gate = Gate()
    if failure == "exception":
        gate.error = RuntimeError("sensitive raw provider payload must not escape")
    elif failure == "wrong_type":
        gate.assessment = {"allowed_ids": ["c1", "c2"]}
    elif failure == "invalid_probability":
        gate.assessment = gate.assessment.model_copy(update={"needed_probabilities": {"c1": float("nan")}})
    else:
        gate.assessment = gate.assessment.model_copy(update={"fallback": True})
    messages, events, chunks = fixture()
    result = evaluate(gate, state=state, data=(messages, events, chunks))
    assert result.fallback and not result.rebuilt
    assert result.messages == render_view(messages, chunks, state.levels)
    assert result.state.levels == state.levels and result.state.epoch == state.epoch
    assert result.state.last_boundary_id == "boundary"
    assert result.no_op_reason == "semantic_gate_failed" and result.error == "SemanticGateError"
    assert "sensitive" not in result.model_dump_json()
    evaluate(gate, state=result.state, data=(messages, events, chunks))
    assert len(gate.calls) == 1


def test_callback_cannot_mutate_canonical_event_or_chunk_metadata():
    messages, events, chunks = fixture()
    before = deepcopy((messages, events, chunks))

    class MutatingGate(Gate):
        def assess(self, *, events, candidates, signals, retained_chunks=()):
            events[0].metadata["path"] = "tampered"
            candidates[0].metadata["tampered"] = True
            return self.assessment

    evaluate(MutatingGate(), data=(messages, events, chunks))
    assert (messages, events, chunks) == before


def test_shadow_records_semantic_decision_without_applying_reduction():
    gate = Gate(SemanticAssessment(allowed_ids=("c1", "c2"), engine="mock", reason="allow"))
    result = evaluate(gate, shadow=True)
    assert result.messages == fixture()[0] and result.state.levels == {}
    assert result.decision.accepted and not result.rebuilt
    assert result.no_op_reason == "shadow_only" and result.semantic_decision["engine"] == "mock"


@pytest.mark.parametrize("field,value", [("cost_usd", Decimal("-1")), ("cost_usd", Decimal("Infinity")),
                                         ("needed_probabilities", {"c1": -0.1}),
                                         ("needed_probabilities", {"c1": 1.1}),
                                         ("needed_probabilities", {"c1": float("nan")}),
                                         ("needed_probabilities", {"c1": True})])
def test_assessment_rejects_invalid_costs_and_probabilities(field, value):
    with pytest.raises(ValidationError):
        SemanticAssessment(engine="mock", reason="invalid", **{field: value})
