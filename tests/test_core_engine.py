from copy import deepcopy
from decimal import Decimal

import pytest

from contextlab.core.controller import ContextController
from contextlab.core.cost_model import decide_rebuild
from contextlab.core.models import Chunk, CompactionConfig, CostInputs, Event, Pricing, ViewState
from contextlab.core.planner import plan_visibility
from contextlab.core.signals import derive_signals
from contextlab.core.signals import test_pass_boundary as find_boundary
from contextlab.core.validator import MessageValidationError, validate_messages
from contextlab.core.view_builder import canonical_bytes, render_view


def prices():
    return Pricing(
        model="synthetic",
        input_per_million=3,
        cache_write_5m_per_million=3.75,
        cache_write_1h_per_million=6,
        cache_read_per_million=0.3,
        output_per_million=15,
    )


def event(seq, kind, **metadata):
    return Event(
        id=f"e{seq}",
        session_id="s1",
        seq=seq,
        kind=kind,
        tool_call_id=f"t{seq}",
        content=(f"original {seq}\n" * 1000),
        metadata=metadata,
    )


def chunk(source, **fields):
    return Chunk(
        id=f"c{source.seq}",
        session_id=source.session_id,
        event_id=source.id,
        seq=source.seq,
        tool_call_id=source.tool_call_id,
        kind=source.kind,
        content=source.content,
        **fields,
    )


def messages_for(events):
    messages = [{"role": "user", "content": "Fix the bug and retain historical evidence."}]
    for source in events:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "opaque", "signature": "do-not-rewrite"},
                        {
                            "type": "tool_use",
                            "id": source.tool_call_id,
                            "name": source.kind,
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": source.tool_call_id,
                            "content": source.content,
                            "cache_control": {"type": "ephemeral"},
                            "is_error": False,
                        }
                    ],
                },
            ]
        )
    return messages


def scenario():
    events = [
        event(1, "file_read", path="a.py", full_read=True),
        event(2, "test_run", command="pytest a", environment="local", exit_code=1),
        event(3, "file_edit", path="a.py"),
        event(4, "file_read", path="a.py", full_read=True),
        event(5, "test_run", command="pytest a", environment="local", exit_code=0),
    ]
    return events, [chunk(source) for source in events], messages_for(events)


def test_superseded_read_requires_actual_covering_read_not_just_edit():
    old = event(1, "file_read", path="a.py", full_read=True)
    edit = event(2, "file_edit", path="a.py")
    partial = event(3, "file_read", path="a.py", start_line=1, end_line=10)
    full = event(4, "file_read", path="a.py", full_read=True)
    assert derive_signals([old, edit], [chunk(old)]) == []
    assert derive_signals([old, edit, partial], [chunk(old)]) == []
    signals = derive_signals([old, edit, partial, full], [chunk(old)])
    assert [(signal.signal_type, signal.triggered_by_event_id) for signal in signals] == [
        ("superseded_read", "e4")
    ]


@pytest.mark.parametrize(
    "new_range, expected", [((1, 20), True), ((5, 15), True), ((6, 15), False), ((1, 14), False)]
)
def test_partial_read_requires_range_coverage(new_range, expected):
    old = event(1, "file_read", path="a.py", start_line=5, end_line=15)
    new = event(2, "file_read", path="a.py", start_line=new_range[0], end_line=new_range[1])
    assert bool(derive_signals([old, new], [chunk(old)])) is expected


def test_unknown_read_range_is_not_guessed():
    old, new = event(1, "file_read", path="a.py"), event(2, "file_read", path="a.py")
    assert derive_signals([old, new], [chunk(old)]) == []


@pytest.mark.parametrize(
    "changes",
    [{"command": "pytest b"}, {"environment": "other"}, {"cwd": "other"}, {"exit_code": 1}],
)
def test_unrelated_or_failing_test_is_not_resolution(changes):
    metadata = {"command": "pytest a", "environment": "local", "cwd": ".", "exit_code": 0}
    old = event(1, "test_run", **(metadata | {"exit_code": 1}))
    new = event(2, "test_run", **(metadata | changes))
    assert derive_signals([old, new], [chunk(old)]) == []
    assert find_boundary([old, new]) is None


def test_smaller_test_enumeration_does_not_resolve_old_failure():
    old = event(1, "test_run", command="pytest a", exit_code=1, test_ids=["one", "two"])
    new = event(2, "test_run", command="pytest a", exit_code=0, test_ids=["one"])
    assert derive_signals([old, new], [chunk(old)]) == []


def test_matching_pass_is_a_boundary_and_retains_evidence():
    events, chunks, _ = scenario()
    signals = derive_signals(events, chunks)
    assert find_boundary(events) == "e5"
    assert [(signal.chunk_id, signal.target) for signal in signals] == [
        ("c1", "hidden"),
        ("c2", "short"),
    ]


def test_repeated_passing_test_does_not_create_repeated_boundary():
    events, _, _ = scenario()
    events.append(event(6, "test_run", command="pytest a", environment="local", exit_code=0))
    assert find_boundary(events) == "e5"


@pytest.mark.parametrize(
    "warm, old, new, turns, overhead, accepted",
    [
        (True, 10000, 1000, 8, "0", True),
        (True, 10000, 9000, 1, "0", False),
        (False, 10000, 1000, 1, "0", True),
        (False, 10000, 9000, 1, "1", False),
        (False, 1000, 1100, 8, "0", False),
    ],
)
def test_warm_and_cold_gate_charges_overhead(warm, old, new, turns, overhead, accepted):
    decision = decide_rebuild(
        CostInputs(
            old_tokens=old,
            new_tokens=new,
            remaining_turns=turns,
            cache_warm=warm,
            expected_recall_cost_usd=Decimal(overhead),
        ),
        prices(),
    )
    assert decision.accepted is accepted


def test_cost_preserves_warm_stable_prefix_and_counts_both_overheads():
    inputs = CostInputs(
        old_tokens=10000,
        new_tokens=5000,
        stable_tokens=4000,
        remaining_turns=3,
        decision_cost_usd=Decimal(".002"),
        expected_recall_cost_usd=Decimal(".003"),
    )
    result = decide_rebuild(inputs, prices())
    assert result.stay_cost_usd == Decimal(".009")
    assert result.move_cost_usd == Decimal(".01295")
    assert not result.accepted


def test_invalid_stable_prefix_fails_loudly_in_pure_cost_function():
    with pytest.raises(ValueError, match="stable_tokens"):
        decide_rebuild(
            CostInputs(old_tokens=10, new_tokens=5, stable_tokens=6, remaining_turns=2), prices()
        )


def test_render_changes_only_tool_result_content_and_is_deterministic():
    events, chunks, original = scenario()
    before = deepcopy(original)
    view = render_view(original, chunks, {"c1": "hidden"})
    assert original == before
    assert canonical_bytes(view) == canonical_bytes(render_view(original, chunks, {"c1": "hidden"}))
    assert view[2]["content"][0]["content"] != original[2]["content"][0]["content"]
    view[2]["content"][0]["content"] = original[2]["content"][0]["content"]
    assert view == original


def test_multimodal_tool_result_is_not_destroyed():
    _, chunks, original = scenario()
    original[2]["content"][0]["content"] = [
        {"type": "image", "source": {"type": "base64", "data": "opaque"}}
    ]
    with pytest.raises(ValueError, match="plain-text"):
        render_view(original, chunks, {"c1": "hidden"})


def test_missing_or_duplicate_mapping_cannot_silently_render():
    _, chunks, original = scenario()
    with pytest.raises(ValueError, match="matching result"):
        render_view(original, chunks, {"absent": "hidden"})
    with pytest.raises(ValueError, match="ambiguous"):
        render_view(original, [chunks[0], chunks[0]], {"c1": "hidden"})


def test_archive_mismatch_does_not_replace_original():
    _, chunks, original = scenario()
    chunks[0] = chunks[0].model_copy(update={"content": "wrong historical data"})
    with pytest.raises(ValueError, match="archive does not match"):
        render_view(original, chunks, {"c1": "hidden"})


def test_validator_accepts_multiple_calls_then_results():
    original = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": name, "name": "read", "input": {}} for name in ["a", "b"]
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": name, "content": "ok"} for name in ["b", "a"]
            ],
        },
    ]
    validate_messages(original)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_role",
        "missing",
        "delayed",
        "duplicate_result",
        "text_before_result",
        "unknown_id",
        "duplicate_call",
    ],
)
def test_validator_rejects_invalid_protocol(mutation):
    messages = messages_for([event(1, "file_read")])
    if mutation == "wrong_role":
        messages[2]["role"] = "assistant"
    elif mutation == "missing":
        messages.pop()
    elif mutation == "delayed":
        messages.insert(2, {"role": "user", "content": "wait"})
    elif mutation == "duplicate_result":
        messages[2]["content"].append(deepcopy(messages[2]["content"][0]))
    elif mutation == "text_before_result":
        messages[2]["content"].insert(0, {"type": "text", "text": "wait"})
    elif mutation == "unknown_id":
        messages[2]["content"][0]["tool_use_id"] = "absent"
    elif mutation == "duplicate_call":
        messages[1]["content"].append(deepcopy(messages[1]["content"][-1]))
    with pytest.raises(MessageValidationError):
        validate_messages(messages)


def test_controller_active_shadow_and_stable_epoch():
    events, chunks, messages = scenario()
    config = CompactionConfig(recent_outputs=0, expected_recall_cost_usd=0, remaining_turns=50)
    controller = ContextController()
    result = controller.project(messages, events, chunks, ViewState(), prices(), config)
    assert result.rebuilt and not result.fallback and result.decision.accepted
    assert result.messages != messages and result.state.epoch == 1
    again = controller.project(messages, events, chunks, result.state, prices(), config)
    assert again.messages == result.messages and not again.rebuilt
    shadow = controller.project(
        messages, events, chunks, ViewState(), prices(), config, shadow=True
    )
    assert shadow.messages == messages and not shadow.rebuilt
    assert shadow.decision.accepted and shadow.state.levels == {}


def test_controller_never_rebuilds_without_matching_boundary():
    events, chunks, messages = scenario()
    result = ContextController().project(
        messages[:-2], events[:-1], chunks[:-1], ViewState(), prices(), CompactionConfig()
    )
    assert not result.rebuilt and result.decision is None


def test_controller_does_not_infer_eligible_cache_prefix_from_identical_bytes():
    events, chunks, messages = scenario()
    config = CompactionConfig(recent_outputs=0, expected_recall_cost_usd=0, remaining_turns=50)
    result = ContextController().project(messages, events, chunks, ViewState(), prices(), config)
    from contextlab.core.view_builder import estimate_tokens

    old_tokens = estimate_tokens(canonical_bytes(messages))
    new_tokens = estimate_tokens(canonical_bytes(result.messages))
    expected = decide_rebuild(
        CostInputs(
            old_tokens=old_tokens, new_tokens=new_tokens, stable_tokens=0, remaining_turns=50
        ),
        prices(),
    )
    assert result.decision == expected


def test_recall_and_explicit_pins_protect_original_chunks():
    events, chunks, _ = scenario()
    events.append(event(6, "recall", chunk_id="c1"))
    levels = plan_visibility(
        events, chunks, ViewState(pinned_ids=("c2",)), CompactionConfig(recent_outputs=0)
    )
    assert levels["c1"] == "full" and levels["c2"] == "full"


def test_controller_falls_back_on_planner_exception_without_mutation():
    events, chunks, messages = scenario()
    before = deepcopy(messages)

    def broken(*args):
        raise RuntimeError("injected planner failure")

    result = ContextController(planner=broken).project(
        messages, events, chunks, ViewState(), prices(), CompactionConfig()
    )
    assert result.fallback and result.messages == before and messages == before
    assert result.error == "RuntimeError: injected planner failure"
    assert result.state.levels == {}


def test_controller_rejects_cross_session_projection():
    events, chunks, messages = scenario()
    chunks[0] = chunks[0].model_copy(update={"session_id": "other"})
    result = ContextController().project(
        messages, events, chunks, ViewState(), prices(), CompactionConfig()
    )
    assert result.fallback and "multiple sessions" in result.error
