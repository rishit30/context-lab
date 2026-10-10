"""Lossless duplicate retirement must retain a proved full copy, including after restart."""

import hashlib
from copy import deepcopy
from decimal import Decimal

import pytest

from contextlab.core.controller import ContextController
from contextlab.core.models import Chunk, CompactionConfig, Event, Pricing, ViewState
from contextlab.core.semantic import SemanticAssessment
from contextlab.core.signals import TRUSTED_BOUNDARY_SOURCE
from contextlab.core.validator import validate_messages


def read(seq, *, content=None, path="a.json", version=None, **metadata):
    text = content if content is not None else "exact original line\n" * 1600
    fields = {
        "path": path,
        "version": version or hashlib.sha256(text.encode()).hexdigest(),
        "start_line": 1,
        "end_line": 1600,
        "full_read": True,
        **metadata,
    }
    return Event(
        id=f"e{seq}",
        session_id="s",
        seq=seq,
        kind="file_read",
        tool_name="read_file",
        tool_call_id=f"t{seq}",
        content=text,
        metadata=fields,
    )


def fixture(*reads):
    if not reads:
        reads = (read(1), read(2))
    events = list(reads)
    chunks = [
        Chunk(
            id=f"c{e.seq}",
            session_id=e.session_id,
            event_id=e.id,
            seq=e.seq,
            kind=e.kind,
            tool_call_id=e.tool_call_id,
            content=e.content,
            metadata=deepcopy(e.metadata),
        )
        for e in reads
    ]
    messages = [{"role": "user", "content": "Complete the next task using this history."}]
    for e in reads:
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "opaque", "signature": "preserve"},
                        {
                            "type": "tool_use",
                            "id": e.tool_call_id,
                            "name": "read_file",
                            "input": {"path": e.metadata["path"]},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": e.tool_call_id, "content": e.content}
                    ],
                },
            ]
        )
    events.append(
        Event(
            id="boundary",
            session_id="s",
            seq=max(e.seq for e in reads) + 1,
            kind="phase_boundary",
            role="system",
            metadata={"source": TRUSTED_BOUNDARY_SOURCE, "trusted": True},
        )
    )
    return messages, events, chunks


class Gate:
    def __init__(self, allowed=()):
        self.allowed = allowed
        self.calls = []

    def assess(self, *, events, candidates, signals, retained_chunks=()):
        self.calls.append((deepcopy(candidates), deepcopy(retained_chunks)))
        return SemanticAssessment(
            engine="mock", reason="keep information", allowed_ids=self.allowed
        )


def evaluate(data, *, state=None, gate=None, recent=0, shadow=False, policy_cost=Decimal(0)):
    messages, events, chunks = data
    pricing = Pricing(
        model="offline",
        input_per_million=3,
        cache_write_5m_per_million=3,
        cache_write_1h_per_million=3,
        cache_read_per_million=3,
        output_per_million=15,
    )
    config = CompactionConfig(recent_outputs=recent, policy_cost_usd=policy_cost)
    return ContextController(semantic_gate=gate).project(
        messages, events, chunks, state or ViewState(), pricing, config, shadow=shadow
    )


def test_identical_verified_read_retires_without_semantic_call_preserves_protocol():
    data = fixture()
    before = deepcopy(data)
    gate = Gate()
    result = evaluate(data, gate=gate, recent=1)
    assert result.rebuilt and not result.fallback and gate.calls == []
    assert result.state.duplicate_sources == {"c1": "c2"}
    assert result.exact_duplicate_retirements == {"c1": "c2"}
    assert result.state.levels == {"c1": "hidden", "c2": "full"}
    assert (
        "identical original text remains fully visible"
        in result.messages[2]["content"][0]["content"]
    )
    assert "t2" in result.messages[2]["content"][0]["content"]
    assert result.messages[3:] == data[0][3:] and result.messages[:2] == data[0][:2]
    assert any(s.signal_type == "exact_duplicate_read" for s in result.signals)
    validate_messages(result.messages)
    assert data == before


def test_identical_reads_require_a_trusted_boundary():
    data = fixture()
    data[1].pop()
    result = evaluate(data, gate=Gate())
    assert not result.rebuilt and result.no_op_reason == "no_boundary"
    assert result.messages == data[0] and result.state.duplicate_sources == {}


@pytest.mark.parametrize(
    "case",
    [
        "different_content",
        "different_version",
        "missing_version",
        "range",
        "invalid_range",
        "missing_range",
        "metadata_only_in_args",
        "tool_error",
    ],
)
def test_incomplete_or_different_provenance_never_bypasses_semantic_veto(case):
    old, new = read(1), read(2)
    if case == "different_content":
        new = new.model_copy(update={"content": "different line\n" * 1600})
    elif case == "different_version":
        new = new.model_copy(update={"metadata": {**new.metadata, "version": "other-version"}})
    elif case == "missing_version":
        old = old.model_copy(update={"metadata": {**old.metadata, "version": None}})
    elif case == "range":
        new = new.model_copy(update={"metadata": {**new.metadata, "end_line": 1700}})
    elif case == "invalid_range":
        old = old.model_copy(update={"metadata": {**old.metadata, "start_line": True}})
    elif case == "missing_range":
        old = old.model_copy(
            update={"metadata": {k: v for k, v in old.metadata.items() if k != "end_line"}}
        )
    elif case == "metadata_only_in_args":
        old = old.model_copy(
            update={"args": old.metadata, "metadata": {"path": "a.json", "full_read": True}}
        )
    elif case == "tool_error":
        new = new.model_copy(update={"metadata": {**new.metadata, "tool_error": True}})
    data = fixture(old, new)
    gate = Gate()
    result = evaluate(data, gate=gate)
    assert (
        result.messages == data[0] and not result.rebuilt and result.state.duplicate_sources == {}
    )
    assert result.exact_duplicate_retirements == {}


def test_same_text_different_file_is_not_a_duplicate():
    data = fixture(read(1), read(2, path="other.json"))
    result = evaluate(data, gate=Gate())
    assert not result.rebuilt and result.messages == data[0]


def test_multiple_duplicates_point_directly_to_latest_full_copy():
    result = evaluate(fixture(read(1), read(2), read(3)), gate=Gate())
    assert result.rebuilt and result.state.duplicate_sources == {"c1": "c3", "c2": "c3"}
    assert result.state.levels["c3"] == "full"
    assert not set(result.state.duplicate_sources).intersection(
        result.state.duplicate_sources.values()
    )


def test_keep_exact_anchor_even_when_newer_different_version_supersedes_it():
    gate = Gate(("c2",))
    data = fixture(read(1), read(2), read(3, content="new version\n" * 1600))
    result = evaluate(data, gate=gate)
    assert result.rebuilt and result.state.duplicate_sources == {"c1": "c2"}
    assert result.state.levels["c2"] == "full" and result.state.levels.get("c3", "full") == "full"
    assert gate.calls == []  # c2 is a required anchor, not a semantic candidate.


def test_persisted_anchor_cannot_be_retired_on_a_later_boundary():
    first = evaluate(fixture(), gate=Gate())
    state = ViewState.model_validate_json(first.state.model_dump_json())
    data = fixture(read(1), read(2), read(3, content="changed snapshot\n" * 1600))
    data[1][-1] = data[1][-1].model_copy(update={"id": "next-boundary"})
    gate = Gate(("c2",))
    result = evaluate(data, state=state, gate=gate)
    assert result.state.duplicate_sources == {"c1": "c2"}
    assert result.state.levels["c2"] == "full" and not result.rebuilt and gate.calls == []
    assert result.messages[4]["content"][0]["content"] == data[0][4]["content"][0]["content"]


@pytest.mark.parametrize("case", ["missing", "modified", "hidden", "pin", "recent"])
def test_persisted_duplicate_restores_before_reused_boundary_if_proof_or_guard_changes(case):
    data = fixture()
    first = evaluate(data, gate=Gate())
    state = first.state
    recent = 0
    if case == "missing":
        data = (data[0][:-2], data[1], data[2])
    elif case == "modified":
        data = fixture(read(1), read(2, content="changed\n" * 1600))
    elif case == "hidden":
        state = state.model_copy(update={"levels": {**state.levels, "c2": "hidden"}})
    elif case == "pin":
        state = state.model_copy(update={"pinned_ids": ("c1",)})
    else:
        recent = 2
    result = evaluate(data, state=state, gate=Gate(), recent=recent)
    assert result.state.levels["c1"] == "full" and result.state.duplicate_sources == {}
    assert result.messages[2]["content"][0]["content"] == data[2][0].content
    assert not result.fallback


@pytest.mark.parametrize("protected", ["pin", "recent"])
def test_guarded_older_copy_is_not_hidden(protected):
    state = ViewState(pinned_ids=("c1",)) if protected == "pin" else ViewState()
    result = evaluate(fixture(), state=state, gate=Gate(), recent=2 if protected == "recent" else 0)
    assert not result.rebuilt and result.state.levels.get("c1", "full") == "full"


def test_semantic_veto_leaves_exact_duplicates_eligible_and_only_full_support_is_sent():
    data = fixture(
        read(1),
        read(2, path="b.json", content="old b\n" * 1600),
        read(3),
        read(4, path="b.json", content="new b\n" * 1600),
    )
    gate = Gate()
    result = evaluate(data, gate=gate)
    assert result.rebuilt and result.state.duplicate_sources == {"c1": "c3"}
    assert [c.id for c in gate.calls[0][0]] == ["c2"]
    assert {c.id for c in gate.calls[0][1]} == {"c3", "c4"}
    assert result.state.levels.get("c2", "full") == "full"
    assert result.semantic_decision is not None and result.exact_duplicate_retirements == {
        "c1": "c3"
    }


@pytest.mark.parametrize("case", ["shadow", "cost"])
def test_duplicate_plan_is_not_persisted_when_not_admitted_by_policy(case):
    data = fixture()
    gate = Gate()
    result = evaluate(
        data,
        gate=gate,
        shadow=case == "shadow",
        policy_cost=Decimal(100) if case == "cost" else Decimal(0),
    )
    assert not result.rebuilt and result.messages == data[0]
    assert result.state.duplicate_sources == {} and result.exact_duplicate_retirements == {}
    assert gate.calls == []


def test_existing_partial_copy_cannot_justify_bypassing_semantic_veto():
    gate = Gate()
    data = fixture()
    result = evaluate(data, state=ViewState(levels={"c2": "short"}), gate=gate)
    assert not result.rebuilt and result.state.duplicate_sources == {}
    assert [chunk.id for chunk in gate.calls[0][0]] == ["c1"]
    assert gate.calls[0][1] == ()
    assert result.messages[2]["content"][0]["content"] == data[2][0].content


def test_failed_semantic_plugin_cannot_partially_commit_mixed_duplicate_plan():
    data = fixture(
        read(1),
        read(2, path="b.json", content="old b\n" * 1600),
        read(3),
        read(4, path="b.json", content="new b\n" * 1600),
    )

    class FailingGate(Gate):
        def assess(self, **kwargs):
            raise RuntimeError("private provider payload")

    result = evaluate(data, gate=FailingGate())
    assert result.fallback and result.error == "SemanticGateError"
    assert result.messages == data[0] and result.state.duplicate_sources == {}
    assert result.state.last_boundary_id == "boundary"
    assert result.exact_duplicate_retirements == {}
