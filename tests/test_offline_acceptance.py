"""Offline acceptance simulations, not model-quality or provider-savings evidence.

The test-only model computes answers from its actual model-visible messages. It
cannot access the workspace, archive, expected answers, or middleware state. The
LangChain graph, file tools, projection, SQLite archive and recall are real.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
from copy import deepcopy

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from contextlab.adapters.langchain_mw import ContextLabMiddleware
from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing, ProjectionResult
from contextlab.core.semantic import SemanticAssessment
from contextlab.ledger.store import LedgerStore
from harness.agent import build_agent
from harness.workspace import Workspace

ROWS = 192


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    """A mistaken provider/telemetry call fails the test instead of spending."""
    def fail(*args, **kwargs):
        raise AssertionError("Offline acceptance attempted network access")

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


def assert_protocol(messages):
    """Independent LC protocol oracle; do not ask the production validator."""
    pending, used = set(), set()
    for message in messages:
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in pending, "unpaired or duplicate result"
            pending.remove(message.tool_call_id)
        else:
            assert not pending, "intervening message before all tool results"
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    assert call["id"] not in used, "reused tool ID"
                    pending.add(call["id"])
                    used.add(call["id"])
                assert not message.invalid_tool_calls
    assert not pending, "request ended with an unresolved tool call"


def visible_documents(messages):
    documents = {}
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            envelope = json.loads(message.content)
            document = json.loads(envelope["content"])
            documents[(document["service"], document["revision"])] = document
        except (ValueError, KeyError, TypeError):
            continue  # Archive pointers and incomplete recall pages are not facts.
    return documents


class EvidenceSimulation(BaseChatModel):
    """Deliberately deterministic fixture consumer, never presented as an LLM."""

    _requests: list = PrivateAttr(default_factory=list)
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "offline-evidence-simulation"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        assert_protocol(messages)
        self._requests.append(deepcopy(messages))
        self._calls += 1
        question = json.loads(next(
            message.content for message in reversed(messages) if isinstance(message, HumanMessage)
        ))
        if isinstance(messages[-1], HumanMessage) and question.get("read_paths"):
            response = AIMessage(content="", tool_calls=[
                {"name": "read_file", "args": {"path": path},
                 "id": f"sim-{self._calls}-read-{index}", "type": "tool_call"}
                for index, path in enumerate(question["read_paths"])
            ])
        else:
            documents = visible_documents(messages)
            requested = [tuple(key) for key in question["facts"]]
            if any(key not in documents for key in requested):
                # Only a supplied pointer grants access; no guessed IDs or archive access.
                pointers = [match for message in messages if isinstance(message, ToolMessage)
                            and isinstance(message.content, str)
                            for match in re.findall(r'recall\("([a-zA-Z0-9_-]+)"', message.content)]
                assert pointers, "required evidence vanished with no recovery pointer"
                response = AIMessage(content="", tool_calls=[{
                    "name": "recall", "args": {"chunk_id": pointers[0]},
                    "id": f"sim-{self._calls}-recall", "type": "tool_call",
                }])
            else:
                answer = []
                for key in requested:
                    document = documents[key]
                    answer.append({
                        "service": document["service"], "revision": document["revision"],
                        "owner": document["owner"], "threshold": document["threshold"],
                        "record_count": len(document["observations"]),
                        "request_total": sum(row["requests"] for row in document["observations"]),
                    })
                response = AIMessage(content=json.dumps(answer),
                                     response_metadata={"stop_reason": "end_turn"})
        return ChatResult(generations=[ChatGeneration(message=response)])


class ConservativeSimulationGate:
    """Test-only veto, not a Jev accuracy claim or a stand-in paid receipt."""

    def __init__(self):
        self.calls = []

    def assess(self, *, events, candidates, signals, retained_chunks=()):
        self.calls.append((deepcopy(candidates), deepcopy(retained_chunks)))
        return SemanticAssessment(
            engine="offline-conservative-simulation", reason="Keep unique historical evidence",
            needed_probabilities={chunk.id: 1.0 for chunk in candidates},
        )


def write_fixture(directory, service, revision=1, owner="Mira", threshold=19):
    document = {
        "service": service, "revision": revision, "owner": owner, "threshold": threshold,
        "observations": [{"shard": index, "requests": index + revision,
                          "annotation": f"fixture-observation-{index}: " + "x" * 50}
                         for index in range(ROWS)],
    }
    (directory / f"{service}.json").write_text(json.dumps(document, indent=2) + "\n")


def expected_fact(service, revision=1, owner="Mira", threshold=19):
    # Independent fixture oracle: arithmetic formula, not the consumer's scan.
    return {"service": service, "revision": revision, "owner": owner, "threshold": threshold,
            "record_count": ROWS, "request_total": ROWS * (ROWS - 1) // 2 + ROWS * revision}


class SimulationRun:
    def __init__(self, directory, *, observe_only=False, gate=None):
        directory.mkdir()
        self.directory = directory
        self.store = LedgerStore(directory / "archive.sqlite")
        self.store.create_session("acceptance", "contextlab", "offline-acceptance",
                                  mode="simulation")
        self.middleware_options = dict(
            controller=ContextController(semantic_gate=gate), session_id="acceptance",
            pricing=Pricing(model="simulation-only", input_per_million=3,
                            cache_write_5m_per_million=3, cache_write_1h_per_million=3,
                            cache_read_per_million=3, output_per_million=15),
            config=CompactionConfig(recent_outputs=1), cache_ttl_seconds=0,
            observe_only=observe_only,
        )
        self.middleware = ContextLabMiddleware(store=self.store, **self.middleware_options)
        self.model = EvidenceSimulation()
        self.workspace = Workspace(directory, recall_reader=self.middleware.recall)
        self.agent = build_agent(
            model=self.model, workspace=self.workspace,
            arm="baseline" if observe_only else "contextlab", context_middleware=self.middleware,
            cache_mode="disabled", max_model_calls=80,
        )
        self.history = []

    def phase(self, label, *, facts, read_paths=(), complete=True):
        self.history.append(HumanMessage(content=json.dumps({
            "phase": label, "facts": facts, "read_paths": list(read_paths),
        })))
        response = self.agent.invoke({"messages": self.history}, {"recursion_limit": 30})
        self.history = response["messages"]
        assert_protocol(self.history)
        answer = json.loads(self.history[-1].content)
        if complete:
            self.middleware.mark_boundary(label)
        return answer

    def verify_archive(self):
        originals = {message.tool_call_id: message.content for message in self.history
                     if isinstance(message, ToolMessage)}
        for chunk in self.store.list_chunks("acceptance"):
            assert chunk.content == originals[chunk.tool_call_id]
        assert self.store.list_model_calls("acceptance") == [], "No fabricated paid usage"
        assert not self.middleware.failures
        assert not any(row["fallback"] for row in self.middleware.telemetry)

    def close(self):
        self.store.close()


def tool_bytes(messages):
    return sum(len(message.content.encode()) for message in messages
               if isinstance(message, ToolMessage) and isinstance(message.content, str))


def test_long_paired_audit_reduces_requests_and_preserves_independent_answers(tmp_path, record_property):
    """Nine phases, 24 real reads and 17 simulated model requests per arm."""
    runs, answers = [], []
    for arm in ("baseline", "contextlab"):
        gate = ConservativeSimulationGate()
        run = SimulationRun(tmp_path / arm, observe_only=arm == "baseline", gate=gate)
        runs.append(run)
        for service in ("gateway", "worker", "queue"):
            write_fixture(run.directory, service)
        facts = [[service, 1] for service in ("gateway", "worker", "queue")]
        expected = [expected_fact(service) for service, _ in facts]
        arm_answers = []
        for phase in range(8):
            answer = run.phase(f"inspection-{phase}", facts=facts,
                               read_paths=[f"{service}.json" for service, _ in facts])
            assert answer == expected
            arm_answers.append(answer)
        arm_answers.append(run.phase("final-handoff", facts=facts))
        assert arm_answers[-1] == expected
        answers.append(arm_answers)
        run.verify_archive()
        assert gate.calls == [], "Proved duplicates must not need a semantic provider"
        assert len(run.model._requests) == 17
    baseline, compacted = runs
    assert answers[0] == answers[1]
    assert len(baseline.store.list_chunks("acceptance")) == 24
    assert len(compacted.store.list_chunks("acceptance")) == 24
    original_bytes = tool_bytes(baseline.model._requests[-1])
    projected_bytes = tool_bytes(compacted.model._requests[-1])
    assert projected_bytes < original_bytes * 0.55
    assert any(row["rebuilt"] for row in compacted.middleware.telemetry)
    assert not any(row["changed"] for row in baseline.middleware.telemetry)
    # Every non-tool field is preserved, even during an intervention.
    canonical = [m for m in compacted.history[:-1] if not isinstance(m, ToolMessage)]
    delivered = [m for m in compacted.model._requests[-1]
                 if not isinstance(m, (ToolMessage, SystemMessage))]
    assert delivered == canonical
    record_property("simulation_original_tool_bytes", original_bytes)
    record_property("simulation_projected_tool_bytes", projected_bytes)
    record_property("simulation_changed_requests", sum(row["changed"]
                                                       for row in compacted.middleware.telemetry))
    for run in runs:
        run.close()


@pytest.mark.parametrize("preserve", [True, False], ids=["semantic-veto", "rules-and-real-recall"])
def test_changed_historical_facts_are_preserved_or_recovered(tmp_path, preserve, record_property):
    gate = ConservativeSimulationGate() if preserve else None
    run = SimulationRun(tmp_path / "history", gate=gate)
    write_fixture(run.directory, "gateway")
    assert run.phase("initial-audit", facts=[["gateway", 1]], read_paths=["gateway.json"]) == [
        expected_fact("gateway")]
    write_fixture(run.directory, "gateway", 2, "Omar", 43)
    assert run.phase("changed-audit", facts=[["gateway", 2]], read_paths=["gateway.json"]) == [
        expected_fact("gateway", 2, "Omar", 43)]
    # The old revision no longer exists on disk. The consumer has no oracle access.
    assert json.loads((run.directory / "gateway.json").read_text())["revision"] == 2
    answer = run.phase("historical-comparison", facts=[["gateway", 1], ["gateway", 2]],
                       complete=False)
    assert answer == [expected_fact("gateway"), expected_fact("gateway", 2, "Omar", 43)]
    recalls = [message for message in run.history if isinstance(message, AIMessage)
               for call in message.tool_calls if call["name"] == "recall"]
    if preserve:
        assert len(gate.calls) == 1
        assert not recalls and not any(row["changed"] for row in run.middleware.telemetry)
        assert run.middleware.telemetry[-1]["no_op_reason"] == "semantic_veto"
        prior_calls = len(gate.calls)
        # Repeated requests at the same considered boundary do not reassess.
        run.middleware.project_messages(run.history)
        assert len(gate.calls) == prior_calls
        run.middleware.project_messages(run.history)
        assert len(gate.calls) == prior_calls
    else:
        assert len(recalls) == 1
        assert any(row["changed"] for row in run.middleware.telemetry)
        first = run.store.list_chunks("acceptance")[0]
        assert first.pinned and run.middleware.state.levels[first.id] == "full"
        assert visible_documents(run.model._requests[-1])[("gateway", 1)]["owner"] == "Mira"
    run.verify_archive()
    record_property("simulation_recall_calls", len(recalls))
    record_property("historical_and_current_facts_exact", True)
    run.close()


def test_unfinished_work_is_a_justified_noop_then_completed_boundary_allows_reduction(tmp_path):
    run = SimulationRun(tmp_path / "unfinished")
    write_fixture(run.directory, "gateway")
    for label in ("inspection", "verification"):
        assert run.phase(label, facts=[["gateway", 1]], read_paths=["gateway.json"],
                         complete=False) == [expected_fact("gateway")]
    assert run.phase("uncompleted-handoff", facts=[["gateway", 1]], complete=False) == [
        expected_fact("gateway")]
    before = tool_bytes(run.model._requests[-1])
    assert all(row["no_op_reason"] == "no_boundary" for row in run.middleware.telemetry)
    assert not any(row["changed"] for row in run.middleware.telemetry)
    # The application now confirms completion; ordinary user text did not do so.
    run.middleware.mark_boundary("verified-inspection-complete")
    assert run.phase("completed-handoff", facts=[["gateway", 1]]) == [expected_fact("gateway")]
    assert tool_bytes(run.model._requests[-1]) < before
    assert run.middleware.telemetry[-1]["rebuilt"]
    assert run.middleware.telemetry[-1]["changed"]
    run.verify_archive()
    run.close()


def test_recall_pagination_and_pin_survive_restart_without_losing_original_bytes(
    tmp_path, record_property,
):
    run = SimulationRun(tmp_path / "recall")
    write_fixture(run.directory, "gateway")
    for label in ("read-once", "read-again"):
        run.phase(label, facts=[["gateway", 1]], read_paths=["gateway.json"])
    projected = run.middleware.project_messages(run.history)
    first = run.store.list_chunks("acceptance")[0]
    assert next(message.content for message in projected if isinstance(message, ToolMessage)
                and message.tool_call_id == first.tool_call_id) != first.content
    assert tool_bytes(projected) < tool_bytes(run.history)
    expected_hash = hashlib.sha256(first.content.encode()).hexdigest()
    recovered, offset, char_offset, pages = "", 0, 0, []
    while True:
        page = json.loads(run.middleware.recall(first.id, offset, 1, char_offset))
        pages.append(page)
        recovered += page["text"]
        if not page["truncated"]:
            break
        continuation = page["next_offset"], page["next_char_offset"]
        assert continuation > (offset, char_offset), "recall must make forward progress"
        offset, char_offset = continuation
        assert len(pages) < 20
    assert len(pages) >= 3 and pages[0]["next_char_offset"] > 0
    assert hashlib.sha256(recovered.encode()).hexdigest() == expected_hash
    assert json.loads(json.loads(recovered)["content"])["threshold"] == 19
    run.close()
    reopened = LedgerStore(run.directory / "archive.sqlite")
    restored = ContextLabMiddleware(store=reopened, **run.middleware_options)
    replay = restored.project_messages(run.history)
    assert replay == run.history  # Recall pin repairs the view without a fresh boundary.
    assert_protocol(replay)
    assert restored.state.duplicate_sources == {}
    assert reopened.get_chunk("acceptance", first.id).pinned
    assert reopened.list_model_calls("acceptance") == []
    record_property("simulation_recall_pages", len(pages))
    record_property("recovered_original_bytes", len(recovered.encode()))
    reopened.close()


def test_corrupt_persisted_anchor_restores_evidence_before_consumer_answers(tmp_path):
    run = SimulationRun(tmp_path / "anchor")
    write_fixture(run.directory, "gateway")
    for label in ("read-once", "read-again"):
        run.phase(label, facts=[["gateway", 1]], read_paths=["gateway.json"])
    run.middleware.project_messages(run.history)
    state = run.store.get_projection_state("acceptance")
    assert state.duplicate_sources
    first = run.store.list_chunks("acceptance")[0]
    # Explicit storage fault injection, not a normal canonical-history edit.
    corrupted = state.model_copy(update={"duplicate_sources": {first.id: "missing-anchor"}})
    run.store.record_projection("acceptance", len(run.middleware.events),
                                ProjectionResult(messages=[], state=corrupted))
    run.close()
    store = LedgerStore(run.directory / "archive.sqlite")
    restarted = ContextLabMiddleware(store=store, **run.middleware_options)
    canonical = deepcopy(run.history)
    projected = restarted.project_messages(canonical)
    assert canonical == run.history and projected == canonical
    assert_protocol(projected)
    assert restarted.state.duplicate_sources == {}
    # An independent consumer sees complete facts even when the saved proof is corrupt.
    consumer = EvidenceSimulation()
    question = HumanMessage(content=json.dumps({"facts": [["gateway", 1]]}))
    answer = consumer.invoke([*projected, question])
    assert json.loads(answer.content) == [expected_fact("gateway")]
    assert store.get_chunk("acceptance", first.id).content == first.content
    assert store.list_model_calls("acceptance") == []
    store.close()
