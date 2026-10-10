"""Real controller/gate/ledger wiring with entirely offline provider receipts."""

from copy import deepcopy
from decimal import Decimal

from contextlab.core.controller import ContextController
from contextlab.core.models import Chunk, CompactionConfig, Event, Pricing, ViewState
from contextlab.core.signals import TRUSTED_BOUNDARY_SOURCE
from contextlab.ledger.budget import BudgetManager
from contextlab.ledger.store import LedgerStore
from harness.config import JevConfig
from harness.jev_decision import JevDecisionGate


def test_jev_veto_and_next_boundary_retirement_are_budgeted_and_archived(tmp_path):
    historical_question = "Compare the original snapshot against the current snapshot."
    current_question = "Historical comparisons are complete. Report only the current snapshot."
    original = "Original: service api, replicas 2, timeout 30.\n" * 160
    current = "Current: service api, replicas 4, timeout 45.\n" * 160
    reads = [
        Event(id=f"read-{index}", session_id="s", seq=index, kind="file_read",
              tool_name="read_file", tool_call_id=f"t{index}", content=content,
              metadata={"path": "services.json", "full_read": True, "version": str(index)})
        for index, content in ((1, original), (2, current))
    ]
    events = [
        Event(id="goal", session_id="s", seq=0, role="user", kind="user_msg",
              content="Inspect the service snapshots."),
        *reads,
        Event(id="historical-question", session_id="s", seq=3, role="user", kind="user_msg",
              content=historical_question),
        Event(id="phase-one", session_id="s", seq=4, role="system", kind="phase_boundary",
              metadata={"source": TRUSTED_BOUNDARY_SOURCE, "trusted": True}),
    ]
    chunks = [Chunk(id=f"c{read.seq}", session_id="s", event_id=read.id, seq=read.seq,
                    tool_call_id=read.tool_call_id, kind=read.kind, content=read.content)
              for read in reads]
    messages = [{"role": "user", "content": events[0].content}]
    for read in reads:
        messages.extend([
            {"role": "assistant", "content": [{"type": "tool_use", "id": read.tool_call_id,
                                                  "name": "read_file", "input": {"path": "services.json"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": read.tool_call_id,
                                             "content": read.content}]},
        ])
    messages.append({"role": "user", "content": historical_question})
    canonical_before = deepcopy(messages)
    pricing = Pricing(model="offline-main", input_per_million=3, cache_write_5m_per_million=3,
                      cache_write_1h_per_million=3, cache_read_per_million=3, output_per_million=15)
    compaction = CompactionConfig(recent_outputs=1, remaining_turns=8, min_saved_tokens=128)

    with LedgerStore(tmp_path / "ledger.sqlite") as store:
        store.create_session("s", "contextlab", "offline-jev-integration")
        for event in events:
            store.append_event(event)
        for chunk in chunks:
            store.add_chunk(chunk)
        budgets = BudgetManager(store, "1", "1")

        class ReceiptClient:
            def __init__(self):
                self.payloads = []

            def infer(self, payload):
                # Exercise the real reservation path: dispatch cannot precede
                # admission, even when the eventual answer vetoes compaction.
                assert budgets.status("s")["session"]["reserved_usd"] > 0
                self.payloads.append(deepcopy(payload))
                probability, input_tokens = ((0.98, 1000) if len(self.payloads) == 1
                                             else (0.05, 1500))
                return {"model": payload["model"],
                        "answers": {key: {"type": "noul", "noul": probability}
                                    for key in payload["questions"]},
                        "usage": {"input_tokens": input_tokens, "output_tokens": 8}}

        client = ReceiptClient()
        gate = JevDecisionGate(client, store, budgets, "s", config=JevConfig())
        controller = ContextController(semantic_gate=gate)
        archived = [store.get_chunk("s", chunk.id) for chunk in chunks]
        kept = controller.project(messages, store.list_events("s"), archived, ViewState(),
                                  pricing, compaction, cache_warm=False)
        store.record_projection("s", 4, kept)
        assert kept.no_op_reason == "semantic_veto" and not kept.rebuilt and not kept.fallback
        assert kept.messages == messages == canonical_before
        assert kept.semantic_decision["needed_probabilities"] == {"c1": 0.98}
        payload = client.payloads[0]
        assert payload["state"]["current_user_request"] == historical_question
        candidate = payload["state"]["candidates"]["candidate_0"]
        assert candidate["original_tool_result"] == original
        assert candidate["lifecycle_evidence"][0]["type"] == "superseded_read"
        call, = store.list_model_calls("s")
        assert call.purpose == "decision" and call.status == "ok"
        assert call.cost_usd == Decimal("0.000042") and call.usage.input_tokens == 1000
        assert call.raw_usage["request"] == payload
        assert budgets.status("s")["experiment"]["spent_usd"] == call.cost_usd
        assert budgets.status("s")["experiment"]["reserved_usd"] == 0

        # A restored projection remembers that the boundary was considered;
        # neither another Jev request nor another charge occurs.
        restored = store.get_projection("s")
        repeated = controller.project(messages, store.list_events("s"), archived, restored.state,
                                      pricing, compaction, cache_warm=False)
        assert repeated.no_op_reason == "boundary_already_considered"
        assert len(client.payloads) == len(store.list_model_calls("s")) == 1

        store.append_event(Event(id="current-question", session_id="s", seq=5, role="user",
                                 kind="user_msg", content=current_question))
        store.append_event(Event(id="phase-two", session_id="s", seq=6, role="system",
                                 kind="phase_boundary",
                                 metadata={"source": TRUSTED_BOUNDARY_SOURCE, "trusted": True}))
        next_messages = [*messages, {"role": "user", "content": current_question}]
        retired = controller.project(next_messages, store.list_events("s"), archived, repeated.state,
                                     pricing, compaction, cache_warm=False)
        store.record_projection("s", 6, retired)
        assert retired.rebuilt and retired.decision.accepted and not retired.fallback
        assert retired.state.levels["c1"] == "hidden" and retired.state.levels["c2"] == "full"
        assert retired.semantic_decision["needed_probabilities"] == {"c1": 0.05}
        assert retired.semantic_decision["cost_usd"] == "0.000063"
        rules_only = ContextController().project(
            next_messages, store.list_events("s"), archived, repeated.state,
            pricing, compaction, cache_warm=False,
        )
        assert retired.messages == rules_only.messages
        assert (retired.decision.move_cost_usd - rules_only.decision.move_cost_usd
                == Decimal("0.000063"))
        assert client.payloads[1]["state"]["current_user_request"] == current_question
        assert len(client.payloads) == 2
        calls = store.list_model_calls("s")
        assert [call.purpose for call in calls] == ["decision", "decision"]
        assert [call.usage.input_tokens for call in calls] == [1000, 1500]
        status = budgets.status("s")["experiment"]
        assert status["spent_usd"] == Decimal("0.000105")
        assert status["reserved_usd"] == 0 and not status["blocked"]
        # The projected tool text changes; the archive and canonical history do not.
        assert retired.messages[2]["content"][0]["content"] != original
        assert store.get_chunk("s", "c1").content == original
        assert messages == canonical_before
