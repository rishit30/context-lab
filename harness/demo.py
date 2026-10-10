"""Offline, explicitly scripted mechanics demonstration. No inference or savings claims."""
from __future__ import annotations

import json
from pathlib import Path

from contextlab.core.controller import ContextController
from contextlab.core.models import (
    Chunk,
    CompactionConfig,
    Event,
    Pricing,
    RunResult,
    Usage,
    ViewState,
)
from contextlab.core.view_builder import estimate_tokens
from contextlab.ledger.pricing import price_usage
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.checks import check_reference_fixture
from harness.tasks import TASKS


def demo_session(directory: Path, arm: str, task_id: str, repeat: int,
                 pricing: Pricing, config: CompactionConfig) -> RunResult:
    """Stress the decision code using labeled synthetic output, then check a reference fixture."""
    from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from contextlab.adapters.langchain_mw import normalize_messages

    task = TASKS[task_id]
    session = f"{task_id}-{repeat}-{arm}"
    run_dir = directory / session
    run_dir.mkdir(parents=True, exist_ok=False)
    workspace = run_dir / "workspace"
    task.create(workspace)
    # Synthetic comments deliberately make the policy branch visible. They are NOT
    # a claim about representative repositories or model behavior.
    old = task.initial + "\n".join(f"# synthetic historical detail {n}: inspect old state before editing" for n in range(600))
    message_list = [HumanMessage(content=task.phase_one)]
    events, chunks, timeline = [], [], []
    state = ViewState()
    controller = ContextController()
    fixture_steps = [
        ("file_read", "read_file", old, {"path": "app.py", "full_read": True, "version": "v0"}),
        ("test_run", "run_tests", "FAILED: public diagnostic assertion", {"test_selection": "tests", "environment": "demo", "exit_code": 1}),
        ("file_edit", "edit_file", "Reference patch applied (scripted)", {"path": "app.py"}),
        ("file_read", "read_file", task.phase_one_solution, {"path": "app.py", "full_read": True, "version": "v1"}),
        ("test_run", "run_tests", "Public checks pass (scripted trace)", {"test_selection": "tests", "environment": "demo", "exit_code": 0}),
    ]
    threshold = ContextEditingMiddleware(edits=[ClearToolUsesEdit(trigger=2000, keep=2)])
    usage_total = {name: 0 for name in Usage.model_fields}
    with LedgerStore(run_dir / "ledger.sqlite") as store:
        store.create_session(session, arm, task_id, repeat, model_main=pricing.model, mode="simulation")
        for seq, (kind, tool, content, metadata) in enumerate(fixture_steps):
            call_id = f"tool-{seq}"
            message_list.extend([
                AIMessage(content="", tool_calls=[{"id": call_id, "name": tool, "args": {}}]),
                ToolMessage(content=content, tool_call_id=call_id),
            ])
            event = Event(id=f"e{seq}", session_id=session, seq=seq, kind=kind, tool_name=tool,
                          tool_call_id=call_id, content=content, metadata=metadata,
                          size_tokens=estimate_tokens(content))
            chunk = Chunk(id=f"c{seq}", session_id=session, event_id=event.id, tool_call_id=call_id,
                          kind=kind, content=content, seq=seq, size_tokens=event.size_tokens,
                          refs=("app.py",), metadata=metadata)
            store.append_event(event)
            store.add_chunk(chunk)
            events.append(event)
            chunks.append(chunk)
            baseline = normalize_messages(message_list)
            projection = controller.project(baseline, events, chunks, state, pricing, config,
                                            cache_warm=True, shadow=arm != "contextlab")
            rendered = baseline
            if arm == "contextlab":
                state, rendered = projection.state, projection.messages
                store.record_projection(session, seq, projection)
            elif arm == "threshold_edit":
                # Invoke the real installed threshold middleware; never impersonate Headroom.
                from langchain.agents.middleware.types import ModelRequest
                from langchain_core.language_models.fake_chat_models import FakeListChatModel
                captured = []
                request = ModelRequest(model=FakeListChatModel(responses=["unused"]),
                                       messages=message_list, state={"messages": message_list},
                                       runtime=None)
                threshold.wrap_model_call(request, lambda req: captured.append(req.messages))
                rendered = normalize_messages(captured[0])
            tokens = estimate_tokens(json.dumps(rendered, sort_keys=True))
            # Deliberately simple counterfactual pricing: no claim that a provider
            # would cache these requests. Displayed as simulated all-uncached usage.
            usage = Usage(input_tokens=tokens, output_tokens=48)
            for name in usage_total:
                usage_total[name] += getattr(usage, name)
            store.record_model_call(ModelCall(
                id=f"{session}-m{seq}", session_id=session, seq=seq, model=pricing.model,
                usage=usage, cost_usd=price_usage(usage, pricing), pricing=pricing,
                cost_basis="simulation",
                raw_usage={"simulated": True, "cache_model": "all_uncached", "provider_calls": 0},
            ))
            timeline.append({"seq": seq, "tool": tool, "baseline_tokens_estimate": estimate_tokens(json.dumps(baseline)),
                             "rendered_tokens_estimate": tokens,
                             "projection": projection.model_dump(mode="json"),
                             "actual_rendered_messages": rendered})
        recall = None
        if arm == "contextlab" and state.levels.get("c0") == "hidden":
            recall = store.recall(session, "c0", 6, offset=0, limit=3).model_dump(mode="json")
        (workspace / "app.py").write_text(task.solution)
        passed = check_reference_fixture(task, workspace)
        store.finish_session(session, passed, "Scripted reference fixture; no agent inference")
        summary = store.session_summary(session)
    (run_dir / "trace.json").write_text(json.dumps({
        "mode": "simulation", "synthetic_stress_case": True, "timeline": timeline,
        "recall_probe": recall, "provider_spend_usd": 0,
    }, indent=2))
    result = RunResult(session_id=session, arm=arm, task_id=task_id, repeat=repeat,
                       mode="simulation", solved=passed, cost_usd=summary["cost_usd"],
                       turns=5, recalls=summary["recalls"], rebuilds=summary["rebuilds"],
                       usage=Usage(**usage_total),
                       notes="Synthetic stress trace + trusted reference checks; not an agent benchmark")
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
    return result
