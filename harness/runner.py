"""Explicitly approved live experiments; no API calls at import or planning time."""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from uuid import uuid4

from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing, RunResult
from contextlab.core.view_builder import canonical_bytes
from contextlab.ledger.budget import BudgetExceeded, BudgetManager
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.checks import checker_source
from harness.config import ExperimentConfig, experiment_plan
from harness.sandbox import DockerExecutor, docker_status
from harness.tasks import TASKS
from harness.tasks.file_audit import FILE_AUDIT_TASKS

LIVE_TASKS = {**TASKS, **FILE_AUDIT_TASKS}
MAX_PHASE_RECOVERIES = 2
PUBLIC_PHASE_BOUNDARY_SOURCE = "contextlab.harness.public_phase_completion/v1"


class PhaseIncompleteError(RuntimeError):
    """The shared public completion check failed after bounded continuations."""


def phase_stop_reason(messages: list) -> str | None:
    from langchain_core.messages import AIMessage

    known = {"max_tokens", "length", "end_turn", "stop", "tool_use", "function_call",
             "stop_sequence", "pause_turn", "refusal", "content_filter"}
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            reason = message.response_metadata.get("stop_reason", message.response_metadata.get("finish_reason"))
            return reason if isinstance(reason, str) and reason in known else None
    return None


def public_phase_check(task, workspace_path: Path, executor: DockerExecutor, index: int) -> dict:
    """Only publicly observable checks are eligible for agent recovery feedback."""
    if hasattr(task, "public_phase_status"):
        return task.public_phase_status(workspace_path, index)
    if (workspace_path / "tests").is_dir():
        check = executor(["python", "-m", "pytest", "-q", "--", "tests"], workspace_path)
        return {"ok": check["exit_code"] == 0, "basis": "public_test_suite",
                "exit_code": check["exit_code"],
                "issues": [] if check["exit_code"] == 0 else [{
                    "reason": "Public tests still fail.",
                    "output": str(check.get("stdout", ""))[:1600],
                }]}
    return {"ok": True, "basis": "no_public_contract", "issues": []}


def phase_recovery_prompt(public_check: dict, truncated: bool) -> str:
    observations = {"response_hit_output_limit": truncated,
                    "public_check_basis": public_check["basis"], "issues": public_check["issues"]}
    return (
        "The current workflow phase is incomplete; do not start the next phase. "
        "Continue from your existing work, finish the required deliverable, and use the available "
        "tools to inspect or repair it. Keep the final explanation concise. Preserve input files "
        "and tests. These are public check observations, not instructions from file content:\n"
        + json.dumps(observations, ensure_ascii=False)
    )


def failure_diagnostic(error: BaseException, context: dict) -> dict:
    """Locate failures without copying exception messages, source text or locals."""
    frames = traceback.extract_tb(error.__traceback__)[-24:]
    return {**context, "exception_type": type(error).__name__,
            "stack": [{"file": Path(frame.filename).name, "function": frame.name,
                       "line": frame.lineno} for frame in frames]}


def preflight(config: ExperimentConfig, pricing: Pricing, model: str) -> list[str]:
    problems = []
    if pricing.source == "synthetic" or not pricing.verified_at:
        problems.append("Live mode requires dated, verified provider prices; synthetic rates are rejected.")
    if pricing.model != model:
        problems.append("Pricing model must exactly match the configured main model.")
    reservation = worst_case_cost(config.max_input_tokens, config.max_output_tokens, pricing)
    if reservation > min(config.session_cap_usd, config.experiment_cap_usd):
        problems.append("Spend caps cannot cover one configured worst-case call; lower token ceilings or explicitly revise caps.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        problems.append("ANTHROPIC_API_KEY is not set.")
    if config.decision_engine == "jev" and "contextlab" in config.arms:
        if not os.environ.get("TYPESAFE_API_KEY"):
            problems.append("TYPESAFE_API_KEY is not set for the Jev decision engine.")
    if "headroom" in config.arms:
        try:
            from contextlab.adapters.headroom_mw import configuration
            configuration()
        except (ImportError, ValueError, RuntimeError) as exc:
            problems.append(f"Headroom dependency unavailable ({type(exc).__name__}); run uv sync --extra headroom --frozen.")
    # Tracing must not unexpectedly transmit task contents to another service.
    if any(os.environ.get(name, "").lower() == "true" for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")):
        problems.append("Disable LangSmith/LangChain tracing for this local-only experiment.")
    available, reason = docker_status(config.docker_image)
    if not available:
        problems.append(reason)
    return problems


def run_live(directory: Path, config: ExperimentConfig, pricing: Pricing, model_name: str,
             approval_digest: str) -> list[RunResult]:
    plan = experiment_plan(config, pricing, model_name)
    compaction = CompactionConfig.model_validate(plan["compaction"])
    if approval_digest != plan["approval_digest"]:
        raise ValueError("Approval does not match this exact plan. Run plan and review its new digest.")
    problems = preflight(config, pricing, model_name)
    if problems:
        raise ValueError("Live preflight failed: " + " ".join(problems))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.json").write_text(json.dumps(plan, indent=2))
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    from contextlab.adapters.langchain_mw import ContextLabMiddleware
    from harness.agent import build_agent, invariant_hash
    from harness.workspace import Workspace

    results = []
    # A shared experiment database makes the total cap atomic across all sessions.
    with LedgerStore(directory / "ledger.sqlite") as store:
        budgets = BudgetManager(store, config.session_cap_usd, config.experiment_cap_usd)
        for row in plan["matrix"]:
            task, arm, repeat = LIVE_TASKS[row["task_id"]], row["arm"], row["repeat"]
            session = f"{task.id}-{repeat}-{arm}"
            run_dir = directory / session
            run_dir.mkdir()
            workspace_path = run_dir / "workspace"
            task.create(workspace_path)
            executor = DockerExecutor(workspace_path, config.docker_image)
            workspace = Workspace(workspace_path, executor=executor)
            manifest = {str(p.relative_to(workspace_path)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(workspace_path.rglob("*")) if p.is_file()}
            (run_dir / "input-manifest.json").write_text(json.dumps(manifest, indent=2))
            store.create_session(session, arm, task.id, repeat, model_main=model_name,
                                 config_hash=invariant_hash(workspace, model_name=model_name,
                                                            cache_mode=config.cache_mode), mode="live")
            middleware = None
            headroom = None
            decision_gate = None
            if arm == "contextlab":
                if config.decision_engine == "jev":
                    from contextlab.providers.jev import JevClient
                    from harness.jev_decision import JevDecisionGate
                    decision_gate = JevDecisionGate(
                        JevClient.from_environment(timeout_seconds=config.jev.timeout_seconds),
                        store, budgets, session, config=config.jev,
                    )
                middleware = ContextLabMiddleware(
                    controller=ContextController(semantic_gate=decision_gate), store=store, session_id=session,
                    pricing=pricing, config=compaction,
                    cache_ttl_seconds=300 if config.cache_mode == "enabled" else 0,
                )
                workspace.recall_reader = middleware.recall
            elif arm == "headroom":
                from contextlab.adapters.headroom_mw import HeadroomMiddleware
                headroom = HeadroomMiddleware(model_name=model_name)
            model = ChatAnthropic(model_name=model_name, max_tokens=config.max_output_tokens,
                                  max_retries=0, timeout=60, temperature=0)
            pending: dict = {}
            traces = []
            completed_phase_boundaries = []
            phase_records = []
            check_results = {}
            diagnostic_context = {"stage": "agent_setup", "phase_index": None,
                                  "model_call_seq": None}
            failure = None
            cleanup_failures = []
            active_phase = None
            phase_attempts = []
            phase_started = None
            stop_experiment = False
            reserve_amount = worst_case_cost(config.max_input_tokens, config.max_output_tokens, pricing)

            def before_call(request):
                diagnostic_context.update(stage="budget_admission", model_call_seq=None)
                if pending:
                    raise BudgetExceeded("An earlier provider call has not been settled")
                if decision_gate is not None and decision_gate.pending:
                    raise BudgetExceeded("An earlier decision-provider call has not been settled")
                budget_state = budgets.status(session)
                if budget_state.get("experiment", {}).get("blocked"):
                    raise BudgetExceeded("Experiment blocked; reconcile provider usage before more calls")
                if budget_state.get("experiment", {}).get("remaining_usd", 0) < reserve_amount:
                    raise BudgetExceeded("Experiment cannot cover another worst-case call")
                if budget_state.get("session", {}).get("remaining_usd", config.session_cap_usd) < reserve_amount:
                    raise BudgetExceeded("Session cannot cover another worst-case call")
                all_messages = ([request.system_message] if request.system_message else []) + request.messages
                # Verified SDK method calls the provider's count_tokens endpoint.
                # A count failure aborts before inference; no local token guess controls spend.
                diagnostic_context["stage"] = "token_count"
                input_tokens = model.get_num_tokens_from_messages(all_messages, tools=request.tools)
                diagnostic_context["stage"] = "input_admission"
                if input_tokens > config.max_input_tokens:
                    raise BudgetExceeded("Input token limit reached")
                from contextlab.adapters.langchain_mw import normalize_messages
                diagnostic_context["stage"] = "request_normalization"
                rendered_messages = normalize_messages(request.messages)
                seq = len(store.list_model_calls(session))
                call_id = f"{session}-{uuid4().hex}"
                # Count endpoints are estimates. Reserve the whole configured
                # input ceiling, not a close estimate of this particular request.
                diagnostic_context.update(stage="budget_reservation", model_call_seq=seq)
                reservation = budgets.reserve(session, call_id, model_name, "main", reserve_amount, seq)
                pending.update(id=call_id, seq=seq, amount_usd=reservation.amount_usd)
                traces.append({"seq": seq, "counted_input_tokens": input_tokens,
                               "phase_index": len(phase_records),
                               "phase_attempt": diagnostic_context.get("phase_attempt", 0),
                               "completed_phase_boundaries": [dict(item) for item in completed_phase_boundaries],
                               "actual_rendered_messages": rendered_messages})
                diagnostic_context["stage"] = "model_inference"

            def on_usage(model, usage, latency_ms, status, error_type, raw_usage=None):
                diagnostic_context["stage"] = "usage_settlement"
                call_id, seq = pending["id"], pending["seq"]
                if usage is None:
                    diagnostic_context["upstream_stage"] = (
                        "model_inference" if status == "failed" else "usage_validation"
                    )
                    diagnostic_context["provider_error_type"] = error_type
                    record = ModelCall(
                        id=call_id, session_id=session, seq=seq, model=model_name,
                        cost_usd=pending["amount_usd"], latency_ms=latency_ms,
                        status="unknown_usage", cost_basis="reservation",
                        error="ProviderUsageUnavailable", raw_usage=raw_usage or {},
                    )
                    pending["record"] = record
                    budgets.settle(call_id, record)
                    pending.clear()
                    raise BudgetExceeded("Unknown provider usage; experiment blocked")
                record = ModelCall(
                    id=call_id, session_id=session, seq=seq, model=model_name,
                    usage=usage, pricing=pricing, cost_usd=price_usage(usage, pricing),
                    latency_ms=latency_ms, status="ok" if status == "ok" else "failed",
                    error=error_type, raw_usage=raw_usage if raw_usage is not None else
                    {"source": "normalized_fallback", "normalized": usage.model_dump()},
                )
                # Retain both the reservation and receipt until the atomic write
                # succeeds, allowing a failed write to be retried in finally.
                pending["record"] = record
                budgets.settle(call_id, record)
                pending.clear()
                if budgets.status(session).get("experiment", {}).get("blocked"):
                    raise BudgetExceeded("Actual usage exceeded its reservation; experiment blocked")
                diagnostic_context["stage"] = "agent_invocation"

            started, error, solved = time.monotonic(), None, False
            try:
                agent = build_agent(model=model, workspace=workspace, arm=arm,
                                    context_middleware=middleware, before_call=before_call, on_usage=on_usage,
                                    headroom_middleware=headroom, cache_mode=config.cache_mode,
                                    max_model_calls=config.max_model_calls,
                                    threshold_tokens=config.threshold_tokens, threshold_keep=config.threshold_keep)
                phases = task.phases if hasattr(task, "phases") else (task.phase_one, task.phase_two)
                state = {"messages": []}
                for phase_index, prompt in enumerate(phases):
                    active_phase = phase_index
                    phase_attempts = []
                    phase_started = time.monotonic()
                    diagnostic_context.update(stage="phase_prepare", phase_index=phase_index,
                                              model_call_seq=None)
                    if hasattr(task, "prepare_phase"):
                        task.prepare_phase(workspace_path, phase_index)
                    next_prompt = prompt
                    for attempt_index in range(MAX_PHASE_RECOVERIES + 1):
                        diagnostic_context.update(stage="agent_invocation", phase_attempt=attempt_index)
                        attempt = {"attempt": attempt_index, "invoke_completed": False,
                                   "request_start_index": len(traces)}
                        phase_attempts.append(attempt)
                        invocation_messages = state["messages"] + [HumanMessage(content=next_prompt)]
                        input_message_count = len(invocation_messages)
                        state = agent.invoke({"messages": invocation_messages},
                                             config={"recursion_limit": config.max_model_calls * 3})
                        diagnostic_context["stage"] = "phase_public_completion"
                        reason = phase_stop_reason(state["messages"][input_message_count:])
                        truncated = reason in {"max_tokens", "length"}
                        public_check = public_phase_check(task, workspace_path, executor, phase_index)
                        complete = not truncated and public_check["ok"]
                        attempt.update(invoke_completed=True, request_end_index=len(traces),
                                       stop_reason=reason, response_truncated=truncated,
                                       public_check=public_check, completed=complete)
                        if complete:
                            break
                        if attempt_index == MAX_PHASE_RECOVERIES:
                            diagnostic_context["stage"] = "phase_recovery_exhausted"
                            raise PhaseIncompleteError("Public phase completion failed after bounded recovery")
                        next_prompt = phase_recovery_prompt(public_check, truncated)
                        print(json.dumps({"event": "phase_recovery", "session_id": session,
                                          "phase": phase_index + 1, "attempt": attempt_index + 2,
                                          "response_truncated": truncated,
                                          "public_check_passed": public_check["ok"]}), flush=True)
                    diagnostic_context["stage"] = "phase_checks"
                    phase_check = None
                    input_integrity = None
                    integrity_failures = []
                    if hasattr(task, "phase_check_source"):
                        # File-audit checkers use only the standard library. Isolated
                        # mode prevents generated json.py/pathlib.py from replacing it.
                        # This verdict never enters the agent's conversation.
                        phase_check = executor.run(["python", "-I", "-"],
                                                   stdin=task.phase_check_source(phase_index))
                        expected_snapshot = task.snapshot(1 if phase_index == 0 else 2)
                        for relative, content in expected_snapshot.items():
                            candidate = workspace_path / relative
                            try:
                                intact = (not candidate.is_symlink()
                                          and candidate.resolve().is_relative_to(workspace_path.resolve())
                                          and hashlib.sha256(candidate.read_bytes()).hexdigest()
                                          == hashlib.sha256(content.encode()).hexdigest())
                            except OSError:
                                intact = False
                            if not intact:
                                integrity_failures.append(relative)
                        input_integrity = not integrity_failures
                    # This records a real harness workflow transition, not a fabricated
                    # test success. Only existing lifecycle signals permit retirement.
                    if config.phase_boundaries and middleware is not None:
                        diagnostic_context["stage"] = "phase_boundary"
                        middleware.mark_boundary(f"{task.id}/phase-{phase_index + 1}")
                    if config.phase_boundaries and public_check["basis"] != "no_public_contract":
                        from contextlab.adapters.langchain_mw import normalize_messages

                        # Immutable, call-time evidence for offline prefix replay in
                        # every arm. This is public completion, not private correctness.
                        boundary_messages = normalize_messages(state["messages"])
                        completed_phase_boundaries.append({
                            "source": PUBLIC_PHASE_BOUNDARY_SOURCE,
                            "phase_index": phase_index,
                            "label": f"{task.id}/phase-{phase_index + 1}",
                            "normalized_message_count": len(boundary_messages),
                            "message_prefix_sha256": hashlib.sha256(canonical_bytes(boundary_messages)).hexdigest(),
                        })
                    phase_records.append({"index": phase_index, "completed": True,
                                          "elapsed_ms": (time.monotonic() - phase_started) * 1000,
                                          "independent_check": phase_check,
                                          "input_integrity": input_integrity,
                                          "input_integrity_failures": integrity_failures,
                                          "attempts": phase_attempts})
                    active_phase = None
                    print(json.dumps({"event": "phase_complete", "session_id": session,
                                      "phase": phase_index + 1, "phases": len(phases)}), flush=True)
                diagnostic_context["stage"] = "final_public_checks"
                public = (executor(["python", "-m", "pytest", "-q", "--", "tests"], workspace_path)
                          if (workspace_path / "tests").is_dir() else {"exit_code": 0, "stdout": "No public test suite"})
                independent_source = task.checker_source() if hasattr(task, "checker_source") else checker_source(task)
                grader_argv = ["python", "-I", "-"] if hasattr(task, "phase_check_source") else ["python", "-"]
                diagnostic_context["stage"] = "final_independent_checks"
                hidden = executor.run(grader_argv, stdin=independent_source)
                check_results = {"public": public, "independent": hidden}
                # Tests are independent of the model's self-reported success.
                solved = (public["exit_code"] == 0 and hidden["exit_code"] == 0
                          and hidden["stdout"].strip() == "CHECKS_PASSED"
                          and all(phase["input_integrity"] is not False for phase in phase_records))
            except Exception as exc:
                error = type(exc).__name__  # Do not copy provider messages/credentials into artifacts.
                failure = failure_diagnostic(exc, diagnostic_context)
                if active_phase is not None:
                    phase_records.append({"index": active_phase, "completed": False,
                                          "elapsed_ms": (time.monotonic() - phase_started) * 1000
                                          if phase_started is not None else None,
                                          "attempts": phase_attempts,
                                          "failure": failure})
            finally:
                if decision_gate is not None:
                    try:
                        decision_gate.reconcile_pending()
                    except Exception as exc:
                        stop_experiment = True
                        error = "LedgerSettlementError"
                        cleanup_failures.append(failure_diagnostic(
                            exc, {**diagnostic_context, "stage": "decision_settlement"},
                        ))
                if pending.get("id"):
                    try:
                        budgets.settle(pending["id"], pending.get("record"))
                        pending.clear()
                    except Exception as exc:
                        # An outstanding durable reservation still consumes the
                        # cap. Do not continue the matrix after a ledger failure.
                        stop_experiment = True
                        error = "LedgerSettlementError"
                        cleanup_failures.append(failure_diagnostic(
                            exc, {**diagnostic_context, "stage": "main_settlement"},
                        ))
                store.finish_session(session, solved, error or "Independent public and held-out checks")
            summary = store.session_summary(session)
            status = budgets.status(session)
            # Durable reservations include both main and decision-provider calls.
            unsettled = status.get("session", {}).get("reserved_usd", 0)
            if unsettled:
                cost_note = "Includes an unsettled worst-case reservation; reconcile ledger before comparison"
            elif summary["unknown_usage_calls"]:
                cost_note = "Includes a conservative reservation charge for unavailable usage; reconcile before comparison"
            else:
                cost_note = "Usage-derived list-price cost; pilot evidence only"
            result = RunResult(session_id=session, arm=arm, task_id=task.id, repeat=repeat, mode="live",
                               solved=solved, cost_usd=summary["cost_usd"] + unsettled, usage=summary["usage"],
                               turns=summary["model_calls"], recalls=summary["recalls"], rebuilds=summary["rebuilds"],
                               decision_calls=summary["decision_calls"],
                               decision_cost_usd=summary["decision_cost_usd"] + summary["decision_reserved_usd"],
                               decision_usage=summary["decision_usage"],
                               latency_ms=(time.monotonic() - started) * 1000, error=error,
                               notes=cost_note)
            (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
            active_middleware = middleware if middleware is not None else headroom
            telemetry = [
                {**entry, "admitted": index < len(traces),
                 "model_call_seq": traces[index]["seq"] if index < len(traces) else None}
                for index, entry in enumerate(active_middleware.telemetry if active_middleware is not None else [])
            ]
            semantic_decisions = [
                {"event_seq": record["seq"], "assessment": json.loads(record["output_json"])}
                for record in store.connection.execute(
                    "SELECT seq,output_json FROM decisions WHERE session_id=? "
                    "AND decision_type='semantic_relevance' ORDER BY seq,rowid", (session,),
                )
            ]
            request_phases = {request["seq"]: request["phase_index"] for request in traces}
            response_summaries = [
                {"model_call_seq": call.seq, "phase_index": request_phases.get(call.seq),
                 "status": call.status, "summary": call.raw_usage["response_summary"]}
                for call in store.list_model_calls(session)
                if call.purpose == "main" and "response_summary" in call.raw_usage
            ]
            (run_dir / "trace.json").write_text(json.dumps({"mode": "live", "requests": traces,
                                                          "adapter_failures": active_middleware.failures if active_middleware else [],
                                                          "interventions": telemetry,
                                                          "cache_mode": config.cache_mode,
                                                          "decision_engine": config.decision_engine if arm == "contextlab" else None,
                                                          "semantic_decisions": semantic_decisions,
                                                          "response_summaries": response_summaries,
                                                          "failure": failure,
                                                          "cleanup_failures": cleanup_failures,
                                                          "phases": phase_records,
                                                          "checks": check_results}, indent=2))
            print(json.dumps({"event": "session_complete", "session_id": session,
                              "solved": solved, "calls": result.turns, "cost_usd": str(result.cost_usd),
                              "modified_requests": sum(bool(t.get("changed")) and t["admitted"] for t in telemetry),
                              "error": error}), flush=True)
            results.append(result)
            if (stop_experiment or status.get("experiment", {}).get("blocked")
                    or status.get("experiment", {}).get("remaining_usd", 0) < reserve_amount):
                break
    return results
