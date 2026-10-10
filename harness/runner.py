"""Explicitly approved live experiments; no API calls at import or planning time."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from uuid import uuid4

from contextlab.core.controller import ContextController
from contextlab.core.models import CompactionConfig, Pricing, RunResult
from contextlab.ledger.budget import BudgetExceeded, BudgetManager
from contextlab.ledger.pricing import price_usage, worst_case_cost
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.checks import checker_source
from harness.config import ExperimentConfig, experiment_plan
from harness.sandbox import DockerExecutor, docker_status
from harness.tasks import TASKS


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
            task, arm, repeat = TASKS[row["task_id"]], row["arm"], row["repeat"]
            session = f"{task.id}-{repeat}-{arm}"
            run_dir = directory / session
            run_dir.mkdir()
            workspace_path = run_dir / "workspace"
            task.create(workspace_path)
            executor = DockerExecutor(workspace_path, config.docker_image)
            workspace = Workspace(workspace_path, executor=executor)
            store.create_session(session, arm, task.id, repeat, model_main=model_name,
                                 config_hash=invariant_hash(workspace, model_name=model_name), mode="live")
            middleware = ContextLabMiddleware(
                controller=ContextController(), store=store, session_id=session,
                pricing=pricing, config=compaction, observe_only=arm != "contextlab",
            )
            if arm == "contextlab":
                workspace.recall_reader = middleware.recall
            model = ChatAnthropic(model_name=model_name, max_tokens=config.max_output_tokens,
                                  max_retries=0, timeout=60, temperature=0)
            pending: dict = {}
            traces = []
            stop_experiment = False
            reserve_amount = worst_case_cost(config.max_input_tokens, config.max_output_tokens, pricing)

            def before_call(request):
                if pending:
                    raise BudgetExceeded("An earlier provider call has not been settled")
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
                input_tokens = model.get_num_tokens_from_messages(all_messages, tools=request.tools)
                if input_tokens > config.max_input_tokens:
                    raise BudgetExceeded("Input token limit reached")
                from contextlab.adapters.langchain_mw import normalize_messages
                rendered_messages = normalize_messages(request.messages)
                seq = len(store.list_model_calls(session))
                call_id = f"{session}-{uuid4().hex}"
                # Count endpoints are estimates. Reserve the whole configured
                # input ceiling, not a close estimate of this particular request.
                reservation = budgets.reserve(session, call_id, model_name, "main", reserve_amount, seq)
                pending.update(id=call_id, seq=seq, amount_usd=reservation.amount_usd)
                traces.append({"seq": seq, "counted_input_tokens": input_tokens,
                               "actual_rendered_messages": rendered_messages})

            def on_usage(model, usage, latency_ms, status, error_type, raw_usage=None):
                call_id, seq = pending["id"], pending["seq"]
                if usage is None:
                    budgets.settle(call_id, None, error=f"Provider usage unavailable ({status}); reconcile before more calls")
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

            started, error, solved = time.monotonic(), None, False
            try:
                agent = build_agent(model=model, workspace=workspace, arm=arm,
                                    context_middleware=middleware, before_call=before_call, on_usage=on_usage,
                                    max_model_calls=config.max_model_calls,
                                    threshold_tokens=config.threshold_tokens, threshold_keep=config.threshold_keep)
                state = agent.invoke({"messages": [HumanMessage(content=task.phase_one)]},
                                     config={"recursion_limit": config.max_model_calls * 3})
                state = agent.invoke({"messages": state["messages"] + [HumanMessage(content=task.phase_two)]},
                                     config={"recursion_limit": config.max_model_calls * 3})
                public = executor(["python", "-m", "pytest", "-q", "--", "tests"], workspace_path)
                hidden = executor.run(["python", "-"], stdin=checker_source(task))
                # Tests are independent of the model's self-reported success.
                solved = (public["exit_code"] == 0 and hidden["exit_code"] == 0
                          and hidden["stdout"].strip() == "CHECKS_PASSED")
            except Exception as exc:
                error = type(exc).__name__  # Do not copy provider messages/credentials into artifacts.
            finally:
                if pending.get("id"):
                    try:
                        budgets.settle(pending["id"], pending.get("record"))
                        pending.clear()
                    except Exception:
                        # An outstanding durable reservation still consumes the
                        # cap. Do not continue the matrix after a ledger failure.
                        stop_experiment = True
                        error = "LedgerSettlementError"
                store.finish_session(session, solved, error or "Independent public and held-out checks")
            summary = store.session_summary(session)
            status = budgets.status(session)
            unsettled = status.get("session", {}).get("reserved_usd", 0) if pending else 0
            if unsettled:
                cost_note = "Includes an unsettled worst-case reservation; reconcile ledger before comparison"
            elif summary["unknown_usage_calls"]:
                cost_note = "Includes a conservative reservation charge for unavailable usage; reconcile before comparison"
            else:
                cost_note = "Usage-derived list-price cost; pilot evidence only"
            result = RunResult(session_id=session, arm=arm, task_id=task.id, repeat=repeat, mode="live",
                               solved=solved, cost_usd=summary["cost_usd"] + unsettled, usage=summary["usage"],
                               turns=summary["model_calls"], recalls=summary["recalls"], rebuilds=summary["rebuilds"],
                               latency_ms=(time.monotonic() - started) * 1000, error=error,
                               notes=cost_note)
            (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
            (run_dir / "trace.json").write_text(json.dumps({"mode": "live", "requests": traces,
                                                          "adapter_failures": middleware.failures}, indent=2))
            results.append(result)
            if (stop_experiment or status.get("experiment", {}).get("blocked")
                    or status.get("experiment", {}).get("remaining_usd", 0) < reserve_amount):
                break
    return results
