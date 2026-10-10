"""Five-arm read-only delegation example: offline demo or explicitly approved live run.

The demo uses deterministic fixture consumers and synthetic accounting. Live run
always uses real providers; it cannot be switched to simulated responses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr

from contextlab.core.models import Pricing
from contextlab.core.routing import (
    EvidenceSource,
    EvidenceTask,
    RoutingConfig,
    build_evidence_brief,
)
from contextlab.session import ContextLabSession, SessionConfig
from examples.file_agent import SYSTEM as FILE_SYSTEM
from examples.file_agent import approval_digest, file_manifest, read_tools
from examples.file_agent import source_manifest as file_source_manifest
from harness.config import ROOT, load_compaction, load_pricing
from harness.environment import load_anthropic_key, load_typesafe_key
from harness.report.routing import build_routing_report
from harness.workspace import Workspace

ARMS = ("baseline", "cl_compact_only", "cl_route_only", "contextlab", "headroom")
ROUTING_ARMS = {"cl_route_only", "contextlab"}
SYSTEM = FILE_SYSTEM + (
    " For a bounded evidence extraction, call analyze_files with specific questions and the "
    "listed source IDs. Its result supplies either verified exact quotes or complete original "
    "evidence; you remain responsible for relevance, completeness, and the final answer. "
    "Exact quotation validation does not establish that an answer is correct."
)
DEFAULT_QUESTIONS = (
    "List available files. Use analyze_files to extract the exact service owners, rollback "
    "criteria, and urgent incident evidence from relevant sources. Assess the incident with citations.",
    "Recheck the source evidence for your assessment. Give exact rollback thresholds, "
    "responsible owners, and any contradictions. Use analyze_files for bounded fact extraction.",
    "Prepare a final handoff with source citations and unresolved uncertainties. "
    "Distinguish facts established by the files from recommendations.",
)


def source_manifest() -> list[tuple[str, str]]:
    sources = dict(file_source_manifest())
    for name in ("requirements.runtime.lock", "Dockerfile", "Dockerfile.task"):
        path = ROOT / name
        if path.is_file():
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return sorted(sources.items())


def source_hash() -> str:
    return hashlib.sha256(json.dumps(source_manifest()).encode()).hexdigest()


def evidence_sources(workspace: Workspace) -> list[EvidenceSource]:
    return [EvidenceSource(id=name, path=name, content=workspace.resolve(name).read_bytes().decode("utf-8"))
            for name in workspace.files()]


def preflight_sources(workspace: Workspace) -> list[EvidenceSource]:
    sources = evidence_sources(workspace)
    if not 1 <= len(sources) <= 64:
        raise ValueError("Select between one and 64 source files")
    for source in sources:
        # Verify every file can fit a minimal complete request. Actual model-
        # chosen question/goal lengths are bounded again by the tool at runtime.
        task = EvidenceTask(goal="Extract requested evidence", questions={"evidence": "Locate relevant evidence"},
                            source_ids=(source.id,))
        build_evidence_brief(task, sources, RoutingConfig())
    return sources


def _settings(main: Pricing, worker: Pricing, arms: tuple[str, ...], *, simulation=False,
              routing_engine="rules") -> dict[str, dict]:
    return {
        arm: SessionConfig(
            mode="simulation" if simulation else "live", arm=arm, pricing=main,
            compaction=load_compaction(), decision_engine="rules",
            routing=RoutingConfig(enabled=arm in ROUTING_ARMS, engine=routing_engine,
                                  worker_pricing=worker, max_calls_per_session=4,
                                  max_output_tokens=256 if simulation else 1024),
            cache_mode="disabled", max_model_calls=20, max_input_tokens=30000,
            max_output_tokens=2048, session_cap_usd="1.00", experiment_cap_usd=str(len(arms)),
        ).model_dump(mode="json") for arm in arms
    }


def plan(*, files: Path, pricing_path: Path, worker_pricing_path: Path, arm="all",
         routing_engine="rules", questions: tuple[str, ...] = DEFAULT_QUESTIONS) -> dict:
    if arm not in (*ARMS, "all") or not questions or any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError("Choose a supported arm and nonempty questions")
    arms = ARMS if arm == "all" else (arm,)
    catalog = preflight_sources(Workspace(files))
    value = {
        "kind": "real-read-only-routing-agent", "mode": "live",
        "settings": _settings(load_pricing(pricing_path), load_pricing(worker_pricing_path), arms,
                              routing_engine=routing_engine),
        "files_directory": str(files.resolve()), "files": file_manifest(Workspace(files)),
        "source_ids": [source.id for source in catalog],
        "evidence_scope": "At most 64 frozen sources; each complete tool request must fit the 32 KB brief limit. Oversized requests return a recoverable tool error.",
        "questions": questions, "system_prompt": SYSTEM,
        "tools": ["list_files", "read_file", "analyze_files", "recall"],
        "maximum_authorized_cost_usd": str(len(arms)),
        "data_sharing": "Selected task evidence goes to Anthropic main/worker models; TypeSafe receives a bounded suitability request only when Jev routing is enabled.",
        "quality_note": "Live answers are not independently graded. Exact worker quotes prove source provenance, not relevance or completeness. No savings claim is made.",
        "source_hash": source_hash(),
    }
    if "headroom" in arms:
        from contextlab.adapters.headroom_mw import configuration
        value["headroom"] = configuration()
    return {**value, "approval_digest": approval_digest(value)}


def _snapshot(source: Path, destination: Path, manifest: dict[str, str]) -> None:
    destination.mkdir()
    workspace = Workspace(source)
    for name, digest in manifest.items():
        data = workspace.resolve(name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Input changed while snapshotting; no inference started")
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def run(directory: Path, reviewed_plan: dict, approval: str) -> dict:
    """Only the reviewed real-provider plan enters this path; no fake injection."""
    if (reviewed_plan.get("kind") != "real-read-only-routing-agent"
            or reviewed_plan.get("mode") != "live"
            or approval != reviewed_plan.get("approval_digest")
            or approval != approval_digest(reviewed_plan)):
        raise ValueError("Approval does not match the real-provider plan")
    if source_hash() != reviewed_plan["source_hash"]:
        raise ValueError("Source changed after review")
    settings = {arm: SessionConfig.model_validate(value) for arm, value in reviewed_plan["settings"].items()}
    if not settings or any(arm not in ARMS or setting.mode != "live" or setting.arm != arm
                           for arm, setting in settings.items()):
        raise ValueError("Live run requires live settings in every arm")
    common = None
    for arm, setting in settings.items():
        if (setting.routing.enabled != (arm in ROUTING_ARMS)
                or setting.decision_engine != "rules" or setting.shadow):
            raise ValueError("The comparison requires the declared routing/compaction ablations")
        comparable = setting.model_dump(mode="json")
        comparable.pop("arm")
        comparable["routing"].pop("enabled")
        if common is not None and comparable != common:
            raise ValueError("Comparison arms must share models, prices, limits and policy settings")
        common = comparable
    ceiling = Decimal(reviewed_plan["maximum_authorized_cost_usd"])
    if (sum((setting.session_cap_usd for setting in settings.values()), Decimal(0)) != ceiling
            or any(setting.experiment_cap_usd != ceiling for setting in settings.values())):
        raise ValueError("Plan spend ceiling and session/experiment caps disagree")
    files = Path(reviewed_plan["files_directory"])
    if file_manifest(Workspace(files)) != reviewed_plan["files"]:
        raise ValueError("Inputs changed after review")
    if [source.id for source in preflight_sources(Workspace(files))] != reviewed_plan["source_ids"]:
        raise ValueError("Source catalog changed after review")
    if directory.resolve().is_relative_to(files.resolve()) or directory.exists():
        raise ValueError("Use a new output directory outside the input directory")
    if "headroom" in settings:
        from contextlab.adapters.headroom_mw import configuration
        if configuration() != reviewed_plan.get("headroom"):
            raise ValueError("Headroom profile changed after review")
    # Credential loading follows every approval/configuration/input check.
    load_anthropic_key(ROOT / ".env")
    if any(setting.routing.enabled and setting.routing.engine == "jev" for setting in settings.values()):
        load_typesafe_key(ROOT / ".env")
    directory.mkdir(parents=True, mode=0o700)
    _snapshot(files, directory / "input", reviewed_plan["files"])
    return _execute(directory, reviewed_plan)


def _execute(directory: Path, reviewed_plan: dict, *, model_factory=None, worker_factory=None,
             expected_answers: list[dict] | None = None) -> dict:
    simulation = reviewed_plan["mode"] == "simulation"
    if not simulation and (model_factory is not None or worker_factory is not None or expected_answers is not None):
        raise ValueError("Simulation providers/checks cannot enter live execution")
    (directory / "plan.json").write_text(json.dumps(reviewed_plan, indent=2))
    (directory / "source-manifest.json").write_text(json.dumps(source_manifest(), indent=2))
    snapshot = directory / "input"
    results, error = [], None
    for arm, raw in reviewed_plan["settings"].items():
        if source_hash() != reviewed_plan["source_hash"]:
            error = "ImplementationChanged"
            break
        if file_manifest(Workspace(snapshot)) != reviewed_plan["files"]:
            error = "InputSnapshotChanged"
            break
        session, summary, failure, close_error, summary_error = None, None, None, None, None
        answers, outcomes, schemas = [], [], []
        try:
            config = SessionConfig.model_validate(raw)
            kwargs = {"model_factory": model_factory, "worker_factory": worker_factory} if simulation else {}
            session = ContextLabSession(directory / "ledger.sqlite", arm, config, **kwargs)
            tools = read_tools(Workspace(snapshot))
            schemas = [convert_to_openai_tool(item) for item in tools]
            graph = session.build_agent(tools=tools, system_prompt=reviewed_plan["system_prompt"],
                                        evidence_sources=evidence_sources(Workspace(snapshot)))
            messages = []
            for index, question in enumerate(reviewed_plan["questions"]):
                before = len(messages)
                state = graph.invoke({"messages": [*messages, HumanMessage(content=question)]},
                                     {"recursion_limit": config.max_model_calls * 3})
                messages = state["messages"]
                final = messages[-1]
                for item in messages[before:]:
                    if isinstance(item, ToolMessage) and item.name == "analyze_files":
                        observed = json.loads(item.content)
                        outcomes.append({key: observed.get(key) for key in ("status", "reason", "validation")})
                if (not isinstance(final, AIMessage) or final.tool_calls or final.invalid_tool_calls
                        or final.response_metadata.get("stop_reason") != "end_turn"):
                    raise RuntimeError("Incomplete main-agent response")
                if file_manifest(Workspace(snapshot)) != reviewed_plan["files"]:
                    raise RuntimeError("Frozen input snapshot changed")
                answers.append(final.content)
                if expected_answers is not None and json.loads(final.content) != expected_answers[index]:
                    raise ValueError("Offline fixture answer failed its independent check")
                session.complete_phase(f"answered-question-{index + 1}")
        except Exception as exc:
            failure = type(exc).__name__
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as exc:
                    close_error = type(exc).__name__
                try:
                    summary = session.summary()
                except Exception as exc:
                    summary_error = type(exc).__name__
        if summary is None:
            summary = {"session_id": arm, "arm": arm, "total_cost_usd": "0" if session is None else None,
                       "budget": {}, "cost_basis": "No invocation" if session is None else "Unknown ledger summary"}
        summary.update(configuration=arm, error_type=failure, close_error_type=close_error,
                       summary_error_type=summary_error, answers=answers, tool_outcomes=outcomes,
                       questions_completed=len(answers), independently_graded=False,
                       mechanical_fixture_passed=(failure is None and len(answers) == len(expected_answers))
                       if expected_answers is not None else None,
                       shared_read_tool_schema_sha256=hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest())
        results.append(summary)
        (directory / f"{arm}.json").write_text(json.dumps(summary, indent=2))
        if summary_error or close_error or session is None:
            error = summary_error or close_error or failure
            break
        if summary.get("budget", {}).get("experiment", {}).get("blocked"):
            error = "ExperimentBudgetBlocked"
            break
    known = sum((Decimal(row["total_cost_usd"]) for row in results if row.get("total_cost_usd") is not None), Decimal(0))
    total = str(known) if all(row.get("total_cost_usd") is not None for row in results) else None
    result = {"mode": reviewed_plan["mode"], "sessions": results, "total_cost_usd": total,
              "known_cost_usd": str(known), "provider_spend_usd": "0" if simulation else None,
              "cost_basis": "Synthetic accounting units, not dollars charged" if simulation else "Usage at reviewed list prices, not an invoice",
              "quality_note": reviewed_plan["quality_note"], "error_type": error,
              "all_arms_attempted": len(results) == len(reviewed_plan["settings"]),
              "report": str((directory / "report.html").resolve())}
    (directory / "summary.json").write_text(json.dumps(result, indent=2))
    build_routing_report(directory, result)
    return result


def _visible_text(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in
                ([item] if key in {"content", "quote"} and isinstance(item, str) else _visible_text(item))]
    if isinstance(value, list):
        return [text for item in value for text in _visible_text(item)]
    return []


class SimulationMain(BaseChatModel):
    """Deterministic tool consumer, not an LLM or a substitute quality benchmark."""
    _calls: int = PrivateAttr(default=0)
    _schemas: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-routing-fixture"

    def bind_tools(self, tools, **kwargs):
        self._schemas = [convert_to_openai_tool(item) for item in tools]
        return self

    def get_num_tokens_from_messages(self, messages, **kwargs):
        return max(1, sum(len(str(message.content).encode()) for message in messages) // 4)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls += 1
        start = max(index for index, message in enumerate(messages) if isinstance(message, HumanMessage))
        question = json.loads(messages[start].content)
        if isinstance(messages[-1], HumanMessage):
            if question["kind"] == "diagnostics":
                calls = [{"name": "read_file", "args": {"path": "diagnostics.txt"},
                          "id": f"sim-{self._calls}-{i}", "type": "tool_call"} for i in range(2)]
            else:
                calls = [{"name": "analyze_files", "args": {"goal": "Extract the requested exact source lines",
                          "questions": question["questions"], "source_ids": ["service.txt"]},
                          "id": f"sim-{self._calls}-extract", "type": "tool_call"}]
            response = AIMessage(content="", tool_calls=calls)
        else:
            texts = [text for item in messages[start + 1:] if isinstance(item, ToolMessage)
                     for text in _visible_text(json.loads(item.content))]
            answer = {}
            for key, prefix in question["questions"].items():
                values = {line[len(prefix):].strip() for text in texts for line in text.splitlines() if line.startswith(prefix)}
                answer[key] = next(iter(values)) if len(values) == 1 else None
            response = AIMessage(content=json.dumps(answer), response_metadata={"stop_reason": "end_turn"})
        usage = {"input_tokens": self.get_num_tokens_from_messages(messages),
                 "output_tokens": max(1, len(str(response.content)) // 4)}
        response.usage_metadata = {**usage, "total_tokens": sum(usage.values())}
        return ChatResult(generations=[ChatGeneration(message=response)])


class SimulationWorker:
    """Read only the received brief, then deliberately fail provenance once."""
    def __init__(self, config):
        self.calls = 0
        self.model_name = config.worker_pricing.model

    def get_num_tokens_from_messages(self, messages):
        return max(1, sum(len(str(item.content).encode()) for item in messages) // 4)

    def invoke(self, messages):
        self.calls += 1
        payload = json.loads(messages[-1].content)
        task = payload.get("task", payload)
        # A deterministic fault injection for the second demo question. Worker
        # factories may create a fresh client per request, so this is explicit.
        inject_invalid_quote = "latency" in task["questions"]
        quotes = []
        for question_id, prefix in task["questions"].items():
            for source in payload["sources"]:
                for line, text in enumerate(source["content"].splitlines(keepends=True), 1):
                    if text.startswith(prefix):
                        quotes.append({"question_id": question_id, "source_id": source["id"],
                                       "start_line": line, "end_line": line,
                                       "quote": text + "NOT_IN_SOURCE" if inject_invalid_quote else text})
        content = json.dumps({"quotes": quotes})
        inputs, outputs = self.get_num_tokens_from_messages(messages), max(1, len(content) // 4)
        return AIMessage(content=content, response_metadata={"stop_reason": "end_turn", "model_name": self.model_name,
                         "usage": {"input_tokens": inputs, "output_tokens": outputs,
                                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}},
                         usage_metadata={"input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs})


def demo(directory: Path) -> dict:
    """Exercise actual graph/tools/router/accounting with explicit fixture providers."""
    if directory.exists():
        raise ValueError("Choose a new demo output directory")
    directory.mkdir(parents=True, mode=0o700)
    snapshot = directory / "input"
    snapshot.mkdir()
    (snapshot / "diagnostics.txt").write_text("generation: demo-v1\n" + "diagnostic sample: stable, no action\n" * 180)
    (snapshot / "service.txt").write_text(
        "owner: service-ops\nrollback: error_rate > 5% for 5 minutes\n"
        + "observation: synthetic fixture traffic, unchanged baseline\n" * 220
        + "latency: p95_ms > 400 for 10 minutes\nescalation: page incident commander\n"
    )
    questions = tuple(json.dumps(value) for value in (
        {"kind": "diagnostics", "questions": {"generation": "generation:"}},
        {"kind": "extract", "questions": {"owner": "owner:", "rollback": "rollback:"}},
        {"kind": "extract", "questions": {"latency": "latency:", "escalation": "escalation:"}},
    ))
    main = Pricing(model="simulation-main", input_per_million=3, output_per_million=15,
                   cache_write_5m_per_million=3, cache_write_1h_per_million=3, cache_read_per_million=3)
    worker = Pricing(model="simulation-worker", input_per_million="0.25", output_per_million="1.25",
                     cache_write_5m_per_million="0.25", cache_write_1h_per_million="0.25", cache_read_per_million="0.25")
    reviewed = {"kind": "offline-routing-mechanics", "mode": "simulation", "settings": _settings(main, worker, ARMS, simulation=True),
                "files": file_manifest(Workspace(snapshot)), "questions": questions, "system_prompt": SYSTEM,
                "source_hash": source_hash(), "maximum_authorized_provider_spend_usd": "0",
                "quality_note": "Deterministic fixture consumers verify mechanics and exact facts only. No LLM quality, real dollars, or savings are measured."}
    return _execute(directory, reviewed, model_factory=lambda config: SimulationMain(), worker_factory=SimulationWorker,
                    expected_answers=[{"generation": "demo-v1"},
                                      {"owner": "service-ops", "rollback": "error_rate > 5% for 5 minutes"},
                                      {"latency": "p95_ms > 400 for 10 minutes", "escalation": "page incident commander"}])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "demo"])
    parser.add_argument("--files", type=Path, default=ROOT / "examples/data")
    parser.add_argument("--pricing", type=Path, default=ROOT / "config/pricing.live.example.yaml")
    parser.add_argument("--worker-pricing", type=Path)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--routing-engine", choices=("rules", "jev"), default="rules")
    parser.add_argument("--question", action="append")
    parser.add_argument("--approve-plan")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            if args.output is None:
                raise ValueError("demo requires --output")
            result = demo(args.output)
        else:
            if args.worker_pricing is None:
                raise ValueError("Provide explicit reviewed worker pricing with --worker-pricing")
            reviewed = plan(files=args.files, pricing_path=args.pricing, worker_pricing_path=args.worker_pricing,
                            arm=args.arm, routing_engine=args.routing_engine,
                            questions=tuple(args.question or DEFAULT_QUESTIONS))
            if args.command == "plan":
                result = reviewed
                print("No provider requests made; review both models, all five configurations, inputs and spend caps.", file=sys.stderr)
            else:
                if args.output is None or args.approve_plan is None:
                    raise ValueError("run requires --output and the exact --approve-plan digest")
                result = run(args.output, reviewed, args.approve_plan)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Routing example stopped ({type(exc).__name__}); inspect local configuration or saved summary.\n")


if __name__ == "__main__":
    main()
