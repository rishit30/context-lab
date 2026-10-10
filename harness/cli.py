"""CLI defaults to offline inspection; live inference requires an exact approved plan."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from harness.config import ROOT, experiment_plan, load_compaction, load_config, load_pricing
from harness.environment import load_anthropic_key, load_typesafe_key
from harness.report import build_report
from harness.scoreboard import load_results, scoreboard


def main() -> None:
    parser = argparse.ArgumentParser(prog="contextlab")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a labeled offline simulation; no API key required")
    demo.add_argument("--output", type=Path)
    report = commands.add_parser("report", help="Regenerate HTML, JSON and CSV reports without inference")
    report.add_argument("directory", type=Path)
    audit = commands.add_parser("audit", help="Read-only ledger/receipt integrity and cost reconciliation")
    audit.add_argument("directory", type=Path)
    xray = commands.add_parser(
        "xray", help="Analyze saved baseline context offline; no model calls or savings claims"
    )
    xray.add_argument("directory", type=Path)
    xray.add_argument(
        "--output", type=Path,
        help="Create JSON and portable HTML in a NEW report directory; preserve the source run",
    )
    commands.add_parser("jev-models", help="List accessible Jev models; no inference is requested")
    for name in ("plan", "run", "check"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=ROOT / "config/experiment.yaml")
        command.add_argument("--pricing", type=Path, required=True)
        command.add_argument("--model", default="claude-sonnet-4-6")
        if name == "run":
            command.add_argument("--approve-plan", required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from harness.demo import demo_session
            output = args.output or ROOT / "runs" / ("demo-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
            output.mkdir(parents=True, exist_ok=False)
            for task in ("retry_policy", "inventory", "config_parser"):
                for arm in ("baseline", "threshold_edit", "contextlab"):
                    demo_session(output, arm, task, 0, load_pricing(), load_compaction())
            print("OFFLINE SIMULATION: no API calls; $0 provider spend; synthetic stress traces.")
            print(json.dumps(scoreboard(load_results(output)), indent=2))
            print(f"Report: {build_report(output)}")
        elif args.command == "report":
            print(build_report(args.directory))
        elif args.command == "audit":
            from harness.audit import audit_run
            result = audit_run(args.directory)
            print(json.dumps(result, indent=2, default=str))
            if not result["ok"]:
                raise SystemExit(2)
        elif args.command == "xray":
            from harness.xray import analyze_run
            result = analyze_run(args.directory)
            print(json.dumps(result, indent=2, default=str))
            if args.output is not None:
                from harness.report.xray import build_xray_report
                destination = build_xray_report(result, args.output, source=args.directory)
                print(f"Report: {destination}", file=sys.stderr)
            if not result["ok"]:
                raise SystemExit(2)
        elif args.command == "jev-models":
            from contextlab.providers.jev import JevClient
            load_typesafe_key(ROOT / ".env")
            client = JevClient.from_environment()
            print(json.dumps(client.list_models(), indent=2))
        else:
            from harness.runner import preflight, run_live
            config, pricing = load_config(args.config), load_pricing(args.pricing)
            plan = experiment_plan(config, pricing, args.model)
            if args.command == "plan":
                print(json.dumps(plan, indent=2))
                print("No inference executed. Review prices, matrix, and maximum spend before approving this digest.", file=sys.stderr)
            elif args.command == "check":
                load_anthropic_key(ROOT / ".env")
                if getattr(config, "decision_engine", "rules") == "jev":
                    load_typesafe_key(ROOT / ".env")
                issues = preflight(config, pricing, args.model)
                print(json.dumps({"ready": not issues, "issues": issues}, indent=2))
                if issues:
                    raise SystemExit(2)
            else:
                load_anthropic_key(ROOT / ".env")
                if getattr(config, "decision_engine", "rules") == "jev":
                    load_typesafe_key(ROOT / ".env")
                run_live(args.output, config, pricing, args.model, args.approve_plan)
                print(build_report(args.output))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"ContextLab: {exc}\n")
