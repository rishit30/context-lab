"""CLI defaults to offline inspection; live inference requires an exact approved plan."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from harness.config import ROOT, experiment_plan, load_compaction, load_config, load_pricing
from harness.environment import load_anthropic_key
from harness.report import build_report
from harness.scoreboard import load_results, scoreboard


def main() -> None:
    parser = argparse.ArgumentParser(prog="contextlab")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a labeled offline simulation; no API key required")
    demo.add_argument("--output", type=Path)
    report = commands.add_parser("report", help="Regenerate a static HTML report")
    report.add_argument("directory", type=Path)
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
        else:
            from harness.runner import preflight, run_live
            config, pricing = load_config(args.config), load_pricing(args.pricing)
            plan = experiment_plan(config, pricing, args.model)
            if args.command == "plan":
                print(json.dumps(plan, indent=2))
                print("No inference executed. Review prices, matrix, and maximum spend before approving this digest.", file=sys.stderr)
            elif args.command == "check":
                load_anthropic_key(ROOT / ".env")
                issues = preflight(config, pricing, args.model)
                print(json.dumps({"ready": not issues, "issues": issues}, indent=2))
                if issues:
                    raise SystemExit(2)
            else:
                load_anthropic_key(ROOT / ".env")
                run_live(args.output, config, pricing, args.model, args.approve_plan)
                print(build_report(args.output))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"ContextLab: {exc}\n")
