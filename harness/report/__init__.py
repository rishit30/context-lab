from __future__ import annotations

import csv
import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from harness.scoreboard import load_results, scoreboard


def build_report(directory: Path) -> Path:
    results = load_results(directory)
    traces = []
    for result in results:
        path = directory / result.session_id / "trace.json"
        data = json.loads(path.read_text()) if path.exists() else {}
        traces.append({"result": result.model_dump(mode="json"), "trace": data})
    by_session = {item["result"]["session_id"]: item["trace"] for item in traces}
    rows = scoreboard(results, by_session)
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    completed = {(item.task_id, item.arm, item.repeat) for item in results}
    missing = [row for row in plan.get("matrix", [])
               if (row["task_id"], row["arm"], row["repeat"]) not in completed]
    # Small portable summaries intentionally omit raw prompts and tool contents.
    summary = {
        "cost_basis": "Provider usage × frozen list prices; not a reconciled invoice. Simulation rows are synthetic.",
        "approval_digest": plan.get("approval_digest"),
        "model": plan.get("model"), "pricing": plan.get("pricing"),
        "headroom": plan.get("headroom"), "rows": rows,
        "decision_engine": plan.get("config", {}).get("decision_engine", "rules"),
        "decision_layer": plan.get("decision_layer"),
        "decision_cost_basis": "Decision-call charges are already included in total cost; the split is not an additional charge.",
        "missing_sessions": missing,
        "sessions": [item.model_dump(mode="json") for item in results],
        "task_rows": [
            {"task_id": task, **row}
            for task in sorted({item.task_id for item in results})
            for row in scoreboard([item for item in results if item.task_id == task], by_session)
        ],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    flat_rows = [{**{key: value for key, value in row.items()
                    if key not in {"usage", "decision_usage", "cache_modes"}},
                  "cache_modes": ",".join(row["cache_modes"]), **row["usage"],
                  **{f"decision_{key}": value for key, value in row["decision_usage"].items()}}
                 for row in summary["task_rows"]]
    with (directory / "summary.csv").open("w", newline="") as stream:
        if flat_rows:
            writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    env = Environment(loader=FileSystemLoader(Path(__file__).parent),
                      autoescape=select_autoescape(["html"]))
    html = env.get_template("index.html").render(
        rows=rows, traces=traces, plan=plan, missing=missing,
        simulation=all(item.mode == "simulation" for item in results),
        total_runs=len(results),
    )
    destination = directory / "report.html"
    destination.write_text(html)
    return destination
