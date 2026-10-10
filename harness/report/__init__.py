from __future__ import annotations

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
    env = Environment(loader=FileSystemLoader(Path(__file__).parent),
                      autoescape=select_autoescape(["html"]))
    html = env.get_template("index.html").render(
        rows=scoreboard(results), traces=traces,
        simulation=all(item.mode == "simulation" for item in results),
        total_runs=len(results),
    )
    destination = directory / "report.html"
    destination.write_text(html)
    return destination
