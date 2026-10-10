"""Portable report for the ungraded, real read-only agent example."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


def build_file_agent_report(directory: Path, results: list[dict], plan: dict) -> Path:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent),
                      autoescape=select_autoescape(["html"]))
    planned = list(plan["settings"])
    total = (str(sum((Decimal(row["total_cost_usd"]) for row in results), Decimal("0")))
             if all(row.get("total_cost_usd") is not None for row in results) else None)
    html = env.get_template("file_agent.html").render(
        results=results, plan=plan, total=total,
        planned=planned, missing=[arm for arm in planned if arm not in {r["arm"] for r in results}],
    )
    path = directory / "report.html"
    path.write_text(html)
    return path
