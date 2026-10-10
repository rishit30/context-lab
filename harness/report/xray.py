"""Portable offline evidence report. Never overwrites a historical experiment."""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (ValueError, InvalidOperation):
        return None


def _percent(value: Any) -> str:
    number = _number(value)
    return f"{number * 100:.2f}%" if number is not None else "Unknown"


def _money(value: Any) -> str:
    number = _number(value)
    return f"${number:.6f}" if number is not None else "Unknown"


def build_xray_report(result: dict[str, Any], directory: Path, *, source: Path) -> Path:
    """Create a separate report, with explicit unknowns and no remote assets."""
    destination, origin = directory.resolve(), source.resolve()
    if destination == origin or destination.is_relative_to(origin):
        raise ValueError("X-ray reports must be outside the source run directory")
    if destination.exists():
        raise ValueError("X-ray output directory already exists; choose a new directory")
    env = Environment(loader=FileSystemLoader(Path(__file__).parent),
                      autoescape=select_autoescape(["html"]))
    env.filters["percent"] = _percent
    env.filters["money"] = _money
    html = env.get_template("xray.html").render(result=result, source_name=origin.name)
    payload = json.dumps(result, indent=2, allow_nan=False)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "xray.json").write_text(payload + "\n", encoding="utf-8")
    path = destination / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
