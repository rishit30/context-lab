"""Failures remain in the denominator and in total spend."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from contextlab.core.models import RunResult, Usage


def load_results(directory: Path) -> list[RunResult]:
    return [RunResult.model_validate_json(path.read_text())
            for path in sorted(directory.glob("*/result.json"))]


def scoreboard(results: list[RunResult], traces: dict[str, dict] | None = None) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[(result.mode, result.arm)].append(result)
    rows = []
    for (mode, arm), group in sorted(groups.items()):
        spent = sum((item.cost_usd for item in group), Decimal("0"))
        solved = sum(item.solved for item in group)
        usage = {field: sum(getattr(item.usage, field) for item in group) for field in Usage.model_fields}
        decision_usage = {field: sum(getattr(item.decision_usage, field) for item in group)
                          for field in Usage.model_fields}
        total_input = sum(value for field, value in usage.items() if field != "output_tokens")
        observed = [traces.get(item.session_id, {}) for item in group] if traces is not None else []
        complete = len(observed) == len(group) and all("interventions" in trace for trace in observed)
        interventions = [entry for trace in observed for entry in trace.get("interventions", [])]
        rows.append({"mode": mode, "arm": arm, "runs": len(group), "solved": solved,
                     "pass_rate": solved / len(group), "total_cost_usd": str(spent),
                     "cost_per_solved_task": str(spent / solved) if solved else None,
                     "cache_hit_ratio": usage["cache_read_tokens"] / total_input if total_input else None,
                     "turns": sum(item.turns for item in group),
                     "decision_calls": sum(item.decision_calls for item in group),
                     "decision_cost_usd": str(sum((item.decision_cost_usd for item in group), Decimal("0"))),
                     "decision_usage": decision_usage,
                     "total_input_tokens": total_input,
                     "latency_ms": sum(item.latency_ms for item in group),
                     "modified_requests": sum(bool(entry.get("changed")) and entry.get("admitted", True)
                                              for entry in interventions) if complete else None,
                     "middleware_ms": sum(entry.get("latency_ms", entry.get("compression_ms", 0))
                                          for entry in interventions) if complete else None,
                     "adapter_failures": sum(len(trace.get("adapter_failures", [])) for trace in observed) if complete else None,
                     "cache_modes": sorted({trace["cache_mode"] for trace in observed if "cache_mode" in trace}),
                     "rebuilds": sum(item.rebuilds for item in group),
                     "recalls": sum(item.recalls for item in group), "usage": usage,
                     "evidence": "SIMULATION — no inference" if mode == "simulation" else "Indicative pilot; no superiority established"})
    return rows
