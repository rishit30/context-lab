import csv
import json
from decimal import Decimal

from contextlab.core.models import RunResult, Usage
from harness.report import build_report
from harness.scoreboard import scoreboard


def test_exports_actual_usage_failures_and_task_specific_totals(tmp_path):
    for index, (task, solved, changed) in enumerate([("long", True, True), ("short", False, False)]):
        session = f"s{index}"
        folder = tmp_path / session
        folder.mkdir()
        result = RunResult(session_id=session, arm="headroom", task_id=task, repeat=0,
                           mode="live", solved=solved, cost_usd=Decimal("0.12"), turns=2,
                           usage=Usage(input_tokens=100, output_tokens=20), latency_ms=1000)
        (folder / "result.json").write_text(result.model_dump_json())
        (folder / "trace.json").write_text(json.dumps({
            "interventions": [{"changed": changed, "compression_ms": 3}],
            "adapter_failures": [] if solved else ["ValueError"], "cache_mode": "disabled",
            "requests": [{"seq": 0, "counted_input_tokens": 100,
                          "actual_rendered_messages": "PRIVATE TRACE MARKER"}],
        }))
    html = build_report(tmp_path).read_text()
    summary = json.loads((tmp_path / "summary.json").read_text())
    row = summary["rows"][0]
    assert row["modified_requests"] == 1 and row["adapter_failures"] == 1
    assert row["cost_per_solved_task"] == "0.24" and row["latency_ms"] == 2000
    assert row["usage"]["input_tokens"] == 200 and row["usage"]["output_tokens"] == 40
    assert "PRIVATE TRACE MARKER" not in (tmp_path / "summary.json").read_text()
    with (tmp_path / "summary.csv").open() as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 2 and {item["task_id"] for item in exported} == {"long", "short"}
    assert exported[1]["cost_per_solved_task"] == ""  # No successful short run.
    assert "Modified requests" in html and "summary.csv" in html


def test_legacy_missing_telemetry_is_unknown_and_baseline_zero_is_explicit():
    result = RunResult(session_id="x", arm="baseline", task_id="t", repeat=0, mode="live", solved=False)
    assert scoreboard([result], {"x": {}})[0]["modified_requests"] is None
    measured = scoreboard([result], {"x": {"interventions": [], "adapter_failures": [], "cache_mode": "disabled"}})[0]
    assert measured["modified_requests"] == 0 and measured["middleware_ms"] == 0
    blocked = scoreboard([result], {"x": {"interventions": [{"changed": True, "admitted": False}]}})[0]
    assert blocked["modified_requests"] == 0


def test_incomplete_matrix_is_explicit_in_export_and_html(tmp_path):
    missing = {"task_id": "long", "arm": "headroom", "repeat": 0}
    (tmp_path / "plan.json").write_text(json.dumps({"matrix": [missing]}))
    html = build_report(tmp_path).read_text()
    assert "INCOMPLETE MATRIX" in html
    assert json.loads((tmp_path / "summary.json").read_text())["missing_sessions"] == [missing]
