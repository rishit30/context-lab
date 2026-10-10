"""Local comparison report; simulation and live accounting stay visibly distinct."""
from __future__ import annotations

from html import escape
from pathlib import Path


def build_routing_report(directory: Path, result: dict) -> Path:
    simulation = result["mode"] == "simulation"
    label = "OFFLINE SIMULATION" if simulation else "LIVE PROVIDER RUN"
    rows = []
    for session in result["sessions"]:
        cells = [session["configuration"], session.get("questions_completed", 0),
                 session.get("provider_calls", "Unknown"), session.get("delegate_calls", "Unknown"),
                 session.get("total_cost_usd", "Unknown"),
                 session.get("mechanical_fixture_passed") if simulation else "Not independently graded",
                 session.get("error_type") or session.get("close_error_type") or "None"]
        rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in cells) + "</tr>")
    details = "".join(
        f"<details><summary>{escape(row['configuration'])} answers and routing outcomes</summary>"
        + "".join(f"<pre>{escape(str(answer))}</pre>" for answer in row.get("answers", []))
        + f"<pre>{escape(str(row.get('tool_outcomes', [])))}</pre></details>"
        for row in result["sessions"]
    )
    cost_label = "Synthetic accounting units" if simulation else "Usage cost at reviewed list prices (USD)"
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ContextLab routing comparison</title><style>
body{{font:16px system-ui,sans-serif;margin:40px auto;max-width:1180px;padding:0 24px;color:#17312b;background:#f7f8f5}}
h1{{font-size:34px;margin-bottom:12px}}.label{{font-weight:750;color:#946315}}p{{line-height:1.6;max-width:85ch}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce4df}}th{{background:#e9f0ec}}
details{{margin:16px 0;padding:14px;background:white}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}.note{{border-left:4px solid #947126;padding:12px;background:#fff8e9}}
</style><main><div class="label">{label}</div><h1>Read-only delegation, five configurations</h1>
<p>Every configuration uses the same main model, prompt, tools and frozen input files. A cheaper worker may select exact cited excerpts; the main agent always continues. Unusable worker output falls back to original evidence.</p>
<p class="note">{escape(result['quality_note'])}</p>
<p><strong>{cost_label}:</strong> {escape(str(result.get('total_cost_usd')))} · {escape(result['cost_basis'])}.
Main, decision and worker calls—including failed work—are included.</p>
<div class="scroll"><table><thead><tr><th>Configuration</th><th>Questions</th><th>All model calls</th><th>Worker calls</th><th>{cost_label}</th><th>Mechanical fixture check</th><th>Error type</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Exact quote validation proves source text and line provenance. It does not prove semantic relevance or exhaustive answers. These results establish no optimization savings.</p>{details}
</main></html>"""
    path = directory / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
