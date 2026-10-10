"""Offline report/CLI checks; no keys, providers, or historical-run writes."""
import json

import pytest

from harness.report.xray import build_xray_report


def analysis(*, ok=True):
    row = {
        "task_id": "audit <script>alert(1)</script>", "mode": "live",
        "session_id": "s", "baseline_calls": 4, "reconstructed_calls": 3,
        "estimated_candidate_context_share": "0.125", "actual_input_cost_usd": "0.02",
        "priced_candidate_input_estimate_usd": {"lower": "0.001", "upper": "0.01"},
        "coverage_complete": False,
    }
    return {
        "ok": ok, "summary": row, "per_task": [row], "per_session": [row],
        "calls": [{"session_id": "s", "seq": 0, "status": "unknown",
                   "reasons": ["<img src=x onerror=alert(1)>"]}],
        "errors": [] if ok else ["Malformed trace"], "warnings": ["Incomplete evidence"],
        "limitations": ["Token estimates are a heuristic, not measured savings."],
    }


def test_xray_report_is_portable_escaped_and_preserves_unknowns(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "trace.json"
    marker.write_text('{"preserve": true}')
    result = analysis()
    result["summary"] = {**result["summary"], "actual_input_cost_usd": None,
                         "estimated_candidate_context_share": None}
    path = build_xray_report(result, tmp_path / "report", source=source)
    html = path.read_text()
    assert json.loads(path.with_name("xray.json").read_text()) == result
    assert marker.read_text() == '{"preserve": true}'
    assert "<script>" not in html and "<img src=x" not in html
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert "Unknown" in html and "12.50%" in html
    assert "Coverage is incomplete" in html
    assert "system prompts and tool definitions" in html
    assert "not achievable savings" in html
    assert 'src="http' not in html and 'href="http' not in html


def test_xray_report_refuses_to_overwrite_or_modify_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for output in (source, source / "child"):
        with pytest.raises(ValueError, match="outside the source"):
            build_xray_report(analysis(), output, source=source)
    output = tmp_path / "report"
    output.mkdir()
    (output / "report.html").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        build_xray_report(analysis(), output, source=source)
    assert (output / "report.html").read_text() == "keep"
    assert not (source / "child").exists()


def test_xray_report_rejects_non_finite_json_before_writing(tmp_path):
    result = analysis()
    result["summary"]["estimated_candidate_context_share"] = float("nan")
    destination = tmp_path / "report"
    with pytest.raises(ValueError):
        build_xray_report(result, destination, source=tmp_path / "source")
    assert not destination.exists()


@pytest.mark.parametrize("ok", [True, False])
def test_xray_cli_never_loads_keys_or_runs_providers(monkeypatch, tmp_path, capsys, ok):
    from harness import cli, xray

    result = analysis(ok=ok)
    source, output = tmp_path / "source", tmp_path / "analysis"
    seen = []

    def analyze(path):
        seen.append(path)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Offline X-ray must not load credentials or contact a provider")

    monkeypatch.setattr(xray, "analyze_run", analyze)
    monkeypatch.setattr(cli, "load_anthropic_key", forbidden)
    monkeypatch.setattr(cli, "load_typesafe_key", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("sys.argv", ["contextlab", "xray", str(source), "--output", str(output)])
    if ok:
        cli.main()
    else:
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert seen == [source]
    assert "Report:" in captured.err and (output / "report.html").exists()


def test_xray_cli_defaults_to_stdout_only(monkeypatch, tmp_path, capsys):
    from harness import cli, xray

    result = analysis()
    monkeypatch.setattr(xray, "analyze_run", lambda _: result)
    monkeypatch.setattr("sys.argv", ["contextlab", "xray", str(tmp_path)])
    cli.main()
    assert json.loads(capsys.readouterr().out) == result
    assert list(tmp_path.iterdir()) == []
