from harness.report.file_agent import build_file_agent_report


def test_file_agent_report_escapes_answers_and_does_not_claim_grade(tmp_path):
    row = {"arm": "contextlab", "total_cost_usd": "0.010042", "decision_cost_usd": "0.000042",
           "provider_calls": 2, "questions_completed": 1, "modified_requests_this_process": 0,
           "error_type": None, "answers": ["<script>fetch('/secret')</script>"]}
    plan = {"questions": ["Inspect files"], "settings": {"contextlab": {}, "baseline": {}}}
    html = build_file_agent_report(tmp_path, [row], plan).read_text()
    assert "<script>fetch" not in html
    assert "&lt;script&gt;" in html
    assert "Not performed" in html and "Incomplete matrix" in html
    assert "do not establish savings" in html
    assert "$0.010042" in html
    assert "Missing arms: baseline" in html
    assert not (tmp_path / "summary.json").exists()  # Reporting never rewrites accounting.


def test_unknown_cost_is_not_rendered_as_zero_and_accounting_is_preserved(tmp_path):
    (tmp_path / "summary.json").write_text('{"total_cost_usd": null}')
    row = {"arm": "baseline", "total_cost_usd": None, "questions_completed": 0,
           "error_type": "DatabaseError", "answers": []}
    plan = {"settings": {"baseline": {}}, "questions": ["Inspect files"]}
    html = build_file_agent_report(tmp_path, [row], plan).read_text()
    assert "Unavailable" in html and "$None" not in html
    assert (tmp_path / "summary.json").read_text() == '{"total_cost_usd": null}'
