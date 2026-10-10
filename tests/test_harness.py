from decimal import Decimal
from pathlib import Path

import pytest

from contextlab.core.models import RunResult
from harness.checks import check_reference_fixture, checker_source
from harness.config import experiment_plan, load_config, load_pricing, run_matrix
from harness.report import build_report
from harness.scoreboard import scoreboard
from harness.tasks import TASKS


def test_interleaved_matrix_rotates_first_arm():
    config = load_config()
    matrix = run_matrix(config)
    assert len(matrix) == 9
    assert matrix[0]["arm"] != matrix[3]["arm"]
    assert {row["arm"] for row in matrix[:3]} == set(config.arms)


def test_approval_digest_binds_prices_and_model_and_caps():
    config, pricing = load_config(), load_pricing()
    original = experiment_plan(config, pricing, "demo")
    assert original == experiment_plan(config, pricing, "demo")
    changed = experiment_plan(config.model_copy(update={"session_cap_usd": Decimal("1")}), pricing, "demo")
    assert original["approval_digest"] != changed["approval_digest"]
    assert original["approval_digest"] != experiment_plan(config, pricing, "other")["approval_digest"]
    assert original["approval_digest"] != experiment_plan(config, pricing.model_copy(update={"output_per_million": Decimal("20")}), "demo")["approval_digest"]


@pytest.mark.parametrize("task", TASKS.values(), ids=TASKS.keys())
def test_reference_passes_and_host_guard_rejects_generated_code(tmp_path, task):
    path = tmp_path / "fixture"
    task.create(path)
    with pytest.raises(ValueError):
        check_reference_fixture(task, path)
    (path / "app.py").write_text(task.solution)
    assert check_reference_fixture(task, path)
    assert "test_" in checker_source(task)


def test_failed_runs_contribute_to_success_cost_and_modes_never_mix():
    good = RunResult(session_id="a", arm="baseline", task_id="task", repeat=0,
                     mode="live", solved=True, cost_usd=Decimal("1"))
    failed = good.model_copy(update={"session_id": "b", "solved": False, "cost_usd": Decimal("2")})
    simulation = good.model_copy(update={"session_id": "c", "mode": "simulation"})
    rows = scoreboard([good, failed, simulation])
    live = next(row for row in rows if row["mode"] == "live")
    assert live["cost_per_solved_task"] == "3"
    assert live["pass_rate"] == 0.5
    assert len(rows) == 2
    assert scoreboard([failed])[0]["cost_per_solved_task"] is None


def test_report_labels_simulation_and_escapes_untrusted_text(tmp_path: Path):
    directory = tmp_path / "s"
    directory.mkdir()
    result = RunResult(session_id="s", arm="baseline", task_id="<script>bad()</script>",
                       repeat=0, mode="simulation", solved=False)
    (directory / "result.json").write_text(result.model_dump_json())
    html = build_report(tmp_path).read_text()
    assert "OFFLINE SIMULATION" in html
    assert "<script>bad()</script>" not in html
    assert "&lt;script&gt;" in html


def test_config_rejects_fake_competitor_arm():
    config = load_config().model_dump()
    config["arms"] = ["deepagents"]
    with pytest.raises(ValueError):
        type(load_config()).model_validate(config)
