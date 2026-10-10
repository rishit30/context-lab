"""Audit fixtures and exact grader contracts; no model or provider calls."""
import ast
import hashlib
import json

import pytest

from harness.tasks.file_audit import CURRENT, FILE_AUDIT_TASKS, INITIAL, OUTPUT_NAMES


@pytest.mark.parametrize("task", FILE_AUDIT_TASKS.values(), ids=FILE_AUDIT_TASKS.keys())
def test_deterministic_inputs_cover_all_services_and_fit_full_read(task, tmp_path):
    workspace = tmp_path / "workspace"
    task.create(workspace)
    services = []
    for path, content in task.snapshot(1).items():
        assert (workspace / path).read_text() == content
        assert len(content.splitlines()) <= 2000
        assert len(content.encode()) <= 512_000
        services.extend(json.loads(content)["services"])
    assert len(services) == task.service_count
    assert len({row["service_id"] for row in services}) == task.service_count
    assert task.snapshot(1) == task.snapshot(1)
    assert not any((workspace / name).exists() for name in OUTPUT_NAMES)


@pytest.mark.parametrize("task", FILE_AUDIT_TASKS.values(), ids=FILE_AUDIT_TASKS.keys())
def test_current_snapshot_overwrites_historical_rollback_facts(task, tmp_path):
    workspace = tmp_path / "workspace"
    task.create(workspace)
    old = {row["service_id"]: row for text in task.snapshot(1).values()
           for row in json.loads(text)["services"]}
    task.prepare_phase(workspace, 1)
    current_text = "\n".join((workspace / path).read_text() for path in task.snapshot_paths)
    for service in task.history_targets:
        assert old[service]["rollback"]["approval_ticket"] not in current_text
    assert all((workspace / path).read_text() == content
               for path, content in task.snapshot(2).items())
    task.prepare_phase(workspace, 2)
    task.prepare_phase(workspace, 3)
    assert current_text == "\n".join((workspace / path).read_text() for path in task.snapshot_paths)


def test_long_inventory_is_substantially_larger_than_short_control():
    long_bytes = sum(len(text.encode()) for text in FILE_AUDIT_TASKS["file_audit_long"].snapshot(1).values())
    short_bytes = sum(len(text.encode()) for text in FILE_AUDIT_TASKS["file_audit_short"].snapshot(1).values())
    assert 90_000 <= long_bytes <= 120_000
    assert long_bytes > 5 * short_bytes


@pytest.mark.parametrize("task", FILE_AUDIT_TASKS.values(), ids=FILE_AUDIT_TASKS.keys())
def test_reference_handoff_reconciles_audits_and_requires_original_history(task):
    outputs = {name: json.loads(content) for name, content in task.reference_outputs().items()}
    initial, current = outputs["audit-initial.json"], outputs["audit-current.json"]
    assert initial["snapshot"] == INITIAL and current["snapshot"] == CURRENT
    a = {(row["service"], rule) for row in initial["findings"] for rule in row["rules"]}
    b = {(row["service"], rule) for row in current["findings"] for rule in row["rules"]}
    handoff = outputs["handoff.json"]
    assert handoff["resolved_findings"] == len(a - b) > 0
    assert handoff["introduced_findings"] == len(b - a) > 0
    assert handoff["unchanged_findings"] == len(a & b) > 0
    assert all(row["initial_image_digest"] != row["current_image_digest"]
               for row in outputs["history.json"]["requests"])


@pytest.mark.parametrize("task", FILE_AUDIT_TASKS.values(), ids=FILE_AUDIT_TASKS.keys())
def test_checker_embeds_exact_answers_and_snapshot_integrity_outside_workspace(task, tmp_path):
    workspace = tmp_path / "workspace"
    task.create(workspace)
    source = task.checker_source()
    tree = ast.parse(source)
    # Inspect literal grader data without executing generated source on the host.
    expected_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                         and node.targets[0].id == "expected")
    expected = json.loads(ast.literal_eval(expected_node.value.args[0]))
    assert expected == {name: json.loads(text) for name, text in task.reference_outputs().items()}
    hash_node = next(node for node in tree.body if isinstance(node, ast.Assign)
                     and node.targets[0].id == "snapshot_hashes")
    assert ast.literal_eval(hash_node.value) == {
        path: hashlib.sha256(text.encode()).hexdigest() for path, text in task.snapshot(2).items()
    }
    assert "json.dumps(actual, sort_keys=True)" in source and "Input snapshot modified" in source
    assert all("expected =" not in path.read_text() for path in workspace.rglob("*.py"))
    for phase in range(4):
        ast.parse(task.phase_check_source(phase))


def test_invalid_versions_and_phase_indexes_are_rejected(tmp_path):
    task = FILE_AUDIT_TASKS["file_audit_short"]
    with pytest.raises(ValueError):
        task.snapshot(0)
    with pytest.raises(ValueError):
        task.prepare_phase(tmp_path, 4)
    with pytest.raises(ValueError):
        task.phase_check_source(-1)


def test_known_services_exercise_all_policy_rules_and_real_new_regressions():
    task = FILE_AUDIT_TASKS["file_audit_long"]
    initial = json.loads(task.reference_outputs()["audit-initial.json"])
    current = json.loads(task.reference_outputs()["audit-current.json"])
    old = {row["service"]: row["rules"] for row in initial["findings"]}
    new = {row["service"]: row["rules"] for row in current["findings"]}
    assert old["svc-000"] == [f"R{index:02d}" for index in range(1, 11)]
    assert new["svc-000"] == [f"R{index:02d}" for index in range(2, 11)]
    assert "svc-001" not in old and "svc-001" not in new
    assert "svc-002" not in old and new["svc-002"] == ["R02"]
    assert "svc-005" not in old and new["svc-005"] == ["R03"]
    assert old["svc-017"] == ["R01"] and "svc-017" not in new
    assert old["svc-029"] == ["R09"] and "svc-029" not in new
