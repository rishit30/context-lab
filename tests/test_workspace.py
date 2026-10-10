import json
from pathlib import Path

import pytest

from harness.workspace import Workspace


def test_read_metadata_tracks_partial_and_version(tmp_path):
    (tmp_path / "sample.py").write_text("one\ntwo\nthree\n")
    workspace = Workspace(tmp_path)
    first = json.loads(workspace.read_file("sample.py", 1, 1))
    assert first["content"] == "two\n" and not first["full_read"]
    second = json.loads(workspace.read_file("sample.py"))
    assert second["full_read"] and first["sha256"] == second["sha256"]
    workspace.edit_file("sample.py", "two", "changed")
    assert json.loads(workspace.read_file("sample.py"))["sha256"] != first["sha256"]


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", ".env", ".aws/credentials", ".git/config"])
def test_paths_are_confined(tmp_path, path):
    with pytest.raises(ValueError):
        Workspace(tmp_path).write_file(path, "bad")


def test_symlink_rejected_even_when_target_inside_workspace(tmp_path):
    (tmp_path / "inside").write_text("original")
    (tmp_path / "link").symlink_to(tmp_path / "inside")
    with pytest.raises(ValueError):
        Workspace(tmp_path).write_file("link", "changed")
    assert (tmp_path / "inside").read_text() == "original"


def test_no_host_execution_and_no_shell_syntax(tmp_path):
    workspace = Workspace(tmp_path)
    with pytest.raises(RuntimeError, match="sandbox"):
        workspace.run_tests()
    with pytest.raises(ValueError):
        workspace.shell("cat /etc/passwd; pwd")


def test_test_execution_uses_fixed_argv_and_fixture_only(tmp_path):
    (tmp_path / "tests").mkdir()
    commands = []

    def executor(argv: list[str], cwd: Path):
        commands.append((argv, cwd))
        return {"exit_code": 1, "stdout": "failed", "stderr": ""}

    workspace = Workspace(tmp_path, executor=executor)
    result = json.loads(workspace.run_tests())
    assert result["exit_code"] == 1
    assert commands == [(["python", "-m", "pytest", "-q", "--", "tests"], tmp_path)]
    with pytest.raises(ValueError):
        workspace.run_tests("../other")


def test_partial_recall_through_read_file(tmp_path):
    seen = []
    workspace = Workspace(tmp_path, recall_reader=lambda *args: seen.append(args) or "old text")
    assert workspace.read_file("/context/c123.txt", 2, 4) == "old text"
    assert seen == [("c123", 2, 4, 0)]
    with pytest.raises(ValueError):
        workspace.read_file("/context/../secrets")


def test_literal_search_cannot_execute_option_or_shell(tmp_path):
    (tmp_path / "sample.txt").write_text("hello $(cat secret)\n--help\n")
    workspace = Workspace(tmp_path)
    assert "sample.txt" in json.loads(workspace.search("$(cat secret)"))["matches"]
    assert "--help" in json.loads(workspace.search("--help"))["matches"]
