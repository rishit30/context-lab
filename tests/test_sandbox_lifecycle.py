"""Exercise timeout/output cleanup with trusted local processes, without Docker."""
import subprocess
import sys

from harness import sandbox


def simulated_docker(monkeypatch, script):
    real_popen = subprocess.Popen
    commands, cleanups = [], []

    def launch(command, **kwargs):
        commands.append(command)
        return real_popen([sys.executable, "-c", script], **kwargs)

    def cleanup(command, **kwargs):
        cleanups.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox.subprocess, "Popen", launch)
    monkeypatch.setattr(sandbox.subprocess, "run", cleanup)
    return commands, cleanups


def test_success_removes_named_container_and_preserves_isolation_flags(tmp_path, monkeypatch):
    commands, cleanups = simulated_docker(monkeypatch, 'print("CHECKS_PASSED")')
    result = sandbox.DockerExecutor(tmp_path, "fixture-image").run(["python", "-"])
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "CHECKS_PASSED"
    command = commands[0]
    for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--user=65534:65534"):
        assert flag in command
    assert cleanups == [["docker", "rm", "-f", command[command.index("--name") + 1]]]


def test_output_is_bounded_and_container_stopped(tmp_path, monkeypatch):
    _, cleanups = simulated_docker(monkeypatch, 'print("x" * 200000)')
    monkeypatch.setattr(sandbox, "OUTPUT_LIMIT_BYTES", 1024)
    result = sandbox.DockerExecutor(tmp_path, "fixture-image").run(["python", "-"])
    assert result["exit_code"] == 125
    assert len(result["stdout"].encode()) <= 1024
    assert len(cleanups) == 1


def test_timeout_stops_container_not_only_client(tmp_path, monkeypatch):
    _, cleanups = simulated_docker(monkeypatch, "import time; time.sleep(10)")
    monkeypatch.setattr(sandbox, "EXECUTION_TIMEOUT_SECONDS", 0.05)
    result = sandbox.DockerExecutor(tmp_path, "fixture-image").run(["python", "-"])
    assert result["exit_code"] == 124
    assert len(cleanups) == 1
