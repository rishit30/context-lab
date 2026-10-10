"""Exercise the real task container with trusted fixtures; no model/API calls.

Run from the project root: uv run python -m scripts.verify_docker
Reference solutions are written only to fresh, disposable runs/ workspaces.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from harness.checks import checker_source
from harness.sandbox import DockerExecutor, docker_status
from harness.tasks import TASKS

ISOLATION_PROBE = '''
import json
import os
from pathlib import Path

assert os.getuid() == 65534, "Unexpected container user"
assert not Path("/var/run/docker.sock").exists(), "Docker socket is exposed"
assert not any(name in os.environ for name in
               ("ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")), \
       "Host credentials are exposed"
interfaces = sorted(path.name for path in Path("/sys/class/net").iterdir())
assert interfaces == ["lo"], f"Unexpected network interfaces: {interfaces}"
status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines()
              if ":" in line)
assert int(status["CapEff"].strip(), 16) == 0, "Unexpected capabilities"
assert status["NoNewPrivs"].strip() == "1", "Privilege escalation is permitted"
for location in ("/workspace/.contextlab-write-probe", "/contextlab-write-probe"):
    try:
        Path(location).write_text("must fail")
    except OSError:
        pass
    else:
        raise AssertionError(f"Read-only path is writable: {location}")
temporary = Path("/tmp/contextlab-write-probe")
temporary.write_text("allowed temporary storage")
assert temporary.read_text() == "allowed temporary storage"
temporary.unlink()
print(json.dumps({"uid": os.getuid(), "interfaces": interfaces,
                  "read_only_workspace": True, "read_only_root": True,
                  "capabilities": 0, "no_new_privileges": True,
                  "temporary_storage": "writable", "credentials": "absent"}))
'''


def container_names() -> set[str]:
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", "name=contextlab-", "--format", "{{.Names}}"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return set(result.stdout.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="contextlab-task:local")
    args = parser.parse_args()
    available, reason = docker_status(args.image)
    if not available:
        print(reason)
        return 1
    project = Path(__file__).resolve().parents[1]
    run_name = datetime.now(timezone.utc).strftime("docker-smoke-%Y%m%dT%H%M%SZ-")
    output = project / "runs" / (run_name + uuid4().hex[:8])
    output.mkdir(parents=True)
    image = subprocess.run(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    initial_containers = container_names()
    records: list[dict] = []

    def record(name: str, result: dict, expected_exit: int, marker: str = "") -> None:
        passed = result["exit_code"] == expected_exit and marker in result["stdout"]
        records.append({"check": name, "passed": passed,
                        "expected_exit": expected_exit, **result})
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    try:
        for task in TASKS.values():
            workspace = output / task.id
            task.create(workspace)
            executor = DockerExecutor(workspace, image)
            command = ["python", "-m", "pytest", "-q", "--", "tests"]
            record(f"{task.id}: seeded public failure", executor(command, workspace), 1)
            (workspace / "app.py").write_text(task.phase_one_solution)
            record(f"{task.id}: phase-one public pass", executor(command, workspace), 0)
            record(f"{task.id}: held-out catches incomplete fix",
                   executor.run(["python", "-"], stdin=checker_source(task)), 1)
            (workspace / "app.py").write_text(task.solution)
            record(f"{task.id}: reference public pass", executor(command, workspace), 0)
            record(f"{task.id}: reference independent pass",
                   executor.run(["python", "-"], stdin=checker_source(task)),
                   0, "CHECKS_PASSED")
        record("container isolation", executor.run(["python", "-"], stdin=ISOLATION_PROBE), 0)
    except Exception as error:
        records.append({"check": "verification execution", "passed": False,
                        "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            leftovers = sorted(container_names() - initial_containers)
            records.append({"check": "container cleanup", "passed": not leftovers,
                            "new_remaining_containers": leftovers})
        except Exception as error:
            records.append({"check": "container cleanup", "passed": False,
                            "error": f"{type(error).__name__}: {error}"})
        report = {"mode": "docker_reference_smoke", "image_tag": args.image,
                  "image_id": image, "provider_calls": 0, "checks": records,
                  "passed": all(item["passed"] for item in records)}
        report_path = output / "verification.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report: {report_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
