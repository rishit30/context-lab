"""Disposable Docker execution. Workspace path confinement alone is not a sandbox."""
from __future__ import annotations

import os
import selectors
import shutil
import subprocess
import time
from pathlib import Path
from uuid import uuid4

EXECUTION_TIMEOUT_SECONDS = 40
OUTPUT_LIMIT_BYTES = 70000


class DockerExecutor:
    def __init__(self, workspace: Path, image: str):
        self.workspace = workspace.resolve()
        self.image = image

    def __call__(self, argv: list[str], cwd: Path) -> dict:
        if cwd.resolve() != self.workspace:
            raise ValueError("Executor cwd must match the isolated task workspace")
        if len(argv) != 6 or argv[:5] != ["python", "-m", "pytest", "-q", "--"]:
            raise ValueError("Only the fixed public-test command is permitted")
        target = (self.workspace / argv[5]).resolve()
        if not target.is_relative_to(self.workspace) or not target.exists():
            raise ValueError("Test selection must remain in the workspace")
        return self.run(argv)

    def run(self, argv: list[str], *, stdin: str | None = None) -> dict:
        # No API credentials, host home, Docker socket or network enter the container.
        container_name = "contextlab-" + uuid4().hex
        command = [
            "docker", "run", "--name", container_name, "--rm", "--init", "--network=none", "--read-only",
            "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64",
            "--memory=256m", "--cpus=1", "--user=65534:65534",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=32m", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "--mount", f"type=bind,src={self.workspace},dst=/workspace,readonly",
            "--workdir=/workspace",
        ]
        if stdin is not None:
            command.append("-i")
        command.extend([self.image, *argv])
        process = subprocess.Popen(command, stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = bytearray()
        deadline, forced_code = time.monotonic() + EXECUTION_TIMEOUT_SECONDS, None
        selector = selectors.DefaultSelector()
        try:
            if stdin is not None and process.stdin is not None:
                process.stdin.write(stdin.encode())
                process.stdin.close()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic() >= deadline:
                    forced_code = 124
                    break
                for key, _ in selector.select(timeout=0.1):
                    block = os.read(key.fd, 8192)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    output.extend(block[:max(0, OUTPUT_LIMIT_BYTES - len(output))])
                    if len(output) >= OUTPUT_LIMIT_BYTES:
                        forced_code = 125
                        break
                if forced_code is not None:
                    break
            if forced_code is not None:
                process.kill()
            return_code = process.wait(timeout=5)
            return {"exit_code": forced_code or return_code,
                    "stdout": output.decode(errors="replace"),
                    "stderr": "Execution time/output limit reached" if forced_code else ""}
        finally:
            selector.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            # Killing the CLI alone does not stop a detached/running container.
            subprocess.run(["docker", "rm", "-f", container_name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)


def docker_status(image: str) -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "Docker is not installed; live generated code requires an isolated executor."
    try:
        check = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False, "Docker daemon is unavailable."
    if check.returncode:
        return False, f"Build the task image first: docker build -f Dockerfile.task -t {image} ."
    return True, "Docker task image available"
