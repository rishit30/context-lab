"""Independent checks never placed in the agent-editable workspace."""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from harness.tasks import TaskSpec


def checker_source(task: TaskSpec) -> str:
    # Include the original public assertions, independent of agent edits to test files.
    names = [node.name for node in ast.parse(task.public_tests).body
             if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")]
    return ("import app\n" + task.hidden_checks + "\n" + task.public_tests + "\n"
            + "\n".join(f"{name}()" for name in names) + '\nprint("CHECKS_PASSED")\n')


def check_reference_fixture(task: TaskSpec, workspace: Path) -> bool:
    """Offline only: execute our trusted reference code, never a model's code on the host."""
    if (workspace / "app.py").read_text() != task.solution:
        raise ValueError("Host execution is permitted only for the bundled reference solution")
    result = subprocess.run(
        [sys.executable, "-c", checker_source(task)], cwd=workspace,
        capture_output=True, text=True, timeout=10,
        env={"PATH": str(Path(sys.executable).parent), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode == 0 and result.stdout.strip() == "CHECKS_PASSED"
