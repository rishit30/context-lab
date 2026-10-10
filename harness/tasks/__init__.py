"""Small diagnostic fixtures. Reference solutions are NEVER copied to live agent workspaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    initial: str
    phase_one_solution: str
    solution: str
    public_tests: str
    hidden_checks: str
    phase_one: str
    phase_two: str

    def create(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "app.py").write_text(self.initial)
        (directory / "tests").mkdir()
        (directory / "tests/test_public.py").write_text(self.public_tests)
        (directory / "README.md").write_text(
            "Diagnostic ContextLab fixture. Implement requirements in app.py; preserve public tests.\n"
        )


TASKS = {
    "retry_policy": TaskSpec(
        id="retry_policy", title="Repair a retry policy",
        initial='def retry_delay(attempt, base=1, cap=30):\n    return min(cap, base * attempt)\n',
        phase_one_solution='def retry_delay(attempt, base=1, cap=30):\n    return min(cap, base * 2 ** attempt)\n',
        solution=('def retry_delay(attempt, base=1, cap=30):\n'
                  '    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:\n'
                  '        raise ValueError("attempt must be a nonnegative integer")\n'
                  '    if base < 0 or cap < 0:\n        raise ValueError("negative bound")\n'
                  '    return min(cap, base * 2 ** attempt)\n'),
        public_tests=('from app import retry_delay\n'
                      'def test_exponential():\n    assert retry_delay(3, 2, 30) == 16\n'
                      'def test_first_attempt():\n    assert retry_delay(0) == 1\n'),
        hidden_checks=('assert app.retry_delay(4, 2, 10) == 10\n'
                       'assert app.retry_delay(2, 0, 10) == 0\n'
                       'for value in [-1, 0.5, True]:\n'
                       '    try: app.retry_delay(value)\n'
                       '    except ValueError: pass\n'
                       '    else: raise AssertionError("invalid attempt accepted")\n'
                       'for kwargs in [{"base": -1}, {"cap": -1}]:\n'
                       '    try: app.retry_delay(1, **kwargs)\n'
                       '    except ValueError: pass\n'
                       '    else: raise AssertionError("negative bound accepted")\n'),
        phase_one='Fix retry_delay: attempt zero waits base seconds, then exponential backoff base*2**attempt capped at cap. Run tests and reread the fixed file.',
        phase_two='Extend the same function: reject negative or non-integer attempts, including booleans, with ValueError. Reject negative base/cap. Preserve cap and zero-base behavior. Run tests.',
    ),
    "inventory": TaskSpec(
        id="inventory", title="Fix inventory reservations",
        initial='def reserve(stock, quantity):\n    return stock - quantity if stock > quantity else None\n',
        phase_one_solution='def reserve(stock, quantity):\n    return stock - quantity if stock >= quantity else None\n',
        solution=('def reserve(stock, quantity):\n'
                  '    if stock < 0 or quantity < 0:\n        raise ValueError("negative inventory")\n'
                  '    return stock - quantity if stock >= quantity else None\n'),
        public_tests=('from app import reserve\n'
                      'def test_exact_stock():\n    assert reserve(4, 4) == 0\n'
                      'def test_shortage():\n    assert reserve(2, 3) is None\n'),
        hidden_checks=('assert app.reserve(4, 0) == 4\nassert app.reserve(0, 0) == 0\n'
                       'for pair in [(-1, 0), (4, -1)]:\n'
                       '    try: app.reserve(*pair)\n'
                       '    except ValueError: pass\n'
                       '    else: raise AssertionError("negative accepted")\n'),
        phase_one='Fix reserve so requesting exactly the available stock succeeds and returns zero. Return None for shortage. Run tests and reread the file.',
        phase_two='Extend reserve to reject negative stock or quantity with ValueError. Zero quantity is valid. Preserve existing behavior and run tests.',
    ),
    "config_parser": TaskSpec(
        id="config_parser", title="Repair a config parser",
        initial='def parse_config(text):\n    return dict(line.split("=") for line in text.splitlines())\n',
        phase_one_solution=('def parse_config(text):\n    result = {}\n'
                            '    for line in text.splitlines():\n'
                            '        key, value = line.split("=", 1)\n'
                            '        result[key.strip()] = value.strip()\n    return result\n'),
        solution=('def parse_config(text):\n    result = {}\n'
                  '    for line in text.splitlines():\n'
                  '        if not line.strip() or line.lstrip().startswith("#"):\n            continue\n'
                  '        key, value = line.split("=", 1)\n'
                  '        if not key.strip():\n            raise ValueError("empty key")\n'
                  '        result[key.strip()] = value.strip()\n    return result\n'),
        public_tests=('from app import parse_config\n'
                      'def test_equals_in_value():\n    assert parse_config("token=a=b") == {"token":"a=b"}\n'
                      'def test_spaces():\n    assert parse_config(" A = B ") == {"A":"B"}\n'),
        hidden_checks=('assert app.parse_config("# hello\\n\\n A=1\\nA=2") == {"A":"2"}\n'
                       'for value in ["=bad", "malformed"]:\n'
                       '    try: app.parse_config(value)\n'
                       '    except ValueError: pass\n'
                       '    else: raise AssertionError("malformed accepted")\n'),
        phase_one='Fix parse_config to split each line at the first equals sign and strip surrounding whitespace from keys and values. Run tests and reread the file.',
        phase_two='Extend parse_config to skip blank lines and lines starting with # after whitespace. Reject empty keys and lines without equals using ValueError. Later duplicate keys win. Preserve embedded equals.',
    ),
}
