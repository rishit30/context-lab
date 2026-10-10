"""Local-key parsing and CLI boundaries; all credentials and providers are fake."""
from pathlib import Path

import pytest

from harness import cli
from harness.environment import load_anthropic_key

KEY = "ANTHROPIC_API_KEY"


@pytest.fixture(autouse=True)
def clear_key(monkeypatch):
    # Track an original absent value too: loader writes os.environ directly.
    monkeypatch.setenv(KEY, "test-placeholder")
    monkeypatch.delenv(KEY, raising=False)


@pytest.mark.parametrize("assignment", [
    "ANTHROPIC_API_KEY=fake-key",
    " export ANTHROPIC_API_KEY = 'fake-key' # comment",
    'ANTHROPIC_API_KEY="fake-key"# comment',
    "ANTHROPIC_API_KEY=fake-key # comment",
])
def test_loads_only_supported_key(tmp_path, monkeypatch, assignment):
    path = tmp_path / ".env"
    path.write_text("# comment\n\nOTHER_SECRET=ignore-me\n" + assignment + "\n")
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    assert load_anthropic_key(path)
    import os
    assert os.environ[KEY] == "fake-key"
    assert "OTHER_SECRET" not in os.environ


@pytest.mark.parametrize("exported", ["already-exported", ""])
def test_exported_value_wins_without_reading_file(monkeypatch, exported):
    monkeypatch.setenv(KEY, exported)
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("Must not read file"))
    assert not load_anthropic_key(Path("unused.env"))


def test_missing_or_unrelated_file_does_not_set_key(tmp_path):
    import os
    assert not load_anthropic_key(tmp_path / "missing")
    path = tmp_path / ".env"
    path.write_text("OTHER=broken '\nANTHROPIC_API_KEY_OTHER=ignored\n")
    assert not load_anthropic_key(path)
    assert KEY not in os.environ


def test_shell_and_variable_syntax_remain_literal(tmp_path):
    import os
    path = tmp_path / ".env"
    value = "$(touch must-not-exist)-${HOME}-`whoami`"
    path.write_text(f"{KEY}='{value}'\n")
    assert load_anthropic_key(path)
    assert os.environ[KEY] == value
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("entry", [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_API_KEY=",
    "ANTHROPIC_API_KEY= # only a comment",
    'ANTHROPIC_API_KEY="fake-sensitive-value',
    'ANTHROPIC_API_KEY="fake-sensitive-value" extra',
    "ANTHROPIC_API_KEY=fake-sensitive-value\nANTHROPIC_API_KEY=duplicate",
    "ANTHROPIC_API_KEY=fake-sensitive-value\x00",
])
def test_malformed_errors_do_not_include_values_or_partially_set_key(tmp_path, entry):
    import os
    path = tmp_path / ".env"
    path.write_text(entry)
    with pytest.raises(ValueError) as caught:
        load_anthropic_key(path)
    assert "fake-sensitive-value" not in str(caught.value)
    assert "duplicate" not in str(caught.value)
    assert KEY not in os.environ


def test_unreadable_file_has_generic_error(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise PermissionError("fake-sensitive-value")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(ValueError, match="Cannot read local .env credential file") as caught:
        load_anthropic_key(tmp_path / ".env")
    assert "fake-sensitive-value" not in str(caught.value)


def wire_cli(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *args: object())
    monkeypatch.setattr(cli, "load_pricing", lambda *args: object())
    monkeypatch.setattr(cli, "load_compaction", lambda: object())
    monkeypatch.setattr(cli, "experiment_plan", lambda *args: {"fake": True})
    monkeypatch.setattr(cli, "build_report", lambda *args: "fake report")


@pytest.mark.parametrize("command", ["check", "run"])
def test_live_cli_loads_local_key_before_runner(tmp_path, monkeypatch, command, capsys):
    import os

    import harness.runner
    wire_cli(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=fake-cli-key\n")
    calls = []
    def called(*args):
        assert os.environ[KEY] == "fake-cli-key"
        calls.append(command)
        return []
    monkeypatch.setattr(harness.runner, "preflight", called)
    monkeypatch.setattr(harness.runner, "run_live", called)
    argv = ["contextlab", command, "--pricing", "unused"]
    if command == "run":
        argv.extend(["--approve-plan", "fake", "--output", str(tmp_path / "output")])
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    assert calls == [command]
    captured = capsys.readouterr()
    assert "fake-cli-key" not in captured.out + captured.err


@pytest.mark.parametrize("command", ["plan", "demo", "report"])
def test_offline_cli_never_loads_credentials(tmp_path, monkeypatch, command):
    import harness.demo
    wire_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "load_anthropic_key", lambda *args: pytest.fail("No credential read"))
    monkeypatch.setattr(harness.demo, "demo_session", lambda *args: None)
    monkeypatch.setattr(cli, "load_results", lambda *args: [])
    argv = ["contextlab", command]
    if command == "plan":
        argv.extend(["--pricing", "unused"])
    elif command == "demo":
        argv.extend(["--output", str(tmp_path / "demo")])
    else:
        argv.append(str(tmp_path))
    monkeypatch.setattr("sys.argv", argv)
    cli.main()


def test_cli_malformed_secret_stops_before_runner_and_reports_generic_error(
    tmp_path, monkeypatch, capsys,
):
    import harness.runner
    wire_cli(monkeypatch, tmp_path)
    (tmp_path / ".env").write_text('ANTHROPIC_API_KEY="fake-sensitive-value\n')
    monkeypatch.setattr(harness.runner, "preflight", lambda *args: pytest.fail("Must not run"))
    monkeypatch.setattr("sys.argv", ["contextlab", "check", "--pricing", "unused"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "Invalid ANTHROPIC_API_KEY entry" in captured.err
    assert "fake-sensitive-value" not in captured.out + captured.err
