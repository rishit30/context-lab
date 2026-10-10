"""Local-key parsing and CLI boundaries; all credentials and providers are fake."""
import os
from pathlib import Path

import pytest

from harness import cli
from harness.environment import load_anthropic_key, load_typesafe_key

KEY = "ANTHROPIC_API_KEY"
TYPESAFE_KEY = "TYPESAFE_API_KEY"
LOADERS = [(KEY, load_anthropic_key), (TYPESAFE_KEY, load_typesafe_key)]


@pytest.fixture(autouse=True)
def clear_key(monkeypatch):
    # Track an original absent value too: loader writes os.environ directly.
    for key, _ in LOADERS:
        monkeypatch.setenv(key, "test-placeholder")
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("key,loader", LOADERS)
@pytest.mark.parametrize("assignment", [
    "{key}=fake-key",
    " export {key} = 'fake-key' # comment",
    '{key}="fake-key"# comment',
    "{key}=fake-key # comment",
])
def test_loads_only_supported_key(tmp_path, monkeypatch, assignment, key, loader):
    path = tmp_path / ".env"
    other_key = TYPESAFE_KEY if key == KEY else KEY
    path.write_text(
        "# comment\n\nOTHER_SECRET=ignore-me\n"
        + f'{other_key}="malformed ignored value\n'
        + assignment.format(key=key) + "\n"
    )
    monkeypatch.delenv("OTHER_SECRET", raising=False)
    assert loader(path)
    assert os.environ[key] == "fake-key"
    assert other_key not in os.environ
    assert "OTHER_SECRET" not in os.environ


@pytest.mark.parametrize("key,loader", LOADERS)
@pytest.mark.parametrize("exported", ["already-exported", ""])
def test_exported_value_wins_without_reading_file(monkeypatch, exported, key, loader):
    monkeypatch.setenv(key, exported)
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("Must not read file"))
    assert not loader(Path("unused.env"))
    assert os.environ[key] == exported


@pytest.mark.parametrize("key,loader", LOADERS)
def test_missing_or_unrelated_file_does_not_set_key(tmp_path, key, loader):
    assert not loader(tmp_path / "missing")
    path = tmp_path / ".env"
    path.write_text(f"OTHER=broken '\n{key}_OTHER=ignored\n")
    assert not loader(path)
    assert key not in os.environ


@pytest.mark.parametrize("key,loader", LOADERS)
@pytest.mark.parametrize("quote", ["", "'", '"'])
def test_shell_and_variable_syntax_remain_literal(tmp_path, monkeypatch, key, loader, quote):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / ".env"
    value = "$(touch must-not-exist)-${HOME}-`whoami`-\\n"
    path.write_text(f"{key}={quote}{value}{quote}\n")
    assert loader(path)
    assert os.environ[key] == value
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("key,loader", LOADERS)
@pytest.mark.parametrize("entry", [
    "{key}",
    "{key}=",
    "{key}= # only a comment",
    '{key}="fake-sensitive-value',
    '{key}="fake-sensitive-value" extra',
    "{key}=fake-sensitive-value\n{key}=duplicate",
    "{key}=fake-sensitive-value\nexport {key} = 'duplicate'",
    "{key}=fake-sensitive-value\x00",
])
def test_malformed_errors_do_not_include_values_or_partially_set_key(tmp_path, entry, key, loader):
    path = tmp_path / ".env"
    path.write_text(entry.format(key=key))
    with pytest.raises(ValueError) as caught:
        loader(path)
    assert str(caught.value) == (
        f"Invalid {key} entry in .env; use one single-line KEY=value assignment."
    )
    assert "fake-sensitive-value" not in str(caught.value)
    assert "duplicate" not in str(caught.value)
    assert key not in os.environ


@pytest.mark.parametrize("key,loader", LOADERS)
@pytest.mark.parametrize("error", [PermissionError, UnicodeError])
def test_unreadable_file_has_generic_error(tmp_path, monkeypatch, key, loader, error):
    def denied(*args, **kwargs):
        raise error("fake-sensitive-value")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(ValueError, match="Cannot read local .env credential file") as caught:
        loader(tmp_path / ".env")
    assert "fake-sensitive-value" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert key not in os.environ


def test_each_loader_independently_selects_its_own_assignment(tmp_path):
    path = tmp_path / ".env"
    path.write_text(f"{KEY}=fake-anthropic\n{TYPESAFE_KEY}=fake-typesafe\n")
    assert load_typesafe_key(path)
    assert os.environ[TYPESAFE_KEY] == "fake-typesafe"
    assert KEY not in os.environ
    assert load_anthropic_key(path)
    assert os.environ[KEY] == "fake-anthropic"
    assert os.environ[TYPESAFE_KEY] == "fake-typesafe"


def test_private_parser_rejects_non_whitelisted_names_without_reading(tmp_path, monkeypatch):
    from harness.environment import _load_key

    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("Must not read file"))
    with pytest.raises(ValueError, match="^Unsupported local credential name\\.$") as caught:
        _load_key(tmp_path / ".env", "fake-sensitive-name")
    assert "fake-sensitive-name" not in str(caught.value)


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
