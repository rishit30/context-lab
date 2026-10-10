"""Optional Jev configuration and CLI contracts; providers are always mocked."""
import json
import sys
from decimal import Decimal
from types import ModuleType

import pytest
from pydantic import ValidationError

from harness import cli
from harness.config import (
    ROOT,
    ExperimentConfig,
    JevConfig,
    experiment_plan,
    load_config,
    load_pricing,
)


def jev_config(**updates):
    values = load_config(ROOT / "config/jev-comparison.yaml").model_dump()
    return ExperimentConfig.model_validate(values | updates)


def test_rules_remain_default_and_have_no_decision_layer_plan():
    config = load_config()
    assert config.decision_engine == "rules"
    assert config.jev == JevConfig()
    assert "decision_layer" not in experiment_plan(config, load_pricing(), "fake-model")


def test_jev_sample_is_a_single_bounded_short_audit():
    config = jev_config()
    assert config.arms == ("contextlab",)
    assert config.tasks == ("file_audit_short",)
    assert config.repeats == 1
    assert config.session_cap_usd == config.experiment_cap_usd == Decimal("3.00")
    assert config.cache_mode == "disabled"
    assert config.phase_boundaries
    assert config.jev == JevConfig()
    assert config.jev.model == "jev-1.13.0"
    assert config.jev.input_per_million == Decimal("0.042")
    assert config.jev.output_per_million == Decimal("0")
    assert config.jev.reservation_input_tokens == 65536


@pytest.mark.parametrize("field,value", [
    ("model", "jev-1.12.0"),
    ("input_per_million", "0"),
    ("input_per_million", "0.043"),
    ("output_per_million", "0.001"),
    ("output_per_million", "-1"),
    ("pricing_source", "https://unreviewed.invalid"),
    ("verified_at", "2027-01-01"),
    ("max_calls_per_session", 0),
    ("max_calls_per_session", 41),
    ("max_state_bytes", 0),
    ("max_state_bytes", 24001),
    ("max_candidates", 0),
    ("max_candidates", 9),
    ("keep_threshold", -0.01),
    ("keep_threshold", 0.51),
    ("timeout_seconds", 0),
    ("timeout_seconds", 31),
    ("reservation_input_tokens", 64000),
    ("reservation_input_tokens", 65537),
    ("unrecognized_setting", "ignored?"),
])
def test_jev_rejects_unreviewed_prices_and_unbounded_settings(field, value):
    with pytest.raises(ValidationError):
        JevConfig.model_validate({field: value})


def test_jev_requires_a_contextlab_arm_but_rules_do_not():
    with pytest.raises(ValidationError, match="requires a contextlab arm"):
        jev_config(arms=("baseline",))
    assert jev_config(arms=("baseline",), decision_engine="rules").arms == ("baseline",)


def test_plan_discloses_price_data_transfer_reservation_and_existing_caps():
    config = jev_config()
    plan = experiment_plan(config, load_pricing(), "fake-model")
    layer = plan["decision_layer"]
    assert layer["provider"] == "TypeSafe"
    assert layer["pricing"] == {
        "input_per_million": "0.042", "output_per_million": "0",
        "source": "https://docs.typesafe.ai/models", "verified_at": "2026-10-10",
    }
    assert layer["settings"] == config.jev.model_dump(mode="json")
    assert layer["reservation_input_tokens"] == 65536
    assert layer["reservation_usd"] == "0.002752512"
    assert layer["maximum_reserved_decision_cost_per_contextlab_session_usd"] == "0.022020096"
    assert "64k" in layer["reservation_basis"]
    assert "neither a tokenizer nor a max_tokens control" in layer["reservation_basis"]
    assert "tool-result text" in layer["data_sent"] and "TypeSafe" in layer["data_sent"]
    assert "no additional spend" in layer["budget"]
    assert "uncalibrated" in layer["threshold_status"]
    assert plan["maximum_authorized_cost_usd"] == "3.00"
    assert plan["matrix"] == [{"arm": "contextlab", "task_id": "file_audit_short", "repeat": 0}]
    assert json.loads(json.dumps(plan)) == plan


@pytest.mark.parametrize("updates", [
    {"decision_engine": "rules"},
    {"jev": {"keep_threshold": 0.2}},
    {"jev": {"max_calls_per_session": 4}},
    {"jev": {"max_state_bytes": 12000}},
    {"jev": {"max_candidates": 4}},
    {"jev": {"timeout_seconds": 5}},
    {"session_cap_usd": "2.00"},
    {"experiment_cap_usd": "2.00"},
])
def test_approval_digest_binds_decision_settings_and_caps(updates):
    pricing = load_pricing()
    original = experiment_plan(jev_config(), pricing, "fake-model")
    changed = experiment_plan(jev_config(**updates), pricing, "fake-model")
    assert original["approval_digest"] != changed["approval_digest"]
    assert original["source_hash"] == changed["source_hash"]


@pytest.mark.parametrize("engine", ["rules", "jev"])
@pytest.mark.parametrize("command", ["plan", "check", "run"])
def test_cli_loads_only_credentials_required_for_selected_live_engine(
    monkeypatch, tmp_path, capsys, engine, command,
):
    import harness.runner

    config = jev_config(decision_engine=engine)
    calls = []
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *args: config)
    monkeypatch.setattr(cli, "load_pricing", lambda *args: object())
    monkeypatch.setattr(cli, "experiment_plan", lambda *args: {"approval_digest": "fake"})
    monkeypatch.setattr(cli, "load_anthropic_key", lambda path: calls.append(("anthropic", path)))
    monkeypatch.setattr(cli, "load_typesafe_key", lambda path: calls.append(("typesafe", path)))
    monkeypatch.setattr(cli, "build_report", lambda *args: "mock report")

    def live(*args):
        calls.append((command, None))
        return []

    monkeypatch.setattr(harness.runner, "preflight", live)
    monkeypatch.setattr(harness.runner, "run_live", live)
    argv = ["contextlab", command, "--pricing", "mock-pricing"]
    if command == "run":
        argv.extend(["--approve-plan", "fake", "--output", str(tmp_path / "output")])
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()
    if command == "plan":
        assert calls == []
        assert json.loads(capsys.readouterr().out) == {"approval_digest": "fake"}
    else:
        expected = [("anthropic", tmp_path / ".env")]
        if engine == "jev":
            expected.append(("typesafe", tmp_path / ".env"))
        assert calls == expected + [(command, None)]


def test_jev_models_only_loads_typesafe_and_lists_models(monkeypatch, tmp_path, capsys):
    calls = []
    response = {"models": [{"id": "jev-1.13.0"}]}

    class FakeClient:
        @classmethod
        def from_environment(cls):
            calls.append("client")
            return cls()

        def list_models(self):
            calls.append("list_models")
            return response

    package = ModuleType("contextlab.providers")
    package.__path__ = []
    provider = ModuleType("contextlab.providers.jev")
    provider.JevClient = FakeClient
    monkeypatch.setitem(sys.modules, "contextlab.providers", package)
    monkeypatch.setitem(sys.modules, "contextlab.providers.jev", provider)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_typesafe_key", lambda path: calls.append(("typesafe", path)))
    monkeypatch.setattr(cli, "load_anthropic_key", lambda *args: pytest.fail("No Anthropic key"))
    monkeypatch.setattr(cli, "load_config", lambda *args: pytest.fail("No experiment configuration"))
    monkeypatch.setattr(sys, "argv", ["contextlab", "jev-models"])
    cli.main()
    assert calls == [("typesafe", tmp_path / ".env"), "client", "list_models"]
    assert json.loads(capsys.readouterr().out) == response


def test_malformed_typesafe_key_stops_before_live_runner(monkeypatch, tmp_path, capsys):
    import harness.runner

    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-export")
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-restore-placeholder")
    monkeypatch.delenv("TYPESAFE_API_KEY")
    (tmp_path / ".env").write_text('TYPESAFE_API_KEY="fake-sensitive-value\n')
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *args: jev_config())
    monkeypatch.setattr(cli, "load_pricing", lambda *args: object())
    monkeypatch.setattr(cli, "experiment_plan", lambda *args: {"approval_digest": "fake"})
    monkeypatch.setattr(harness.runner, "preflight", lambda *args: pytest.fail("Must not run"))
    monkeypatch.setattr(sys, "argv", ["contextlab", "check", "--pricing", "mock"])
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "Invalid TYPESAFE_API_KEY entry" in captured.err
    assert "fake-sensitive-value" not in captured.out + captured.err
