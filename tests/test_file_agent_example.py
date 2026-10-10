"""Offline orchestration tests, never used as real model-output evidence.

Production example builds real ChatAnthropic. Test-only session doubles below
verify approval, immutable inputs, completion and accounting without inference.
"""

import json
from copy import deepcopy
from decimal import Decimal

import pytest
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from examples import file_agent
from harness.workspace import Workspace


@pytest.fixture
def setup(tmp_path, monkeypatch):
    files = tmp_path / "files"
    files.mkdir()
    (files / "incident.txt").write_text("owner: team-a\nrollback: error_rate > 5%\n")
    pricing = tmp_path / "pricing.yaml"
    pricing.write_text(
        "model: claude-sonnet-4-6\ninput_per_million: '3'\ncache_write_5m_per_million: '3.75'\ncache_write_1h_per_million: '6'\ncache_read_per_million: '0.30'\noutput_per_million: '15'\nsource: https://official.example/pricing\nverified_at: '2026-10-10'\n"
    )
    monkeypatch.setattr(
        file_agent, "source_manifest", lambda: [("unit-fixture.py", "frozen-source")]
    )
    monkeypatch.setattr(
        "contextlab.adapters.headroom_mw.configuration",
        lambda: {
            "distribution": "headroom-ai",
            "profile": "test-only-configuration",
            "version": "unit",
        },
    )
    loaded = []
    monkeypatch.setattr(file_agent, "load_anthropic_key", lambda path: loaded.append("anthropic"))
    monkeypatch.setattr(file_agent, "load_typesafe_key", lambda path: loaded.append("typesafe"))
    return files, pricing, loaded


def make_plan(setup, **kwargs):
    files, pricing, _ = setup
    return file_agent.plan(
        files=files,
        pricing_path=pricing,
        arm=kwargs.get("arm", "all"),
        decision_engine=kwargs.get("decision_engine", "jev"),
        questions=kwargs.get("questions", ("First question", "Second question")),
    )


def install_session_double(monkeypatch, *, mode="normal", mutate=None, blocked=False):
    instances = []

    class SessionDouble:
        def __init__(self, path, arm, config):
            self.path, self.arm, self.config = path, arm, config
            self.phases = []
            self.inputs = []
            self.close_calls = 0
            self.closed = False
            instances.append(self)

        def build_agent(self, *, tools, system_prompt):
            self.tools = tools
            self.system_prompt = system_prompt
            owner = self

            class Graph:
                def invoke(self, state, config):
                    owner.inputs.append(deepcopy(state["messages"]))
                    output = json.loads(tools[1].invoke({"path": "incident.txt"}))
                    owner.read_text = output["content"]
                    if mutate is not None:
                        mutate(owner)
                    if mode == "failure":
                        raise RuntimeError("private provider error details")
                    if mode == "truncated":
                        response = AIMessage(
                            content="Incomplete", response_metadata={"stop_reason": "max_tokens"}
                        )
                    elif mode == "invalid_tool_call":
                        response = AIMessage(
                            content="Incomplete",
                            response_metadata={"stop_reason": "end_turn"},
                            invalid_tool_calls=[
                                {
                                    "name": "read_file",
                                    "args": "{bad",
                                    "id": "invalid",
                                    "type": "invalid_tool_call",
                                    "error": "bad JSON",
                                }
                            ],
                        )
                    else:
                        response = AIMessage(
                            content="Unit-only answer",
                            response_metadata={"stop_reason": "end_turn"},
                        )
                    return {"messages": [*state["messages"], response]}

            return Graph()

        def complete_phase(self, label):
            self.phases.append(label)

        def close(self):
            self.close_calls += 1
            self.closed = True
            if mode == "close_failure":
                raise RuntimeError("private settlement failure")

        def summary(self):
            assert self.closed  # Reporting must follow final settlement.
            if mode == "summary_failure":
                raise RuntimeError("private database failure")
            fee = Decimal("0.001") if self.config.decision_engine == "jev" else Decimal(0)
            return {
                "session_id": self.arm,
                "arm": self.arm,
                "total_cost_usd": str(Decimal("0.010") + fee),
                "decision_cost_usd": str(fee),
                "decision_calls": int(bool(fee)),
                "budget": {"experiment": {"blocked": blocked}},
                "cost_basis": "Test fixture accounting, not provider evidence",
            }

    monkeypatch.setattr(file_agent, "ContextLabSession", SessionDouble)
    return instances


def test_plan_has_three_real_configurations_and_makes_no_provider_or_key_requests(setup):
    plan = make_plan(setup)
    assert list(plan["settings"]) == ["baseline", "contextlab", "headroom"]
    assert plan["maximum_authorized_cost_usd"] == "3"
    assert all(
        s["session_cap_usd"] == "1.00" and s["cache_mode"] == "disabled"
        for s in plan["settings"].values()
    )
    assert plan["settings"]["contextlab"]["decision_engine"] == "jev"
    assert plan["settings"]["baseline"]["decision_engine"] == "rules"
    assert plan["headroom"]["distribution"] == "headroom-ai"
    assert plan["approval_digest"] == file_agent.approval_digest(plan)
    assert setup[2] == []


@pytest.mark.parametrize("change", ["digest", "settings", "question", "source", "input"])
def test_rejected_plan_stops_before_keys_output_or_session(setup, tmp_path, monkeypatch, change):
    plan = make_plan(setup, arm="baseline")
    approval = plan["approval_digest"]
    if change == "digest":
        approval = "wrong"
    elif change == "settings":
        plan["settings"]["baseline"]["session_cap_usd"] = "9"
    elif change == "question":
        plan["questions"] = ("Unreviewed question",)
    elif change == "source":
        monkeypatch.setattr(file_agent, "source_manifest", lambda: [("unit-fixture.py", "changed")])
    elif change == "input":
        (setup[0] / "incident.txt").write_text("changed input")
    instances = install_session_double(monkeypatch)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        file_agent.run(output, plan, approval)
    assert not output.exists() and instances == [] and setup[2] == []


def test_all_arms_use_identical_snapshot_tools_questions_and_final_accounting(
    setup, tmp_path, monkeypatch
):
    plan = make_plan(setup)
    original = (setup[0] / "incident.txt").read_text()

    def change_live_inputs(_session):
        (setup[0] / "incident.txt").write_text("external input changed after snapshot")

    instances = install_session_double(monkeypatch, mutate=change_live_inputs)
    result = file_agent.run(tmp_path / "output", plan, plan["approval_digest"])
    assert result["total_cost_usd"] == "0.031" and result["all_arms_attempted"]
    assert all(
        row["questions_completed"] == 2 and not row["independently_graded"]
        for row in result["sessions"]
    )
    assert [instance.read_text for instance in instances] == [original] * 3
    assert all(instance.close_calls == 1 for instance in instances)
    assert all(
        instance.phases == ["answered-question-1", "answered-question-2"] for instance in instances
    )
    schemas = [[convert_to_openai_tool(tool) for tool in instance.tools] for instance in instances]
    assert schemas[0] == schemas[1] == schemas[2]
    assert all(instance.system_prompt == plan["system_prompt"] for instance in instances)
    assert json.loads((tmp_path / "output/summary.json").read_text()) == result
    assert setup[2] == ["anthropic", "typesafe"]
    assert (tmp_path / "output/source-manifest.json").exists()
    report = (tmp_path / "output/report.html").read_text()
    assert "Independent grading" in report and "Not performed" in report
    assert result["report"] == str((tmp_path / "output/report.html").resolve())


@pytest.mark.parametrize("mode", ["truncated", "invalid_tool_call", "failure"])
def test_invalid_completion_is_not_marked_as_a_finished_phase(setup, tmp_path, monkeypatch, mode):
    plan = make_plan(setup, arm="baseline")
    instances = install_session_double(monkeypatch, mode=mode)
    result = file_agent.run(tmp_path / "output", plan, plan["approval_digest"])
    assert result["sessions"][0]["questions_completed"] == 0
    assert result["sessions"][0]["error_type"] == "RuntimeError"
    assert instances[0].phases == [] and len(instances[0].inputs) == 1
    assert "private" not in json.dumps(result["sessions"])
    assert result["total_cost_usd"] == "0.010"


def test_snapshot_mutation_stops_other_arms_and_retains_incurred_cost(setup, tmp_path, monkeypatch):
    plan = make_plan(setup)
    output = tmp_path / "output"

    def mutate_snapshot(_session):
        (output / "input/incident.txt").write_text("unexpected mutation")

    instances = install_session_double(monkeypatch, mutate=mutate_snapshot)
    result = file_agent.run(output, plan, plan["approval_digest"])
    assert len(instances) == 1 and instances[0].phases == []
    assert not result["all_arms_attempted"] and result["error_type"] == "InputSnapshotChanged"
    assert result["total_cost_usd"] == "0.010"


@pytest.mark.parametrize("mode", ["close_failure", "summary_failure"])
def test_final_accounting_failure_stops_matrix_and_never_invents_zero_cost(
    setup, tmp_path, monkeypatch, mode
):
    plan = make_plan(setup)
    instances = install_session_double(monkeypatch, mode=mode)
    result = file_agent.run(tmp_path / "output", plan, plan["approval_digest"])
    assert len(instances) == 1 and instances[0].close_calls == 1
    assert result["error_type"] == "RuntimeError" and not result["all_arms_attempted"]
    assert result["total_cost_usd"] == (None if mode == "summary_failure" else "0.010")
    assert "private" not in json.dumps(result["sessions"])


def test_experiment_block_stops_matrix(setup, tmp_path, monkeypatch):
    plan = make_plan(setup)
    instances = install_session_double(monkeypatch, blocked=True)
    result = file_agent.run(tmp_path / "output", plan, plan["approval_digest"])
    assert len(instances) == 1 and result["error_type"] == "ExperimentBudgetBlocked"


def test_read_tools_have_no_write_or_execution_capability_and_recover_expected_bad_inputs(setup):
    tools = file_agent.read_tools(Workspace(setup[0]))
    assert [tool.name for tool in tools] == ["list_files", "read_file"]
    data = json.loads(tools[1].invoke({"path": "incident.txt"}))
    assert data["full_read"] and data["content"].startswith("owner: team-a")
    assert "existing UTF-8 file" in tools[1].invoke({"path": "../outside.txt"})
    assert "offset >= 0" in tools[1].invoke({"path": "incident.txt", "offset": -1})


def test_plan_stdout_is_pure_json_for_redirection(setup, monkeypatch, capsys):
    files, pricing, _ = setup
    monkeypatch.setattr(
        "sys.argv",
        [
            "file_agent",
            "plan",
            "--files",
            str(files),
            "--pricing",
            str(pricing),
            "--arm",
            "baseline",
        ],
    )
    file_agent.main()
    captured = capsys.readouterr()
    assert json.loads(captured.out)["kind"] == "real-read-only-file-agent"
    assert "No provider requests made" in captured.err and setup[2] == []
