"""Offline checks exercise installed Headroom, never a substitute compressor."""
import json
import subprocess
import sys
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from contextlab.adapters import headroom_mw
from contextlab.adapters.langchain_mw import MeteringMiddleware, normalize_messages
from harness.tasks.file_audit import FileAuditTask
from harness.workspace import Workspace


@pytest.fixture
def middleware():
    try:
        version("headroom-ai")
    except PackageNotFoundError:
        pytest.skip("Install optional comparison SDK with uv sync --extra headroom")
    return headroom_mw.HeadroomMiddleware()


def request(body=None, tool_name="run_tests"):
    if body is None:
        body = json.dumps([
            {"test_id": f"test_{i}", "status": "passed", "service": "payments",
             "duration": i % 4, "path": "tests/test_payments.py"}
            for i in range(60)
        ], indent=2)
    return ModelRequest(
        model=FakeMessagesListChatModel(responses=[AIMessage(content="ok")]),
        messages=[
            HumanMessage(content="Find the inconsistent service setting."),
            AIMessage(content=[{"type": "thinking", "thinking": "opaque reasoning",
                                "signature": "keep-verbatim"}],
                      tool_calls=[{"name": tool_name, "args": {}, "id": "call1",
                                   "type": "tool_call"}],
                      response_metadata={"provider_marker": "intact"}),
            ToolMessage(content=body, tool_call_id="call1", name=tool_name,
                        artifact={"metadata": "intact"}, id="result1"),
        ],
        system_message=SystemMessage(content="A fixed system instruction."),
        tools=[{"name": tool_name, "description": "test", "input_schema": {"type": "object"}}],
        model_settings={"temperature": 0},
    )


def collect(middleware, value):
    captured = []
    response = ModelResponse(result=[AIMessage(content="ok")])
    assert middleware.wrap_model_call(value, lambda r: captured.append(r) or response) is response
    assert len(captured) == 1
    return captured[0]


def test_genuine_headroom_changes_json_and_preserves_every_other_field(middleware):
    original = request()
    snapshot = deepcopy(original.messages)
    projected = collect(middleware, original)
    assert projected.messages[-1].content != original.messages[-1].content
    assert len(projected.messages[-1].content) < len(original.messages[-1].content)
    assert projected.messages[:-1] == original.messages[:-1]
    assert projected.messages[-1].model_dump(exclude={"content"}) == original.messages[-1].model_dump(exclude={"content"})
    assert projected.system_message == original.system_message
    assert projected.tools == original.tools
    assert projected.model_settings == original.model_settings
    assert original.messages == snapshot
    assert middleware.failures == []
    assert middleware.stats["changed_calls"] == 1
    assert middleware.stats["recalls"] == 0
    assert middleware.telemetry[-1]["lossless_verified"] is True
    assert middleware.telemetry[-1]["changed"] is True
    assert middleware.telemetry[-1]["tokens_before_estimate"] > middleware.telemetry[-1]["tokens_after_estimate"]
    assert middleware.telemetry[-1]["transforms_applied"]


def test_repeated_history_densifies_to_identical_bytes(middleware):
    original = request()
    first = collect(middleware, original)
    second = collect(middleware, original)
    assert first.messages == second.messages
    assert middleware.stats["changed_calls"] == 2


def test_real_workspace_read_envelope_honors_upstream_recent_read_protection(middleware, tmp_path):
    task = FileAuditTask(id="headroom-check", title="Headroom fixture", service_count=24)
    directory = tmp_path / "fixture"
    task.create(directory)
    body = Workspace(directory).read_file(task.snapshot_paths[0])
    assert json.loads(body)["full_read"] is True
    original = request(body, tool_name="read_file")
    projected = collect(middleware, original)
    assert projected.messages == original.messages
    assert middleware.stats["changed_calls"] == 0
    assert middleware.telemetry[-1]["changed"] is False
    assert middleware.failures == []
    assert "router:excluded:tool" in middleware.telemetry[-1]["transforms_applied"]


def test_meter_sees_final_compressed_input_and_original_provider_usage(middleware):
    original = request()
    seen = []
    receipts = []
    meter = MeteringMiddleware(model_name="offline", before_call=lambda r: seen.append(r),
                              on_usage=lambda *args: receipts.append(args))
    answer = AIMessage(content="done", usage_metadata={
        "input_tokens": 123, "output_tokens": 7, "total_tokens": 130,
    })
    response = ModelResponse(result=[answer])
    result = middleware.wrap_model_call(
        original, lambda r: meter.wrap_model_call(r, lambda _: response),
    )
    assert result is response
    assert seen[0].messages[-1].content != original.messages[-1].content
    assert receipts[0][1].input_tokens == 123
    assert receipts[0][1].output_tokens == 7


def test_provider_failure_propagates_once_without_baseline_retry(middleware):
    called = []

    def fail(value):
        called.append(value)
        raise RuntimeError("provider error")

    with pytest.raises(RuntimeError, match="provider error"):
        middleware.wrap_model_call(request(), fail)
    assert len(called) == 1
    assert middleware.failures == []


@pytest.mark.parametrize("failure", ["exception", "unverified", "instructions", "opaque"])
def test_compression_errors_forward_original_and_record_sanitized_failure(middleware, monkeypatch, failure):
    original = request()

    def invalid(messages, **_kwargs):
        if failure == "exception":
            raise RuntimeError("secret body must not be logged")
        result = SimpleNamespace(messages=deepcopy(messages), lossless=True,
                                 tokens_before=100, tokens_after=50,
                                 transforms_applied=[], reverted_messages=0, diagnostics=[])
        if failure == "unverified":
            result.lossless = None
        elif failure == "instructions":
            result.messages[0]["content"] = "changed instructions"
        else:
            result.messages[1]["content"][0]["signature"] = "changed signature"
        return result

    monkeypatch.setattr(middleware, "_compress", invalid)
    projected = collect(middleware, original)
    assert projected is original
    assert middleware.stats["fallback_calls"] == 1
    assert len(middleware.failures) == 1
    assert "secret" not in json.dumps(middleware.telemetry)
    assert middleware.telemetry[0]["fallback"] is True
    assert middleware.telemetry[0]["changed"] is False


def test_parallel_results_and_multimodal_tool_blocks_remain_protocol_valid(middleware):
    original = request()
    original.messages[1].tool_calls.append(
        {"name": "image", "args": {}, "id": "call2", "type": "tool_call"})
    image_result = ToolMessage(tool_call_id="call2", content=[
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "opaque"}},
    ])
    original.messages.append(image_result)
    projected = collect(middleware, original)
    assert projected.messages[-1] == image_result
    assert projected.messages[2].tool_call_id == "call1"
    assert projected.messages[3].tool_call_id == "call2"
    assert len(normalize_messages(projected.messages)[-1]["content"]) == 2
    assert middleware.failures == []


def test_missing_optional_package_and_version_mismatch_are_clear(monkeypatch):
    def missing(_):
        raise PackageNotFoundError("headroom-ai")
    monkeypatch.setattr(headroom_mw, "version", missing)
    with pytest.raises(RuntimeError, match="uv sync --extra headroom"):
        headroom_mw.configuration()
    monkeypatch.setattr(headroom_mw, "version", lambda _: "unexpected")
    with pytest.raises(RuntimeError, match="requires headroom-ai==0.40.0"):
        headroom_mw.configuration()


def test_profile_metadata_is_frozen_copy_and_explicit_about_estimates(middleware):
    config = headroom_mw.configuration()
    assert config["version"] == "0.40.0"
    assert config["token_counts_are_estimates"] is True
    assert config["ccr_enabled"] is False
    assert config["recall_applicable"] is False
    config["compress_config"]["kompress_model"] = "bad"
    assert headroom_mw.configuration()["compress_config"]["kompress_model"] == "disabled"


def test_cold_import_and_genuine_compression_attempt_no_socket_connections(middleware):
    script = '''
import json, socket
attempts = []
def blocked(*args, **kwargs):
    attempts.append("socket")
    raise AssertionError("network must not be used")
socket.socket.connect = blocked
socket.create_connection = blocked
from contextlab.adapters.headroom_mw import HeadroomMiddleware
mw = HeadroomMiddleware()
rows = [{"id": i, "status": "passed", "path": "tests/same.py"} for i in range(80)]
body = json.dumps(rows, indent=2)
messages = [{"role":"assistant", "content":[{"type":"tool_use", "id":"a", "name":"run_tests", "input":{}}]},
            {"role":"user", "content":[{"type":"tool_result", "tool_use_id":"a", "content":body}]}]
result = mw._compress(messages, model=mw.model_name, config=mw._config)
assert result.lossless is True
assert result.tokens_after < result.tokens_before
assert not attempts, attempts
print("cold-compression-offline-ok")
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                            timeout=45, check=False)
    assert result.returncode == 0, result.stderr
    assert "cold-compression-offline-ok" in result.stdout
