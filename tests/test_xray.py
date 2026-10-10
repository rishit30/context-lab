from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest

from contextlab.core.models import Chunk, Event, Pricing, Usage
from contextlab.core.view_builder import canonical_bytes
from contextlab.ledger.pricing import price_usage
from contextlab.ledger.store import LedgerStore, ModelCall
from harness.xray import BOUNDARY_SOURCE, _priced, analyze_run


def rates():
    return Pricing(model="claude-test", input_per_million=3, cache_write_5m_per_million=Decimal("3.75"),
                   cache_write_1h_per_million=6, cache_read_per_million=Decimal(".3"), output_per_million=15)


def observation(call_id, text, *, path="data.json", tool="read_file", metadata=None):
    payload = ({"content": text, "path": path, "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "offset": 0, "limit": 2000, "total_lines": 1, "full_read": True}
               if metadata is None else metadata)
    return [
        {"role": "assistant", "content": [{"type": "tool_use", "id": call_id, "name": tool,
                                               "input": {"path": path} if tool == "read_file" else {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id,
                                         "content": json.dumps(payload, sort_keys=True)}]},
    ]


def marker(messages):
    return {"source": BOUNDARY_SOURCE, "phase_index": 0, "label": "completed",
            "normalized_message_count": len(messages),
            "message_prefix_sha256": hashlib.sha256(canonical_bytes(messages)).hexdigest()}


def write_run(root: Path, requests, *, recent=0, session="s", task="audit", mode="live", per_session=False,
              solved=False, usage=None, unknown_usage=False, failed=False, archive=None):
    root.mkdir(exist_ok=True)
    directory = root / session
    directory.mkdir(exist_ok=True)
    ledger = directory / "ledger.sqlite" if per_session else root / "ledger.sqlite"
    usage = usage or Usage(input_tokens=10000, output_tokens=10)
    with LedgerStore(ledger) as store:
        store.create_session(session, "baseline", task, mode=mode, model_main="claude-test")
        for event, chunk in archive or []:
            store.append_event(event)
            store.add_chunk(chunk)
        for seq, request in enumerate(requests):
            raw = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                   "cache_creation_input_tokens": usage.cache_write_5m_tokens + usage.cache_write_1h_tokens,
                   "cache_creation": {"ephemeral_5m_input_tokens": usage.cache_write_5m_tokens,
                                      "ephemeral_1h_input_tokens": usage.cache_write_1h_tokens},
                   "cache_read_input_tokens": usage.cache_read_tokens}
            call = ModelCall(id=f"{session}-{seq}", session_id=session, seq=seq, model="claude-test", usage=usage,
                             pricing=rates(), cost_usd=price_usage(usage, rates()),
                             raw_usage={"messages": [{"provider_usage": raw}]},
                             status="unknown_usage" if unknown_usage else "failed" if failed else "ok",
                             cost_basis="reservation" if unknown_usage else "simulation" if mode == "simulation" else "usage")
            store.record_model_call(call)
            request["seq"] = seq
        store.finish_session(session, solved)
    (directory / "trace.json").write_text(json.dumps({"mode": mode, "timeline" if per_session else "requests": requests}))
    (root / "plan.json").write_text(json.dumps({"compaction": {"recent_outputs": recent}, "source_hash": "historical-source"}))


def snapshots():
    first = [{"role": "user", "content": "Audit current data, then compare its original values."},
             *observation("old", "old value = 42 " * 40)]
    second = first + observation("new", "new value = 99")
    third = second + [{"role": "assistant", "content": "Now compare historical values."},
                      {"role": "user", "content": "Continue the historical comparison."}]
    return first, second, third


def test_future_read_does_not_change_earlier_call_and_repeated_exposure_counts(tmp_path):
    first, second, third = snapshots()
    requests = [{"actual_rendered_messages": messages, "phase_index": 1} for messages in (first, second, third)]
    write_run(tmp_path, requests, failed=True)
    result = analyze_run(tmp_path)
    assert result["ok"] and result["summary"]["baseline_calls"] == 3
    before, after, again = result["calls"]
    assert before["candidate_text_bytes"] == 0
    expected = len(first[-1]["content"][0]["content"].encode())
    assert after["candidate_text_bytes"] == again["candidate_text_bytes"] == expected
    assert result["summary"]["candidate_text_bytes"] == 2 * expected
    assert after["boundary_eligibility"] == "unknown"  # A phase index is not a trusted marker.
    assert result["summary"]["receipt_verified_calls"] == 3  # Known failed calls still count.
    assert result["summary"]["actual_input_cost_usd"] == "0.09"
    assert result["execution_source_hash"] == "historical-source"
    assert "old value = 42" not in json.dumps(result)
    assert any("historical comparison" in limitation for limitation in result["limitations"])


def test_exact_duplicate_and_superseded_signal_count_content_once(tmp_path):
    content = "same observation" * 20
    history = [{"role": "user", "content": "Audit"}, *observation("a", content), *observation("b", content)]
    write_run(tmp_path, [{"actual_rendered_messages": history}])
    call, = analyze_run(tmp_path)["calls"]
    assert call["candidate_count"] == call["exact_duplicate_count"] == 1
    assert call["candidate_text_bytes"] == len(history[2]["content"][0]["content"].encode())


def test_conditional_bucket_allocation_uses_observed_capacity_and_excludes_output():
    usage = Usage(input_tokens=100, cache_write_5m_tokens=400, cache_write_1h_tokens=300,
                  cache_read_tokens=1000, output_tokens=9000)
    raw = {"input_tokens": 100, "cache_creation_input_tokens": 700,
           "cache_creation": {"ephemeral_5m_input_tokens": 400, "ephemeral_1h_input_tokens": 300},
           "cache_read_input_tokens": 1000, "output_tokens": 9000}
    call = ModelCall(id="c", session_id="s", seq=0, model="claude-test", pricing=rates(), usage=usage,
                     cost_usd=price_usage(usage, rates()), raw_usage=raw)
    result = _priced(call, 200)
    assert result["actual_input_cost_usd"] == Decimal(".003900")
    assert result["priced_candidate_input_estimate_usd"] == {"lower": Decimal(".000060"), "upper": Decimal(".001200")}
    assert result["total_input_tokens"] == 1800
    assert _priced(call, 1801)["priced_candidate_input_estimate_usd"] is None
    assert _priced(call, 1801)["estimated_candidate_provider_input_share"] is None


def test_unknown_usage_preserves_byte_measurement_but_not_financial_attribution(tmp_path):
    _, history, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": history}], unknown_usage=True)
    result = analyze_run(tmp_path)
    assert result["ok"] and result["summary"]["candidate_text_bytes"] > 0
    assert result["summary"]["receipt_verified_calls"] == 0
    assert result["summary"]["actual_input_cost_usd"] is None
    assert result["summary"]["priced_candidate_input_estimate_usd"] is None


@pytest.mark.parametrize("mutation", ["missing", "broken_protocol", "non_prefix"])
def test_unreconstructable_history_is_unknown_not_zero(tmp_path, mutation):
    first, second, _ = snapshots()
    requests = [{"actual_rendered_messages": first}, {"actual_rendered_messages": second}]
    if mutation == "missing":
        requests[1] = {}
    elif mutation == "broken_protocol":
        requests[1]["actual_rendered_messages"] = second[:-1]
    else:
        requests[1]["actual_rendered_messages"] = [{"role": "user", "content": "Different history"}]
    write_run(tmp_path, requests)
    result = analyze_run(tmp_path)
    assert not result["ok"] and result["readiness"] == "partial"
    assert result["summary"]["baseline_calls"] == 2
    assert result["summary"]["unknown_calls"] == 1
    assert result["summary"]["estimated_candidate_context_share"] is None
    assert result["calls"][1]["candidate_text_bytes"] is None


def test_invalid_call_record_is_not_double_counted_as_orphan_request(tmp_path):
    first, _, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": first}])
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        db.execute("UPDATE model_calls SET record_json='secret-malformed'")
    result = analyze_run(tmp_path)
    assert not result["ok"] and result["summary"]["baseline_calls"] == 1
    assert result["summary"]["unknown_calls"] == 1
    assert "secret" not in json.dumps(result)


def test_future_test_pass_does_not_create_boundary_for_earlier_call(tmp_path):
    _, history, _ = snapshots()
    failed = history + observation("fail", "", tool="run_tests", metadata={"exit_code": 1, "test_selection": "tests"})
    passed = failed + observation("pass", "", tool="run_tests", metadata={"exit_code": 0, "test_selection": "tests"})
    write_run(tmp_path, [{"actual_rendered_messages": messages} for messages in (failed, passed)])
    result = analyze_run(tmp_path)
    assert result["calls"][0]["boundary_eligibility"] == "unknown"
    assert result["calls"][1]["boundary_source"] == "test_fail_to_pass"
    assert result["calls"][1]["verified_boundary_candidate_text_bytes"] > 0


def test_boundary_snapshot_preserves_recent_protection_when_chunk_ages_later(tmp_path):
    _, at_boundary, _ = snapshots()
    later = at_boundary + observation("extra", "unrelated", path="other.json")
    boundary = marker(at_boundary)
    write_run(tmp_path, [{"actual_rendered_messages": at_boundary, "completed_phase_boundaries": [boundary]},
                         {"actual_rendered_messages": later, "completed_phase_boundaries": [boundary]}], recent=2)
    result = analyze_run(tmp_path)
    assert result["ok"]
    assert result["calls"][1]["candidate_count"] == 1
    assert result["calls"][1]["verified_boundary_candidate_text_bytes"] == 0


def test_later_covering_read_does_not_erase_earlier_boundary_evidence(tmp_path):
    _, at_boundary, _ = snapshots()
    later = at_boundary + observation("newer", "even newer value")
    write_run(tmp_path, [{"actual_rendered_messages": later, "completed_phase_boundaries": [marker(at_boundary)]}])
    call, = analyze_run(tmp_path)["calls"]
    old_size = len(at_boundary[2]["content"][0]["content"].encode())
    assert call["candidate_count"] == 2
    assert call["verified_boundary_candidate_text_bytes"] == old_size


def test_forged_or_future_boundary_prefix_is_rejected(tmp_path):
    first, second, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": first, "completed_phase_boundaries": [marker(second)]}])
    result = analyze_run(tmp_path)
    assert not result["ok"] and result["calls"][0]["status"] == "unknown"


def test_future_labeled_boundary_is_rejected_even_when_prefix_hash_matches(tmp_path):
    first, _, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": first, "phase_index": 0,
                         "completed_phase_boundaries": [marker(first)]}])
    result = analyze_run(tmp_path)
    assert not result["ok"] and result["calls"][0]["status"] == "unknown"


def test_final_ledger_pins_and_future_invalid_chunk_do_not_change_earlier_call(tmp_path):
    first, second, _ = snapshots()
    raw = first[2]["content"][0]["content"]
    parsed = json.loads(raw)
    metadata = {**parsed, "version": parsed["sha256"], "start_line": 1, "end_line": 1}
    event = Event(id="original", session_id="s", seq=0, role="tool", kind="file_read", tool_name="read_file",
                  tool_call_id="old", args={"path": "data.json"}, content=raw, metadata=metadata)
    chunk = Chunk(id="c0", session_id="s", event_id=event.id, tool_call_id="old", seq=0,
                  kind="file_read", content=raw, metadata=metadata)
    future = event.model_copy(update={"id": "future", "seq": 99, "tool_call_id": "not-yet-seen"})
    future_chunk = chunk.model_copy(update={"id": "c99", "seq": 99, "event_id": "future", "tool_call_id": "not-yet-seen"})
    write_run(tmp_path, [{"actual_rendered_messages": second}], archive=[(event, chunk), (future, future_chunk)])
    with sqlite3.connect(tmp_path / "ledger.sqlite") as db:
        db.execute("UPDATE chunks SET pinned=1 WHERE id='c0'")
        db.execute("DROP TRIGGER chunks_no_content_update")
        db.execute("UPDATE chunks SET content_sha256='invalid' WHERE id='c99'")
    result = analyze_run(tmp_path)
    assert result["ok"] and result["calls"][0]["candidate_count"] == 1


def test_simulation_per_session_layout_and_mixed_modes_stay_separate(tmp_path):
    first, second, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": first}, {"actual_rendered_messages": second}],
              session="demo", mode="simulation", per_session=True)
    simulation = analyze_run(tmp_path)
    assert simulation["ok"] and simulation["summary"]["mode"] == "simulation"
    assert simulation["summary"]["estimated_candidate_context_share"] > 0
    assert simulation["summary"]["actual_input_cost_usd"] is None
    write_run(tmp_path, [{"actual_rendered_messages": deepcopy(second)}], session="live", per_session=True)
    mixed = analyze_run(tmp_path)
    assert mixed["ok"] and mixed["summary"]["mode"] == "mixed"
    assert mixed["summary"]["estimated_candidate_context_share"] is None
    assert mixed["summary"]["actual_input_cost_usd"] is None
    assert {row["mode"] for row in mixed["per_mode"]} == {"simulation", "live"}


@pytest.mark.parametrize("which", ["plan", "trace"])
def test_symlink_cannot_read_outside_run(tmp_path, which):
    root = tmp_path / "run"
    first, _, _ = snapshots()
    write_run(root, [{"actual_rendered_messages": first}])
    external = tmp_path / "external.json"
    external.write_text("secret-unreadable-invalid-json")
    target = root / "plan.json" if which == "plan" else root / "s" / "trace.json"
    target.unlink()
    target.symlink_to(external)
    result = analyze_run(root)
    assert not result["ok"]
    assert "secret" not in json.dumps(result)


def test_read_only_and_missing_evidence(tmp_path):
    missing = analyze_run(tmp_path)
    assert not missing["ok"] and list(tmp_path.iterdir()) == []
    first, _, _ = snapshots()
    write_run(tmp_path, [{"actual_rendered_messages": first}])
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert analyze_run(tmp_path)["ok"]
    assert all(path.read_bytes() == original for path, original in files.items())


def test_empty_baseline_session_is_unavailable_not_zero_opportunity(tmp_path):
    write_run(tmp_path, [])
    result = analyze_run(tmp_path)
    assert not result["ok"] and result["readiness"] == "unavailable"
    assert result["summary"]["estimated_candidate_context_share"] is None
