"""Deterministic synthetic deployment audits; model responses are never scripted.

Fixture generation is local. Reference answers stay outside the agent workspace.
The runner replaces current snapshots before phase two, preserving historical
observations only in the conversation/archive or in notes the agent chose to make.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INITIAL = "2026-10-01"
CURRENT = "2026-10-10"
REGIONS = ("ap-south-1", "eu-west-1", "us-east-1")
OUTPUT_NAMES = ("audit-initial.json", "audit-current.json", "history.json", "handoff.json")
RULES = {
    "R01": "image.digest must be a nonempty sha256 digest; tags alone are insufficient.",
    "R02": "replicas must be at least 3 for critical tier and at least 2 for standard tier.",
    "R03": "health.readiness_probe must be true.",
    "R04": "security.run_as_non_root must be true.",
    "R05": "security.privileged must be false.",
    "R06": "resources.memory_limit_mb must be greater than zero.",
    "R07": "network.admin_ingress_cidrs must not contain 0.0.0.0/0.",
    "R08": "secrets.provider must be external_store.",
    "R09": "resources.cpu_limit_millis must be at least resources.cpu_request_millis.",
    "R10": "For critical tier, rollout.max_unavailable must equal zero.",
}


def _digest(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _record(index: int, version: int) -> dict[str, Any]:
    service = f"svc-{index:03d}"
    critical = index % 4 == 0
    record: dict[str, Any] = {
        "service_id": service,
        "description": f"Deployment inventory for {'transaction' if critical else 'support'} service {service}.",
        "owner_team": ("payments", "identity", "fulfilment", "observability")[index % 4],
        "environment": "production",
        "region": REGIONS[index % len(REGIONS)],
        "tier": "critical" if critical else "standard",
        "replicas": 1 if index % 13 == 0 else 3 + index % 3,
        "image": {"repository": f"registry.internal/platform/{service}",
                  "tag": "latest" if index % 17 == 0 else f"v1.{index}.0",
                  "digest": "" if index % 17 == 0 else _digest(f"{service}/image/v1")},
        "health": {"readiness_probe": index % 7 != 0, "liveness_probe": True,
                   "path": "/health/ready", "initial_delay_seconds": 5 + index % 6,
                   "timeout_seconds": 2, "failure_threshold": 3},
        "security": {"run_as_non_root": index % 19 != 0, "privileged": index % 23 == 0,
                     "read_only_root_filesystem": True, "service_account": f"sa-{service}"},
        "resources": {"memory_limit_mb": 0 if index % 11 == 0 else 512 + 128 * (index % 4),
                      "cpu_request_millis": 100 + 50 * (index % 4),
                      "cpu_limit_millis": 50 if index % 29 == 0 else 500},
        "network": {"admin_ingress_cidrs": ["0.0.0.0/0"] if index % 31 == 0 else ["10.20.0.0/16"],
                    "application_port": 8080, "admin_port": 9090,
                    "egress_dependencies": [f"db-{index % 8:02d}.internal", "telemetry.internal"]},
        "secrets": {"provider": "plain_environment" if index % 37 == 0 else "external_store",
                    "reference": f"platform/{service}/production", "rotation_days": 30},
        "rollout": {"max_unavailable": 1 if index % 20 == 0 else 0,
                    "max_surge": 1, "progress_deadline_seconds": 600},
        "rollback": {"approval_ticket": f"CHG-{hashlib.sha256((service + '/v1').encode()).hexdigest()[:12]}",
                     "window_minutes": 15 + (index * 7) % 46,
                     "previous_digest": _digest(f"{service}/previous/v1"),
                     "owner": f"oncall-{index % 8:02d}", "strategy": "restore_previous_immutable_image"},
        "observability": {"p99_latency_budget_ms": 150 + 25 * (index % 7),
                          "availability_target_percent": 99.95 if critical else 99.9,
                          "dashboard": f"dashboards.internal/services/{service}",
                          "alert_route": f"oncall-{index % 8:02d}"},
        "snapshot": INITIAL,
    }
    if version == 2:
        # Most services receive a remediation; some retain known issues. A later
        # misconfiguration also introduces new risk, so 'all fixed' is incorrect.
        if index % 3 != 0:
            record["replicas"] = max(record["replicas"], 3 if critical else 2)
            record["health"]["readiness_probe"] = True
            record["security"].update(run_as_non_root=True, privileged=False)
            record["resources"].update(memory_limit_mb=768, cpu_limit_millis=500)
            record["network"]["admin_ingress_cidrs"] = ["10.20.0.0/16"]
            record["secrets"]["provider"] = "external_store"
            record["rollout"]["max_unavailable"] = 0
        record["image"].update(tag=f"v2.{index}.0", digest=_digest(f"{service}/image/v2"))
        if index % 26 == 2:
            record["replicas"] = 1
        if index % 22 == 5:
            record["health"]["readiness_probe"] = False
        record["rollback"].update(
            approval_ticket=f"CHG-{hashlib.sha256((service + '/v2').encode()).hexdigest()[:12]}",
            window_minutes=90 + index % 17,
            previous_digest=_digest(f"{service}/previous/v2"),
        )
        record["snapshot"] = CURRENT
    return record


def _violations(record: dict[str, Any]) -> list[str]:
    checks = (
        bool(record["image"]["digest"]),
        record["replicas"] >= (3 if record["tier"] == "critical" else 2),
        record["health"]["readiness_probe"],
        record["security"]["run_as_non_root"],
        not record["security"]["privileged"],
        record["resources"]["memory_limit_mb"] > 0,
        "0.0.0.0/0" not in record["network"]["admin_ingress_cidrs"],
        record["secrets"]["provider"] == "external_store",
        record["resources"]["cpu_limit_millis"] >= record["resources"]["cpu_request_millis"],
        record["tier"] != "critical" or record["rollout"]["max_unavailable"] == 0,
    )
    return [f"R{index:02d}" for index, passed in enumerate(checks, 1) if not passed]


def _encode(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


@dataclass(frozen=True)
class FileAuditTask:
    id: str
    title: str
    service_count: int

    @property
    def snapshot_paths(self) -> tuple[str, ...]:
        return tuple(f"deployments/shard-{letter}.json" for letter in "abcd")

    @property
    def history_targets(self) -> tuple[str, ...]:
        indexes = (1, 11, 17, 29, 35, 47) if self.service_count >= 48 else (1, 3, 5, 7)
        return tuple(f"svc-{index:03d}" for index in indexes if index < self.service_count)

    def snapshot(self, version: int) -> dict[str, str]:
        if version not in (1, 2):
            raise ValueError("Only fixture snapshot versions 1 and 2 exist")
        records = [_record(index, version) for index in range(self.service_count)]
        size = (self.service_count + 3) // 4
        return {path: _encode({"snapshot": INITIAL if version == 1 else CURRENT,
                              "services": records[offset * size:(offset + 1) * size]})
                for offset, path in enumerate(self.snapshot_paths)}

    @property
    def phases(self) -> tuple[str, ...]:
        paths = ", ".join(self.snapshot_paths)
        return (
            "Audit the initial synthetic production-deployment snapshot. Read policy.md and "
            f"all four snapshot files in full: {paths}. This is a file-reading workload: "
            "use read_file to inspect each full file (continue at the returned offset if needed). "
            "Check every service against every applicable R01-R10 rule. Write audit-initial.json "
            f'with exactly {{"snapshot":"{INITIAL}","findings":[{{"service":"svc-000",'
            '"rules":["R01"]}}]}. Include only violating services, sorted by service; sort rules. '
            "The example is a schema, not an answer. Run tests/test_initial.py. Original deployment "
            "files will be replaced at the next phase; you may keep useful notes. Do not edit input "
            "snapshots, policy, or tests. Explain actual checks performed.",

            f"The operator has now replaced all four deployment files with snapshot {CURRENT}. "
            f"Reread the complete current contents of every previous path: {paths}. Do not reuse "
            "the old snapshot as current. Apply the same policy to all current services and write "
            "audit-current.json with the same schema and current snapshot date. Preserve the "
            "initial audit. Run tests/test_current.py. Distinguish remaining and new issues; "
            "do not assume the rollout fixed everything.",

            "A rollback reviewer needs historical facts that the current snapshot overwrote. "
            f"For these services, sorted by ID: {', '.join(self.history_targets)}, retrieve from "
            "your original observations, archived tool outputs, or notes: initial image.digest, "
            "initial rollback.approval_ticket, and initial rollback.window_minutes. Also include "
            "the current image.digest. Do not substitute current values for historical values. "
            'Write history.json with exactly {"initial_snapshot":"2026-10-01",'
            '"current_snapshot":"2026-10-10","requests":[{"service":"svc-000",'
            '"initial_image_digest":"...","initial_rollback_ticket":"...",'
            '"initial_rollback_window_minutes":15,"current_image_digest":"..."}]}. '
            "An initial empty digest must remain an empty string. Run tests/test_history.py.",

            "Finish the operator handoff from the two audits and current deployment snapshot. "
            "Treat each (service, rule) pair as one finding. Count resolved pairs (initial only), "
            "introduced pairs (current only), and unchanged pairs (both). A critical service is "
            "blocked if it has ANY current finding; otherwise it is ready. Write handoff.json "
            'with exactly {"resolved_findings":0,"introduced_findings":0,"unchanged_findings":0,'
            '"blocked_critical_services":["svc-000"],"ready_critical_by_region":'
            '{"ap-south-1":[],"eu-west-1":[],"us-east-1":[]}}. '
            "Numbers and IDs shown are schema examples. Sort every service list. Include all "
            "three region keys, even if empty. Run all public tests and review all four output "
            "JSON files before reporting completion. Do not change snapshots, policy, or tests.",
        )

    def create(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "deployments").mkdir()
        (directory / "tests").mkdir()
        (directory / "policy.md").write_text(
            "# Synthetic local DevOps audit fixture\n\n"
            "These generated records are a diagnostic dataset, not customer telemetry. "
            "No real credentials or network access are involved. Apply only the following rules; "
            "other fields provide deployment and rollback context.\n\n"
            + "\n".join(f"- {key}: {value}" for key, value in RULES.items()) + "\n"
        )
        (directory / "README.md").write_text(
            f"Synthetic deployment audit with {self.service_count} services and two snapshots.\n"
            "Use the shared read_file tool to inspect complete snapshots. Produce the JSON files "
            "requested in each phase. Public tests verify structure only; the independent grader "
            "checks exact answers and input integrity. You may write notes or helper files. "
            "Do not edit deployments/, policy.md, or tests/.\n"
        )
        self.prepare_phase(directory, 0)
        for name, output in zip(("initial", "current", "history", "handoff"), OUTPUT_NAMES, strict=True):
            source = (
                "import json\nfrom pathlib import Path\n\n"
                f"def test_{name}_artifact_is_json_object():\n"
                f"    value = json.loads(Path({output!r}).read_text())\n"
                "    assert isinstance(value, dict) and value\n"
            )
            (directory / "tests" / f"test_{name}.py").write_text(source)

    def prepare_phase(self, directory: Path, index: int) -> None:
        if not 0 <= index < len(self.phases):
            raise ValueError("Invalid audit phase index")
        if index in (0, 1):
            for path, content in self.snapshot(index + 1).items():
                (directory / path).write_text(content)

    def public_phase_status(self, directory: Path, index: int) -> dict[str, Any]:
        """Check only schemas disclosed in prompts; never consult reference answers.

        These structural observations may be returned to the agent. They do not
        establish that findings, counts, or historical values are correct.
        """
        if not 0 <= index < len(self.phases):
            raise ValueError("Invalid audit phase index")
        issues = []
        for name in OUTPUT_NAMES[:index + 1]:
            path = directory / name
            try:
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    reason = "Output must be a regular file inside the task workspace."
                elif not path.is_file():
                    reason = "Required JSON deliverable is missing."
                elif path.stat().st_size > 512_000:
                    reason = "JSON deliverable exceeds the 512000-byte workspace limit."
                else:
                    value = json.loads(path.read_text())
                    reason = self._public_schema_error(name, value)
            except (OSError, UnicodeError, ValueError, RecursionError):
                reason = "Deliverable must contain a readable complete JSON object."
            if reason:
                issues.append({"file": name, "reason": reason})
        return {"ok": not issues, "basis": "public_json_schema_only", "issues": issues}

    def _public_schema_error(self, name: str, value: Any) -> str | None:
        def service_list(items: Any) -> bool:
            return (isinstance(items, list)
                    and all(isinstance(item, str) and re.fullmatch(r"svc-\d{3}", item)
                            for item in items)
                    and items == sorted(set(items)))

        def nonnegative_integer(item: Any) -> bool:
            return type(item) is int and item >= 0

        if not isinstance(value, dict):
            return "Deliverable must be a JSON object with exactly the requested keys."
        if name in {"audit-initial.json", "audit-current.json"}:
            date = INITIAL if name == "audit-initial.json" else CURRENT
            if set(value) != {"snapshot", "findings"} or value.get("snapshot") != date:
                return f"Use exactly snapshot and findings keys, with snapshot {date}."
            findings = value["findings"]
            if not isinstance(findings, list):
                return "findings must be a list of service/rules objects."
            services = []
            for finding in findings:
                if not isinstance(finding, dict) or set(finding) != {"service", "rules"}:
                    return "Each finding must contain exactly service and rules."
                services.append(finding["service"])
                rules = finding["rules"]
                if (not isinstance(rules, list) or not rules
                        or any(not isinstance(rule, str) or rule not in RULES for rule in rules)
                        or rules != sorted(set(rules))):
                    return "Each rules list must be nonempty, unique and sorted, using R01-R10."
            if not service_list(services):
                return "Findings must use unique svc-NNN service IDs sorted by service."
        elif name == "history.json":
            if (set(value) != {"initial_snapshot", "current_snapshot", "requests"}
                    or value.get("initial_snapshot") != INITIAL
                    or value.get("current_snapshot") != CURRENT):
                return "Use initial_snapshot, current_snapshot and requests with the requested dates."
            requests = value["requests"]
            fields = {"service", "initial_image_digest", "initial_rollback_ticket",
                      "initial_rollback_window_minutes", "current_image_digest"}
            if not isinstance(requests, list):
                return "requests must be a list of historical lookup objects."
            for request in requests:
                if not isinstance(request, dict) or set(request) != fields:
                    return "Each historical lookup must contain exactly the five requested fields."
                if (not isinstance(request["initial_image_digest"], str)
                        or not isinstance(request["current_image_digest"], str)
                        or not isinstance(request["initial_rollback_ticket"], str)
                        or not request["initial_rollback_ticket"]
                        or not nonnegative_integer(request["initial_rollback_window_minutes"])):
                    return "Historical digests/tickets must be strings; window minutes a nonnegative integer."
            if [request["service"] for request in requests] != list(self.history_targets):
                return "Include exactly the service IDs requested in this phase, in sorted order."
        else:
            counters = {"resolved_findings", "introduced_findings", "unchanged_findings"}
            if set(value) != counters | {"blocked_critical_services", "ready_critical_by_region"}:
                return "Use exactly the five handoff keys requested in the prompt."
            if any(not nonnegative_integer(value[key]) for key in counters):
                return "Finding counts must be nonnegative integers."
            regions = value["ready_critical_by_region"]
            if (not isinstance(regions, dict) or set(regions) != set(REGIONS)
                    or not all(service_list(items) for items in regions.values())
                    or not service_list(value["blocked_critical_services"])):
                return "Use all three requested region keys and unique sorted svc-NNN service lists."
        return None

    def reference_outputs(self) -> dict[str, str]:
        initial = [_record(index, 1) for index in range(self.service_count)]
        current = [_record(index, 2) for index in range(self.service_count)]
        def findings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{"service": record["service_id"], "rules": rules}
                    for record in records if (rules := _violations(record))]
        initial_findings, current_findings = findings(initial), findings(current)
        initial_pairs = {(row["service"], rule) for row in initial_findings for rule in row["rules"]}
        current_pairs = {(row["service"], rule) for row in current_findings for rule in row["rules"]}
        initial_by_id = {row["service_id"]: row for row in initial}
        current_by_id = {row["service_id"]: row for row in current}
        history = [{"service": service,
                    "initial_image_digest": initial_by_id[service]["image"]["digest"],
                    "initial_rollback_ticket": initial_by_id[service]["rollback"]["approval_ticket"],
                    "initial_rollback_window_minutes": initial_by_id[service]["rollback"]["window_minutes"],
                    "current_image_digest": current_by_id[service]["image"]["digest"]}
                   for service in self.history_targets]
        blocked = [row["service_id"] for row in current
                   if row["tier"] == "critical" and _violations(row)]
        ready = {region: [row["service_id"] for row in current
                          if row["tier"] == "critical" and row["region"] == region
                          and not _violations(row)] for region in REGIONS}
        return {
            "audit-initial.json": _encode({"snapshot": INITIAL, "findings": initial_findings}),
            "audit-current.json": _encode({"snapshot": CURRENT, "findings": current_findings}),
            "history.json": _encode({"initial_snapshot": INITIAL, "current_snapshot": CURRENT,
                                     "requests": history}),
            "handoff.json": _encode({"resolved_findings": len(initial_pairs - current_pairs),
                                     "introduced_findings": len(current_pairs - initial_pairs),
                                     "unchanged_findings": len(initial_pairs & current_pairs),
                                     "blocked_critical_services": blocked,
                                     "ready_critical_by_region": ready}),
        }

    def phase_check_source(self, index: int) -> str:
        if not 0 <= index < len(self.phases):
            raise ValueError("Invalid audit phase index")
        expected = {name: json.loads(content) for name, content in self.reference_outputs().items()
                    if name in OUTPUT_NAMES[:index + 1]}
        snapshot_hashes = {path: hashlib.sha256(content.encode()).hexdigest()
                           for path, content in self.snapshot(1 if index == 0 else 2).items()}
        return (
            "import hashlib\nimport json\nfrom pathlib import Path\n"
            f"expected = json.loads({json.dumps(expected)!r})\n"
            "for name, value in expected.items():\n"
            "    actual = json.loads(Path(name).read_text())\n"
            "    assert json.dumps(actual, sort_keys=True) == json.dumps(value, sort_keys=True), "
            "f'Incorrect audit artifact: {name}'\n"
            f"snapshot_hashes = {snapshot_hashes!r}\n"
            "for name, digest in snapshot_hashes.items():\n"
            "    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, "
            "f'Input snapshot modified: {name}'\n"
            "print('CHECKS_PASSED')\n"
        )

    def checker_source(self) -> str:
        return self.phase_check_source(len(self.phases) - 1)


FILE_AUDIT_TASKS = {
    "file_audit_long": FileAuditTask("file_audit_long", "Audit an evolving deployment inventory", 48),
    "file_audit_short": FileAuditTask("file_audit_short", "Short deployment audit control", 8),
}
