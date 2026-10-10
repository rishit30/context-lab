"""Strict configuration and canonical experiment plans."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from contextlab.core.models import CompactionConfig, Pricing, Record
from contextlab.providers.jev_config import JevConfig as JevConfig

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE_ARMS = ("baseline", "threshold_edit", "contextlab", "headroom")


class ExperimentConfig(Record):
    arms: tuple[Literal["baseline", "threshold_edit", "contextlab", "headroom"], ...]
    tasks: tuple[str, ...]
    repeats: int = Field(ge=1, le=100)
    session_cap_usd: Decimal = Field(gt=0)
    experiment_cap_usd: Decimal = Field(gt=0)
    max_model_calls: int = Field(ge=1, le=200)
    max_output_tokens: int = Field(ge=1, le=8192)
    max_input_tokens: int = Field(ge=1, le=200000)
    threshold_tokens: int = Field(ge=1)
    threshold_keep: int = Field(ge=1)
    docker_image: str
    cache_mode: Literal["enabled", "disabled"] = "enabled"
    phase_boundaries: bool = False
    decision_engine: Literal["rules", "jev"] = "rules"
    jev: JevConfig = Field(default_factory=JevConfig)

    @model_validator(mode="after")
    def distinct(self) -> ExperimentConfig:
        if not self.arms or not self.tasks:
            raise ValueError("At least one arm and task are required")
        if len(set(self.arms)) != len(self.arms) or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("Arms and tasks must be distinct")
        if self.decision_engine == "jev" and "contextlab" not in self.arms:
            raise ValueError("The Jev decision engine requires a contextlab arm")
        return self


def read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def load_config(path: Path = ROOT / "config/experiment.yaml") -> ExperimentConfig:
    return ExperimentConfig.model_validate(read_yaml(path))


def load_compaction() -> CompactionConfig:
    return CompactionConfig.model_validate(read_yaml(ROOT / "config/compaction.yaml"))


def load_pricing(path: Path = ROOT / "config/pricing.yaml") -> Pricing:
    return Pricing.model_validate(read_yaml(path))


def run_matrix(config: ExperimentConfig) -> list[dict]:
    # Rotate which arm runs first to avoid always assigning drift to the same arm.
    rows = []
    for repeat in range(config.repeats):
        for task_index, task in enumerate(config.tasks):
            offset = (repeat + task_index) % len(config.arms)
            order = config.arms[offset:] + config.arms[:offset]
            rows.extend({"arm": arm, "task_id": task, "repeat": repeat} for arm in order)
    return rows


def experiment_plan(config: ExperimentConfig, pricing: Pricing, model: str) -> dict:
    from harness.tasks import TASKS
    from harness.tasks.file_audit import FILE_AUDIT_TASKS

    unknown = set(config.tasks) - (TASKS.keys() | FILE_AUDIT_TASKS.keys())
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    payload = {
        "mode": "live", "config": config.model_dump(mode="json"),
        "pricing": pricing.model_dump(mode="json"), "model": model,
        "compaction": load_compaction().model_dump(mode="json"),
        "matrix": run_matrix(config),
        "maximum_authorized_cost_usd": str(min(
            config.experiment_cap_usd,
            config.session_cap_usd * len(run_matrix(config)),
        )),
        "evidence": "indicative pilot; repeat count alone does not establish significance",
    }
    if "headroom" in config.arms:
        from contextlab.adapters.headroom_mw import configuration
        payload["headroom"] = configuration()
    if config.decision_engine == "jev":
        reservation = (
            Decimal(config.jev.reservation_input_tokens)
            * config.jev.input_per_million / Decimal(1_000_000)
        )
        payload["decision_layer"] = {
            "engine": "jev",
            "provider": "TypeSafe",
            "model": config.jev.model,
            "settings": config.jev.model_dump(mode="json"),
            "pricing": {
                "input_per_million": str(config.jev.input_per_million),
                "output_per_million": str(config.jev.output_per_million),
                "source": config.jev.pricing_source,
                "verified_at": config.jev.verified_at,
            },
            "reservation_input_tokens": config.jev.reservation_input_tokens,
            "reservation_usd": str(reservation),
            "maximum_reserved_decision_cost_per_contextlab_session_usd": str(
                reservation * config.jev.max_calls_per_session
            ),
            "reservation_basis": (
                "Reserve 65,536 input tokens before each call, conservatively interpreting the "
                "documented 64k total-request limit. The provider supplies neither a tokenizer "
                "nor a max_tokens control; the serialized body is separately bounded to 32 KB, "
                "with at most 24,000 state bytes and eight candidates. Actual usage above "
                "the reserved bound blocks further calls."
            ),
            "data_sent": "Bounded task state and selected tool-result text are sent to TypeSafe.",
            "budget": (
                "Decision calls share the configured session and experiment spend caps; "
                "no additional spend is authorized."
            ),
            "threshold_status": (
                "The keep threshold is uncalibrated and experimental; "
                "no quality or savings advantage is established."
            ),
        }
    # Bind approval to implementation and task definitions as well as configuration.
    sources = []
    for base in (ROOT / "contextlab", ROOT / "harness"):
        for file in sorted(base.rglob("*.py")):
            sources.append((str(file.relative_to(ROOT)), hashlib.sha256(file.read_bytes()).hexdigest()))
    for filename in ("pyproject.toml", "uv.lock", "requirements.lock", "Dockerfile.task"):
        file = ROOT / filename
        if file.exists():
            sources.append((filename, hashlib.sha256(file.read_bytes()).hexdigest()))
    payload["source_hash"] = hashlib.sha256(json.dumps(sources).encode()).hexdigest()
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "approval_digest": digest}
