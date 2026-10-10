"""Framework-independent records shared by the runtime, ledger and harness."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Message = dict[str, Any]
Level = Literal["full", "long", "short", "hidden"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Event(Record):
    id: str
    session_id: str
    seq: int = Field(ge=0)
    role: str = "tool"
    kind: str
    tool_name: str = ""
    tool_call_id: str = ""
    args: dict[str, Any] = Field(default_factory=dict)
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    size_tokens: int = Field(default=0, ge=0)
    created_at: str = ""


class Chunk(Record):
    id: str
    session_id: str
    event_id: str
    tool_call_id: str
    kind: str
    content: str
    seq: int = Field(ge=0)
    size_tokens: int = Field(default=0, ge=0)
    refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)
    long_text: str = ""
    short_text: str = ""
    signpost_text: str = ""
    pinned: bool = False


class Signal(Record):
    chunk_id: str
    signal_type: str
    triggered_by_event_id: str
    reason: str
    target: Level = "hidden"


class Pricing(Record):
    """USD per million tokens; synthetic demo prices must be labeled as such."""
    model: str
    input_per_million: Decimal = Field(ge=0)
    cache_write_5m_per_million: Decimal = Field(ge=0)
    cache_write_1h_per_million: Decimal = Field(ge=0)
    cache_read_per_million: Decimal = Field(ge=0)
    output_per_million: Decimal = Field(ge=0)
    source: str = "synthetic"
    verified_at: str = ""


class Usage(Record):
    input_tokens: int = Field(default=0, ge=0)
    cache_write_5m_tokens: int = Field(default=0, ge=0)
    cache_write_1h_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total_input_tokens(self) -> int:
        return (self.input_tokens + self.cache_write_5m_tokens
                + self.cache_write_1h_tokens + self.cache_read_tokens)


class CompactionConfig(Record):
    remaining_turns: int = Field(default=8, ge=1)
    safety_factor: Decimal = Field(default=Decimal("1.2"), ge=1)
    recent_outputs: int = Field(default=2, ge=0)
    recall_probability: Decimal = Field(default=Decimal("0.1"), ge=0, le=1)
    expected_recall_cost_usd: Decimal = Field(default=Decimal("0.01"), ge=0)
    policy_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    min_saved_tokens: int = Field(default=128, ge=0)


class CostInputs(Record):
    old_tokens: int = Field(ge=0)
    new_tokens: int = Field(ge=0)
    stable_tokens: int = Field(default=0, ge=0)
    remaining_turns: int = Field(ge=1)
    cache_warm: bool = True
    decision_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    expected_recall_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    safety_factor: Decimal = Field(default=Decimal("1.2"), ge=1)


class CostDecision(Record):
    accepted: bool
    stay_cost_usd: Decimal
    move_cost_usd: Decimal
    reason: str


class ViewState(Record):
    epoch: int = 0
    levels: dict[str, Level] = Field(default_factory=dict)
    pinned_ids: tuple[str, ...] = ()
    last_boundary_id: str = ""
    duplicate_sources: dict[str, str] = Field(default_factory=dict)


class ProjectionResult(Record):
    messages: list[Message]
    state: ViewState
    signals: list[Signal] = Field(default_factory=list)
    decision: CostDecision | None = None
    rebuilt: bool = False
    fallback: bool = False
    error: str | None = None
    estimated_tokens_saved: int = 0
    no_op_reason: str | None = None
    semantic_decision: dict[str, Any] | None = None
    exact_duplicate_retirements: dict[str, str] = Field(default_factory=dict)


class RunResult(Record):
    session_id: str
    arm: str
    task_id: str
    repeat: int
    mode: Literal["simulation", "live"]
    solved: bool
    cost_usd: Decimal = Decimal("0")
    turns: int = 0
    recalls: int = 0
    rebuilds: int = 0
    latency_ms: float = 0
    usage: Usage = Field(default_factory=Usage)
    decision_calls: int = Field(default=0, ge=0)
    decision_cost_usd: Decimal = Field(default=Decimal("0"), ge=0)
    decision_usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    notes: str = ""
