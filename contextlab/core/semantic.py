"""Framework-neutral semantic vetoes; rules retain authority over eligibility."""

from collections.abc import Sequence
from decimal import Decimal
from typing import Annotated, Protocol

from pydantic import Field

from contextlab.core.models import Chunk, Event, Record, Signal

NeededProbability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False, strict=True)]
CoverageCount = Annotated[int, Field(ge=0, strict=True)]


class SemanticAssessment(Record):
    """Allow-list of proposed reductions, with actual decision cost and risk.

    Probabilities describe whether removal would lose needed information given
    the complete evidence guaranteed to remain visible. A gate
    may remove candidates from a plan but cannot make other chunks eligible.
    ``fallback`` means the gate could not return a usable semantic decision.
    """

    allowed_ids: tuple[str, ...] = ()
    needed_probabilities: dict[str, NeededProbability] = Field(default_factory=dict)
    cost_usd: Decimal = Field(default=Decimal("0"), ge=0, allow_inf_nan=False)
    engine: str
    reason: str
    fallback: bool = False
    coverage: dict[str, CoverageCount] = Field(default_factory=dict)


class SemanticGate(Protocol):
    def assess(
        self,
        *,
        events: Sequence[Event],
        candidates: Sequence[Chunk],
        signals: Sequence[Signal],
        retained_chunks: Sequence[Chunk] = (),
    ) -> SemanticAssessment:
        """Assess rule-eligible reductions without mutating canonical evidence."""
        ...
