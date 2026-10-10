"""Explicit configuration for the optional Jev semantic decision service."""

from decimal import Decimal
from typing import Literal

from pydantic import Field

from contextlab.core.models import Record


class JevConfig(Record):
    """Explicit experimental decision service settings and reviewed price snapshot."""

    model: Literal["jev-1.13.0"] = "jev-1.13.0"
    input_per_million: Decimal = Field(
        default=Decimal("0.042"), ge=Decimal("0.042"), le=Decimal("0.042"),
    )
    output_per_million: Decimal = Field(default=Decimal("0"), ge=0, le=0)
    pricing_source: Literal["https://docs.typesafe.ai/models"] = "https://docs.typesafe.ai/models"
    verified_at: Literal["2026-10-10"] = "2026-10-10"
    max_calls_per_session: int = Field(default=8, ge=1, le=40)
    max_state_bytes: int = Field(default=24000, ge=1, le=24000)
    max_candidates: int = Field(default=8, ge=1, le=8)
    keep_threshold: float = Field(default=0.15, ge=0, le=0.5)
    timeout_seconds: float = Field(default=10, gt=0, le=30)
    reservation_input_tokens: Literal[65536] = 65536
