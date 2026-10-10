"""Exact usage accounting. Prices are supplied by configuration, never guessed."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from contextlab.core.models import Pricing, Usage

MILLION = Decimal("1000000")


def money(value: Any) -> Decimal:
    """Parse a nonnegative finite amount without a binary floating point sum."""
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Invalid monetary amount") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError("Monetary amounts must be finite and nonnegative")
    return amount


def _tokens(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def normalize_anthropic_usage(raw: Mapping[str, Any], cache_ttl: str = "5m") -> Usage:
    """Anthropic input_tokens excludes cached input; the cache total is not a bucket.

    A cache_creation breakdown, when present, must add up to the aggregate field.
    This avoids charging the aggregate *and* its two TTL sub-buckets.
    """
    if cache_ttl not in {"5m", "1h"}:
        raise ValueError("cache_ttl must be 5m or 1h")
    if "input_tokens" not in raw or "output_tokens" not in raw:
        raise ValueError("Usage is incomplete: input_tokens and output_tokens required")
    base = _tokens(raw["input_tokens"], "input_tokens")
    output = _tokens(raw["output_tokens"], "output_tokens")
    read = _tokens(raw.get("cache_read_input_tokens", 0), "cache_read_input_tokens")
    total = _tokens(raw.get("cache_creation_input_tokens", 0), "cache_creation_input_tokens")
    detail = raw.get("cache_creation")
    if detail is not None:
        if not isinstance(detail, Mapping):
            raise ValueError("cache_creation must be a mapping")
        short = _tokens(detail.get("ephemeral_5m_input_tokens", 0), "ephemeral_5m_input_tokens")
        long = _tokens(detail.get("ephemeral_1h_input_tokens", 0), "ephemeral_1h_input_tokens")
        if "cache_creation_input_tokens" in raw and short + long != total:
            raise ValueError("Cache creation total disagrees with TTL breakdown")
    else:
        short, long = (total, 0) if cache_ttl == "5m" else (0, total)
    return Usage(input_tokens=base, cache_write_5m_tokens=short,
                 cache_write_1h_tokens=long, cache_read_tokens=read, output_tokens=output)


def price_usage(usage: Usage, rates: Pricing) -> Decimal:
    return (
        usage.input_tokens * rates.input_per_million
        + usage.cache_write_5m_tokens * rates.cache_write_5m_per_million
        + usage.cache_write_1h_tokens * rates.cache_write_1h_per_million
        + usage.cache_read_tokens * rates.cache_read_per_million
        + usage.output_tokens * rates.output_per_million
    ) / MILLION


def worst_case_cost(input_tokens: int, max_output_tokens: int, rates: Pricing) -> Decimal:
    """Reserve at the most expensive input bucket, including a possible cold cache.

    The caller must supply a safe input upper bound and enforce max_output_tokens.
    Retries and separate decision/delegation calls require separate reservations.
    """
    _tokens(input_tokens, "input_tokens")
    _tokens(max_output_tokens, "max_output_tokens")
    highest_input_rate = max(
        rates.input_per_million,
        rates.cache_write_5m_per_million,
        rates.cache_write_1h_per_million,
        rates.cache_read_per_million,
    )
    return (input_tokens * highest_input_rate + max_output_tokens * rates.output_per_million) / MILLION
