from decimal import Decimal

import pytest

from contextlab.core.models import Pricing
from contextlab.ledger.pricing import normalize_anthropic_usage, price_usage, worst_case_cost


@pytest.fixture
def prices():
    return Pricing(model="test", input_per_million=Decimal("3"), cache_write_5m_per_million=Decimal("3.75"),
                   cache_write_1h_per_million=Decimal("6"), cache_read_per_million=Decimal("0.3"),
                   output_per_million=Decimal("15"))


def test_cache_write_total_is_not_double_counted(prices):
    raw = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 500,
           "cache_creation_input_tokens": 300,
           "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 200}}
    usage = normalize_anthropic_usage(raw)
    assert usage.total_input_tokens == 900
    assert price_usage(usage, prices) == Decimal("0.002175")


def test_aggregate_only_and_partial_breakdown(prices):
    usage = normalize_anthropic_usage({"input_tokens": 0, "output_tokens": 0,
                                     "cache_creation_input_tokens": 1000}, "1h")
    assert price_usage(usage, prices) == Decimal("0.006")
    with pytest.raises(ValueError, match="disagrees"):
        normalize_anthropic_usage({"input_tokens": 0, "output_tokens": 0,
                                  "cache_creation_input_tokens": 2,
                                  "cache_creation": {"ephemeral_5m_input_tokens": 1}})
    assert normalize_anthropic_usage({"input_tokens": 0, "output_tokens": 0,
                                     "cache_creation": {"ephemeral_1h_input_tokens": 100}}).cache_write_1h_tokens == 100


@pytest.mark.parametrize("raw", [{}, {"input_tokens": 1}, {"input_tokens": True, "output_tokens": 0},
                                 {"input_tokens": -1, "output_tokens": 0},
                                 {"input_tokens": 1.5, "output_tokens": 0}])
def test_incomplete_or_invalid_usage_is_not_silently_free(raw):
    with pytest.raises(ValueError):
        normalize_anthropic_usage(raw)


def test_reservation_includes_cold_cache_and_max_output(prices):
    assert worst_case_cost(1000, 100, prices) == Decimal("0.0075")
