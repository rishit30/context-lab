"""Pure, explicit warm/cold cache forecasts. These are not billed savings."""

from decimal import Decimal

from contextlab.core.models import CostDecision, CostInputs, Pricing

MILLION = Decimal(1_000_000)


def decide_rebuild(inputs: CostInputs, pricing: Pricing) -> CostDecision:
    if inputs.stable_tokens > min(inputs.old_tokens, inputs.new_tokens):
        raise ValueError("stable_tokens must be contained in both request views")
    read = pricing.cache_read_per_million / MILLION
    write = pricing.cache_write_5m_per_million / MILLION
    turns = inputs.remaining_turns
    stable = inputs.stable_tokens
    old_suffix = inputs.old_tokens - stable
    new_suffix = inputs.new_tokens - stable
    # The unchanged eligible prefix is still read, not rewritten, on a warm
    # cache. A cold cache writes that prefix in both branches.
    stable_cost = Decimal(stable) * (
        (turns * read) if inputs.cache_warm else (write + (turns - 1) * read)
    )
    stay_suffix = Decimal(old_suffix) * (
        (turns * read) if inputs.cache_warm else (write + (turns - 1) * read)
    )
    move_suffix = Decimal(new_suffix) * (write + (turns - 1) * read)
    stay = stable_cost + stay_suffix
    move = stable_cost + move_suffix + inputs.decision_cost_usd + inputs.expected_recall_cost_usd
    accepted = inputs.new_tokens < inputs.old_tokens and stay > inputs.safety_factor * move
    reason = (
        "Estimated savings exceed the configured safety margin."
        if accepted
        else "Keep the current view: estimated savings do not cover rebuild, recall, and safety margin."
    )
    return CostDecision(accepted=accepted, stay_cost_usd=stay, move_cost_usd=move, reason=reason)
