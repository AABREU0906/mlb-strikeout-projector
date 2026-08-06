"""
American odds conversion and vig (overround) removal.

Method:
1. Convert each side's American odds to a raw implied probability.
2. Add the two implied probabilities to calculate the market overround.
3. Remove the vig by dividing each raw probability by the total.
4. Convert each vig-free probability back into fair American odds.

The proportional two-way de-vig method is transparent and easy to audit,
although more advanced methods may produce different estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def american_odds_to_implied_probability(odds: int) -> float:
    """Convert American odds into raw implied probability."""
    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds > 0:
        return 100.0 / (odds + 100.0)

    return (-odds) / ((-odds) + 100.0)


def implied_probability_to_fair_american_odds(prob: float) -> int:
    """Convert a probability into rounded fair American odds."""
    if not 0.0 < prob < 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")

    if prob >= 0.5:
        odds = -(prob / (1.0 - prob)) * 100.0
    else:
        odds = ((1.0 - prob) / prob) * 100.0

    return int(round(odds))


@dataclass(frozen=True)
class VigRemovalResult:
    """Result of removing vig from a two-way betting market."""

    raw_over_prob: float
    raw_under_prob: float
    overround: float
    vig_free_over_prob: float
    vig_free_under_prob: float
    fair_over_odds: int
    fair_under_odds: int
    method: str = "proportional_two_way_devig"


def remove_vig_two_way(over_odds: int, under_odds: int) -> VigRemovalResult:
    """Remove vig from a two-way market using proportional normalization."""
    raw_over = american_odds_to_implied_probability(over_odds)
    raw_under = american_odds_to_implied_probability(under_odds)

    total = raw_over + raw_under

    if total <= 0.0:
        raise ValueError("Invalid odds: implied probabilities sum to <= 0.")

    fair_over = raw_over / total
    fair_under = raw_under / total

    return VigRemovalResult(
        raw_over_prob=raw_over,
        raw_under_prob=raw_under,
        overround=total - 1.0,
        vig_free_over_prob=fair_over,
        vig_free_under_prob=fair_under,
        fair_over_odds=implied_probability_to_fair_american_odds(fair_over),
        fair_under_odds=implied_probability_to_fair_american_odds(fair_under),
    )


def moneyline_to_implied_probability(
    moneyline: Optional[int],
) -> Optional[float]:
    """Convert an optional moneyline into implied probability."""
    if moneyline is None:
        return None

    return american_odds_to_implied_probability(moneyline)


def expected_value_per_dollar(
    model_prob: float,
    american_odds: int,
) -> float:
    """
    Calculate expected value per $1 staked.

    A positive value indicates positive estimated expected value according
    to the supplied model probability. It does not guarantee a profit.
    """
    if not 0.0 <= model_prob <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")

    if american_odds == 0:
        raise ValueError("American odds cannot be 0.")

    if american_odds > 0:
        profit_per_dollar = american_odds / 100.0
    else:
        profit_per_dollar = 100.0 / abs(american_odds)

    win_value = model_prob * profit_per_dollar
    loss_value = (1.0 - model_prob) * -1.0

    return win_value + loss_value


def classify_edge(model_prob: float, vig_free_prob: float) -> str:
    """
    Classify the absolute difference between model and market probability.

    Thresholds:
    - Below 1 percentage point: no meaningful edge
    - 1 through 5 percentage points: small estimated edge
    - More than 5 through 10 percentage points: moderate estimated edge
    - More than 10 percentage points: large estimated edge

    The edge is rounded before classification to prevent floating-point
    precision issues at exact boundaries such as 0.55 - 0.45.
    """
    if not 0.0 <= model_prob <= 1.0:
        raise ValueError("Model probability must be between 0 and 1.")

    if not 0.0 <= vig_free_prob <= 1.0:
        raise ValueError("Vig-free probability must be between 0 and 1.")

    edge = round(abs(model_prob - vig_free_prob), 10)

    if edge < 0.01:
        return "No meaningful edge"

    if edge <= 0.05:
        return "Small estimated edge"

    if edge <= 0.10:
        return "Moderate estimated edge"

    return "Large estimated edge with elevated risk"