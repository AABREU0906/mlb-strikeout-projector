"""
American odds conversion and vig (overround) removal.

Method (documented):
1. Convert each side's American odds to a raw implied probability:
     positive odds (e.g. +150):  implied = 100 / (odds + 100)
     negative odds (e.g. -130):  implied = -odds / (-odds + 100)
2. Sum raw implied probabilities for the two-way market (over + under).
   This sum is > 1.0 because of the sportsbook's built-in margin ("vig").
3. Remove the vig by normalizing (the standard "multiplicative" / basic
   de-vig method): fair_prob_side = raw_prob_side / sum(raw_probs).
   This is the simplest defensible de-vig method. It assumes the vig is
   distributed proportionally across both sides, which is a documented
   simplification -- more advanced methods (Shin's method, power-method)
   exist but require additional assumptions; we use the transparent
   proportional method and disclose that choice everywhere it's used.
4. Fair American odds are then derived by inverting the American-odds
   formula on the de-vigged probability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def american_odds_to_implied_probability(odds: int) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0.")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def implied_probability_to_fair_american_odds(prob: float) -> int:
    if not (0.0 < prob < 1.0):
        raise ValueError("Probability must be strictly between 0 and 1.")
    if prob >= 0.5:
        odds = -1 * (prob / (1 - prob)) * 100.0
    else:
        odds = ((1 - prob) / prob) * 100.0
    return int(round(odds))


@dataclass
class VigRemovalResult:
    raw_over_prob: float
    raw_under_prob: float
    overround: float  # total "vig" -- raw_over_prob + raw_under_prob - 1.0
    vig_free_over_prob: float
    vig_free_under_prob: float
    fair_over_odds: int
    fair_under_odds: int
    method: str = "proportional_two_way_devig"


def remove_vig_two_way(over_odds: int, under_odds: int) -> VigRemovalResult:
    raw_over = american_odds_to_implied_probability(over_odds)
    raw_under = american_odds_to_implied_probability(under_odds)
    total = raw_over + raw_under
    overround = total - 1.0

    if total <= 0:
        raise ValueError("Invalid odds: implied probabilities sum to <= 0.")

    fair_over = raw_over / total
    fair_under = raw_under / total

    return VigRemovalResult(
        raw_over_prob=raw_over,
        raw_under_prob=raw_under,
        overround=overround,
        vig_free_over_prob=fair_over,
        vig_free_under_prob=fair_under,
        fair_over_odds=implied_probability_to_fair_american_odds(fair_over),
        fair_under_odds=implied_probability_to_fair_american_odds(fair_under),
    )


def moneyline_to_implied_probability(moneyline: Optional[int]) -> Optional[float]:
    if moneyline is None:
        return None
    return american_odds_to_implied_probability(moneyline)


def expected_value_per_dollar(model_prob: float, american_odds: int) -> float:
    """EV per $1 staked at the given American odds, using model_prob as the
    true win probability. Positive = the bet has positive expected value
    under the model's own probability estimate (not a guarantee)."""
    if american_odds > 0:
        payout_per_dollar = american_odds / 100.0
    else:
        payout_per_dollar = 100.0 / (-american_odds)
    win_ev = model_prob * payout_per_dollar
    lose_ev = (1 - model_prob) * (-1.0)
    return win_ev + lose_ev


def classify_edge(model_prob: float, vig_free_prob: float) -> str:
    """Neutral, non-promissory edge labels per project rules."""
    diff = model_prob - vig_free_prob
    abs_diff = abs(diff)
    if abs_diff < 0.02:
        return "No meaningful edge"
    if abs_diff < 0.05:
        return "Small estimated edge"
    if abs_diff < 0.09:
        return "Moderate estimated edge"
    return "Large estimated edge with elevated uncertainty"
