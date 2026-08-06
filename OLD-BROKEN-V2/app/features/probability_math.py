"""
Shared probability math used across the strikeout and NRFI/YRFI models.

log5() lived privately inside stage2_batter_probability.py; it's pulled out
here so the NRFI/YRFI half-inning scoring model can reuse the exact same,
already-tested implementation instead of a second copy, per project rules
against duplicating utility functions.
"""
from __future__ import annotations


def log5(rate_a: float, rate_b: float, league_rate: float) -> float:
    """Bill James' log5 method: combines two independent rates against a
    league baseline. Used for pitcher-vs-batter strikeout matchups AND for
    team-offense-vs-opposing-pitcher first-inning scoring matchups -- same
    math, different inputs. Deliberately NOT a simple average of rate_a and
    rate_b (that's explicitly disallowed for both models)."""
    rate_a = min(max(rate_a, 1e-4), 1 - 1e-4)
    rate_b = min(max(rate_b, 1e-4), 1 - 1e-4)
    league_rate = min(max(league_rate, 1e-4), 1 - 1e-4)

    num = (rate_a * rate_b) / league_rate
    den = num + ((1 - rate_a) * (1 - rate_b)) / (1 - league_rate)
    return num / den


def combine_independent_no_score_probabilities(
    away_no_score_prob: float, home_no_score_prob: float
) -> tuple[float, float]:
    """P(NRFI) = P(away doesn't score) * P(home doesn't score) -- treating
    the two half-innings as independent events, per project spec. This is
    explicitly NOT an average of the two probabilities.

    Returns (nrfi_probability, yrfi_probability), which always sum to 1.0.
    """
    away_no_score_prob = min(max(away_no_score_prob, 0.0), 1.0)
    home_no_score_prob = min(max(home_no_score_prob, 0.0), 1.0)

    nrfi_prob = away_no_score_prob * home_no_score_prob
    yrfi_prob = 1.0 - nrfi_prob
    return nrfi_prob, yrfi_prob
