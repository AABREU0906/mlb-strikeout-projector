"""
Shared probability math used across the strikeout and NRFI/YRFI models.

log5() lived privately inside stage2_batter_probability.py; it's pulled out
here so the NRFI/YRFI half-inning scoring model can reuse the exact same,
already-tested implementation instead of a second copy, per project rules
against duplicating utility functions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


def is_valid_rate(value: object) -> bool:
    """A rate is usable only if it is not None, an actual numeric type
    (int/float, not a string that merely happens to parse as one -- a
    stray string reaching here is itself a sign of a type-confusion bug
    upstream and should be rejected, not silently coerced), finite, and
    within [0, 1]. Deliberately an explicit validity check rather than a
    truthiness check -- `0.0` is a perfectly valid rate and must never be
    treated as "missing" (a bare `if value:` check would incorrectly
    reject it)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    if not math.isfinite(numeric):
        return False
    return 0.0 <= numeric <= 1.0


@dataclass
class RateResolution:
    """Structured record of how a rate was resolved through the
    documented fallback hierarchy, so callers can surface exactly what
    happened rather than silently swallowing a fallback."""
    value: float
    source: str  # e.g. "split" | "season" | "career" | "league_average"
    fallback_used: bool
    notes: list[str] = field(default_factory=list)


def resolve_rate_with_fallback(
    candidates: list[tuple[str, Optional[float]]],
    label: str,
) -> RateResolution:
    """Walks an ORDERED list of (source_name, value) candidates -- e.g.
    [("split", pitcher_split_rate), ("season", pitcher_season_rate),
    ("career", pitcher_career_rate), ("league_average", league_rate)] --
    and returns the first one that passes `is_valid_rate`. This is the
    single, shared implementation of the documented fallback hierarchy
    (split -> season -> career -> league average) used for BOTH the
    pitcher and batter side of a matchup, so the two can never drift into
    different validation logic.

    Raises ValueError only if NOT EVEN the league-average candidate is
    valid, which would indicate a deeper league_constants problem, not
    something to paper over with a fabricated number.
    """
    notes: list[str] = []
    for i, (source, value) in enumerate(candidates):
        if is_valid_rate(value):
            fallback_used = i > 0
            if fallback_used:
                notes.append(
                    f"{label}: higher-priority rate(s) were unavailable or invalid; "
                    f"used the {source} rate ({float(value):.3f}) instead."
                )
            return RateResolution(value=float(value), source=source, fallback_used=fallback_used, notes=notes)

    raise ValueError(
        f"No valid rate could be resolved for {label} at any fallback tier "
        f"(including league average) -- candidates: {candidates!r}"
    )


def log5(rate_a: float, rate_b: float, league_rate: float) -> float:
    """Bill James' log5 method: combines two independent rates against a
    league baseline. Used for pitcher-vs-batter strikeout matchups AND for
    team-offense-vs-opposing-pitcher first-inning scoring matchups -- same
    math, different inputs. Deliberately NOT a simple average of rate_a and
    rate_b (that's explicitly disallowed for both models).

    Raises ValueError (rather than crashing deep inside a TypeError on
    `None`) if any input is missing or invalid -- callers MUST resolve a
    valid rate (see resolve_rate_with_fallback) before calling this;
    log5() itself does not guess or substitute a default, since silently
    picking a fallback here would hide the problem from the caller that's
    actually responsible for its fallback policy.
    """
    for name, value in (("pitcher", rate_a), ("batter", rate_b), ("league", league_rate)):
        if not is_valid_rate(value):
            raise ValueError(
                f"log5 received an invalid pitcher or batter probability "
                f"({name}={value!r}); rates must be non-None, numeric, finite, "
                f"and within [0, 1]."
            )

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
