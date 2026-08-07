"""
Shrinkage / stabilization utilities.

Method: empirical-Bayes-style shrinkage toward a prior mean using a
"stabilization point" expressed in the same units as the sample size (e.g.
plate appearances). This is the standard sabermetric approach (popularized
by Russell Carleton's "how many PA until X stabilizes" research) and is
simple, transparent, and documented, in line with the project's requirement
for a defensible, explainable stabilization method rather than an opaque one.

  shrunk_rate = (observed_n * observed_rate + stabilization_n * prior_rate)
                / (observed_n + stabilization_n)

As observed_n -> infinity, shrunk_rate -> observed_rate.
As observed_n -> 0, shrunk_rate -> prior_rate.

Stabilization points (in PA, approximate, documented defaults; can be
overridden):
  - Batter strikeout rate (overall):      ~60 PA
  - Batter strikeout rate (vs one hand):  ~110 PA (smaller-sample splits
                                            need more games to stabilize)
  - Pitcher strikeout rate:                ~70 batters faced
  - Pitcher strikeout rate (vs one hand): ~120 batters faced
  - Walk rate (batter or pitcher):         ~120 PA/BF
"""
from __future__ import annotations

from dataclasses import dataclass

STABILIZATION_POINTS = {
    "batter_k_rate_overall": 60,
    "batter_k_rate_split": 110,
    "batter_bb_rate": 120,
    "pitcher_k_rate_overall": 70,
    "pitcher_k_rate_split": 120,
    "pitcher_bb_rate": 120,
    "contact_rate": 100,
    "chase_rate": 150,
    "swstr_rate": 100,
}


@dataclass
class ShrinkageResult:
    observed_rate: float
    observed_n: float
    prior_rate: float
    stabilization_n: float
    shrunk_rate: float
    reliability: float  # observed_n / (observed_n + stabilization_n), 0-1

    @property
    def is_small_sample(self) -> bool:
        return self.reliability < 0.5


def shrink_rate(
    observed_rate: float,
    observed_n: float,
    prior_rate: float,
    stabilization_n: float,
) -> ShrinkageResult:
    observed_n = max(observed_n, 0.0)
    if observed_n + stabilization_n <= 0:
        shrunk = prior_rate
        reliability = 0.0
    else:
        shrunk = (observed_n * observed_rate + stabilization_n * prior_rate) / (
            observed_n + stabilization_n
        )
        reliability = observed_n / (observed_n + stabilization_n)
    return ShrinkageResult(
        observed_rate=observed_rate,
        observed_n=observed_n,
        prior_rate=prior_rate,
        stabilization_n=stabilization_n,
        shrunk_rate=shrunk,
        reliability=reliability,
    )


def shrink_named(observed_rate: float, observed_n: float, prior_rate: float, key: str) -> ShrinkageResult:
    stab_n = STABILIZATION_POINTS.get(key)
    if stab_n is None:
        raise KeyError(f"No stabilization point defined for '{key}'")
    return shrink_rate(observed_rate, observed_n, prior_rate, stab_n)
