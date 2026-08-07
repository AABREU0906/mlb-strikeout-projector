"""
Umpire tendency adjustments.

Honest limitation: there is no free, structured, officially documented API
for home-plate umpire ball/strike tendencies. Sites that compile this
(e.g. umpire scorecards) are HTML-only and not universally ToS-clear for
scraping. Per project rules, we do not scrape without a confirmed-allowed
structured alternative.

This module therefore:
1. Accepts manually-entered umpire tendency data when the user has it
   (e.g. from a publicly available umpire scorecard they looked up).
2. Always regresses hard toward league-neutral (documented shrinkage),
   consistent with the requirement to not give umpire effects excessive
   weight.
3. Defaults to fully neutral (no adjustment) when no data is supplied --
   never fabricates a tendency.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from app.features.shrinkage import shrink_rate

UMPIRE_STABILIZATION_GAMES = 400  # heavy regression: umpire effects are small


class UmpireProfile(BaseModel):
    name: Optional[str] = None
    games_sampled: Optional[int] = None
    called_strike_rate_observed: Optional[float] = None
    called_strike_rate_adjusted: Optional[float] = None
    k_effect_multiplier: float = 1.00  # regressed multiplicative effect on K rate
    bb_effect_multiplier: float = 1.00
    data_available: bool = False


LEAGUE_AVG_CALLED_STRIKE_RATE = 0.32  # documented approximate league mean


def build_umpire_profile(
    name: Optional[str] = None,
    games_sampled: Optional[int] = None,
    called_strike_rate_observed: Optional[float] = None,
) -> UmpireProfile:
    if name is None or games_sampled is None or called_strike_rate_observed is None:
        return UmpireProfile(data_available=False)

    result = shrink_rate(
        observed_rate=called_strike_rate_observed,
        observed_n=games_sampled,
        prior_rate=LEAGUE_AVG_CALLED_STRIKE_RATE,
        stabilization_n=UMPIRE_STABILIZATION_GAMES,
    )
    # Convert the (heavily regressed) called-strike deviation into a modest
    # multiplicative K/BB effect. Deliberately compressed: a full deviation
    # from league average only moves K rate by a few tenths of a percent.
    deviation = result.shrunk_rate - LEAGUE_AVG_CALLED_STRIKE_RATE
    k_multiplier = 1.0 + (deviation * 0.35)
    bb_multiplier = 1.0 - (deviation * 0.25)

    return UmpireProfile(
        name=name,
        games_sampled=games_sampled,
        called_strike_rate_observed=called_strike_rate_observed,
        called_strike_rate_adjusted=result.shrunk_rate,
        k_effect_multiplier=round(k_multiplier, 4),
        bb_effect_multiplier=round(bb_multiplier, 4),
        data_available=True,
    )
