"""
Stage 4: Market-Informed Adjustment.

The statistics-only model (Stages 1-3 + Stage 5 Monte Carlo on
statistics-only inputs) NEVER sees sportsbook data. This stage computes a
*separate*, small, capped adjustment multiplier derived only from market
features, which is applied to a second, parallel run of the batter
probabilities / workload to produce the "market-informed" projection.

Until enough graded history exists to fit real regression weights (see
app/training/), the market-feature weights below are conservative,
documented defaults -- not arbitrary blending, but a transparent starting
formula that the evaluation/retraining system is explicitly built to
replace once walk-forward validation shows better weights (see
app/training/model_promotion.py).

Market features used (only those with a real value are applied):
  - Consensus strikeout line vs. the model's own raw statistics-only median:
      if the market line sits meaningfully above/below the stats-only
      projection, nudge modestly toward the market (the market may embed
      information the stats model lacks, e.g. imminent workload news).
  - Opponent implied runs vs. league average: higher implied runs against
    the pitcher's team's opponent often correlates with a tougher offensive
    matchup (fewer Ks); lower implied runs the opposite.
  - Game total vs. league-average total: very low totals often correlate
    with pitcher-friendly conditions.
  - Line movement (opening -> current): meaningful movement can reflect
    late-breaking information (e.g. wind, workload news) not yet reflected
    in the stats-only model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.schemas.market import MarketSnapshot

LEAGUE_AVG_GAME_TOTAL = 8.6
LEAGUE_AVG_TEAM_IMPLIED_RUNS = 4.4

MAX_MARKET_ADJUSTMENT = 0.12  # cap: market path can move the projection by at most +/-12%


@dataclass
class MarketAdjustment:
    multiplier: float
    components: dict = field(default_factory=dict)
    features_used: list[str] = field(default_factory=list)
    disagreement_note: Optional[str] = None


def compute_market_adjustment(
    market: Optional[MarketSnapshot],
    statistics_only_projection: Optional[float],
) -> MarketAdjustment:
    if market is None:
        return MarketAdjustment(multiplier=1.0, components={}, features_used=[])

    components: dict[str, float] = {}
    features_used: list[str] = []

    if market.strikeout_line is not None and statistics_only_projection:
        diff = market.strikeout_line - statistics_only_projection
        # Pull 20% of the way toward the market line, capped.
        pull = max(min(diff * 0.20 / max(statistics_only_projection, 0.1), 0.06), -0.06)
        components["market_line_pull"] = pull
        features_used.append("consensus_strikeout_line")

    if market.opponent_implied_runs is not None:
        dev = (market.opponent_implied_runs - LEAGUE_AVG_TEAM_IMPLIED_RUNS) / LEAGUE_AVG_TEAM_IMPLIED_RUNS
        components["opponent_implied_runs"] = max(min(-dev * 0.10, 0.05), -0.05)
        features_used.append("opponent_implied_runs")

    if market.game_total is not None:
        dev = (market.game_total - LEAGUE_AVG_GAME_TOTAL) / LEAGUE_AVG_GAME_TOTAL
        components["game_total"] = max(min(-dev * 0.06, 0.03), -0.03)
        features_used.append("game_total")

    if market.line_movement is not None and abs(market.line_movement) >= 0.5:
        components["line_movement"] = max(min(market.line_movement * 0.03, 0.03), -0.03)
        features_used.append("line_movement")

    total_adjustment = sum(components.values())
    total_adjustment = max(min(total_adjustment, MAX_MARKET_ADJUSTMENT), -MAX_MARKET_ADJUSTMENT)
    multiplier = 1.0 + total_adjustment

    disagreement_note = None
    if market.market_disagreement is not None and market.market_disagreement >= 0.5:
        disagreement_note = (
            f"Sportsbooks disagree meaningfully on the strikeout line "
            f"(stdev={market.market_disagreement:.2f}); market-informed adjustment "
            f"treated with extra caution."
        )

    return MarketAdjustment(
        multiplier=round(multiplier, 4),
        components={k: round(v, 4) for k, v in components.items()},
        features_used=features_used,
        disagreement_note=disagreement_note,
    )
