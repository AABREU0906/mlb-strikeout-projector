"""
Projection Explanation.

Builds a ranked list of the factors that most increased or decreased the
projection relative to a neutral (league-average) baseline, with a
quantified estimated effect wherever the underlying stage produced a
numeric adjustment (Stage 2 per-batter adjustments, Stage 1 workload
notes, Stage 4 market components).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.features.league_constants import get_league_average
from app.projections.stage1_workload import WorkloadEstimate
from app.projections.stage2_batter_probability import BatterMatchupResult
from app.projections.stage4_market_adjustment import MarketAdjustment


@dataclass
class ExplanationFactor:
    direction: str  # "positive" | "negative"
    description: str
    estimated_effect: str


def build_explanation(
    batter_results: list[BatterMatchupResult],
    workload: WorkloadEstimate,
    market_adjustment: MarketAdjustment,
) -> dict:
    positives: list[ExplanationFactor] = []
    negatives: list[ExplanationFactor] = []

    league_k = get_league_average("league_k_rate")
    above_avg = [b for b in batter_results if b.adjusted_probability > league_k * 1.08]
    below_avg = [b for b in batter_results if b.adjusted_probability < league_k * 0.92]

    if len(above_avg) >= 3:
        positives.append(
            ExplanationFactor(
                direction="positive",
                description=f"{len(above_avg)} lineup hitters project above league-average strikeout rate for this matchup.",
                estimated_effect=f"+{len(above_avg)} elevated-K matchups",
            )
        )
    if len(below_avg) >= 3:
        negatives.append(
            ExplanationFactor(
                direction="negative",
                description=f"{len(below_avg)} lineup hitters project below league-average strikeout rate for this matchup.",
                estimated_effect=f"-{len(below_avg)} suppressed-K matchups",
            )
        )

    for b in batter_results:
        if b.sample_size_warning:
            continue  # sample warnings surface elsewhere, not as a factor
        park = b.adjustments_applied.get("ballpark")
        if park and abs(park - 1.0) >= 0.03:
            direction = "positive" if park > 1.0 else "negative"
            (positives if direction == "positive" else negatives).append(
                ExplanationFactor(
                    direction=direction,
                    description="Ballpark strikeout factor",
                    estimated_effect=f"x{park:.2f}",
                )
            )
            break  # one park note is enough; it applies to the whole lineup

    for note in workload.notes:
        direction = "negative" if any(
            w in note.lower() for w in ["shorten", "cap", "limit", "reduc", "rest", "uncertain"]
        ) else "positive"
        (negatives if direction == "negative" else positives).append(
            ExplanationFactor(direction=direction, description=note, estimated_effect="workload adjustment")
        )

    for component, value in market_adjustment.components.items():
        if abs(value) < 0.005:
            continue
        direction = "positive" if value > 0 else "negative"
        (positives if direction == "positive" else negatives).append(
            ExplanationFactor(
                direction=direction,
                description=f"Market feature: {component.replace('_', ' ')}",
                estimated_effect=f"{value*100:+.1f}%",
            )
        )

    return {
        "positive_factors": [f.__dict__ for f in positives],
        "negative_factors": [f.__dict__ for f in negatives],
    }
