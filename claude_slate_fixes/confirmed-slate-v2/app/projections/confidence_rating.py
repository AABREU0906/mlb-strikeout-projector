"""
Confidence Rating.

Produces High / Medium / Low / Avoid from a documented weighted penalty
score (0 = perfect confidence, higher = worse), so the rating is always
traceable to specific, listed factors rather than a black-box judgment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

RATING_THRESHOLDS = {
    "High": 0.20,
    "Medium": 0.45,
    "Low": 0.70,
    # anything above Low's threshold => "Avoid"
}


@dataclass
class ConfidenceResult:
    rating: str
    total_penalty: float
    factors: dict = field(default_factory=dict)


def compute_confidence(
    lineup_is_confirmed: bool,
    pitcher_confirmed: bool,
    pitcher_data_completeness: float,
    batter_avg_data_completeness: float,
    workload_confidence_penalty: float,
    news_confidence_penalty: float,
    weather_delay_risk: float,  # 0-1
    market_disagreement_flag: bool,
    stats_vs_market_disagreement_pct: Optional[float],
    simulation_std_dev: float,
    simulation_mean: float,
    model_recent_calibration_penalty: float = 0.0,
) -> ConfidenceResult:
    factors: dict[str, float] = {}

    factors["projected_lineup"] = 0.0 if lineup_is_confirmed else 0.18
    factors["unconfirmed_pitcher"] = 0.0 if pitcher_confirmed else 0.30
    factors["pitcher_data_gaps"] = round((1 - pitcher_data_completeness) * 0.15, 4)
    factors["batter_data_gaps"] = round((1 - batter_avg_data_completeness) * 0.10, 4)
    factors["workload_uncertainty"] = round(workload_confidence_penalty * 0.20, 4)
    factors["injury_news_uncertainty"] = round(news_confidence_penalty * 0.20, 4)
    factors["weather_delay_risk"] = round(weather_delay_risk * 0.10, 4)
    factors["market_disagreement"] = 0.08 if market_disagreement_flag else 0.0

    if stats_vs_market_disagreement_pct is not None and stats_vs_market_disagreement_pct >= 0.15:
        factors["stats_vs_market_model_disagreement"] = 0.10
    else:
        factors["stats_vs_market_model_disagreement"] = 0.0

    if simulation_mean > 0:
        cv = simulation_std_dev / simulation_mean
        factors["simulation_variance"] = round(min(cv * 0.10, 0.10), 4)
    else:
        factors["simulation_variance"] = 0.10

    factors["historical_calibration"] = round(model_recent_calibration_penalty, 4)

    total_penalty = round(sum(factors.values()), 4)

    if total_penalty <= RATING_THRESHOLDS["High"]:
        rating = "High"
    elif total_penalty <= RATING_THRESHOLDS["Medium"]:
        rating = "Medium"
    elif total_penalty <= RATING_THRESHOLDS["Low"]:
        rating = "Low"
    else:
        rating = "Avoid"

    return ConfidenceResult(rating=rating, total_penalty=total_penalty, factors=factors)
