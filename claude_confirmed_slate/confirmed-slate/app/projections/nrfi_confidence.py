"""
NRFI/YRFI Confidence Score (0-100).

Per spec: "A 70% NRFI prediction does not automatically mean high
confidence... keep prediction strength and data confidence separate."
This module never looks at the predicted probability itself -- only at
data quality/certainty signals -- so a strong prediction built on shaky
data still surfaces a low confidence score, and a modest-edge prediction
built on excellent data can surface high confidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

DEDUCTIONS = {
    "pitcher_unconfirmed": 22,
    "lineup_projected": 15,
    "small_pitcher_sample": 12,
    "small_team_sample": 8,
    "missing_bvp_data": 6,
    "missing_weather": 4,
    "injury_uncertainty": 12,
    "opener_risk": 10,
    "stale_data": 10,
    "poor_recent_calibration": 10,
}


@dataclass
class ConfidenceScoreResult:
    score: float
    factors: dict = field(default_factory=dict)


def compute_nrfi_confidence(
    pitcher_confirmed: bool,
    lineup_confirmed: bool,
    home_pitcher_reliability: float,
    away_pitcher_reliability: float,
    home_team_reliability: float,
    away_team_reliability: float,
    bvp_data_missing_count: int,
    weather_available: bool,
    injury_warning_present: bool,
    opener_risk_present: bool,
    data_freshness_minutes: Optional[float] = None,
    recent_calibration_penalty: float = 0.0,
) -> ConfidenceScoreResult:
    factors: dict[str, float] = {}
    score = 100.0

    if not pitcher_confirmed:
        factors["pitcher_unconfirmed"] = -DEDUCTIONS["pitcher_unconfirmed"]
        score -= DEDUCTIONS["pitcher_unconfirmed"]
    if not lineup_confirmed:
        factors["lineup_projected"] = -DEDUCTIONS["lineup_projected"]
        score -= DEDUCTIONS["lineup_projected"]

    min_pitcher_reliability = min(home_pitcher_reliability, away_pitcher_reliability)
    if min_pitcher_reliability < 0.5:
        deduction = DEDUCTIONS["small_pitcher_sample"] * (1 - min_pitcher_reliability / 0.5)
        factors["small_pitcher_sample"] = -round(deduction, 1)
        score -= deduction

    min_team_reliability = min(home_team_reliability, away_team_reliability)
    if min_team_reliability < 0.5:
        deduction = DEDUCTIONS["small_team_sample"] * (1 - min_team_reliability / 0.5)
        factors["small_team_sample"] = -round(deduction, 1)
        score -= deduction

    if bvp_data_missing_count > 0:
        deduction = min(DEDUCTIONS["missing_bvp_data"], bvp_data_missing_count * 2)
        factors["missing_bvp_data"] = -deduction
        score -= deduction

    if not weather_available:
        factors["missing_weather"] = -DEDUCTIONS["missing_weather"]
        score -= DEDUCTIONS["missing_weather"]

    if injury_warning_present:
        factors["injury_uncertainty"] = -DEDUCTIONS["injury_uncertainty"]
        score -= DEDUCTIONS["injury_uncertainty"]

    if opener_risk_present:
        factors["opener_risk"] = -DEDUCTIONS["opener_risk"]
        score -= DEDUCTIONS["opener_risk"]

    if data_freshness_minutes is not None and data_freshness_minutes > 60:
        deduction = min(DEDUCTIONS["stale_data"], (data_freshness_minutes - 60) / 30)
        factors["stale_data"] = -round(deduction, 1)
        score -= deduction

    if recent_calibration_penalty > 0:
        deduction = DEDUCTIONS["poor_recent_calibration"] * recent_calibration_penalty
        factors["poor_recent_calibration"] = -round(deduction, 1)
        score -= deduction

    score = max(0.0, min(100.0, score))
    return ConfidenceScoreResult(score=round(score, 1), factors=factors)
