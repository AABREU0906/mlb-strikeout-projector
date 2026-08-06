"""
NRFI/YRFI Projection Engine -- orchestrates the half-inning model, Threat
Score, confidence score, and explanation generator into one NrfiEngineResult.
Mirrors the strikeout model's engine.py structure (Stages -> one orchestrator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.features.league_constants import get_league_average
from app.features.nrfi_league_constants import get_nrfi_league_average
from app.projections.nrfi_confidence import ConfidenceScoreResult, compute_nrfi_confidence
from app.projections.nrfi_explanation import build_nrfi_explanation
from app.projections.nrfi_half_inning_model import NrfiGameResult, compute_nrfi_probability
from app.projections.nrfi_threat_score import ThreatScoreResult, compute_threat_score
from app.schemas.nrfi import BvPProfile, PitcherFirstInningProfile, TeamFirstInningProfile

MODEL_VERSION_LABEL = "nrfi-baseline-v0.1.0"


@dataclass
class NrfiEngineResult:
    game_result: NrfiGameResult
    away_threat: ThreatScoreResult
    home_threat: ThreatScoreResult
    confidence: ConfidenceScoreResult
    explanation: dict
    model_version_label: str = MODEL_VERSION_LABEL


def run_nrfi_projection(
    away_pitcher: PitcherFirstInningProfile,
    home_pitcher: PitcherFirstInningProfile,
    away_offense: TeamFirstInningProfile,
    home_offense: TeamFirstInningProfile,
    pitcher_confirmed: bool,
    lineup_confirmed: bool,
    away_top_order_bvp: Optional[list[BvPProfile]] = None,
    home_top_order_bvp: Optional[list[BvPProfile]] = None,
    ballpark_run_factor: float = 1.0,
    weather_run_multiplier: float = 1.0,
    weather_available: bool = True,
    umpire_run_multiplier: float = 1.0,
    injury_warning_present: bool = False,
    opener_risk_present: bool = False,
    data_freshness_minutes: Optional[float] = None,
) -> NrfiEngineResult:
    league_scoreless = get_nrfi_league_average("league_scoreless_half_inning_rate")
    league_scoring_rate = 1.0 - league_scoreless

    game_result = compute_nrfi_probability(
        away_offense=away_offense,
        home_pitcher=home_pitcher,
        home_offense=home_offense,
        away_pitcher=away_pitcher,
        league_scoring_rate=league_scoring_rate,
        away_top_order_bvp=away_top_order_bvp,
        home_top_order_bvp=home_top_order_bvp,
        ballpark_run_factor=ballpark_run_factor,
        weather_run_multiplier=weather_run_multiplier,
        umpire_run_multiplier=umpire_run_multiplier,
    )

    league_obp = get_league_average("league_obp")
    league_slg = get_league_average("league_slg")
    league_k_pct = get_league_average("league_k_rate")
    league_bb_pct = get_league_average("league_bb_rate")
    league_hr_rate = get_nrfi_league_average("league_first_inning_hr_rate")

    away_threat = compute_threat_score(
        away_offense.season_slash_line, league_obp, league_slg, league_k_pct, league_bb_pct, league_hr_rate,
        top_order_bvp=away_top_order_bvp,
        recent_form_rate=away_offense.last_10_scoring_rate.shrunk_rate,
        season_form_rate=away_offense.season_scoring_rate.shrunk_rate,
    )
    home_threat = compute_threat_score(
        home_offense.season_slash_line, league_obp, league_slg, league_k_pct, league_bb_pct, league_hr_rate,
        top_order_bvp=home_top_order_bvp,
        recent_form_rate=home_offense.last_10_scoring_rate.shrunk_rate,
        season_form_rate=home_offense.season_scoring_rate.shrunk_rate,
    )

    bvp_missing = 0
    for group in (away_top_order_bvp, home_top_order_bvp):
        if group:
            bvp_missing += sum(1 for b in group if "no_bvp_history" in b.missing_fields)

    confidence = compute_nrfi_confidence(
        pitcher_confirmed=pitcher_confirmed,
        lineup_confirmed=lineup_confirmed,
        home_pitcher_reliability=home_pitcher.season_scoreless_rate.reliability,
        away_pitcher_reliability=away_pitcher.season_scoreless_rate.reliability,
        home_team_reliability=home_offense.season_scoring_rate.reliability,
        away_team_reliability=away_offense.season_scoring_rate.reliability,
        bvp_data_missing_count=bvp_missing,
        weather_available=weather_available,
        injury_warning_present=injury_warning_present,
        opener_risk_present=opener_risk_present,
        data_freshness_minutes=data_freshness_minutes,
    )

    explanation = build_nrfi_explanation(
        away_pitcher=away_pitcher, home_pitcher=home_pitcher,
        away_offense=away_offense, home_offense=home_offense,
        away_half=game_result.away_half, home_half=game_result.home_half,
        away_threat=away_threat, home_threat=home_threat,
        league_scoreless_rate=league_scoreless,
    )

    return NrfiEngineResult(
        game_result=game_result,
        away_threat=away_threat,
        home_threat=home_threat,
        confidence=confidence,
        explanation=explanation,
    )
