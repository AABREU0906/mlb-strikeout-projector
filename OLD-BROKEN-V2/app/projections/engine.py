"""
Projection Engine -- ties Stages 1-5 together into a single reproducible
ProjectionResult, producing both the statistics-only and market-informed
projections side by side (never silently replacing one with the other).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import settings
from app.data_sources.news_api import WarningLog
from app.features.league_constants import get_league_average
from app.projections.confidence_rating import compute_confidence
from app.projections.explanation import build_explanation
from app.projections.stage1_workload import WorkloadEstimate, estimate_workload
from app.projections.stage2_batter_probability import (
    BatterMatchupResult,
    compute_batter_matchup_probability,
)
from app.projections.stage4_market_adjustment import compute_market_adjustment
from app.schemas.market import MarketSnapshot
from app.schemas.player import BatterProfile, PitcherProfile, TeamProfile
from app.simulation.monte_carlo import SimulationResult, run_monte_carlo

MODEL_VERSION_LABEL = "baseline-v0.1.0"  # bumped by app/training when a new model is promoted


@dataclass
class ProjectionResult:
    statistics_only_projection: float
    market_informed_projection: float
    final_blended_projection: float
    median_strikeouts: float
    std_dev: float
    percentiles: dict
    probability_by_k: dict
    most_likely_k: int
    expected_innings: float
    expected_batters_faced: float
    expected_pitch_count: float
    batter_results: list[BatterMatchupResult]
    workload: WorkloadEstimate
    confidence_rating: str
    confidence_factors: dict
    explanation: dict
    market_used: dict
    model_version_label: str = MODEL_VERSION_LABEL
    n_simulations: int = 25000
    random_seed: Optional[int] = None


def run_projection(
    pitcher: PitcherProfile,
    lineup: list[BatterProfile],
    opponent_team: Optional[TeamProfile],
    lineup_is_confirmed: bool,
    pitcher_confirmed: bool,
    ballpark_k_factor: float = 1.0,
    weather_k_multiplier: float = 1.0,
    umpire_k_multiplier: float = 1.0,
    weather_delay_risk: float = 0.0,
    market: Optional[MarketSnapshot] = None,
    warning_log: Optional[WarningLog] = None,
    is_opener: bool = False,
    is_tandem_risk: bool = False,
    announced_pitch_limit: Optional[int] = None,
    short_rest: bool = False,
    extra_rest: bool = False,
    recent_rehab_assignment: bool = False,
    recent_skipped_start: bool = False,
    n_simulations: Optional[int] = None,
    seed: Optional[int] = None,
) -> ProjectionResult:
    n_simulations = n_simulations or settings.default_monte_carlo_iterations
    seed = seed if seed is not None else settings.default_random_seed
    warning_log = warning_log or WarningLog()

    # ---- Stage 1: Workload ----
    workload = estimate_workload(
        pitcher=pitcher,
        opponent_team=opponent_team,
        is_opener=is_opener,
        is_tandem_risk=is_tandem_risk,
        announced_pitch_limit=announced_pitch_limit,
        short_rest=short_rest,
        extra_rest=extra_rest,
        recent_rehab_assignment=recent_rehab_assignment,
        recent_skipped_start=recent_skipped_start,
        lineup_batters=len(lineup) or 9,
    )

    # ---- Stage 2: Batter matchup probabilities (statistics-only) ----
    batter_results: list[BatterMatchupResult] = []
    for batter in lineup:
        res = compute_batter_matchup_probability(
            pitcher=pitcher,
            batter=batter,
            ballpark_k_factor=ballpark_k_factor,
            weather_k_multiplier=weather_k_multiplier,
            umpire_k_multiplier=umpire_k_multiplier,
        )
        batter_results.append(res)

    lineup_size = max(len(batter_results), 1)
    stats_only_probs = [b.adjusted_probability for b in batter_results] or [get_league_average("league_k_rate")]

    bf_per_inning = (workload.expected_batters_faced / workload.expected_innings) if workload.expected_innings else 4.3
    workload_spread = 1.1 + workload.workload_confidence_penalty * 1.6

    # ---- Stage 5 (statistics-only run) ----
    sim_stats_only: SimulationResult = run_monte_carlo(
        expected_innings=workload.expected_innings,
        workload_spread=workload_spread,
        bf_per_inning=bf_per_inning,
        batter_probabilities=stats_only_probs,
        n_simulations=n_simulations,
        seed=seed,
    )

    # ---- Stage 4: Market-informed adjustment ----
    market_adj = compute_market_adjustment(market, sim_stats_only.mean)
    market_probs = [min(max(p * market_adj.multiplier, 0.01), 0.75) for p in stats_only_probs]
    market_workload_innings = workload.expected_innings  # workload itself stays stats-derived; market adjusts probability only, per spec (independent stats model untouched)

    sim_market_informed: SimulationResult = run_monte_carlo(
        expected_innings=market_workload_innings,
        workload_spread=workload_spread,
        bf_per_inning=bf_per_inning,
        batter_probabilities=market_probs,
        n_simulations=n_simulations,
        seed=(seed + 1) if seed is not None else None,
    )

    # ---- Blended projection ----
    # Conservative documented default weights until enough graded history
    # exists for walk-forward-validated weights (see app/training/blend_weights.py).
    stats_weight, market_weight = 0.65, 0.35
    if market is None or (market.strikeout_line is None and market.over_odds is None):
        stats_weight, market_weight = 1.0, 0.0  # no market data at all -> pure stats-only
    blended = stats_weight * sim_stats_only.mean + market_weight * sim_market_informed.mean

    # ---- Confidence ----
    avg_batter_completeness = (
        sum(b_p.data_completeness for b_p in lineup) / len(lineup) if lineup else 0.5
    )
    stats_vs_market_disagreement = None
    if sim_stats_only.mean > 0:
        stats_vs_market_disagreement = abs(sim_market_informed.mean - sim_stats_only.mean) / sim_stats_only.mean

    confidence = compute_confidence(
        lineup_is_confirmed=lineup_is_confirmed,
        pitcher_confirmed=pitcher_confirmed,
        pitcher_data_completeness=pitcher.data_completeness,
        batter_avg_data_completeness=avg_batter_completeness,
        workload_confidence_penalty=workload.workload_confidence_penalty,
        news_confidence_penalty=warning_log.confidence_penalty(),
        weather_delay_risk=weather_delay_risk,
        market_disagreement_flag=bool(market and market.market_disagreement and market.market_disagreement >= 0.5),
        stats_vs_market_disagreement_pct=stats_vs_market_disagreement,
        simulation_std_dev=sim_stats_only.std_dev,
        simulation_mean=sim_stats_only.mean,
    )

    explanation = build_explanation(batter_results, workload, market_adj)

    market_used = {
        "snapshot": market.model_dump() if market else None,
        "adjustment_multiplier": market_adj.multiplier,
        "adjustment_components": market_adj.components,
        "features_used": market_adj.features_used,
        "disagreement_note": market_adj.disagreement_note,
    }

    return ProjectionResult(
        statistics_only_projection=round(sim_stats_only.mean, 2),
        market_informed_projection=round(sim_market_informed.mean, 2),
        final_blended_projection=round(blended, 2),
        median_strikeouts=sim_stats_only.median,
        std_dev=sim_stats_only.std_dev,
        percentiles=sim_stats_only.percentiles,
        probability_by_k=sim_stats_only.probability_by_k,
        most_likely_k=sim_stats_only.most_likely_k,
        expected_innings=workload.expected_innings,
        expected_batters_faced=workload.expected_batters_faced,
        expected_pitch_count=workload.expected_pitch_count,
        batter_results=batter_results,
        workload=workload,
        confidence_rating=confidence.rating,
        confidence_factors=confidence.factors,
        explanation=explanation,
        market_used=market_used,
        n_simulations=n_simulations,
        random_seed=seed,
    )
