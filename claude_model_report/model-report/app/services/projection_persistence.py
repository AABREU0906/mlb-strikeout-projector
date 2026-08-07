from __future__ import annotations

from typing import Optional

from app.data_sources.news_api import WarningLog
from app.database.models import Projection
from app.projections.engine import ProjectionResult
from app.schemas.player import BatterProfile, PitcherProfile, TeamProfile


def build_projection_row(
    game_id: str,
    game_date: str,
    game_start_utc: Optional[str],
    pitcher: PitcherProfile,
    pitcher_team: Optional[str],
    opponent_team_name: Optional[str],
    ballpark: Optional[str],
    lineup_status: str,
    lineup_source: Optional[str],
    lineup_retrieved_at: Optional[str],
    lineup: list[BatterProfile],
    team_profile: Optional[TeamProfile],
    weather_data: Optional[dict],
    umpire_data: Optional[dict],
    result: ProjectionResult,
    warning_log: WarningLog,
    manual_line_override: bool,
    data_source_versions: dict,
) -> Projection:
    return Projection(
        game_id=game_id,
        game_date=game_date,
        game_start_utc=game_start_utc,
        pitcher_id=pitcher.player_id,
        pitcher_name=pitcher.name,
        pitcher_team=pitcher_team,
        opponent_team=opponent_team_name,
        ballpark=ballpark,
        lineup_status=lineup_status,
        lineup_source=lineup_source,
        lineup_retrieved_at=lineup_retrieved_at,
        lineup_json=[b.model_dump() for b in lineup],
        pitcher_inputs_json=pitcher.model_dump(),
        batter_inputs_json=[b.model_dump() for b in lineup],
        team_inputs_json=team_profile.model_dump() if team_profile else None,
        weather_inputs_json=weather_data,
        umpire_inputs_json=umpire_data,
        workload_inputs_json=result.workload.__dict__,
        news_warnings_json=warning_log.to_json(),
        injury_warnings_json=[w.model_dump() for w in warning_log.all() if "injur" in w.issue.lower() or "il" in w.issue.lower()],
        workload_warnings_json=[w.model_dump() for w in warning_log.all() if "pitch" in w.issue.lower() or "rest" in w.issue.lower() or "limit" in w.issue.lower()],
        market_snapshot_json=result.market_used.get("snapshot"),
        market_timestamp_utc=(result.market_used.get("snapshot") or {}).get("retrieved_at") if result.market_used.get("snapshot") else None,
        market_source=(result.market_used.get("snapshot") or {}).get("source") if result.market_used.get("snapshot") else None,
        manual_line_override=manual_line_override,
        statistics_only_projection=result.statistics_only_projection,
        market_informed_projection=result.market_informed_projection,
        final_blended_projection=result.final_blended_projection,
        median_strikeouts=result.median_strikeouts,
        std_dev=result.std_dev,
        simulation_distribution_json=result.probability_by_k,
        percentiles_json=result.percentiles,
        expected_innings=result.expected_innings,
        expected_batters_faced=result.expected_batters_faced,
        expected_pitch_count=result.expected_pitch_count,
        confidence_rating=result.confidence_rating,
        confidence_factors_json=result.confidence_factors,
        explanation_json=result.explanation,
        model_version_label=result.model_version_label,
        data_source_versions_json=data_source_versions,
        random_seed=result.random_seed,
        n_simulations=result.n_simulations,
    )


def update_projection_edge_outcome(
    projection_id: str,
    *,
    validation_status: Optional[str] = None,
    recommended_side: Optional[str] = None,
    edge_grade: Optional[str] = None,
    betting_confidence: Optional[str] = None,
    estimated_ev: Optional[float] = None,
    model_over_probability: Optional[float] = None,
    model_under_probability: Optional[float] = None,
    projection_minus_line: Optional[float] = None,
    pitcher_confirmed: Optional[bool] = None,
    home_or_away: Optional[str] = None,
) -> None:
    """Updates a saved Projection row with the betting-edge outcome, once
    analyze_betting_edge() has run (which happens one layer above
    pipeline.run(), in the CLI/display layer -- the projection itself is
    already saved by that point). Only non-None fields are written, so a
    partial call (e.g. just validation_status for a validation-failed
    projection with no market line at all) never clobbers other fields
    with None."""
    from app.database.repositories import ProjectionRepository
    from app.database.session import session_scope

    fields = {
        "validation_status": validation_status,
        "recommended_side": recommended_side,
        "edge_grade": edge_grade,
        "betting_confidence": betting_confidence,
        "estimated_ev": estimated_ev,
        "model_over_probability": model_over_probability,
        "model_under_probability": model_under_probability,
        "projection_minus_line": projection_minus_line,
        "pitcher_confirmed": pitcher_confirmed,
        "home_or_away": home_or_away,
    }
    fields = {k: v for k, v in fields.items() if v is not None}
    if not fields:
        return

    with session_scope() as session:
        ProjectionRepository.update_edge_outcome(session, projection_id, **fields)
