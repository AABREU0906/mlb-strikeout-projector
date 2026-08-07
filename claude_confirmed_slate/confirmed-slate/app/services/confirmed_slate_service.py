"""
Orchestration for `python main.py project-confirmed-slate`.

Ties together, in order:
  1. MLB Stats API schedule (existing provider, unchanged).
  2. Confirmed-lineup eligibility (existing get_confirmed_lineup, unchanged).
  3. Incremental "already projected today" tracking (new repository lookup).
  4. Credit-conserving Odds API orchestration (app.services.confirmed_slate_odds).
  5. Event matching (app.services.odds_event_matcher) and pitcher/line
     matching (app.services.pitcher_prop_matcher).
  6. The EXISTING projection pipeline, validator, edge analysis, and
     persistence -- called exactly the same way the interactive CLI flow
     calls them -- nothing about projection math, validation, or edge
     grading is reimplemented here.

Every per-pitcher and per-game step is wrapped so one failure (API
timeout, ambiguous match, missing data, projection error) cannot abort
the rest of the slate -- it's recorded and processing continues.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

from app.config.logging_config import get_logger
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.database.repositories import ProjectionRepository
from app.database.session import session_scope
from app.services.confirmed_slate_odds import OddsRunSession, extract_fanduel_pitcher_outcomes
from app.services.odds_event_matcher import match_event_to_game
from app.services.pipeline import ProjectionPipeline
from app.services.pitcher_prop_matcher import match_pitcher_name, pair_over_under
from app.services.projection_persistence import update_projection_edge_outcome
from app.validation.projection_validator import validate_projection

logger = get_logger(__name__)


@dataclass
class SlateRow:
    pitcher_name: str
    pitcher_id: int
    game_id: str
    game_date: str
    game_time: str
    team: str
    opponent: str
    home_or_away: str
    lineup_status: str
    sportsbook: Optional[str]
    strikeout_line: Optional[float]
    over_odds: Optional[int]
    under_odds: Optional[int]
    statistics_only_projection: float
    market_informed_projection: float
    final_blended_projection: float
    median_strikeouts: float
    std_dev: float
    model_over_probability: Optional[float]
    model_under_probability: Optional[float]
    projection_minus_line: Optional[float]
    recommended_side: Optional[str]
    edge_grade: Optional[str]
    confidence: Optional[str]
    estimated_ev: Optional[float]
    expected_innings: float
    expected_batters_faced: float
    expected_pitch_count: float
    workload_source: Optional[str]
    workload_role: Optional[str]
    validation_status: str
    odds_timestamp: Optional[str]
    projection_timestamp: str
    projection_id: str
    fanduel_market_status: str


@dataclass
class SkippedGame:
    matchup: str
    reason: str


@dataclass
class FailedItem:
    label: str
    reason: str


@dataclass
class ConfirmedSlateSummary:
    games_today: int = 0
    games_confirmed: int = 0
    already_projected_today: int = 0
    new_games_projected: int = 0
    games_skipped: int = 0
    pitchers_projected: int = 0
    fanduel_markets_found: int = 0
    fanduel_markets_unavailable: int = 0
    actionable_recommendations: int = 0
    pass_count: int = 0
    projections_saved: int = 0
    credits_used_this_run: Optional[int] = None
    credits_remaining: Optional[int] = None
    events_list_calls: int = 0
    event_odds_calls: int = 0
    odds_api_configured: bool = True


@dataclass
class ConfirmedSlateResult:
    rows: list = field(default_factory=list)
    skipped_games: list = field(default_factory=list)
    failed_items: list = field(default_factory=list)
    summary: ConfirmedSlateSummary = field(default_factory=ConfirmedSlateSummary)


def _both_lineups_confirmed(mlb: MlbStatsApiProvider, game: dict) -> bool:
    home_confirmed = mlb.get_confirmed_lineup(game["game_id"], game["home_team_id"]) is not None
    away_confirmed = mlb.get_confirmed_lineup(game["game_id"], game["away_team_id"]) is not None
    return home_confirmed and away_confirmed


def _build_manual_market(line, sportsbook: str = "FanDuel"):
    from app.schemas.market import ManualMarketEntry

    return ManualMarketEntry(
        strikeout_line=line.line,
        over_odds=line.over_odds,
        under_odds=line.under_odds,
        sportsbook_name=sportsbook,
    )


def _project_one_pitcher(
    pipeline: ProjectionPipeline,
    game: dict,
    pitcher_id: int,
    is_home: bool,
    season: int,
    manual_market,
    lineup_status_for_display: str,
):
    from app.markets.line_probability import compute_line_probabilities
    from app.reporting.display import print_market_comparison

    try:
        result, projection_id, lineup_status, lineup_source, pitcher_name = pipeline.run(
            game=game, pitcher_id=pitcher_id, pitcher_is_home=is_home, season=season, manual_market=manual_market,
        )
    except Exception as exc:
        return None, f"Projection failed: {exc}"

    market_snapshot = None
    if result.market_used.get("snapshot"):
        from app.schemas.market import MarketSnapshot
        market_snapshot = MarketSnapshot(**result.market_used["snapshot"])

    line_probs = None
    if market_snapshot is not None and market_snapshot.strikeout_line is not None:
        line_probs = compute_line_probabilities(result.probability_by_k, float(market_snapshot.strikeout_line))

    try:
        validation_report = validate_projection(
            expected_innings=result.expected_innings,
            expected_batters_faced=result.expected_batters_faced,
            expected_pitch_count=result.expected_pitch_count,
            final_projection=result.statistics_only_projection,
            probability_by_k=result.probability_by_k,
            percentiles=result.percentiles,
            std_dev=result.std_dev,
            prob_complete_5=result.workload.prob_complete_5,
            prob_complete_6=result.workload.prob_complete_6,
            prob_complete_7=result.workload.prob_complete_7,
            prob_early_exit=result.workload.prob_early_exit,
            over_probability=line_probs.over_probability if line_probs else None,
            under_probability=line_probs.under_probability if line_probs else None,
            push_probability=line_probs.push_probability if line_probs else 0.0,
            lineup_confirmed=(lineup_status == "confirmed"),
            pitcher_confirmed=True,
            workload_fallback_used=result.workload.workload_fallback_used,
            workload_fallback_count=result.workload.workload_fallback_count,
            workload_all_metrics_fallback=result.workload.workload_all_metrics_fallback,
        )
    except Exception as exc:
        return None, f"Validation failed: {exc}"

    home_or_away = "home" if is_home else "away"
    validation_status = "valid" if validation_report.is_valid else "invalid"
    if validation_report.is_valid and validation_report.has_warnings:
        validation_status = "valid_with_warnings"

    opponent = game["away_team"] if is_home else game["home_team"]
    team = game["home_team"] if is_home else game["away_team"]

    if not validation_report.is_valid:
        update_projection_edge_outcome(
            projection_id, validation_status=validation_status, pitcher_confirmed=True, home_or_away=home_or_away,
        )
        row = SlateRow(
            pitcher_name=pitcher_name, pitcher_id=pitcher_id, game_id=game["game_id"],
            game_date=game.get("game_date", ""), game_time=game.get("scheduled_start_utc", ""),
            team=team, opponent=opponent, home_or_away=home_or_away, lineup_status=lineup_status_for_display,
            sportsbook=(market_snapshot.source if market_snapshot else None),
            strikeout_line=(market_snapshot.strikeout_line if market_snapshot else None),
            over_odds=(market_snapshot.over_odds if market_snapshot else None),
            under_odds=(market_snapshot.under_odds if market_snapshot else None),
            statistics_only_projection=result.statistics_only_projection,
            market_informed_projection=result.market_informed_projection,
            final_blended_projection=result.final_blended_projection,
            median_strikeouts=result.median_strikeouts, std_dev=result.std_dev,
            model_over_probability=None, model_under_probability=None, projection_minus_line=None,
            recommended_side=None, edge_grade=None, confidence=None, estimated_ev=None,
            expected_innings=result.expected_innings, expected_batters_faced=result.expected_batters_faced,
            expected_pitch_count=result.expected_pitch_count,
            workload_source=result.workload.workload_source, workload_role=result.workload.workload_role,
            validation_status=validation_status,
            odds_timestamp=(market_snapshot.retrieved_at if market_snapshot else None),
            projection_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
            projection_id=projection_id,
            fanduel_market_status=("found" if manual_market is not None else "unavailable"),
        )
        return row, None

    try:
        edge_analysis = print_market_comparison(
            market_snapshot, result, lineup_confirmed=(lineup_status == "confirmed"), pitcher_confirmed=True,
        )
    except Exception as exc:
        edge_analysis = None
        logger.warning("Edge analysis display failed for pitcher %s: %s", pitcher_id, exc)

    if edge_analysis is not None:
        update_projection_edge_outcome(
            projection_id, validation_status=validation_status, pitcher_confirmed=True, home_or_away=home_or_away,
            recommended_side=edge_analysis.recommended_side, edge_grade=edge_analysis.grade,
            betting_confidence=edge_analysis.confidence,
            estimated_ev=(edge_analysis.selected.expected_value if edge_analysis.selected else None),
            model_over_probability=edge_analysis.over.model_probability,
            model_under_probability=edge_analysis.under.model_probability,
            projection_minus_line=(
                result.statistics_only_projection - float(market_snapshot.strikeout_line)
                if market_snapshot and market_snapshot.strikeout_line is not None else None
            ),
        )
    else:
        update_projection_edge_outcome(
            projection_id, validation_status=validation_status, pitcher_confirmed=True, home_or_away=home_or_away,
        )

    row = SlateRow(
        pitcher_name=pitcher_name, pitcher_id=pitcher_id, game_id=game["game_id"],
        game_date=game.get("game_date", ""), game_time=game.get("scheduled_start_utc", ""),
        team=team, opponent=opponent, home_or_away=home_or_away, lineup_status=lineup_status_for_display,
        sportsbook=(market_snapshot.source if market_snapshot else None),
        strikeout_line=(market_snapshot.strikeout_line if market_snapshot else None),
        over_odds=(market_snapshot.over_odds if market_snapshot else None),
        under_odds=(market_snapshot.under_odds if market_snapshot else None),
        statistics_only_projection=result.statistics_only_projection,
        market_informed_projection=result.market_informed_projection,
        final_blended_projection=result.final_blended_projection,
        median_strikeouts=result.median_strikeouts, std_dev=result.std_dev,
        model_over_probability=(edge_analysis.over.model_probability if edge_analysis else None),
        model_under_probability=(edge_analysis.under.model_probability if edge_analysis else None),
        projection_minus_line=(
            result.statistics_only_projection - float(market_snapshot.strikeout_line)
            if market_snapshot and market_snapshot.strikeout_line is not None else None
        ),
        recommended_side=(edge_analysis.recommended_side if edge_analysis else None),
        edge_grade=(edge_analysis.grade if edge_analysis else None),
        confidence=(edge_analysis.confidence if edge_analysis else None),
        estimated_ev=(edge_analysis.selected.expected_value if edge_analysis and edge_analysis.selected else None),
        expected_innings=result.expected_innings, expected_batters_faced=result.expected_batters_faced,
        expected_pitch_count=result.expected_pitch_count,
        workload_source=result.workload.workload_source, workload_role=result.workload.workload_role,
        validation_status=validation_status,
        odds_timestamp=(market_snapshot.retrieved_at if market_snapshot else None),
        projection_timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        projection_id=projection_id,
        fanduel_market_status=("found" if manual_market is not None else "unavailable"),
    )
    return row, None


def run_confirmed_slate(
    game_date: Optional[str] = None,
    refresh: bool = False,
    no_odds: bool = False,
) -> ConfirmedSlateResult:
    game_date = game_date or dt.date.today().isoformat()
    season = dt.date.fromisoformat(game_date).year

    pipeline = ProjectionPipeline()
    mlb = MlbStatsApiProvider()
    odds_session = OddsRunSession()
    result = ConfirmedSlateResult()
    result.summary.odds_api_configured = odds_session.is_configured()

    games = pipeline.get_schedule(game_date)
    result.summary.games_today = len(games)

    eligible_games = []
    for game in games:
        try:
            confirmed = _both_lineups_confirmed(mlb, game)
        except Exception as exc:
            result.skipped_games.append(SkippedGame(
                matchup=f"{game.get('away_team', '?')} @ {game.get('home_team', '?')}",
                reason=f"Could not check lineup status: {exc}",
            ))
            result.summary.games_skipped += 1
            continue

        if not confirmed:
            result.skipped_games.append(SkippedGame(
                matchup=f"{game['away_team']} @ {game['home_team']}", reason="lineup not confirmed",
            ))
            result.summary.games_skipped += 1
            continue

        result.summary.games_confirmed += 1
        eligible_games.append(game)

    events = [] if no_odds else odds_session.get_events()

    for game in eligible_games:
        pitchers = [
            (game.get("probable_home_pitcher_id"), True, game.get("probable_home_pitcher_name")),
            (game.get("probable_away_pitcher_id"), False, game.get("probable_away_pitcher_name")),
        ]
        pitchers = [p for p in pitchers if p[0] is not None]

        already_done = set()
        for pitcher_id, _is_home, _name in pitchers:
            with session_scope() as session:
                exists = ProjectionRepository.exists_for_game_pitcher_date(session, game["game_id"], pitcher_id, game_date)
            if exists:
                already_done.add(pitcher_id)

        to_process = pitchers if refresh else [p for p in pitchers if p[0] not in already_done]
        if not refresh:
            result.summary.already_projected_today += len(pitchers) - len(to_process)
        if not to_process:
            continue

        game_outcomes = []
        if not no_odds and odds_session.is_configured():
            try:
                matched = match_event_to_game(events, game["home_team"], game["away_team"], game.get("scheduled_start_utc", ""))
                if matched is not None:
                    event_odds = odds_session.get_event_odds(matched.event_id)
                    if event_odds is not None:
                        game_outcomes = extract_fanduel_pitcher_outcomes(event_odds)
            except Exception as exc:
                logger.warning("Odds lookup failed for game %s: %s", game["game_id"], exc)

        game_had_any_success = False
        for pitcher_id, is_home, pitcher_name in to_process:
            try:
                manual_market = None
                fanduel_found = False
                if game_outcomes and pitcher_name:
                    candidate_names = list({o["pitcher_name"] for o in game_outcomes if o.get("pitcher_name")})
                    matched_name = match_pitcher_name(pitcher_name, candidate_names)
                    if matched_name:
                        this_pitcher_outcomes = [o for o in game_outcomes if o.get("pitcher_name") == matched_name]
                        line = pair_over_under(this_pitcher_outcomes)
                        if line is not None:
                            manual_market = _build_manual_market(line)
                            fanduel_found = True

                if fanduel_found:
                    result.summary.fanduel_markets_found += 1
                else:
                    result.summary.fanduel_markets_unavailable += 1

                row, error = _project_one_pitcher(
                    pipeline, game, pitcher_id, is_home, season, manual_market, lineup_status_for_display="confirmed",
                )
            except Exception as exc:
                error = str(exc)
                row = None

            if error is not None:
                result.failed_items.append(FailedItem(label=f"{pitcher_name} ({game['game_id']})", reason=error))
                continue

            result.rows.append(row)
            result.summary.pitchers_projected += 1
            result.summary.projections_saved += 1
            game_had_any_success = True
            if row.recommended_side in ("OVER", "UNDER"):
                result.summary.actionable_recommendations += 1
            elif row.recommended_side == "PASS":
                result.summary.pass_count += 1

        if game_had_any_success:
            result.summary.new_games_projected += 1

    result.summary.credits_used_this_run = odds_session.credits_used_this_run
    result.summary.credits_remaining = odds_session.credits_remaining
    result.summary.events_list_calls = odds_session.events_list_calls_made
    result.summary.event_odds_calls = odds_session.event_odds_calls_made

    return result
