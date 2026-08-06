"""
Builds a TeamFirstInningProfile from stored FirstInningGameResult rows,
mirroring the pitcher builder's shrinkage approach and leakage-safety
guarantee (repository-level before_date filtering).
"""
from __future__ import annotations

from typing import Optional

from app.database.models import FirstInningGameResult
from app.database.repositories import FirstInningGameResultRepository
from app.database.session import session_scope
from app.features.nrfi_league_constants import get_nrfi_league_average
from app.features.nrfi_rate_calculations import (
    compute_slash_line,
    last_n,
    rate_of,
    split_day_night,
    split_home_away,
    split_season,
)
from app.schemas.nrfi import FirstInningSlashLineSchema, ShrunkRate, TeamFirstInningProfile

SCORING_RATE_STABILIZATION_GAMES = 20


def _to_shrunk_rate(rate_result, prior: float, stabilization_n: float) -> ShrunkRate:
    return ShrunkRate.from_rate_result(rate_result, prior, stabilization_n)


def _row_to_team_record(row: FirstInningGameResult, team_id: int) -> dict:
    is_home = row.home_team_id == team_id
    if is_home:
        runs_scored = row.home_first_inning_runs
        hits = row.home_hits
        walks = row.home_walks
        strikeouts = row.home_strikeouts
        hr = row.home_home_runs
        ab = row.home_at_bats
        tb = row.home_total_bases
        pa = row.home_plate_appearances
    else:
        runs_scored = row.away_first_inning_runs
        hits = row.away_hits
        walks = row.away_walks
        strikeouts = row.away_strikeouts
        hr = row.away_home_runs
        ab = row.away_at_bats
        tb = row.away_total_bases
        pa = row.away_plate_appearances

    return {
        "game_date": row.game_date,
        "season": row.season,
        "is_home": is_home,
        "day_night": row.day_night,
        "scored": (runs_scored > 0) if runs_scored is not None else None,
        "runs_allowed": runs_scored,
        "hits_allowed": hits,
        "walks_allowed": walks,
        "strikeouts": strikeouts,
        "home_runs_allowed": hr,
        "at_bats_faced": ab,
        "total_bases_allowed": tb,
        "plate_appearances_faced": pa,
        "pitches_thrown": None,
    }


class TeamFirstInningFeatureBuilder:
    def build(
        self,
        team_id: int,
        team_name: str,
        as_of_date: str,
        current_season: int,
        history_limit: int = 300,
    ) -> TeamFirstInningProfile:
        missing: list[str] = []

        with session_scope() as session:
            rows = FirstInningGameResultRepository.list_team_history(
                session, team_id, before_date=as_of_date, limit=history_limit
            )
        records = [_row_to_team_record(r, team_id) for r in rows]

        if not records:
            missing.append("no_prior_games_found")

        scoring_prior = 1.0 - get_nrfi_league_average("league_scoreless_half_inning_rate")

        this_season, other_seasons = split_season(records, current_season)
        prev_season_year = current_season - 1
        previous_season_records = [r for r in other_seasons if r["season"] == prev_season_year]

        season_rate = _to_shrunk_rate(rate_of(this_season, "scored"), scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        prev_season_rate = _to_shrunk_rate(rate_of(previous_season_records, "scored"), scoring_prior, SCORING_RATE_STABILIZATION_GAMES)

        overall_rate_result = rate_of(records, "scored")
        overall_shrunk = _to_shrunk_rate(overall_rate_result, scoring_prior, SCORING_RATE_STABILIZATION_GAMES)

        last5 = _to_shrunk_rate(rate_of(last_n(records, 5), "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        last10 = _to_shrunk_rate(rate_of(last_n(records, 10), "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        last20 = _to_shrunk_rate(rate_of(last_n(records, 20), "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        last30 = _to_shrunk_rate(rate_of(last_n(records, 30), "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)

        home_recs, away_recs = split_home_away(records)
        home_rate = _to_shrunk_rate(rate_of(home_recs, "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        away_rate = _to_shrunk_rate(rate_of(away_recs, "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)

        day_recs, night_recs = split_day_night(records)
        day_rate = _to_shrunk_rate(rate_of(day_recs, "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)
        night_rate = _to_shrunk_rate(rate_of(night_recs, "scored"), overall_shrunk.shrunk_rate or scoring_prior, SCORING_RATE_STABILIZATION_GAMES)

        slash_args = ("runs_allowed", "hits_allowed", "walks_allowed", "strikeouts",
                      "home_runs_allowed", "at_bats_faced", "total_bases_allowed",
                      "plate_appearances_faced", "pitches_thrown")
        season_slash = compute_slash_line(this_season, *slash_args)

        avg_runs = None
        known_runs = [r["runs_allowed"] for r in records if r["runs_allowed"] is not None]
        if known_runs:
            avg_runs = round(sum(known_runs) / len(known_runs), 3)

        nrfi_apps = sum(1 for r in records if r.get("scored") is False)
        yrfi_apps = sum(1 for r in records if r.get("scored") is True)

        total_expected = 2
        completeness = max(0.0, 1.0 - (len(missing) / total_expected))

        return TeamFirstInningProfile(
            team_id=team_id,
            team_name=team_name,
            season_scoring_rate=season_rate,
            previous_season_scoring_rate=prev_season_rate,
            last_5_scoring_rate=last5,
            last_10_scoring_rate=last10,
            last_20_scoring_rate=last20,
            last_30_scoring_rate=last30,
            home_scoring_rate=home_rate,
            away_scoring_rate=away_rate,
            day_scoring_rate=day_rate,
            night_scoring_rate=night_rate,
            season_slash_line=FirstInningSlashLineSchema(**season_slash.__dict__),
            avg_first_inning_runs=avg_runs,
            leadoff_reach_rate=None,
            team_nrfi_record=f"{nrfi_apps}-{yrfi_apps}" if records else None,
            team_yrfi_record=f"{yrfi_apps}-{nrfi_apps}" if records else None,
            data_completeness=completeness,
            missing_fields=missing,
        )
