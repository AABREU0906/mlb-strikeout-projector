"""
Builds a PitcherFirstInningProfile from stored FirstInningGameResult rows
(via the repository), applying the same empirical-Bayes shrinkage approach
used by the strikeout model (app/features/shrinkage.py) so small samples
(e.g. a pitcher's first 3 starts of a season) pull toward career/league
rates rather than swinging wildly.

Leakage safety: callers MUST pass `as_of_date` and this builder only uses
FirstInningGameResultRepository.list_pitcher_history(before_date=as_of_date),
which is already date-filtered strictly before that date at the repository
level -- see repositories.py.
"""
from __future__ import annotations

import datetime as dt
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
from app.schemas.nrfi import FirstInningSlashLineSchema, PitcherFirstInningProfile, ShrunkRate

SCORELESS_RATE_STABILIZATION_STARTS = 20
GAME_NRFI_RATE_STABILIZATION_STARTS = 20


def _to_shrunk_rate(rate_result, prior: float, stabilization_n: float) -> ShrunkRate:
    return ShrunkRate.from_rate_result(rate_result, prior, stabilization_n)


def _row_to_pitcher_record(row: FirstInningGameResult, pitcher_id: int) -> dict:
    is_home = row.home_starting_pitcher_id == pitcher_id
    if is_home:
        scoreless = row.home_pitcher_scoreless_first
        runs_allowed = row.away_first_inning_runs
        hits_allowed = row.away_hits
        walks_allowed = row.away_walks
        strikeouts = row.away_strikeouts
        hr_allowed = row.away_home_runs
        ab_faced = row.away_at_bats
        tb_allowed = row.away_total_bases
        pa_faced = row.away_plate_appearances
        pitches = row.home_pitcher_first_inning_pitches
    else:
        scoreless = row.away_pitcher_scoreless_first
        runs_allowed = row.home_first_inning_runs
        hits_allowed = row.home_hits
        walks_allowed = row.home_walks
        strikeouts = row.home_strikeouts
        hr_allowed = row.home_home_runs
        ab_faced = row.home_at_bats
        tb_allowed = row.home_total_bases
        pa_faced = row.home_plate_appearances
        pitches = row.away_pitcher_first_inning_pitches

    return {
        "game_date": row.game_date,
        "season": row.season,
        "is_home": is_home,
        "day_night": row.day_night,
        "scoreless": scoreless,
        "game_is_nrfi": row.is_nrfi,
        "runs_allowed": runs_allowed,
        "hits_allowed": hits_allowed,
        "walks_allowed": walks_allowed,
        "strikeouts": strikeouts,
        "home_runs_allowed": hr_allowed,
        "at_bats_faced": ab_faced,
        "total_bases_allowed": tb_allowed,
        "plate_appearances_faced": pa_faced,
        "pitches_thrown": pitches,
    }


class PitcherFirstInningFeatureBuilder:
    def build(
        self,
        pitcher_id: int,
        pitcher_name: str,
        throws: Optional[str],
        as_of_date: str,
        current_season: int,
        history_limit: int = 200,
    ) -> PitcherFirstInningProfile:
        missing: list[str] = []

        with session_scope() as session:
            rows = FirstInningGameResultRepository.list_pitcher_history(
                session, pitcher_id, before_date=as_of_date, limit=history_limit
            )
        records = [_row_to_pitcher_record(r, pitcher_id) for r in rows]

        if not records:
            missing.append("no_prior_starts_found")

        scoreless_prior = get_nrfi_league_average("league_scoreless_half_inning_rate")
        nrfi_prior = get_nrfi_league_average("league_game_nrfi_rate")

        career_scoreless = _to_shrunk_rate(rate_of(records, "scoreless"), scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        game_nrfi = _to_shrunk_rate(rate_of(records, "game_is_nrfi"), nrfi_prior, GAME_NRFI_RATE_STABILIZATION_STARTS)

        this_season, other_seasons = split_season(records, current_season)
        prev_season_year = current_season - 1
        previous_season_records = [r for r in other_seasons if r["season"] == prev_season_year]

        season_scoreless = _to_shrunk_rate(rate_of(this_season, "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        prev_season_scoreless = _to_shrunk_rate(rate_of(previous_season_records, "scoreless"), scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)

        last5 = _to_shrunk_rate(rate_of(last_n(records, 5), "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        last10 = _to_shrunk_rate(rate_of(last_n(records, 10), "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        last20 = _to_shrunk_rate(rate_of(last_n(records, 20), "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)

        home_recs, away_recs = split_home_away(records)
        home_scoreless = _to_shrunk_rate(rate_of(home_recs, "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        away_scoreless = _to_shrunk_rate(rate_of(away_recs, "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)

        day_recs, night_recs = split_day_night(records)
        day_scoreless = _to_shrunk_rate(rate_of(day_recs, "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)
        night_scoreless = _to_shrunk_rate(rate_of(night_recs, "scoreless"), career_scoreless.shrunk_rate or scoreless_prior, SCORELESS_RATE_STABILIZATION_STARTS)

        slash_args = ("runs_allowed", "hits_allowed", "walks_allowed", "strikeouts",
                      "home_runs_allowed", "at_bats_faced", "total_bases_allowed",
                      "plate_appearances_faced", "pitches_thrown")
        season_slash = compute_slash_line(this_season, *slash_args)
        career_slash = compute_slash_line(records, *slash_args)

        days_of_rest = self._compute_rest_days(records, as_of_date)
        previous_start_pitches = self._previous_start_pitch_count(rows, pitcher_id)

        total_expected = 3
        completeness = max(0.0, 1.0 - (len(missing) / total_expected))

        return PitcherFirstInningProfile(
            pitcher_id=pitcher_id,
            name=pitcher_name,
            throws=throws,
            career_scoreless_rate=career_scoreless,
            season_scoreless_rate=season_scoreless,
            previous_season_scoreless_rate=prev_season_scoreless,
            last_5_scoreless_rate=last5,
            last_10_scoreless_rate=last10,
            last_20_scoreless_rate=last20,
            home_scoreless_rate=home_scoreless,
            away_scoreless_rate=away_scoreless,
            day_scoreless_rate=day_scoreless,
            night_scoreless_rate=night_scoreless,
            game_nrfi_rate_in_starts=game_nrfi,
            season_slash_line=FirstInningSlashLineSchema(**season_slash.__dict__),
            career_slash_line=FirstInningSlashLineSchema(**career_slash.__dict__),
            days_of_rest=days_of_rest,
            previous_start_pitch_count=previous_start_pitches,
            recent_velocity_change=None,
            data_completeness=completeness,
            missing_fields=missing,
        )

    @staticmethod
    def _compute_rest_days(records: list[dict], as_of_date: str) -> Optional[int]:
        if not records:
            return None
        most_recent = max(records, key=lambda r: r["game_date"])
        try:
            last_start = dt.date.fromisoformat(most_recent["game_date"])
            today = dt.date.fromisoformat(as_of_date)
            return (today - last_start).days
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _previous_start_pitch_count(rows: list[FirstInningGameResult], pitcher_id: int) -> Optional[int]:
        if not rows:
            return None
        most_recent = max(rows, key=lambda r: r.game_date)
        if most_recent.home_starting_pitcher_id == pitcher_id:
            return most_recent.home_pitcher_first_inning_pitches
        return most_recent.away_pitcher_first_inning_pitches
