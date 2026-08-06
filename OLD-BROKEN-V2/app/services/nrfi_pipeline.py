"""
NrfiPipeline -- the single place that wires together data sources, feature
builders, and the NRFI engine for one game, then persists the result.
Deliberately reuses ProjectionPipeline.resolve_lineup() (the strikeout
model's already-built confirmed/projected lineup logic) for BOTH teams'
lineups rather than writing a second lineup resolver, per project rules
against duplicating shared services.
"""
from __future__ import annotations

from typing import Optional

from app.data_sources.base import utc_now_iso
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.data_sources.weather_api import OpenMeteoWeatherProvider, get_ballpark_reference
from app.database.repositories import GameRepository, NrfiProjectionRepository
from app.database.session import session_scope
from app.features.nrfi_bvp_features import BvPFeatureBuilder
from app.features.nrfi_pitcher_features import PitcherFirstInningFeatureBuilder
from app.features.nrfi_team_features import TeamFirstInningFeatureBuilder
from app.features.umpire_features import UmpireProfile, build_umpire_profile
from app.projections.nrfi_engine import NrfiEngineResult, run_nrfi_projection
from app.services.pipeline import ProjectionPipeline

TOP_ORDER_SPOTS = (1, 2, 3, 4, 5)

DATA_SOURCE_VERSIONS = {
    "mlb_stats_api": "v1/v1.1 (live)",
    "open_meteo": "v1 forecast",
    "nrfi_model": "baseline-v0.1.0",
}


class NrfiPipeline:
    def __init__(self):
        self.mlb = MlbStatsApiProvider()
        self.weather = OpenMeteoWeatherProvider()
        self.strikeout_pipeline = ProjectionPipeline()
        self.pitcher_builder = PitcherFirstInningFeatureBuilder()
        self.team_builder = TeamFirstInningFeatureBuilder()
        self.bvp_builder = BvPFeatureBuilder(self.mlb)

    def get_schedule(self, game_date: str) -> list[dict]:
        return self.strikeout_pipeline.get_schedule(game_date)

    def run(
        self,
        game: dict,
        season: int,
        home_pitcher_id: Optional[int] = None,
        away_pitcher_id: Optional[int] = None,
        pitchers_confirmed: bool = False,
        umpire: Optional[UmpireProfile] = None,
        injury_warning_present: bool = False,
        opener_risk_present: bool = False,
    ) -> tuple[NrfiEngineResult, str]:
        as_of_date = game["game_date"]
        umpire = umpire or build_umpire_profile()

        home_pitcher_id = home_pitcher_id or game.get("probable_home_pitcher_id")
        away_pitcher_id = away_pitcher_id or game.get("probable_away_pitcher_id")
        if home_pitcher_id is None or away_pitcher_id is None:
            raise ValueError("Both starting pitchers must be known (confirmed or probable) to run an NRFI projection.")

        home_pitcher_hand = self.mlb.get_person_handedness(home_pitcher_id)
        away_pitcher_hand = self.mlb.get_person_handedness(away_pitcher_id)

        away_lineup_res = self.strikeout_pipeline.resolve_lineup(
            game_id=game["game_id"], opponent_team_id=game["away_team_id"],
            pitcher_throws=(home_pitcher_hand or {}).get("pitch_hand"), season=season,
        )
        home_lineup_res = self.strikeout_pipeline.resolve_lineup(
            game_id=game["game_id"], opponent_team_id=game["home_team_id"],
            pitcher_throws=(away_pitcher_hand or {}).get("pitch_hand"), season=season,
        )
        lineup_confirmed = (away_lineup_res.status == "confirmed") and (home_lineup_res.status == "confirmed")

        away_pitcher_profile = self.pitcher_builder.build(
            pitcher_id=away_pitcher_id,
            pitcher_name=(away_pitcher_hand or {}).get("full_name", f"Pitcher {away_pitcher_id}"),
            throws=(away_pitcher_hand or {}).get("pitch_hand"),
            as_of_date=as_of_date, current_season=season,
        )
        home_pitcher_profile = self.pitcher_builder.build(
            pitcher_id=home_pitcher_id,
            pitcher_name=(home_pitcher_hand or {}).get("full_name", f"Pitcher {home_pitcher_id}"),
            throws=(home_pitcher_hand or {}).get("pitch_hand"),
            as_of_date=as_of_date, current_season=season,
        )

        away_offense = self.team_builder.build(
            team_id=game["away_team_id"], team_name=game["away_team"], as_of_date=as_of_date, current_season=season
        )
        home_offense = self.team_builder.build(
            team_id=game["home_team_id"], team_name=game["home_team"], as_of_date=as_of_date, current_season=season
        )

        away_top_order_bvp = self._build_top_order_bvp(
            away_lineup_res.lineup, opposing_pitcher_id=home_pitcher_id,
            opposing_pitcher_hand=(home_pitcher_hand or {}).get("pitch_hand"), season=season,
        )
        home_top_order_bvp = self._build_top_order_bvp(
            home_lineup_res.lineup, opposing_pitcher_id=away_pitcher_id,
            opposing_pitcher_hand=(away_pitcher_hand or {}).get("pitch_hand"), season=season,
        )

        ballpark_ref = get_ballpark_reference(game.get("venue_id"))
        weather_available = False
        weather_run_multiplier = 1.0
        if ballpark_ref.get("latitude") is not None and game.get("scheduled_start_utc"):
            weather_payload = self.weather.get_game_weather(
                ballpark_ref["latitude"], ballpark_ref["longitude"], game["scheduled_start_utc"]
            )
            if weather_payload.data.get("available"):
                weather_available = True
                wind = weather_payload.data.get("wind_speed_mph") or 0
                if wind and wind > 15:
                    weather_run_multiplier = 1.02

        result = run_nrfi_projection(
            away_pitcher=away_pitcher_profile,
            home_pitcher=home_pitcher_profile,
            away_offense=away_offense,
            home_offense=home_offense,
            pitcher_confirmed=pitchers_confirmed,
            lineup_confirmed=lineup_confirmed,
            away_top_order_bvp=away_top_order_bvp,
            home_top_order_bvp=home_top_order_bvp,
            ballpark_run_factor=ballpark_ref.get("run_factor", 1.0),
            weather_run_multiplier=weather_run_multiplier,
            weather_available=weather_available,
            umpire_run_multiplier=umpire.bb_effect_multiplier,
            injury_warning_present=injury_warning_present,
            opener_risk_present=opener_risk_present,
        )

        projection_id = self._persist(
            game=game, home_pitcher_id=home_pitcher_id, away_pitcher_id=away_pitcher_id,
            home_pitcher_name=home_pitcher_profile.name, away_pitcher_name=away_pitcher_profile.name,
            lineup_confirmed=lineup_confirmed,
            away_lineup_res=away_lineup_res, home_lineup_res=home_lineup_res,
            result=result,
        )

        return result, projection_id

    def _build_top_order_bvp(self, lineup, opposing_pitcher_id, opposing_pitcher_hand, season):
        top_order = [b for b in lineup if b.batting_order in TOP_ORDER_SPOTS]
        return [
            self.bvp_builder.build(
                batter_id=b.player_id, batter_name=b.name, pitcher_id=opposing_pitcher_id,
                pitcher_hand=opposing_pitcher_hand, season=season,
            )
            for b in top_order
        ]

    def _persist(self, game, home_pitcher_id, away_pitcher_id, home_pitcher_name, away_pitcher_name,
                 lineup_confirmed, away_lineup_res, home_lineup_res, result: NrfiEngineResult) -> str:
        from app.database.models import NrfiProjection

        row = NrfiProjection(
            game_id=game["game_id"],
            game_date=game["game_date"],
            game_start_utc=game.get("scheduled_start_utc"),
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_pitcher_id=home_pitcher_id,
            away_pitcher_id=away_pitcher_id,
            home_pitcher_name=home_pitcher_name,
            away_pitcher_name=away_pitcher_name,
            ballpark=game.get("ballpark"),
            lineup_status="confirmed" if lineup_confirmed else "projected",
            lineup_source=f"away:{away_lineup_res.source}|home:{home_lineup_res.source}",
            lineup_retrieved_at=utc_now_iso(),
            away_lineup_json=[b.model_dump() for b in away_lineup_res.lineup],
            home_lineup_json=[b.model_dump() for b in home_lineup_res.lineup],
            away_scoring_probability=result.game_result.away_half.scoring_probability,
            home_scoring_probability=result.game_result.home_half.scoring_probability,
            nrfi_probability=result.game_result.nrfi_probability,
            yrfi_probability=result.game_result.yrfi_probability,
            expected_first_inning_runs=result.game_result.expected_first_inning_runs,
            away_threat_score=result.away_threat.score,
            home_threat_score=result.home_threat.score,
            confidence_score=result.confidence.score,
            confidence_factors_json=result.confidence.factors,
            explanation_json=result.explanation,
            model_version_label=result.model_version_label,
            feature_version="v0.1.0",
            data_source_versions_json=DATA_SOURCE_VERSIONS,
        )

        with session_scope() as session:
            GameRepository.upsert(session, game)
            saved = NrfiProjectionRepository.save(session, row)
            return saved.id
