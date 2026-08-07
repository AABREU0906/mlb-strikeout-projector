"""
ProjectionPipeline: the single place that wires together every data source,
feature builder, and the projection engine for one pitcher/game, then
persists the result. This is what `python main.py project` calls after the
user has picked a game and a pitcher.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

from app.config.logging_config import get_logger
from app.data_sources.base import utc_now_iso
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.data_sources.news_api import WarningLog
from app.data_sources.weather_api import OpenMeteoWeatherProvider, get_ballpark_reference
from app.database.repositories import GameRepository, ProjectionRepository
from app.database.session import session_scope
from app.features.batter_features import BatterFeatureBuilder
from app.features.pitcher_features import PitcherFeatureBuilder
from app.features.team_features import TeamFeatureBuilder
from app.features.umpire_features import UmpireProfile, build_umpire_profile
from app.markets.market_service import MarketService
from app.projections.engine import ProjectionResult, run_projection
from app.schemas.market import ManualMarketEntry
from app.schemas.player import BatterProfile, PitcherProfile
from app.services.projection_persistence import build_projection_row

logger = get_logger(__name__)

DATA_SOURCE_VERSIONS = {
    "mlb_stats_api": "v1/v1.1 (live)",
    "open_meteo": "v1 forecast",
    "the_odds_api": "v4",
    "ballpark_reference_table": "seed-2025-07",
    "league_constants": "seed-2025-season",
}


@dataclass
class LineupResolution:
    lineup: list[BatterProfile]
    status: str  # "confirmed" | "projected"
    source: str
    retrieved_at: str


class ProjectionPipeline:
    def __init__(self):
        self.mlb = MlbStatsApiProvider()
        self.weather = OpenMeteoWeatherProvider()
        self.market = MarketService()
        self.batter_builder = BatterFeatureBuilder(self.mlb)
        self.pitcher_builder = PitcherFeatureBuilder(self.mlb)
        self.team_builder = TeamFeatureBuilder()

    # ---------------------------------------------------------------- schedule
    def get_schedule(self, game_date: str) -> list[dict]:
        payload = self.mlb.get_schedule(game_date)
        with session_scope() as session:
            for g in payload.data.get("games", []):
                GameRepository.upsert(session, g)
        return payload.data.get("games", [])

    # ------------------------------------------------------------------ lineup
    def resolve_lineup(
        self,
        game_id: str,
        opponent_team_id: int,
        pitcher_throws: Optional[str],
        season: int,
        manual_lineup: Optional[list[dict]] = None,
    ) -> LineupResolution:
        if manual_lineup is not None:
            profiles = [
                self._build_batter_from_manual(entry, pitcher_throws, season) for entry in manual_lineup
            ]
            return LineupResolution(
                lineup=profiles, status="projected", source="manual_entry", retrieved_at=utc_now_iso()
            )

        confirmed = self.mlb.get_confirmed_lineup(game_id, opponent_team_id)
        if confirmed is not None:
            profiles = [
                self.batter_builder.build(
                    batter_id=entry["player_id"],
                    season=season,
                    batting_order=entry["batting_order"],
                    pitcher_hand_today=pitcher_throws,
                )
                for entry in confirmed.data["lineup"]
                if entry.get("player_id")
            ]
            return LineupResolution(
                lineup=profiles, status="confirmed", source=confirmed.source, retrieved_at=confirmed.retrieved_at
            )

        projected = self.mlb.get_projected_lineup(opponent_team_id, pitcher_throws)
        profiles = [
            self.batter_builder.build(
                batter_id=entry["player_id"],
                season=season,
                batting_order=entry["batting_order"],
                pitcher_hand_today=pitcher_throws,
            )
            for entry in projected.data.get("lineup", [])
            if entry.get("player_id")
        ]
        return LineupResolution(
            lineup=profiles, status="projected", source=projected.source, retrieved_at=projected.retrieved_at
        )

    def _build_batter_from_manual(self, entry: dict, pitcher_throws, season) -> BatterProfile:
        # Manual lineup entries still pull real season stats for the named
        # player_id -- "manual" refers to the batting-order assignment, not
        # fabricated statistics.
        return self.batter_builder.build(
            batter_id=entry["player_id"],
            season=season,
            batting_order=entry["batting_order"],
            pitcher_hand_today=pitcher_throws,
        )

    # ---------------------------------------------------------------- pitcher
    def build_pitcher(self, pitcher_id: int, season: int) -> PitcherProfile:
        return self.pitcher_builder.build(pitcher_id, season)

    # ------------------------------------------------------------------ full run
    def run(
        self,
        game: dict,
        pitcher_id: int,
        pitcher_is_home: bool,
        season: int,
        manual_lineup: Optional[list[dict]] = None,
        manual_market: Optional[ManualMarketEntry] = None,
        warning_log: Optional[WarningLog] = None,
        umpire: Optional[UmpireProfile] = None,
        is_opener: bool = False,
        is_tandem_risk: bool = False,
        announced_pitch_limit: Optional[int] = None,
        short_rest: bool = False,
        extra_rest: bool = False,
        recent_rehab_assignment: bool = False,
        recent_skipped_start: bool = False,
        n_simulations: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> tuple[ProjectionResult, str]:
        warning_log = warning_log or WarningLog()
        umpire = umpire or build_umpire_profile()

        pitcher = self.build_pitcher(pitcher_id, season)
        pitcher_team_id = game["home_team_id"] if pitcher_is_home else game["away_team_id"]
        pitcher_team_name = game["home_team"] if pitcher_is_home else game["away_team"]
        opponent_team_id = game["away_team_id"] if pitcher_is_home else game["home_team_id"]
        opponent_team_name = game["away_team"] if pitcher_is_home else game["home_team"]

        lineup_res = self.resolve_lineup(
            game_id=game["game_id"],
            opponent_team_id=opponent_team_id,
            pitcher_throws=pitcher.throws,
            season=season,
            manual_lineup=manual_lineup,
        )

        opponent_team_profile = self.team_builder.build(opponent_team_id, opponent_team_name, season)

        ballpark_ref = get_ballpark_reference(game.get("venue_id"))
        weather_data = None
        weather_k_mult = 1.0
        weather_delay_risk = 0.0
        if ballpark_ref.get("latitude") is not None and game.get("scheduled_start_utc"):
            weather_payload = self.weather.get_game_weather(
                ballpark_ref["latitude"], ballpark_ref["longitude"], game["scheduled_start_utc"]
            )
            weather_data = weather_payload.data
            if weather_data.get("available"):
                precip = weather_data.get("precipitation_probability_pct") or 0
                weather_delay_risk = min(precip / 100.0, 1.0) if precip else 0.0
                wind = weather_data.get("wind_speed_mph") or 0
                # Modest, capped effect: strong wind blowing out is a very
                # small documented drag on K rate (more balls in play get
                # extra carry -> hitters occasionally more aggressive); kept
                # tiny and capped per project rules against exaggeration.
                if wind and wind > 15:
                    weather_k_mult = 0.99

        market_snapshot = self.market.build_snapshot(
            game_id=game["game_id"], pitcher_name=pitcher.name, manual=manual_market
        )

        lineup_batters = len(lineup_res.lineup) or 9

        result = run_projection(
            pitcher=pitcher,
            lineup=lineup_res.lineup,
            opponent_team=opponent_team_profile,
            lineup_is_confirmed=(lineup_res.status == "confirmed"),
            pitcher_confirmed=True,  # caller only reaches here after explicit pitcher selection
            ballpark_k_factor=ballpark_ref.get("k_factor", 1.0),
            weather_k_multiplier=weather_k_mult,
            umpire_k_multiplier=umpire.k_effect_multiplier,
            weather_delay_risk=weather_delay_risk,
            market=market_snapshot,
            warning_log=warning_log,
            is_opener=is_opener,
            is_tandem_risk=is_tandem_risk,
            announced_pitch_limit=announced_pitch_limit,
            short_rest=short_rest,
            extra_rest=extra_rest,
            recent_rehab_assignment=recent_rehab_assignment,
            recent_skipped_start=recent_skipped_start,
            n_simulations=n_simulations,
            seed=seed,
        )

        row = build_projection_row(
            game_id=game["game_id"],
            game_date=game["game_date"],
            game_start_utc=game.get("scheduled_start_utc"),
            pitcher=pitcher,
            pitcher_team=pitcher_team_name,
            opponent_team_name=opponent_team_name,
            ballpark=game.get("ballpark"),
            lineup_status=lineup_res.status,
            lineup_source=lineup_res.source,
            lineup_retrieved_at=lineup_res.retrieved_at,
            lineup=lineup_res.lineup,
            team_profile=opponent_team_profile,
            weather_data=weather_data,
            umpire_data=umpire.model_dump(),
            result=result,
            warning_log=warning_log,
            manual_line_override=manual_market is not None,
            data_source_versions=DATA_SOURCE_VERSIONS,
        )

        with session_scope() as session:
            GameRepository.upsert(session, game)
            saved = ProjectionRepository.save(session, row)
            projection_id = saved.id

        return result, projection_id, lineup_res.status, lineup_res.source, pitcher.name
