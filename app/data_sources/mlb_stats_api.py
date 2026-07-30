"""
MLB Stats API provider (statsapi.mlb.com).

This is a public, documented, widely-used JSON API (no key required). It is
the backbone data source for schedule, probable pitchers, official confirmed
lineups (via boxscore battingOrder once posted), player bio/handedness, and
season/career/splits statistics.

Docs are not officially published by MLB but the endpoint shapes are stable
and long-established; this module isolates all endpoint knowledge so any
schema drift only needs fixing here.
"""
from __future__ import annotations

from typing import Any, Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.base import (
    LineupProvider,
    PlayerStatsProvider,
    ScheduleProvider,
    SourcedPayload,
    utc_now_iso,
)
from app.utilities.http_client import http_client

logger = get_logger(__name__)

SOURCE_NAME = "mlb_stats_api"


class MlbStatsApiProvider(ScheduleProvider, LineupProvider, PlayerStatsProvider):
    def __init__(self):
        self.base = settings.mlb_stats_api_base_url
        self.base_v1_1 = settings.mlb_stats_api_base_url_v1_1

    # ---------------------------------------------------------------- schedule
    def get_schedule(self, game_date: str) -> SourcedPayload:
        url = f"{self.base}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team,linescore,venue,weather,status",
        }
        resp = http_client.get_json(
            url,
            params=params,
            cache_category="schedule",
            cache_ttl_seconds=settings.cache_ttl_schedule_minutes * 60,
        )
        if resp is None:
            return SourcedPayload(
                source=SOURCE_NAME, retrieved_at=utc_now_iso(), from_cache=False
            )._replace_data({"games": []})
        games = []
        for date_block in resp.json_body.get("dates", []):
            for g in date_block.get("games", []):
                games.append(self._normalize_game(g))
        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=resp.retrieved_at, from_cache=resp.from_cache)
        return payload._replace_data({"games": games})

    def _normalize_game(self, g: dict) -> dict:
        teams = g.get("teams", {})
        home = teams.get("home", {})
        away = teams.get("away", {})
        venue = g.get("venue", {})
        return {
            "game_id": str(g.get("gamePk")),
            "game_date": g.get("officialDate"),
            "scheduled_start_utc": g.get("gameDate"),
            "status": (g.get("status", {}) or {}).get("detailedState"),
            "abstract_state": (g.get("status", {}) or {}).get("abstractGameState"),
            "doubleheader": g.get("doubleHeader"),
            "game_number": g.get("gameNumber", 1),
            "home_team": (home.get("team", {}) or {}).get("name"),
            "home_team_id": (home.get("team", {}) or {}).get("id"),
            "away_team": (away.get("team", {}) or {}).get("name"),
            "away_team_id": (away.get("team", {}) or {}).get("id"),
            "ballpark": venue.get("name"),
            "venue_id": venue.get("id"),
            "probable_home_pitcher_id": ((home.get("probablePitcher") or {}).get("id")),
            "probable_home_pitcher_name": ((home.get("probablePitcher") or {}).get("fullName")),
            "probable_away_pitcher_id": ((away.get("probablePitcher") or {}).get("id")),
            "probable_away_pitcher_name": ((away.get("probablePitcher") or {}).get("fullName")),
            "raw": g,
        }

    # ------------------------------------------------------------------ lineup
    def get_confirmed_lineup(self, game_id: str, team_id: int) -> Optional[SourcedPayload]:
        """Confirmed lineups post to the live boxscore roughly ~1-2.5 hours
        before first pitch. We detect confirmation via presence of a
        battingOrder on each player entry for the given team."""
        url = f"{self.base_v1_1}/game/{game_id}/feed/live"
        resp = http_client.get_json(
            url,
            cache_category="confirmed_lineup",
            cache_ttl_seconds=settings.cache_ttl_confirmed_lineup_minutes * 60,
        )
        if resp is None:
            return None

        live = resp.json_body
        box = (live.get("liveData", {}) or {}).get("boxscore", {}) or {}
        teams = box.get("teams", {}) or {}
        side = "home" if (teams.get("home", {}).get("team", {}) or {}).get("id") == team_id else "away"
        team_box = teams.get(side, {}) or {}
        batting_order_ids = team_box.get("battingOrder") or []
        if not batting_order_ids:
            return None  # not confirmed yet

        players = team_box.get("players", {}) or {}
        lineup = []
        for spot_index, pid in enumerate(batting_order_ids, start=1):
            key = f"ID{pid}"
            p = players.get(key, {}) or {}
            person = p.get("person", {}) or {}
            bat_side = (p.get("batSide", {}) or {}).get("code")
            position = (p.get("position", {}) or {}).get("abbreviation")
            lineup.append(
                {
                    "batting_order": spot_index,
                    "player_id": person.get("id"),
                    "player_name": person.get("fullName"),
                    "bat_side": bat_side,
                    "position": position,
                }
            )
        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=resp.retrieved_at, from_cache=resp.from_cache)
        return payload._replace_data({"lineup": lineup, "status": "confirmed"})

    def get_projected_lineup(self, team_id: int, vs_pitcher_hand: Optional[str] = None) -> SourcedPayload:
        """Best-effort projected lineup: use the team's most recent completed
        game's actual batting order as a proxy. This is a documented,
        transparent fallback -- NOT presented as confirmed."""
        url = f"{self.base}/schedule"
        params = {"sportId": 1, "teamId": team_id, "hydrate": "team", "gameType": "R"}
        resp = http_client.get_json(
            url, params=params, cache_category="team_schedule", cache_ttl_seconds=6 * 3600
        )
        recent_game_pk = None
        if resp is not None:
            dates = resp.json_body.get("dates", [])
            candidates = []
            for d in dates:
                for g in d.get("games", []):
                    if (g.get("status", {}) or {}).get("abstractGameState") == "Final":
                        candidates.append(g)
            if candidates:
                candidates.sort(key=lambda g: g.get("gameDate", ""))
                recent_game_pk = candidates[-1].get("gamePk")

        payload = SourcedPayload(source=f"{SOURCE_NAME}:projected_from_last_lineup", retrieved_at=utc_now_iso())
        if recent_game_pk is None:
            return payload._replace_data({"lineup": [], "status": "projected", "basis": "no_recent_game_found"})

        confirmed = self.get_confirmed_lineup(str(recent_game_pk), team_id)
        if confirmed is None:
            return payload._replace_data({"lineup": [], "status": "projected", "basis": "recent_game_lineup_unavailable"})
        data = confirmed.data
        return payload._replace_data(
            {"lineup": data["lineup"], "status": "projected", "basis": f"most_recent_lineup_game_{recent_game_pk}"}
        )

    # -------------------------------------------------------------- player stats
    def get_pitcher_stats(self, pitcher_id: int, season: int) -> SourcedPayload:
        return self._get_person_stats(pitcher_id, season, group="pitching")

    def get_batter_stats(self, batter_id: int, season: int) -> SourcedPayload:
        return self._get_person_stats(batter_id, season, group="hitting")

    def _get_person_stats(self, person_id: int, season: int, group: str) -> SourcedPayload:
        url = f"{self.base}/people/{person_id}"
        params = {
            "hydrate": (
                f"stats(group=[{group}],type=[season,career,"
                f"vsTeam,gameLog],season={season}),currentTeam"
            )
        }
        resp = http_client.get_json(
            url,
            params=params,
            cache_category="player_stats",
            cache_ttl_seconds=settings.cache_ttl_player_stats_hours * 3600,
        )
        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=utc_now_iso() if resp is None else resp.retrieved_at,
                                  from_cache=False if resp is None else resp.from_cache)
        if resp is None:
            return payload._replace_data({"person": None, "stats": []})
        people = resp.json_body.get("people", [])
        person = people[0] if people else None
        return payload._replace_data({"person": person, "stats": (person or {}).get("stats", [])})

    def get_person_handedness(self, person_id: int) -> Optional[dict]:
        url = f"{self.base}/people/{person_id}"
        resp = http_client.get_json(
            url, cache_category="player_bio", cache_ttl_seconds=24 * 3600
        )
        if resp is None:
            return None
        people = resp.json_body.get("people", [])
        if not people:
            return None
        p = people[0]
        return {
            "bat_side": (p.get("batSide", {}) or {}).get("code"),
            "pitch_hand": (p.get("pitchHand", {}) or {}).get("code"),
            "full_name": p.get("fullName"),
        }

    def get_boxscore_for_result_capture(self, game_id: str) -> Optional[dict]:
        """Used by update-results: pull final boxscore/linescore for a
        completed game to extract actual pitcher performance."""
        url = f"{self.base_v1_1}/game/{game_id}/feed/live"
        resp = http_client.get_json(url, cache_category="final_boxscore", cache_ttl_seconds=0)
        if resp is None:
            return None
        return resp.json_body
