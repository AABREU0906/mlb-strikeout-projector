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
                f"vsTeam,gameLog],season={season},sportId=1,gameType=R),currentTeam"
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

    def get_batter_handedness_splits_raw(self, batter_id: int, season: int) -> Optional[dict]:
        """Raw vs-RHP/vs-LHP stat blocks for a batter (includes MLB's own
        precomputed avg/obp/slg/ops/strikeOuts/etc. fields). Shared by the
        strikeout model's batter feature builder (which reads K-rate fields)
        and the NRFI/YRFI BvP builder (which reads avg/obp/slg fields) so
        neither duplicates this HTTP call."""
        url = f"{self.base}/people/{batter_id}/stats"
        params = {"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": "vr,vl",
                   "sportId": 1, "gameType": "R"}
        resp = http_client.get_json(
            url, params=params, cache_category="batter_splits",
            cache_ttl_seconds=settings.cache_ttl_player_stats_hours * 3600,
        )
        if resp is None:
            return None
        stats = resp.json_body.get("stats", [])
        out = {}
        for block in stats:
            for split in block.get("splits", []):
                code = (split.get("split", {}) or {}).get("code")
                stat = split.get("stat", {})
                if code == "vr":
                    out["vs_rhp"] = stat
                elif code == "vl":
                    out["vs_lhp"] = stat
        return out or None

    def get_batter_vs_pitcher(self, batter_id: int, pitcher_id: int) -> SourcedPayload:
        """Career head-to-head hitting stats for one batter against one
        specific pitcher, via the vsPlayer stat type. This is a real but
        thinly-documented MLB Stats API endpoint; the field mapping below
        is defensive (missing fields -> None, never fabricated) since the
        exact response shape can't be verified without a live request in
        this environment -- see PROJECT NOTE in nrfi_bvp_features.py."""
        url = f"{self.base}/people/{batter_id}/stats"
        params = {
            "stats": "vsPlayer",
            "opposingPlayerId": pitcher_id,
            "group": "hitting",
            "sportId": 1,
        }
        resp = http_client.get_json(
            url, params=params, cache_category="bvp_stats",
            cache_ttl_seconds=settings.cache_ttl_player_stats_hours * 3600,
        )
        payload = SourcedPayload(
            source=SOURCE_NAME,
            retrieved_at=utc_now_iso() if resp is None else resp.retrieved_at,
            from_cache=False if resp is None else resp.from_cache,
        )
        if resp is None:
            return payload._replace_data({"stat": None})

        stats = resp.json_body.get("stats", [])
        stat_line = None
        for block in stats:
            splits = block.get("splits", [])
            if splits:
                stat_line = splits[0].get("stat", {})
                break
        return payload._replace_data({"stat": stat_line})

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

    def get_home_plate_umpire(self, game_id: str) -> Optional[dict]:
        """Returns {"id": int, "name": str} for the home-plate umpire if the
        live feed has posted official assignments, else None. Reused by
        both the strikeout model (umpire K/BB effect) and the NRFI/YRFI
        model (umpire run-environment effect)."""
        url = f"{self.base_v1_1}/game/{game_id}/feed/live"
        resp = http_client.get_json(
            url, cache_category="umpire_assignment",
            cache_ttl_seconds=settings.cache_ttl_confirmed_lineup_minutes * 60,
        )
        if resp is None:
            return None
        live = resp.json_body
        officials = (live.get("liveData", {}) or {}).get("boxscore", {}) or {}
        officials_list = officials.get("officials", [])
        for o in officials_list:
            official_type = o.get("officialType", "")
            if "home plate" in official_type.lower():
                person = o.get("official", {}) or {}
                if person.get("id") and person.get("fullName"):
                    return {"id": person["id"], "name": person["fullName"]}
        return None

    def get_schedule_range(self, start_date: str, end_date: str) -> SourcedPayload:
        """Fetch all games between two dates (inclusive) in a single call --
        used by the NRFI/YRFI historical backfill instead of looping one
        request per day, to minimize request volume per the project's
        rate-limit/caching rules."""
        url = f"{self.base}/schedule"
        params = {
            "sportId": 1,
            "startDate": start_date,
            "endDate": end_date,
            "hydrate": "probablePitcher,team,venue,status",
        }
        resp = http_client.get_json(
            url, params=params, cache_category="schedule_range",
            cache_ttl_seconds=6 * 3600,
        )
        if resp is None:
            return SourcedPayload(source=SOURCE_NAME, retrieved_at=utc_now_iso())._replace_data({"games": []})
        games = []
        for date_block in resp.json_body.get("dates", []):
            for g in date_block.get("games", []):
                games.append(self._normalize_game(g))
        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=resp.retrieved_at, from_cache=resp.from_cache)
        return payload._replace_data({"games": games})

    def get_first_inning_result(self, game_id: str) -> Optional[dict]:
        """Extracts ground-truth first-inning data for a completed game:
        runs scored by each side in the 1st (from linescore, the stable
        source), the actual starting pitchers, and -- separately -- whether
        each starter personally avoided allowing a run before being pulled,
        vs. whether the *team's* half-inning was scoreless (these differ
        only when a starter is removed mid-1st-inning, e.g. injury/opener
        situations, which this detects via the play-by-play pitching-change
        events rather than assuming the starter faced the whole inning).

        Returns None if the game isn't far enough along to have a complete
        1st inning yet, or if the feed can't be read. Never guesses --
        fields this can't determine confidently are left as None."""
        url = f"{self.base_v1_1}/game/{game_id}/feed/live"
        resp = http_client.get_json(url, cache_category="first_inning_result", cache_ttl_seconds=0)
        if resp is None:
            return None

        live = resp.json_body
        game_data = live.get("gameData", {}) or {}
        live_data = live.get("liveData", {}) or {}
        status = (game_data.get("status", {}) or {}).get("abstractGameState")

        linescore = live_data.get("linescore", {}) or {}
        innings = linescore.get("innings", [])
        if not innings:
            return None  # game hasn't started / no innings posted yet
        first = innings[0]
        away_runs = (first.get("away", {}) or {}).get("runs")
        home_runs = (first.get("home", {}) or {}).get("runs")
        if away_runs is None or home_runs is None:
            return None  # 1st inning not yet complete

        box = live_data.get("boxscore", {}) or {}
        teams_box = box.get("teams", {}) or {}

        def _starting_pitcher(side: str) -> Optional[dict]:
            team_box = teams_box.get(side, {}) or {}
            pitcher_ids = team_box.get("pitchers", [])
            if not pitcher_ids:
                return None
            first_pitcher_id = pitcher_ids[0]
            player = team_box.get("players", {}).get(f"ID{first_pitcher_id}", {}) or {}
            person = player.get("person", {}) or {}
            if not person.get("id"):
                return None
            return {"id": person["id"], "name": person.get("fullName")}

        away_pitcher = _starting_pitcher("away")
        home_pitcher = _starting_pitcher("home")

        # Detect mid-1st-inning pitching changes via play-by-play; if one
        # occurred, we cannot confidently attribute the runs allowed to a
        # single pitcher's personal line without deeper event parsing, so
        # that side's "pitcher_scoreless_first" is left as None (unknown)
        # rather than assumed.
        all_plays = live_data.get("plays", {}).get("allPlays", []) if live_data.get("plays") else []
        first_inning_sub_occurred = {"away": False, "home": False}
        for play in all_plays:
            about = play.get("about", {}) or {}
            if about.get("inning") != 1:
                continue
            for event in play.get("playEvents", []) or []:
                details = event.get("details", {}) or {}
                if "substitution" in (details.get("event") or "").lower() and "pitch" in (details.get("event") or "").lower():
                    half = "home" if about.get("isTopInning") else "away"
                    # A pitching sub during the top of the 1st means the
                    # AWAY pitcher (who was pitching to the home... wait:
                    # top of 1st = away team batting, home team pitching)
                    pitching_side = "home" if about.get("isTopInning") else "away"
                    first_inning_sub_occurred[pitching_side] = True

        # The home team's pitcher faces the away lineup in the top of the
        # 1st (so a pitching change flagged under the "home" pitching side
        # means the home pitcher's personal figure is uncertain), and the
        # away team's pitcher faces the home lineup in the bottom of the
        # 1st (uncertain if a sub is flagged under "away").
        home_pitcher_scoreless = None if first_inning_sub_occurred["home"] else (away_runs == 0)
        away_pitcher_scoreless = None if first_inning_sub_occurred["away"] else (home_runs == 0)

        venue = game_data.get("venue", {}) or {}
        game_datetime = game_data.get("datetime", {}) or {}

        away_bat, home_bat, away_pitches, home_pitches = self._first_inning_batting_aggregates(all_plays)

        return {
            "game_status": status,
            "away_first_inning_runs": away_runs,
            "home_first_inning_runs": home_runs,
            "is_nrfi": (away_runs == 0 and home_runs == 0),
            "away_starting_pitcher": away_pitcher,
            "home_starting_pitcher": home_pitcher,
            "away_pitcher_scoreless_first": away_pitcher_scoreless,
            "home_pitcher_scoreless_first": home_pitcher_scoreless,
            "day_night": game_datetime.get("dayNight"),
            "venue_id": venue.get("id"),
            "away_batting": away_bat,
            "home_batting": home_bat,
            "away_pitcher_first_inning_pitches": away_pitches,
            "home_pitcher_first_inning_pitches": home_pitches,
        }

    @staticmethod
    def _first_inning_batting_aggregates(all_plays: list) -> tuple[dict, dict, Optional[int], Optional[int]]:
        """Parses 1st-inning plate appearances from play-by-play into PA/AB/
        H/BB/K/HR/total-bases for each side's offense, plus pitches thrown
        by each starter in the 1st. Each `play` in allPlays represents one
        complete plate appearance; `about.isTopInning`=True means the away
        team is batting (against the home starter).

        Returns (away_batting_dict, home_batting_dict, away_pitcher_pitches,
        home_pitcher_pitches). Any field this can't confidently determine
        stays at its zero/None default rather than being guessed -- callers
        should treat an entirely-zero aggregate on a game with no 1st-inning
        plays as "no data" rather than "confirmed zero", though in practice
        every completed 1st inning has at least 3 plate appearances.
        """
        HIT_EVENTS = {"single", "double", "triple", "home_run"}
        TOTAL_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
        WALK_EVENTS = {"walk", "intent_walk"}
        K_EVENTS = {"strikeout", "strikeout_double_play"}

        def _blank():
            return {"plate_appearances": 0, "at_bats": 0, "hits": 0, "walks": 0,
                    "strikeouts": 0, "home_runs": 0, "total_bases": 0}

        away_bat, home_bat = _blank(), _blank()
        away_pitches, home_pitches = 0, 0
        any_play_found = False

        for play in all_plays:
            about = play.get("about", {}) or {}
            if about.get("inning") != 1:
                continue
            any_play_found = True
            is_top = about.get("isTopInning")
            batting_side = away_bat if is_top else home_bat
            # Pitching side is the OTHER team's starter (top of 1st = home
            # team pitching to away batters).
            pitching_is_home = bool(is_top)

            result = play.get("result", {}) or {}
            event_type = (result.get("eventType") or "").lower()

            batting_side["plate_appearances"] += 1
            if event_type in WALK_EVENTS or event_type == "hit_by_pitch":
                batting_side["walks"] += 1
            else:
                batting_side["at_bats"] += 1
                if event_type in K_EVENTS:
                    batting_side["strikeouts"] += 1
                if event_type in HIT_EVENTS:
                    batting_side["hits"] += 1
                    batting_side["total_bases"] += TOTAL_BASES.get(event_type, 0)
                    if event_type == "home_run":
                        batting_side["home_runs"] += 1

            pitch_events = [e for e in (play.get("playEvents") or []) if e.get("isPitch")]
            if pitching_is_home:
                home_pitches += len(pitch_events)
            else:
                away_pitches += len(pitch_events)

        if not any_play_found:
            return away_bat, home_bat, None, None
        return away_bat, home_bat, away_pitches or None, home_pitches or None
