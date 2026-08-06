"""
`python main.py update-results`

For every saved Projection lacking an ActualResult, checks whether the game
is Final and, if so, extracts the pitcher's actual line from the MLB Stats
API live-feed boxscore and stores it as a separate ActualResult row.
Pregame Projection rows are never modified.
"""
from __future__ import annotations

import datetime as dt

from app.config.logging_config import get_logger
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.database.models import ActualResult
from app.database.repositories import ActualResultRepository, ProjectionRepository
from app.database.session import session_scope

logger = get_logger(__name__)


def _extract_pitcher_line(live_feed: dict, pitcher_id: int) -> dict | None:
    box = (live_feed.get("liveData", {}) or {}).get("boxscore", {}) or {}
    teams = box.get("teams", {}) or {}
    for side in ("home", "away"):
        players = (teams.get(side, {}) or {}).get("players", {}) or {}
        key = f"ID{pitcher_id}"
        if key in players:
            stats = (players[key].get("stats", {}) or {}).get("pitching", {}) or {}
            if not stats:
                return None
            return {
                "strikeouts": stats.get("strikeOuts"),
                "innings_pitched": stats.get("inningsPitched"),
                "batters_faced": stats.get("battersFaced"),
                "pitch_count": stats.get("numberOfPitches"),
            }
    return None


def _game_is_final(live_feed: dict) -> bool:
    status = (live_feed.get("gameData", {}) or {}).get("status", {}) or {}
    return status.get("abstractGameState") == "Final"


def update_all_pending_results() -> int:
    provider = MlbStatsApiProvider()
    updated = 0

    with session_scope() as session:
        pending = ProjectionRepository.list_without_results(session)
        game_ids_to_projections: dict[str, list] = {}
        for p in pending:
            game_ids_to_projections.setdefault(p.game_id, []).append(p)

        for game_id, projections in game_ids_to_projections.items():
            live_feed = provider.get_boxscore_for_result_capture(game_id)
            if live_feed is None:
                logger.info("Could not fetch live feed for game %s; skipping.", game_id)
                continue
            if not _game_is_final(live_feed):
                continue

            for proj in projections:
                line = _extract_pitcher_line(live_feed, proj.pitcher_id)
                if line is None:
                    logger.info(
                        "No pitching line found for pitcher %s in game %s (may not have appeared).",
                        proj.pitcher_id, game_id,
                    )
                    continue
                if ActualResultRepository.exists_for_projection(session, proj.id):
                    continue

                ip = _parse_innings(line.get("innings_pitched"))
                result = ActualResult(
                    projection_id=proj.id,
                    actual_strikeouts=line.get("strikeouts"),
                    actual_innings_pitched=ip,
                    actual_batters_faced=line.get("batters_faced"),
                    actual_pitch_count=line.get("pitch_count"),
                    game_result_json={"raw_status": "Final"},
                    source="mlb_stats_api",
                )
                ActualResultRepository.save(session, result)
                updated += 1

    return updated


def _parse_innings(ip_str):
    if ip_str is None:
        return None
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + (int(frac) / 3.0 if frac else 0.0)
    except (ValueError, TypeError):
        return None
