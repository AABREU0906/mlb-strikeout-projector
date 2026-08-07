"""
`python main.py update-results`

For every saved Projection lacking an ActualResult, checks whether the game
is Final and, if so, extracts the pitcher's actual line from the MLB Stats
API live-feed boxscore and stores it as a separate ActualResult row.
Pregame Projection rows are never modified.

Enhanced to categorize every pending projection into exactly one bucket
(updated, skipped-not-final, postponed/suspended/cancelled, unavailable,
or already-settled) so the CLI can show a clear breakdown rather than a
single opaque count, and to support an explicit --force option to
re-fetch and overwrite an already-recorded result (never happens by
default -- a finalized result is never silently overwritten).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from app.config.logging_config import get_logger
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.database.models import ActualResult
from app.database.repositories import ActualResultRepository, ProjectionRepository
from app.database.session import session_scope

logger = get_logger(__name__)

# detailedState fragments that mean "this game will not produce a live
# result right now" but are NOT the same as "still in progress" -- surfaced
# as their own category rather than silently lumped into "not final."
_NON_FINAL_TERMINAL_STATES = ("postponed", "suspended", "cancelled", "canceled")


@dataclass
class ResultsUpdateSummary:
    updated: list[tuple[str, str]] = field(default_factory=list)  # (pitcher_name, game_id)
    skipped_not_final: list[tuple[str, str, str]] = field(default_factory=list)  # (pitcher_name, game_id, detailed_state)
    postponed_or_suspended: list[tuple[str, str, str]] = field(default_factory=list)  # (pitcher_name, game_id, detailed_state)
    unavailable: list[tuple[str, str]] = field(default_factory=list)  # (pitcher_name, game_id)
    already_settled_skipped: list[tuple[str, str]] = field(default_factory=list)  # (pitcher_name, game_id)

    @property
    def total_pending(self) -> int:
        return (
            len(self.updated) + len(self.skipped_not_final) + len(self.postponed_or_suspended)
            + len(self.unavailable) + len(self.already_settled_skipped)
        )


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


def _game_status(live_feed: dict) -> tuple[bool, str]:
    """Returns (is_final, detailed_state_lowercase)."""
    status = (live_feed.get("gameData", {}) or {}).get("status", {}) or {}
    abstract = status.get("abstractGameState", "")
    detailed = (status.get("detailedState") or abstract or "unknown").lower()
    return abstract == "Final", detailed


def update_all_pending_results(force: bool = False) -> ResultsUpdateSummary:
    provider = MlbStatsApiProvider()
    summary = ResultsUpdateSummary()

    with session_scope() as session:
        pending = ProjectionRepository.list_all(session) if force else ProjectionRepository.list_without_results(session)
        game_ids_to_projections: dict[str, list] = {}
        for p in pending:
            game_ids_to_projections.setdefault(p.game_id, []).append(p)

        for game_id, projections in game_ids_to_projections.items():
            live_feed = provider.get_boxscore_for_result_capture(game_id)

            if live_feed is None:
                logger.info("Could not fetch live feed for game %s; skipping.", game_id)
                for proj in projections:
                    summary.unavailable.append((proj.pitcher_name, game_id))
                continue

            is_final, detailed_state = _game_status(live_feed)

            if not is_final:
                for proj in projections:
                    entry = (proj.pitcher_name, game_id, detailed_state)
                    if any(term in detailed_state for term in _NON_FINAL_TERMINAL_STATES):
                        summary.postponed_or_suspended.append(entry)
                    else:
                        summary.skipped_not_final.append(entry)
                continue

            for proj in projections:
                has_existing = ActualResultRepository.exists_for_projection(session, proj.id)
                if has_existing and not force:
                    summary.already_settled_skipped.append((proj.pitcher_name, game_id))
                    continue

                line = _extract_pitcher_line(live_feed, proj.pitcher_id)
                if line is None:
                    logger.info(
                        "No pitching line found for pitcher %s in game %s (may not have appeared).",
                        proj.pitcher_id, game_id,
                    )
                    summary.unavailable.append((proj.pitcher_name, game_id))
                    continue

                ip = _parse_innings(line.get("innings_pitched"))

                if has_existing and force:
                    ActualResultRepository.delete_for_projection(session, proj.id)

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
                ProjectionRepository.update_edge_outcome(session, proj.id, game_final=True)
                summary.updated.append((proj.pitcher_name, game_id))

    return summary


def _parse_innings(ip_str):
    if ip_str is None:
        return None
    try:
        whole, _, frac = str(ip_str).partition(".")
        return int(whole) + (int(frac) / 3.0 if frac else 0.0)
    except (ValueError, TypeError):
        return None
