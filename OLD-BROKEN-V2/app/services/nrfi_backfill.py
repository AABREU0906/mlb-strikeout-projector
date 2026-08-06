"""
Historical first-inning backfill.

Design for resumability: every game's result is upserted keyed by game_id
as soon as it's fetched, and `is_nrfi is not None` is the completeness
marker. If the process is interrupted (Ctrl+C, crash, closed terminal) and
re-run with the same date range, already-complete games are skipped
immediately (a fast DB check, no network call), so progress is never lost
and no game is re-fetched unnecessarily -- this satisfies "resume after
interruption" and "skip existing complete records" without needing a
separate checkpoint file.

Games that come back with no result yet (postponed, in progress, or the
feed hasn't posted innings data) are left with is_nrfi=None and will be
retried on the next backfill run that covers that date, satisfying "update
only missing or incomplete records."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from app.config.logging_config import get_logger
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.database.models import FirstInningGameResult
from app.database.repositories import FirstInningGameResultRepository, GameRepository
from app.database.session import session_scope

logger = get_logger(__name__)
console = Console()

# MLB Stats API has no published hard rate limit for this endpoint, but we
# self-limit to be a considerate client per the project's data-source rules
# (avoid unnecessary repeated/rapid requests).
REQUEST_PAUSE_SECONDS = 0.15


@dataclass
class BackfillReport:
    games_seen: int = 0
    already_complete_skipped: int = 0
    newly_completed: int = 0
    not_yet_final: int = 0
    fetch_failures: int = 0
    failed_game_ids: list[str] = field(default_factory=list)


class NrfiBackfillService:
    def __init__(self, provider: Optional[MlbStatsApiProvider] = None):
        self.provider = provider or MlbStatsApiProvider()

    def backfill_date_range(self, start_date: str, end_date: str, season: int) -> BackfillReport:
        report = BackfillReport()

        console.print(f"[bold]Fetching schedule for {start_date} to {end_date}...[/bold]")
        schedule = self.provider.get_schedule_range(start_date, end_date)
        games = schedule.data.get("games", [])
        report.games_seen = len(games)

        if not games:
            console.print("[yellow]No games found in this date range.[/yellow]")
            return report

        # Only regular/postseason games that have actually been played or
        # are in progress are worth fetching; skip clearly-future scheduled
        # games (no result to backfill yet) to save requests.
        eligible = [g for g in games if g.get("abstract_state") in ("Final", "Live")]
        console.print(f"[dim]{len(games)} scheduled game(s), {len(eligible)} eligible (Final/Live) to check.[/dim]")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Backfilling first-inning results", total=len(eligible))

            for g in eligible:
                game_id = g["game_id"]

                with session_scope() as session:
                    GameRepository.upsert(session, g)
                    already_done = FirstInningGameResultRepository.exists_complete(session, game_id)

                if already_done:
                    report.already_complete_skipped += 1
                    progress.advance(task)
                    continue

                try:
                    result = self.provider.get_first_inning_result(game_id)
                except Exception as exc:
                    logger.warning("Failed to fetch first-inning result for game %s: %s", game_id, exc)
                    report.fetch_failures += 1
                    report.failed_game_ids.append(game_id)
                    progress.advance(task)
                    continue

                time.sleep(REQUEST_PAUSE_SECONDS)

                if result is None or result.get("is_nrfi") is None:
                    report.not_yet_final += 1
                    progress.advance(task)
                    continue

                row = self._build_row(g, result, season)
                with session_scope() as session:
                    FirstInningGameResultRepository.upsert(session, row)
                report.newly_completed += 1
                progress.advance(task)

        self._print_report(report)
        return report

    def backfill_season(self, season: int) -> BackfillReport:
        # Regular season runs roughly late March through early October;
        # postseason through early November. Using a generous fixed window
        # per season rather than guessing exact opening day per year.
        start_date = f"{season}-03-15"
        end_date = f"{season}-11-05"
        return self.backfill_date_range(start_date, end_date, season)

    def update_recent(self, days_back: int = 3) -> BackfillReport:
        import datetime as dt

        today = dt.date.today()
        start = (today - dt.timedelta(days=days_back)).isoformat()
        end = today.isoformat()
        return self.backfill_date_range(start, end, season=today.year)

    def _build_row(self, game: dict, result: dict, season: int) -> FirstInningGameResult:
        away_p = result.get("away_starting_pitcher") or {}
        home_p = result.get("home_starting_pitcher") or {}
        away_bat = result.get("away_batting") or {}
        home_bat = result.get("home_batting") or {}
        return FirstInningGameResult(
            game_id=game["game_id"],
            game_date=game["game_date"],
            season=season,
            home_team=game["home_team"],
            away_team=game["away_team"],
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            home_starting_pitcher_id=home_p.get("id"),
            away_starting_pitcher_id=away_p.get("id"),
            home_starting_pitcher_name=home_p.get("name"),
            away_starting_pitcher_name=away_p.get("name"),
            away_first_inning_runs=result.get("away_first_inning_runs"),
            home_first_inning_runs=result.get("home_first_inning_runs"),
            away_pitcher_scoreless_first=result.get("away_pitcher_scoreless_first"),
            home_pitcher_scoreless_first=result.get("home_pitcher_scoreless_first"),
            is_nrfi=result.get("is_nrfi"),
            day_night=result.get("day_night"),
            venue_id=result.get("venue_id"),
            game_status=result.get("game_status"),
            away_plate_appearances=away_bat.get("plate_appearances"),
            away_at_bats=away_bat.get("at_bats"),
            away_hits=away_bat.get("hits"),
            away_walks=away_bat.get("walks"),
            away_strikeouts=away_bat.get("strikeouts"),
            away_home_runs=away_bat.get("home_runs"),
            away_total_bases=away_bat.get("total_bases"),
            home_plate_appearances=home_bat.get("plate_appearances"),
            home_at_bats=home_bat.get("at_bats"),
            home_hits=home_bat.get("hits"),
            home_walks=home_bat.get("walks"),
            home_strikeouts=home_bat.get("strikeouts"),
            home_home_runs=home_bat.get("home_runs"),
            home_total_bases=home_bat.get("total_bases"),
            away_pitcher_first_inning_pitches=result.get("away_pitcher_first_inning_pitches"),
            home_pitcher_first_inning_pitches=result.get("home_pitcher_first_inning_pitches"),
        )

    def _print_report(self, report: BackfillReport) -> None:
        console.print("\n[bold]Backfill summary[/bold]")
        console.print(f"  Games seen (scheduled):     {report.games_seen}")
        console.print(f"  Already complete, skipped:  {report.already_complete_skipped}")
        console.print(f"  Newly completed:            [green]{report.newly_completed}[/green]")
        console.print(f"  Not yet final (will retry): {report.not_yet_final}")
        if report.fetch_failures:
            console.print(f"  [red]Fetch failures: {report.fetch_failures}[/red]")
            for gid in report.failed_game_ids[:10]:
                console.print(f"    - {gid}")
