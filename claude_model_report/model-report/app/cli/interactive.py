from __future__ import annotations

import datetime as dt
from typing import Optional

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from app.data_sources.news_api import Warning, WarningLog
from app.schemas.market import ManualMarketEntry

console = Console()


def prompt_game_date() -> str:
    default = dt.date.today().isoformat()
    return Prompt.ask("Enter game date (YYYY-MM-DD)", default=default)


def select_game(games: list[dict]) -> Optional[dict]:
    if not games:
        console.print("[yellow]No games scheduled for this date.[/yellow]")
        return None

    table = Table(title="Scheduled Games")
    table.add_column("#", justify="right")
    table.add_column("Matchup")
    table.add_column("Time (UTC)")
    table.add_column("Ballpark")
    table.add_column("Status")

    for i, g in enumerate(games, start=1):
        table.add_row(
            str(i),
            f"{g['away_team']} @ {g['home_team']}",
            g.get("scheduled_start_utc", "?"),
            g.get("ballpark", "?"),
            g.get("status", "?"),
        )
    console.print(table)

    choice = IntPrompt.ask("Select a game number", default=1)
    if 1 <= choice <= len(games):
        return games[choice - 1]
    console.print("[red]Invalid selection.[/red]")
    return None


def select_pitcher(game: dict) -> Optional[tuple[int, bool]]:
    """Returns (pitcher_id, is_home)."""
    options = []
    if game.get("probable_home_pitcher_id"):
        options.append((game["probable_home_pitcher_id"], game.get("probable_home_pitcher_name", "Home probable"), True))
    if game.get("probable_away_pitcher_id"):
        options.append((game["probable_away_pitcher_id"], game.get("probable_away_pitcher_name", "Away probable"), False))

    if not options:
        console.print("[yellow]No probable pitchers found from the schedule feed.[/yellow]")
        manual_id = IntPrompt.ask("Enter the MLB player ID of the starting pitcher to analyze")
        is_home = Confirm.ask("Is this pitcher on the home team?", default=True)
        return manual_id, is_home

    table = Table(title="Probable Starting Pitchers")
    table.add_column("#")
    table.add_column("Pitcher")
    table.add_column("Team")
    for i, (pid, name, is_home) in enumerate(options, start=1):
        team = game["home_team"] if is_home else game["away_team"]
        table.add_row(str(i), name, team)
    console.print(table)

    choice = IntPrompt.ask("Select pitcher to analyze", default=1)
    if 1 <= choice <= len(options):
        pid, _, is_home = options[choice - 1]
        return pid, is_home
    console.print("[red]Invalid selection.[/red]")
    return None


def prompt_manual_market(skip_prompt_default: bool = True) -> Optional[ManualMarketEntry]:
    if not Confirm.ask(
        "Enter/override sportsbook strikeout line and odds manually?", default=not skip_prompt_default
    ):
        return None

    line = None
    over = None
    under = None
    if Confirm.ask("Do you have a strikeout prop line?", default=True):
        line = FloatPrompt.ask("Strikeout line (e.g. 6.5)")
        over = IntPrompt.ask("Over odds (American, e.g. -115)")
        under = IntPrompt.ask("Under odds (American, e.g. -105)")

    game_total = None
    if Confirm.ask("Enter game total?", default=False):
        game_total = FloatPrompt.ask("Game total")
    opp_implied = None
    if Confirm.ask("Enter opponent implied runs?", default=False):
        opp_implied = FloatPrompt.ask("Opponent implied runs")

    return ManualMarketEntry(
        strikeout_line=line,
        over_odds=over,
        under_odds=under,
        game_total=game_total,
        opponent_implied_runs=opp_implied,
    )


def prompt_warnings() -> WarningLog:
    log = WarningLog()
    if not Confirm.ask("Add any news/injury/workload warnings you're aware of?", default=False):
        return log

    while True:
        player = Prompt.ask("Player name")
        issue = Prompt.ask("Issue (e.g. 'returning from IL', 'possible pitch limit')")
        source = Prompt.ask("Source (outlet/reporter)")
        confidence = Prompt.ask(
            "Confidence", choices=["confirmed", "reported", "inferred", "speculative"], default="reported"
        )
        effect = Prompt.ask("Expected effect on projection (short description)")
        pub_date = Prompt.ask("Publication date (YYYY-MM-DD, optional)", default="")
        log.add_raw(
            player=player,
            issue=issue,
            source=source,
            confidence=confidence,
            expected_effect=effect,
            published_date=pub_date or None,
        )
        if not Confirm.ask("Add another warning?", default=False):
            break
    return log
