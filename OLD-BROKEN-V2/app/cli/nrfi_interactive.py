from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.prompt import Confirm, IntPrompt

console = Console()


def prompt_nrfi_odds() -> Optional[tuple[int, int]]:
    """Returns (nrfi_odds, yrfi_odds) or None if the user has none to enter."""
    if not Confirm.ask("Enter NRFI/YRFI odds if you have them?", default=False):
        return None
    nrfi_odds = IntPrompt.ask("NRFI odds (American, e.g. -130)")
    yrfi_odds = IntPrompt.ask("YRFI odds (American, e.g. +108)")
    return nrfi_odds, yrfi_odds


def confirm_pitchers_known(game: dict) -> bool:
    """NRFI needs BOTH starters; returns True if both probable pitchers are
    present on the game record (from the schedule feed's probablePitcher
    hydration), which is the normal case once starters are announced."""
    return bool(game.get("probable_home_pitcher_id") and game.get("probable_away_pitcher_id"))
