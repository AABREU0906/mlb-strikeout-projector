"""
Matches The Odds API's event objects to MLB Stats API games.

The two APIs use different identifier spaces entirely (Odds API events
have their own UUID-style ids; MLB Stats API uses gamePk integers), so
matching has to go through team names + date/time, not a shared id.

Deliberately conservative throughout: an uncertain match returns None
rather than guessing, per the explicit requirement not to guess ambiguous
matches. This module never calls any API itself -- it's pure matching
logic over already-fetched data, so it's fully unit-testable without
mocks.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

_TEAM_ALIASES: dict[str, str] = {}


def _register(canonical: str, *variants: str) -> None:
    _TEAM_ALIASES[canonical.lower()] = canonical
    for v in variants:
        _TEAM_ALIASES[v.lower()] = canonical


_register("Arizona Diamondbacks", "Diamondbacks", "D-backs", "Arizona")
_register("Atlanta Braves", "Braves", "Atlanta")
_register("Baltimore Orioles", "Orioles", "Baltimore")
_register("Boston Red Sox", "Red Sox", "Boston")
_register("Chicago Cubs", "Cubs")
_register("Chicago White Sox", "White Sox")
_register("Cincinnati Reds", "Reds", "Cincinnati")
_register("Cleveland Guardians", "Guardians", "Cleveland")
_register("Colorado Rockies", "Rockies", "Colorado")
_register("Detroit Tigers", "Tigers", "Detroit")
_register("Houston Astros", "Astros", "Houston")
_register("Kansas City Royals", "Royals", "Kansas City")
_register("Los Angeles Angels", "Angels", "LA Angels", "Anaheim Angels")
_register("Los Angeles Dodgers", "Dodgers", "LA Dodgers")
_register("Miami Marlins", "Marlins", "Miami")
_register("Milwaukee Brewers", "Brewers", "Milwaukee")
_register("Minnesota Twins", "Twins", "Minnesota")
_register("New York Mets", "Mets")
_register("New York Yankees", "Yankees")
_register("Athletics", "Oakland Athletics", "Las Vegas Athletics", "A's", "Oakland A's")
_register("Philadelphia Phillies", "Phillies", "Philadelphia")
_register("Pittsburgh Pirates", "Pirates", "Pittsburgh")
_register("San Diego Padres", "Padres", "San Diego")
_register("San Francisco Giants", "Giants", "San Francisco", "SF Giants")
_register("Seattle Mariners", "Mariners", "Seattle")
_register("St. Louis Cardinals", "Cardinals", "St Louis Cardinals", "Saint Louis Cardinals")
_register("Tampa Bay Rays", "Rays", "Tampa Bay")
_register("Texas Rangers", "Rangers", "Texas")
_register("Toronto Blue Jays", "Blue Jays", "Toronto")
_register("Washington Nationals", "Nationals", "Washington")


def canonical_team_name(raw_name: Optional[str]) -> Optional[str]:
    """Returns the canonical franchise name, or None if the input isn't a
    recognized variant -- callers must treat None as "cannot safely
    match", not fall back to a raw string comparison."""
    if not raw_name:
        return None
    return _TEAM_ALIASES.get(raw_name.strip().lower())


@dataclass
class MatchedEvent:
    event_id: str
    home_team: str
    away_team: str
    commence_time: str


def match_event_to_game(
    events: list[dict],
    game_home_team: str,
    game_away_team: str,
    game_scheduled_start_utc: str,
    max_start_time_diff_minutes: int = 180,
) -> Optional[MatchedEvent]:
    """Finds the single Odds API event corresponding to an MLB game, by
    canonical home/away team name AND a start-time proximity check (the
    time check disambiguates doubleheaders, where both games share the
    same two teams on the same date). Returns None -- never a guess --
    if zero or more than one event satisfies both team names within the
    time window."""
    canonical_home = canonical_team_name(game_home_team)
    canonical_away = canonical_team_name(game_away_team)
    if canonical_home is None or canonical_away is None:
        return None

    try:
        game_start = dt.datetime.fromisoformat(game_scheduled_start_utc.replace("Z", "+00:00"))
    except (ValueError, AttributeError, TypeError):
        game_start = None

    candidates = []
    for event in events:
        event_home = canonical_team_name(event.get("home_team"))
        event_away = canonical_team_name(event.get("away_team"))
        if event_home != canonical_home or event_away != canonical_away:
            continue

        if game_start is not None and event.get("commence_time"):
            try:
                event_start = dt.datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
                diff_minutes = abs((event_start - game_start).total_seconds()) / 60
                if diff_minutes > max_start_time_diff_minutes:
                    continue
            except (ValueError, AttributeError, TypeError):
                pass

        candidates.append(event)

    if len(candidates) != 1:
        return None

    chosen = candidates[0]
    return MatchedEvent(
        event_id=chosen["id"],
        home_team=chosen.get("home_team", ""),
        away_team=chosen.get("away_team", ""),
        commence_time=chosen.get("commence_time", ""),
    )
