"""
Credit-conserving Odds API orchestration for `project-confirmed-slate`.

Per The Odds API's actual documented behavior (player props, including
pitcher_strikeouts, are non-featured markets only queryable one event at a
time via /v4/sports/baseball_mlb/events/{eventId}/odds -- NOT retrievable
in one sport-wide request), this module enforces exactly two call shapes
for an entire run:

  1. GET /v4/sports/baseball_mlb/events  -- called AT MOST ONCE per run,
     regardless of how many games are being processed.
  2. GET /v4/sports/baseball_mlb/events/{eventId}/odds?markets=pitcher_strikeouts
     &bookmakers=fanduel&oddsFormat=american -- called AT MOST ONCE PER
     EVENT per run, reused for BOTH starting pitchers in that game.

Both calls go through the shared http_client, but this module tracks
per-run call counts and credit headers itself via an in-memory,
per-session cache -- the "at most once" guarantee holds by construction,
not by hoping a file-cache TTL doesn't expire mid-run.

On the events-endpoint quota question: The Odds API's own documentation
states the /events endpoint does not consume quota, but this module does
not hardcode that as an assumption baked into the credit math -- it
simply reports whatever x-requests-remaining/x-requests-used headers each
call actually returns, and the CLI displays the events-list call count
and the event-odds call count separately so you can see directly whether
credits moved between them on your plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.config.settings import settings
from app.utilities.http_client import http_client

FANDUEL_BOOKMAKER_KEY = "fanduel"
STRIKEOUTS_MARKET_KEY = "pitcher_strikeouts"


@dataclass
class OddsRunSession:
    """Tracks everything about ONE run's Odds API usage: the events list
    (fetched at most once), each event's odds response (fetched at most
    once per event), and credit-usage headers as of the most recent LIVE
    (non-cached) call."""
    api_key: Optional[str] = field(default_factory=lambda: settings.odds_api_key)
    base_url: str = field(default_factory=lambda: settings.odds_api_base_url)

    _events: Optional[list[dict]] = field(default=None, repr=False)
    _event_odds_cache: dict[str, Optional[dict]] = field(default_factory=dict, repr=False)

    events_list_calls_made: int = 0
    event_odds_calls_made: int = 0
    credits_remaining_after_events_call: Optional[int] = None
    credits_remaining_after_last_event_odds_call: Optional[int] = None
    credits_used_this_run: Optional[int] = None

    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def credits_remaining(self) -> Optional[int]:
        """The most recent (lowest-staleness) remaining-credits value
        observed this run, from whichever call happened most recently."""
        return (
            self.credits_remaining_after_last_event_odds_call
            if self.credits_remaining_after_last_event_odds_call is not None
            else self.credits_remaining_after_events_call
        )

    def _parse_int_header(self, headers: dict, name: str) -> Optional[int]:
        raw = headers.get(name)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    def get_events(self) -> list[dict]:
        """Fetches /sports/baseball_mlb/events -- at most once per
        OddsRunSession instance, regardless of how many times this is
        called during the run."""
        if self._events is not None:
            return self._events
        if not self.is_configured():
            self._events = []
            return self._events

        url = f"{self.base_url}/sports/baseball_mlb/events"
        params = {"apiKey": self.api_key}
        resp = http_client.get_json(url, params=params, cache_category="confirmed_slate_events", cache_ttl_seconds=0)
        self.events_list_calls_made += 1
        if resp is None:
            self._events = []
            return self._events

        remaining = self._parse_int_header(resp.headers, "x-requests-remaining")
        used = self._parse_int_header(resp.headers, "x-requests-used")
        if remaining is not None:
            self.credits_remaining_after_events_call = remaining
        if used is not None:
            self.credits_used_this_run = used

        self._events = resp.json_body if isinstance(resp.json_body, list) else []
        return self._events

    def get_event_odds(self, event_id: str) -> Optional[dict]:
        """Fetches pitcher_strikeouts odds for ONE event, filtered to
        FanDuel only, at most once per event_id for the lifetime of this
        session -- a second call for the same event_id (e.g. because two
        pitchers in the same game both need odds) returns the cached
        response instead of making a second request."""
        if event_id in self._event_odds_cache:
            return self._event_odds_cache[event_id]

        if not self.is_configured():
            self._event_odds_cache[event_id] = None
            return None

        url = f"{self.base_url}/sports/baseball_mlb/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": settings.odds_api_region,
            "markets": STRIKEOUTS_MARKET_KEY,
            "bookmakers": FANDUEL_BOOKMAKER_KEY,
            "oddsFormat": "american",
        }
        resp = http_client.get_json(url, params=params, cache_category="confirmed_slate_event_odds", cache_ttl_seconds=0)
        self.event_odds_calls_made += 1
        if resp is None:
            self._event_odds_cache[event_id] = None
            return None

        remaining = self._parse_int_header(resp.headers, "x-requests-remaining")
        used = self._parse_int_header(resp.headers, "x-requests-used")
        if remaining is not None:
            self.credits_remaining_after_last_event_odds_call = remaining
        if used is not None:
            self.credits_used_this_run = used

        self._event_odds_cache[event_id] = resp.json_body
        return resp.json_body


def extract_fanduel_pitcher_outcomes(event_odds: dict) -> list[dict]:
    """Flattens an event-odds response down to just FanDuel's
    pitcher_strikeouts outcomes, each tagged with the pitcher name from
    the outcome's `description` field (The Odds API's convention for
    player-prop outcomes). Any other bookmaker or market present in the
    response (e.g. if the API returns more than requested) is ignored --
    this is the explicit FanDuel-only, pitcher_strikeouts-only filter."""
    if not event_odds:
        return []

    outcomes = []
    for bookmaker in event_odds.get("bookmakers", []):
        if bookmaker.get("key") != FANDUEL_BOOKMAKER_KEY:
            continue
        for market in bookmaker.get("markets", []):
            if market.get("key") != STRIKEOUTS_MARKET_KEY:
                continue
            for outcome in market.get("outcomes", []):
                outcomes.append({
                    "pitcher_name": outcome.get("description"),
                    "name": outcome.get("name"),
                    "point": outcome.get("point"),
                    "price": outcome.get("price"),
                    "last_update": market.get("last_update"),
                })
    return outcomes
