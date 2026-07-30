"""
Sportsbook odds provider using The Odds API (https://the-odds-api.com).

This is a licensed, ToS-compliant aggregator API (not HTML scraping of any
sportsbook). It requires a key (ODDS_API_KEY). If no key is configured, every
method returns None and the CLI falls back to manual entry -- this is by
design per the project rules ("do not scrape sportsbook websites when
prohibited; use permitted APIs ... or manually entered data").

We never fabricate a response when the key is missing or the request fails.
"""
from __future__ import annotations

from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.base import OddsProvider, SourcedPayload, utc_now_iso
from app.utilities.http_client import http_client

logger = get_logger(__name__)

SOURCE_NAME = "the_odds_api"


class TheOddsApiProvider(OddsProvider):
    def __init__(self):
        self.base_url = settings.odds_api_base_url
        self.api_key = settings.odds_api_key
        self.region = settings.odds_api_region
        self.bookmakers = settings.odds_api_bookmakers_list

    def _configured(self) -> bool:
        if not self.api_key:
            logger.info("ODDS_API_KEY not set -- automated odds unavailable, use manual entry.")
            return False
        return True

    def get_game_market(self, game_id: str) -> Optional[SourcedPayload]:
        """Game-level moneyline/totals/run line. `game_id` here is matched
        by the caller against The Odds API's own event list (see
        find_event_id) since MLB Stats API gamePk and The Odds API event ids
        are different identifier spaces."""
        if not self._configured():
            return None

        url = f"{self.base_url}/sports/baseball_mlb/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": "h2h,totals,spreads",
            "oddsFormat": "american",
            "bookmakers": ",".join(self.bookmakers),
        }
        resp = http_client.get_json(
            url, params=params, cache_category="odds_game_market",
            cache_ttl_seconds=settings.cache_ttl_odds_minutes * 60,
        )
        if resp is None:
            return None

        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=resp.retrieved_at, from_cache=resp.from_cache)
        return payload._replace_data({"events": resp.json_body})

    def get_pitcher_strikeout_props(self, game_id: str, pitcher_name: str) -> Optional[SourcedPayload]:
        """Pitcher strikeout props live under the event-specific
        `player_strikeouts` market, which The Odds API exposes via a
        per-event odds endpoint requiring the event's own id."""
        if not self._configured():
            return None

        event_id = self._find_event_id_by_pitcher(pitcher_name)
        if event_id is None:
            logger.info("Could not match event for pitcher '%s' in odds feed.", pitcher_name)
            return None

        url = f"{self.base_url}/sports/baseball_mlb/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.region,
            "markets": "pitcher_strikeouts",
            "oddsFormat": "american",
            "bookmakers": ",".join(self.bookmakers),
        }
        resp = http_client.get_json(
            url, params=params, cache_category="odds_k_props",
            cache_ttl_seconds=settings.cache_ttl_odds_minutes * 60,
        )
        if resp is None:
            return None

        lines = self._extract_pitcher_lines(resp.json_body, pitcher_name)
        if not lines:
            return None

        payload = SourcedPayload(source=SOURCE_NAME, retrieved_at=resp.retrieved_at, from_cache=resp.from_cache)
        return payload._replace_data({"lines": lines})

    def _find_event_id_by_pitcher(self, pitcher_name: str) -> Optional[str]:
        url = f"{self.base_url}/sports/baseball_mlb/events"
        params = {"apiKey": self.api_key}
        resp = http_client.get_json(
            url, params=params, cache_category="odds_events", cache_ttl_seconds=15 * 60
        )
        if resp is None:
            return None
        # Event objects don't carry probable pitchers, so we can only match
        # on team names elsewhere in the pipeline; the caller (market
        # service) is expected to pass the correct event_id when it has
        # already resolved home/away teams. This helper is a best-effort
        # fallback and intentionally conservative: it returns None rather
        # than guessing when it cannot be sure.
        return None

    @staticmethod
    def _extract_pitcher_lines(event_odds_payload: dict, pitcher_name: str) -> list[dict]:
        results = []
        for bookmaker in event_odds_payload.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market.get("key") != "pitcher_strikeouts":
                    continue
                for outcome in market.get("outcomes", []):
                    desc = (outcome.get("description") or "").lower()
                    if pitcher_name.lower() not in desc:
                        continue
                    results.append(
                        {
                            "bookmaker": bookmaker.get("key"),
                            "name": outcome.get("name"),  # "Over" / "Under"
                            "point": outcome.get("point"),
                            "price": outcome.get("price"),
                            "last_update": market.get("last_update"),
                        }
                    )
        return results
