"""
Provider interfaces.

Every concrete data source implements one of these ABCs. The rest of the
application depends only on these interfaces (dependency inversion), so a
provider can be replaced (e.g. swapping odds vendors) without touching the
projection engine, CLI, or database code.

Every return type carries `source` and `retrieved_at` so provenance is never
lost between fetch and storage.
"""
from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SourcedPayload(BaseModel):
    """Every provider result carries its data plus provenance."""

    source: str
    retrieved_at: str = Field(default_factory=utc_now_iso)
    from_cache: bool = False
    data: dict[str, Any] = Field(default_factory=dict)

    def _replace_data(self, data: dict[str, Any]) -> "SourcedPayload":
        return self.model_copy(update={"data": data})


class ScheduleProvider(ABC):
    @abstractmethod
    def get_schedule(self, game_date: str) -> SourcedPayload:
        """Return games scheduled for game_date (YYYY-MM-DD)."""


class LineupProvider(ABC):
    @abstractmethod
    def get_confirmed_lineup(self, game_id: str, team_id: int) -> Optional[SourcedPayload]:
        """Return the confirmed lineup if the official lineup has posted, else None."""

    @abstractmethod
    def get_projected_lineup(self, team_id: int, vs_pitcher_hand: Optional[str]) -> SourcedPayload:
        """Return a best-effort projected lineup (e.g. team's most recent
        actual lineup) when the confirmed lineup is not yet available."""


class PlayerStatsProvider(ABC):
    @abstractmethod
    def get_pitcher_stats(self, pitcher_id: int, season: int) -> SourcedPayload: ...

    @abstractmethod
    def get_batter_stats(self, batter_id: int, season: int) -> SourcedPayload: ...


class WeatherProvider(ABC):
    @abstractmethod
    def get_game_weather(
        self, latitude: float, longitude: float, game_datetime_utc: str
    ) -> SourcedPayload: ...


class OddsProvider(ABC):
    @abstractmethod
    def get_pitcher_strikeout_props(self, game_id: str, pitcher_name: str) -> Optional[SourcedPayload]:
        """Return strikeout prop line/odds if available from the configured
        odds vendor, else None (caller falls back to manual entry)."""

    @abstractmethod
    def get_game_market(self, game_id: str) -> Optional[SourcedPayload]:
        """Moneyline / totals / run line for the game."""


class NewsProvider(ABC):
    @abstractmethod
    def search_player_news(self, player_name: str, since_days: int = 7) -> SourcedPayload: ...
