from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ManualMarketEntry(BaseModel):
    """User-entered sportsbook data. Any field left as None is simply
    unavailable for this projection -- never inferred or defaulted."""

    strikeout_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    opening_strikeout_line: Optional[float] = None
    opening_over_odds: Optional[int] = None
    opening_under_odds: Optional[int] = None

    game_total: Optional[float] = None
    team_moneyline: Optional[int] = None
    opponent_moneyline: Optional[int] = None
    team_implied_runs: Optional[float] = None
    opponent_implied_runs: Optional[float] = None
    first_five_total: Optional[float] = None
    run_line: Optional[float] = None

    sportsbook_name: str = "manual_entry"
    entered_at: Optional[str] = None


class MarketSnapshot(BaseModel):
    """The resolved, as-used market picture for a single projection --
    combining automated feed data with any manual override, plus vig-free
    math already applied. This whole object is stored verbatim on the
    Projection row for full reproducibility."""

    source: str
    retrieved_at: str
    manual_override: bool = False

    strikeout_line: Optional[float] = None
    over_odds: Optional[int] = None
    under_odds: Optional[int] = None
    opening_strikeout_line: Optional[float] = None
    opening_over_odds: Optional[int] = None
    opening_under_odds: Optional[int] = None
    line_movement: Optional[float] = None
    odds_movement_over: Optional[int] = None
    odds_movement_under: Optional[int] = None

    raw_over_prob: Optional[float] = None
    raw_under_prob: Optional[float] = None
    overround: Optional[float] = None
    vig_free_over_prob: Optional[float] = None
    vig_free_under_prob: Optional[float] = None
    fair_over_odds: Optional[int] = None
    fair_under_odds: Optional[int] = None

    game_total: Optional[float] = None
    team_moneyline: Optional[int] = None
    opponent_moneyline: Optional[int] = None
    team_implied_runs: Optional[float] = None
    opponent_implied_runs: Optional[float] = None
    first_five_total: Optional[float] = None
    run_line: Optional[float] = None

    consensus_book_count: int = 0
    market_disagreement: Optional[float] = None  # stdev of line across books, if multiple
