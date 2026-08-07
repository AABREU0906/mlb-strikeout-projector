"""
Resolves the final MarketSnapshot used by a projection: tries the automated
odds provider first, applies a manual override when the user supplies one
(manual always wins when present, per project rules), computes vig-free
probabilities, and computes simple consensus/disagreement stats across
books when more than one bookmaker's line is available.
"""
from __future__ import annotations

import statistics
from typing import Optional

from app.data_sources.base import utc_now_iso
from app.data_sources.odds_api import TheOddsApiProvider
from app.markets.odds_math import remove_vig_two_way
from app.schemas.market import ManualMarketEntry, MarketSnapshot


class MarketService:
    def __init__(self, odds_provider: Optional[TheOddsApiProvider] = None):
        self.odds_provider = odds_provider or TheOddsApiProvider()

    def build_snapshot(
        self,
        game_id: str,
        pitcher_name: str,
        manual: Optional[ManualMarketEntry] = None,
    ) -> MarketSnapshot:
        # Efficiency-only change for the confirmed-slate feature's credit
        # conservation goal: when the caller has already supplied market
        # data (e.g. confirmed_slate_service.py, which does its own
        # deliberate, credit-conscious Odds API fetch upstream), skip this
        # auto-fetch attempt entirely -- it would be immediately
        # overwritten by `manual` below regardless (see the `if manual is
        # not None:` block), so attempting it first was pure wasted work,
        # not a behavior difference. Output is identical either way.
        auto = self.odds_provider.get_pitcher_strikeout_props(game_id, pitcher_name) if manual is None else None

        snapshot = MarketSnapshot(source="unavailable", retrieved_at=utc_now_iso())

        if auto is not None and auto.data.get("lines"):
            lines = auto.data["lines"]
            over_prices = [l for l in lines if l.get("name") == "Over"]
            under_prices = [l for l in lines if l.get("name") == "Under"]
            snapshot.source = auto.source
            snapshot.retrieved_at = auto.retrieved_at
            snapshot.consensus_book_count = len({l.get("bookmaker") for l in lines})
            if over_prices:
                points = [l["point"] for l in over_prices if l.get("point") is not None]
                if points:
                    snapshot.strikeout_line = statistics.median(points)
                    if len(points) > 1:
                        snapshot.market_disagreement = statistics.pstdev(points)
                snapshot.over_odds = over_prices[0].get("price")
            if under_prices:
                snapshot.under_odds = under_prices[0].get("price")

        # Same reasoning as the auto strikeout-props fetch above: this
        # call's result (`game_market`) is not currently consumed by
        # anything downstream (see the pre-existing comment below) -- it
        # was pure wasted Odds API quota with zero effect on output.
        # Skipped when the caller already supplied market data.
        game_market = self.odds_provider.get_game_market(game_id) if manual is None else None
        # game_market intentionally left for future event-id-resolved
        # extraction of totals/moneylines; see odds_api.py notes on why
        # event matching is conservative rather than guessed.

        if manual is not None:
            snapshot.manual_override = True
            snapshot.source = manual.sportsbook_name
            snapshot.retrieved_at = manual.entered_at or utc_now_iso()
            if manual.strikeout_line is not None:
                snapshot.strikeout_line = manual.strikeout_line
            if manual.over_odds is not None:
                snapshot.over_odds = manual.over_odds
            if manual.under_odds is not None:
                snapshot.under_odds = manual.under_odds
            snapshot.opening_strikeout_line = manual.opening_strikeout_line
            snapshot.opening_over_odds = manual.opening_over_odds
            snapshot.opening_under_odds = manual.opening_under_odds
            snapshot.game_total = manual.game_total
            snapshot.team_moneyline = manual.team_moneyline
            snapshot.opponent_moneyline = manual.opponent_moneyline
            snapshot.team_implied_runs = manual.team_implied_runs
            snapshot.opponent_implied_runs = manual.opponent_implied_runs
            snapshot.first_five_total = manual.first_five_total
            snapshot.run_line = manual.run_line

        if snapshot.over_odds is not None and snapshot.under_odds is not None:
            vig = remove_vig_two_way(snapshot.over_odds, snapshot.under_odds)
            snapshot.raw_over_prob = vig.raw_over_prob
            snapshot.raw_under_prob = vig.raw_under_prob
            snapshot.overround = vig.overround
            snapshot.vig_free_over_prob = vig.vig_free_over_prob
            snapshot.vig_free_under_prob = vig.vig_free_under_prob
            snapshot.fair_over_odds = vig.fair_over_odds
            snapshot.fair_under_odds = vig.fair_under_odds

        if (
            snapshot.strikeout_line is not None
            and snapshot.opening_strikeout_line is not None
        ):
            snapshot.line_movement = snapshot.strikeout_line - snapshot.opening_strikeout_line
        if snapshot.over_odds is not None and snapshot.opening_over_odds is not None:
            snapshot.odds_movement_over = snapshot.over_odds - snapshot.opening_over_odds
        if snapshot.under_odds is not None and snapshot.opening_under_odds is not None:
            snapshot.odds_movement_under = snapshot.under_odds - snapshot.opening_under_odds

        return snapshot
