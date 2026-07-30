"""
Transforms raw MLB Stats API hitting stat blocks into a shrinkage-adjusted
BatterProfile, including 7/14/30-day recent-form K rates computed from the
gameLog split (summed manually since the API does not expose rolling
windows directly).
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.features.league_constants import get_league_average
from app.features.shrinkage import shrink_named
from app.schemas.player import BatterProfile, SampleStat
from app.utilities.http_client import http_client

logger = get_logger(__name__)


def _find_stat_block(stats: list[dict], stat_type: str, group: str) -> Optional[dict]:
    for block in stats:
        btype = (block.get("type", {}) or {}).get("displayName", "").lower()
        bgroup = (block.get("group", {}) or {}).get("displayName", "").lower()
        if stat_type.lower() in btype and group.lower() in bgroup:
            splits = block.get("splits", [])
            if splits:
                return splits[0].get("stat", {})
    return None


def _find_gamelog_splits(stats: list[dict]) -> list[dict]:
    for block in stats:
        btype = (block.get("type", {}) or {}).get("displayName", "").lower()
        if "gamelog" in btype:
            return block.get("splits", [])
    return []


def _sample_stat_generic(observed_events, observed_n, prior_value, shrink_key) -> SampleStat:
    if observed_events is None or observed_n is None or observed_n <= 0:
        return SampleStat(observed_rate=None, observed_n=observed_n, shrunk_rate=None, reliability=0.0, is_small_sample=True)
    rate = observed_events / observed_n
    result = shrink_named(rate, observed_n, prior_value, shrink_key)
    return SampleStat(
        observed_rate=result.observed_rate,
        observed_n=result.observed_n,
        shrunk_rate=result.shrunk_rate,
        reliability=result.reliability,
        is_small_sample=result.is_small_sample,
    )


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


class BatterFeatureBuilder:
    def __init__(self, provider: Optional[MlbStatsApiProvider] = None):
        self.provider = provider or MlbStatsApiProvider()

    def build(
        self,
        batter_id: int,
        season: int,
        batting_order: Optional[int] = None,
        pitcher_hand_today: Optional[str] = None,
        as_of_date: Optional[dt.date] = None,
    ) -> BatterProfile:
        missing: list[str] = []
        as_of_date = as_of_date or dt.date.today()

        result = self.provider.get_batter_stats(batter_id, season)
        person = result.data.get("person") or {}
        stats = result.data.get("stats") or []

        name = person.get("fullName", f"Batter {batter_id}")
        bat_side = (person.get("batSide", {}) or {}).get("code")
        is_switch = bat_side == "S"

        expected_side = bat_side
        if is_switch and pitcher_hand_today:
            # Switch hitters bat from the side opposite the pitcher's throwing hand.
            expected_side = "L" if pitcher_hand_today == "R" else "R"
        elif is_switch:
            missing.append("pitcher_hand_for_switch_hitter_resolution")

        season_block = _find_stat_block(stats, "season", "hitting") or {}
        career_block = _find_stat_block(stats, "career", "hitting") or {}

        season_pa = _to_int(season_block.get("plateAppearances"))
        season_so = _to_int(season_block.get("strikeOuts"))
        season_bb = _to_int(season_block.get("baseOnBalls"))
        career_pa = _to_int(career_block.get("plateAppearances"))

        if season_pa is None:
            missing.append("season_plateAppearances")
        if season_so is None:
            missing.append("season_strikeOuts")

        k_rate_overall = _sample_stat_generic(
            season_so, season_pa, get_league_average("league_k_rate"), "batter_k_rate_overall"
        )
        bb_rate = _sample_stat_generic(
            season_bb, season_pa, get_league_average("league_bb_rate"), "batter_bb_rate"
        )

        splits = self._get_handedness_splits(batter_id, season)
        if splits is None:
            missing.append("handedness_splits")

        k_vs_r = k_vs_l = None
        pa_vs_r = pa_vs_l = None
        if splits:
            vr = splits.get("vs_rhp", {})
            vl = splits.get("vs_lhp", {})
            pa_vs_r = _to_int(vr.get("plateAppearances"))
            pa_vs_l = _to_int(vl.get("plateAppearances"))
            prior_r = get_league_average("league_k_rate_vs_rhp")
            prior_l = get_league_average("league_k_rate_vs_lhp")
            k_vs_r = _sample_stat_generic(_to_int(vr.get("strikeOuts")), pa_vs_r, prior_r, "batter_k_rate_split")
            k_vs_l = _sample_stat_generic(_to_int(vl.get("strikeOuts")), pa_vs_l, prior_l, "batter_k_rate_split")

        gamelog = _find_gamelog_splits(stats)
        k7, k14, k30 = self._recent_k_rates(gamelog, as_of_date)
        if k7 is None and k14 is None and k30 is None:
            missing.append("recent_gamelog_form")

        total_expected_fields = 8
        completeness = max(0.0, 1.0 - (len(missing) / total_expected_fields))

        return BatterProfile(
            player_id=batter_id,
            name=name,
            batting_order=batting_order,
            bat_side=bat_side,
            is_switch_hitter=is_switch,
            expected_side_today=expected_side,
            season_pa=season_pa,
            career_pa=career_pa,
            k_rate_overall=k_rate_overall,
            k_rate_vs_rhp=k_vs_r,
            k_rate_vs_lhp=k_vs_l,
            pa_vs_rhp=pa_vs_r,
            pa_vs_lhp=pa_vs_l,
            k_rate_last_7d=k7,
            k_rate_last_14d=k14,
            k_rate_last_30d=k30,
            bb_rate=bb_rate,
            data_completeness=completeness,
            missing_fields=missing,
        )

    def _get_handedness_splits(self, batter_id: int, season: int) -> Optional[dict]:
        url = f"{self.provider.base}/people/{batter_id}/stats"
        params = {"stats": "statSplits", "group": "hitting", "season": season, "sitCodes": "vr,vl"}
        resp = http_client.get_json(
            url, params=params, cache_category="batter_splits",
            cache_ttl_seconds=settings.cache_ttl_player_stats_hours * 3600,
        )
        if resp is None:
            return None
        stats = resp.json_body.get("stats", [])
        out = {}
        for block in stats:
            for split in block.get("splits", []):
                code = (split.get("split", {}) or {}).get("code")
                stat = split.get("stat", {})
                if code == "vr":
                    out["vs_rhp"] = stat
                elif code == "vl":
                    out["vs_lhp"] = stat
        return out or None

    @staticmethod
    def _recent_k_rates(gamelog_splits: list[dict], as_of_date: dt.date):
        windows = {7: [0, 0], 14: [0, 0], 30: [0, 0]}  # [strikeouts, PA]
        for split in gamelog_splits:
            game_date_str = (split.get("date"))
            if not game_date_str:
                continue
            try:
                game_date = dt.date.fromisoformat(game_date_str)
            except ValueError:
                continue
            days_ago = (as_of_date - game_date).days
            if days_ago < 0:
                continue
            stat = split.get("stat", {})
            so = _to_int(stat.get("strikeOuts")) or 0
            pa = _to_int(stat.get("plateAppearances")) or 0
            for w in windows:
                if days_ago <= w:
                    windows[w][0] += so
                    windows[w][1] += pa

        def _rate(w):
            so, pa = windows[w]
            return round(so / pa, 3) if pa > 0 else None

        return _rate(7), _rate(14), _rate(30)
