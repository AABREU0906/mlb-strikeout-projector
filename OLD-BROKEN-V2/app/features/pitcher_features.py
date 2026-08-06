"""
Transforms raw MLB Stats API pitching stat blocks into a shrinkage-adjusted
PitcherProfile.

Note on field mapping: the MLB Stats API's exact JSON shape for split stats
(vs RHB/LHB) depends on which `sitCodes` are requested alongside the season
stat group; this module requests them separately via
`get_pitcher_handedness_splits` and defensively handles missing keys rather
than assuming a fixed shape, since undocumented API responses can drift.
Any field that can't be found is left as None and added to
`missing_fields` -- it is never fabricated or defaulted silently.
"""
from __future__ import annotations

from typing import Optional

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.features.league_constants import get_league_average
from app.features.shrinkage import shrink_named
from app.schemas.player import PitcherProfile, SampleStat
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


def _sample_stat(observed_events: Optional[float], observed_n: Optional[float], prior_key: str) -> SampleStat:
    if observed_events is None or observed_n is None or observed_n <= 0:
        return SampleStat(observed_rate=None, observed_n=observed_n, shrunk_rate=None, reliability=0.0, is_small_sample=True)
    rate = observed_events / observed_n
    prior = get_league_average(
        {
            "batter_k_rate_overall": "league_k_rate",
            "batter_k_rate_split": "league_k_rate",
            "pitcher_k_rate_overall": "league_k_rate",
            "pitcher_k_rate_split": "league_k_rate",
            "batter_bb_rate": "league_bb_rate",
            "pitcher_bb_rate": "league_bb_rate",
            "contact_rate": "league_contact_rate",
            "chase_rate": "league_chase_rate",
            "swstr_rate": "league_swstr_rate",
        }[prior_key]
    )
    result = shrink_named(rate, observed_n, prior, prior_key)
    return SampleStat(
        observed_rate=result.observed_rate,
        observed_n=result.observed_n,
        shrunk_rate=result.shrunk_rate,
        reliability=result.reliability,
        is_small_sample=result.is_small_sample,
    )


class PitcherFeatureBuilder:
    def __init__(self, provider: Optional[MlbStatsApiProvider] = None):
        self.provider = provider or MlbStatsApiProvider()

    def build(self, pitcher_id: int, season: int) -> PitcherProfile:
        missing: list[str] = []

        result = self.provider.get_pitcher_stats(pitcher_id, season)
        person = result.data.get("person") or {}
        stats = result.data.get("stats") or []

        name = person.get("fullName", f"Pitcher {pitcher_id}")
        throws = (person.get("pitchHand", {}) or {}).get("code")

        season_block = _find_stat_block(stats, "season", "pitching") or {}
        career_block = _find_stat_block(stats, "career", "pitching") or {}

        season_bf = _to_int(season_block.get("battersFaced"))
        season_so = _to_int(season_block.get("strikeOuts"))
        season_bb = _to_int(season_block.get("baseOnBalls"))
        starts = _to_int(season_block.get("gamesStarted")) or 0
        innings = _parse_innings(season_block.get("inningsPitched"))
        pitches = _to_int(season_block.get("numberOfPitches"))

        if season_bf is None:
            missing.append("season_battersFaced")
        if season_so is None:
            missing.append("season_strikeOuts")

        k_rate_season = _sample_stat(season_so, season_bf, "pitcher_k_rate_overall")

        career_bf = _to_int(career_block.get("battersFaced"))
        career_so = _to_int(career_block.get("strikeOuts"))
        k_rate_career = _sample_stat(career_so, career_bf, "pitcher_k_rate_overall")

        splits = self._get_handedness_splits(pitcher_id, season)
        if splits is None:
            missing.append("handedness_splits")

        k_vs_r = k_vs_l = bb_vs_r = bb_vs_l = None
        bf_vs_r = bf_vs_l = None
        if splits:
            vr = splits.get("vs_rhb", {})
            vl = splits.get("vs_lhb", {})
            bf_vs_r = _to_int(vr.get("battersFaced"))
            bf_vs_l = _to_int(vl.get("battersFaced"))
            k_vs_r = _sample_stat(_to_int(vr.get("strikeOuts")), bf_vs_r, "pitcher_k_rate_split")
            k_vs_l = _sample_stat(_to_int(vl.get("strikeOuts")), bf_vs_l, "pitcher_k_rate_split")
            bb_vs_r = _sample_stat(_to_int(vr.get("baseOnBalls")), bf_vs_r, "pitcher_bb_rate")
            bb_vs_l = _sample_stat(_to_int(vl.get("baseOnBalls")), bf_vs_l, "pitcher_bb_rate")

        bb_rate_season = _sample_stat(season_bb, season_bf, "pitcher_bb_rate")

        avg_ip_per_start = round(innings / starts, 2) if innings and starts else None
        avg_bf_per_start = round(season_bf / starts, 1) if season_bf and starts else None
        avg_pitches_per_start = round(pitches / starts, 1) if pitches and starts else None

        total_expected_fields = 10
        completeness = max(0.0, 1.0 - (len(missing) / total_expected_fields))

        return PitcherProfile(
            player_id=pitcher_id,
            name=name,
            throws=throws,
            season_bf=season_bf,
            career_bf=career_bf,
            k_rate_season=k_rate_season,
            k_rate_career=k_rate_career,
            k_per_9=_safe_ratio(season_so, innings, 9) if innings else None,
            k_per_bf=(season_so / season_bf) if (season_so and season_bf) else None,
            k_per_start=(season_so / starts) if (season_so and starts) else None,
            k_rate_vs_rhb=k_vs_r,
            k_rate_vs_lhb=k_vs_l,
            bb_rate_vs_rhb=bb_vs_r,
            bb_rate_vs_lhb=bb_vs_l,
            bf_vs_rhb=bf_vs_r,
            bf_vs_lhb=bf_vs_l,
            bb_rate_season=bb_rate_season,
            avg_innings_per_start=avg_ip_per_start,
            avg_bf_per_start=avg_bf_per_start,
            avg_pitches_per_start=avg_pitches_per_start,
            data_completeness=completeness,
            missing_fields=missing,
        )

    def _get_handedness_splits(self, pitcher_id: int, season: int) -> Optional[dict]:
        """Requests sitCodes-based splits directly. Returns None (not a
        fabricated split) if the endpoint doesn't return usable data."""
        url = f"{self.provider.base}/people/{pitcher_id}/stats"
        params = {
            "stats": "statSplits",
            "group": "pitching",
            "season": season,
            "sitCodes": "vr,vl",
        }
        resp = http_client.get_json(
            url, params=params, cache_category="pitcher_splits",
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
                    out["vs_rhb"] = stat
                elif code == "vl":
                    out["vs_lhb"] = stat
        return out or None


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _parse_innings(ip_str: Optional[str]) -> Optional[float]:
    """MLB represents partial innings as e.g. '123.1' meaning 123 1/3 innings."""
    if ip_str is None:
        return None
    try:
        whole, _, frac = str(ip_str).partition(".")
        whole_i = int(whole)
        frac_i = int(frac) if frac else 0
        return whole_i + (frac_i / 3.0)
    except (ValueError, TypeError):
        return None


def _safe_ratio(numerator, denominator, multiplier=1) -> Optional[float]:
    if numerator is None or not denominator:
        return None
    return round((numerator / denominator) * multiplier, 2)
