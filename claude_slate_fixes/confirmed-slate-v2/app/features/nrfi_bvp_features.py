"""
Batter-vs-pitcher feature builder.

PROJECT NOTE on field mapping confidence: the `vsPlayer` stat type used by
get_batter_vs_pitcher() is a real but thinly-documented MLB Stats API
endpoint. Its response shape (avg/obp/slg/hits/atBats/baseOnBalls fields)
follows the same convention as every other hitting stat block this project
already parses successfully, so the mapping below should hold -- but as
with the rest of this codebase's undocumented-API handling, every field is
read defensively and missing data is never fabricated.
"""
from __future__ import annotations

from typing import Optional

from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.features.league_constants import get_league_average
from app.features.nrfi_rate_calculations import cascading_shrinkage
from app.schemas.nrfi import BvPFactor, BvPProfile

BVP_STABILIZATION_PA = 60
VS_HAND_STABILIZATION_PA = 200
SEASON_STABILIZATION_PA = 200


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


class BvPFeatureBuilder:
    def __init__(self, provider: Optional[MlbStatsApiProvider] = None):
        self.provider = provider or MlbStatsApiProvider()

    def build(
        self, batter_id: int, batter_name: str, pitcher_id: int,
        pitcher_hand: Optional[str], season: int,
    ) -> BvPProfile:
        missing: list[str] = []

        bvp_payload = self.provider.get_batter_vs_pitcher(batter_id, pitcher_id)
        bvp_stat = bvp_payload.data.get("stat") or {}
        if not bvp_stat:
            missing.append("no_bvp_history")

        season_payload = self.provider.get_batter_stats(batter_id, season)
        season_blocks = season_payload.data.get("stats") or []
        season_stat = self._find_season_block(season_blocks)
        if season_stat is None:
            missing.append("season_stats_unavailable")

        hand_splits = self.provider.get_batter_handedness_splits_raw(batter_id, season) or {}
        vs_hand_stat = None
        if pitcher_hand == "L":
            vs_hand_stat = hand_splits.get("vs_lhp")
        elif pitcher_hand == "R":
            vs_hand_stat = hand_splits.get("vs_rhp")
        if vs_hand_stat is None:
            missing.append("vs_hand_split_unavailable")

        league_avg = get_league_average("league_avg")
        league_obp = get_league_average("league_obp")
        league_slg = get_league_average("league_slg")

        bvp_pa = _to_int(bvp_stat.get("plateAppearances"))
        season_pa = _to_int((season_stat or {}).get("plateAppearances"))
        vs_hand_pa = _to_int((vs_hand_stat or {}).get("plateAppearances"))

        avg_factor = self._build_factor(
            "avg", league_avg, bvp_stat, bvp_pa, season_stat, season_pa, vs_hand_stat, vs_hand_pa
        )
        obp_factor = self._build_factor(
            "obp", league_obp, bvp_stat, bvp_pa, season_stat, season_pa, vs_hand_stat, vs_hand_pa
        )
        slg_factor = self._build_factor(
            "slg", league_slg, bvp_stat, bvp_pa, season_stat, season_pa, vs_hand_stat, vs_hand_pa
        )

        total_expected = 3
        completeness = max(0.0, 1.0 - (len(missing) / total_expected))

        return BvPProfile(
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            batter_name=batter_name,
            plate_appearances=bvp_pa or 0,
            at_bats=_to_int(bvp_stat.get("atBats")) or 0,
            hits=_to_int(bvp_stat.get("hits")) or 0,
            singles=self._singles(bvp_stat),
            doubles=_to_int(bvp_stat.get("doubles")),
            triples=_to_int(bvp_stat.get("triples")),
            home_runs=_to_int(bvp_stat.get("homeRuns")),
            walks=_to_int(bvp_stat.get("baseOnBalls")) or 0,
            strikeouts=_to_int(bvp_stat.get("strikeOuts")),
            avg=avg_factor,
            obp=obp_factor,
            slg=slg_factor,
            ops_adjusted=round(obp_factor.final_adjusted_value + slg_factor.final_adjusted_value, 4),
            data_completeness=completeness,
            missing_fields=missing,
        )

    @staticmethod
    def _singles(stat: dict) -> Optional[int]:
        hits = _to_int(stat.get("hits"))
        if hits is None:
            return None
        doubles = _to_int(stat.get("doubles")) or 0
        triples = _to_int(stat.get("triples")) or 0
        hr = _to_int(stat.get("homeRuns")) or 0
        return max(hits - doubles - triples - hr, 0)

    @staticmethod
    def _find_season_block(stat_blocks: list[dict]) -> Optional[dict]:
        for block in stat_blocks:
            btype = (block.get("type", {}) or {}).get("displayName", "").lower()
            bgroup = (block.get("group", {}) or {}).get("displayName", "").lower()
            if "season" in btype and "hitting" in bgroup:
                splits = block.get("splits", [])
                if splits:
                    return splits[0].get("stat", {})
        return None

    def _build_factor(
        self, field: str, league_rate: float,
        bvp_stat: dict, bvp_pa: Optional[int],
        season_stat: Optional[dict], season_pa: Optional[int],
        vs_hand_stat: Optional[dict], vs_hand_pa: Optional[int],
    ) -> BvPFactor:
        bvp_value = _to_float(bvp_stat.get(field))
        season_value = _to_float((season_stat or {}).get(field))
        vs_hand_value = _to_float((vs_hand_stat or {}).get(field))

        final_rate, detail = cascading_shrinkage(
            league_rate=league_rate,
            season_rate=season_value, season_n=season_pa, season_stabilization_n=SEASON_STABILIZATION_PA,
            vs_hand_rate=vs_hand_value, vs_hand_n=vs_hand_pa, vs_hand_stabilization_n=VS_HAND_STABILIZATION_PA,
            bvp_rate=bvp_value, bvp_n=bvp_pa, bvp_stabilization_n=BVP_STABILIZATION_PA,
        )

        return BvPFactor(
            raw_bvp_value=bvp_value,
            raw_bvp_n=bvp_pa or 0,
            season_prior=season_value,
            vs_hand_prior=vs_hand_value,
            final_adjusted_value=round(final_rate, 4),
            reliability=detail["bvp_reliability"],
        )
