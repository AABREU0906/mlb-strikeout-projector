"""
Transforms raw MLB Stats API pitching stat blocks into a shrinkage-adjusted
PitcherProfile.

Note on field mapping: the MLB Stats API's exact JSON shape for split stats
(vs RHB/LHB) depends on which `sitCodes` are requested alongside the season
stat group; this module requests them separately via
`get_pitcher_handedness_splits` and defensively handles missing keys rather
than assuming a fixed shape, since undocumented API responses can drift.

Any field that cannot be found is left as None and added to
`missing_fields`. It is never fabricated or defaulted silently.
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


def _find_stat_block(
    stats: list[dict],
    stat_type: str,
    group: str,
) -> Optional[dict]:
    for block in stats:
        block_type = (
            block.get("type", {}) or {}
        ).get("displayName", "").lower()

        block_group = (
            block.get("group", {}) or {}
        ).get("displayName", "").lower()

        if (
            stat_type.lower() in block_type
            and group.lower() in block_group
        ):
            splits = block.get("splits", [])

            if splits:
                return splits[0].get("stat", {})

    return None


def _sample_stat(
    observed_events: Optional[float],
    observed_n: Optional[float],
    prior_key: str,
) -> SampleStat:
    if (
        observed_events is None
        or observed_n is None
        or observed_n <= 0
    ):
        return SampleStat(
            observed_rate=None,
            observed_n=observed_n,
            shrunk_rate=None,
            reliability=0.0,
            is_small_sample=True,
        )

    rate = observed_events / observed_n

    prior_mapping = {
        "batter_k_rate_overall": "league_k_rate",
        "batter_k_rate_split": "league_k_rate",
        "pitcher_k_rate_overall": "league_k_rate",
        "pitcher_k_rate_split": "league_k_rate",
        "batter_bb_rate": "league_bb_rate",
        "pitcher_bb_rate": "league_bb_rate",
        "contact_rate": "league_contact_rate",
        "chase_rate": "league_chase_rate",
        "swstr_rate": "league_swstr_rate",
    }

    prior = get_league_average(
        prior_mapping[prior_key]
    )

    result = shrink_named(
        rate,
        observed_n,
        prior,
        prior_key,
    )

    return SampleStat(
        observed_rate=result.observed_rate,
        observed_n=result.observed_n,
        shrunk_rate=result.shrunk_rate,
        reliability=result.reliability,
        is_small_sample=result.is_small_sample,
    )


class PitcherFeatureBuilder:
    def __init__(
        self,
        provider: Optional[MlbStatsApiProvider] = None,
    ) -> None:
        self.provider = provider or MlbStatsApiProvider()

    def build(
        self,
        pitcher_id: int,
        season: int,
    ) -> PitcherProfile:
        missing: list[str] = []

        result = self.provider.get_pitcher_stats(
            pitcher_id,
            season,
        )

        person = result.data.get("person") or {}
        stats = result.data.get("stats") or []

        name = person.get(
            "fullName",
            f"Pitcher {pitcher_id}",
        )

        throws = (
            person.get("pitchHand", {}) or {}
        ).get("code")

        season_block = (
            _find_stat_block(
                stats,
                "season",
                "pitching",
            )
            or {}
        )

        career_block = (
            _find_stat_block(
                stats,
                "career",
                "pitching",
            )
            or {}
        )

        season_bf = _to_int(
            season_block.get("battersFaced")
        )

        season_so = _to_int(
            season_block.get("strikeOuts")
        )

        season_bb = _to_int(
            season_block.get("baseOnBalls")
        )

        starts = (
            _to_int(
                season_block.get("gamesStarted")
            )
            or 0
        )

        innings = _parse_innings(
            season_block.get("inningsPitched")
        )

        pitches = _to_int(
            season_block.get("numberOfPitches")
        )

        if season_bf is None:
            missing.append("season_battersFaced")

        if season_so is None:
            missing.append("season_strikeOuts")

        if starts <= 0:
            missing.append("season_gamesStarted")

        if innings is None:
            missing.append("season_inningsPitched")

        if pitches is None:
            missing.append("season_numberOfPitches")

        k_rate_season = _sample_stat(
            season_so,
            season_bf,
            "pitcher_k_rate_overall",
        )

        career_bf = _to_int(
            career_block.get("battersFaced")
        )

        career_so = _to_int(
            career_block.get("strikeOuts")
        )

        k_rate_career = _sample_stat(
            career_so,
            career_bf,
            "pitcher_k_rate_overall",
        )

        splits = self._get_handedness_splits(
            pitcher_id,
            season,
        )

        if splits is None:
            missing.append("handedness_splits")

        k_vs_r: Optional[SampleStat] = None
        k_vs_l: Optional[SampleStat] = None
        bb_vs_r: Optional[SampleStat] = None
        bb_vs_l: Optional[SampleStat] = None
        bf_vs_r: Optional[int] = None
        bf_vs_l: Optional[int] = None

        if splits:
            versus_right = splits.get(
                "vs_rhb",
                {},
            )

            versus_left = splits.get(
                "vs_lhb",
                {},
            )

            bf_vs_r = _to_int(
                versus_right.get("battersFaced")
            )

            bf_vs_l = _to_int(
                versus_left.get("battersFaced")
            )

            k_vs_r = _sample_stat(
                _to_int(
                    versus_right.get("strikeOuts")
                ),
                bf_vs_r,
                "pitcher_k_rate_split",
            )

            k_vs_l = _sample_stat(
                _to_int(
                    versus_left.get("strikeOuts")
                ),
                bf_vs_l,
                "pitcher_k_rate_split",
            )

            bb_vs_r = _sample_stat(
                _to_int(
                    versus_right.get("baseOnBalls")
                ),
                bf_vs_r,
                "pitcher_bb_rate",
            )

            bb_vs_l = _sample_stat(
                _to_int(
                    versus_left.get("baseOnBalls")
                ),
                bf_vs_l,
                "pitcher_bb_rate",
            )

        bb_rate_season = _sample_stat(
            season_bb,
            season_bf,
            "pitcher_bb_rate",
        )

        avg_ip_per_start = (
            round(innings / starts, 2)
            if innings is not None and starts > 0
            else None
        )

        avg_bf_per_start = (
            round(season_bf / starts, 1)
            if season_bf is not None and starts > 0
            else None
        )

        avg_pitches_per_start = (
            round(pitches / starts, 1)
            if pitches is not None and starts > 0
            else None
        )

        # Reject impossible per-start workload values.
        # This protects the workload model when season totals are
        # accidentally paired with an incorrect or incomplete starts count.
        if (
            avg_ip_per_start is not None
            and not 0.5 <= avg_ip_per_start <= 9.0
        ):
            logger.warning(
                "Invalid innings per start for pitcher %s: %s",
                pitcher_id,
                avg_ip_per_start,
            )
            missing.append(
                "invalid_avg_innings_per_start"
            )
            avg_ip_per_start = None

        if (
            avg_bf_per_start is not None
            and not 3.0 <= avg_bf_per_start <= 45.0
        ):
            logger.warning(
                "Invalid batters faced per start for pitcher %s: %s",
                pitcher_id,
                avg_bf_per_start,
            )
            missing.append(
                "invalid_avg_bf_per_start"
            )
            avg_bf_per_start = None

        if (
            avg_pitches_per_start is not None
            and not 10.0 <= avg_pitches_per_start <= 130.0
        ):
            logger.warning(
                "Invalid pitches per start for pitcher %s: %s",
                pitcher_id,
                avg_pitches_per_start,
            )
            missing.append(
                "invalid_avg_pitches_per_start"
            )
            avg_pitches_per_start = None

        total_expected_fields = 10

        completeness = max(
            0.0,
            1.0 - (
                len(set(missing))
                / total_expected_fields
            ),
        )

        return PitcherProfile(
            player_id=pitcher_id,
            name=name,
            throws=throws,
            season_bf=season_bf,
            career_bf=career_bf,
            k_rate_season=k_rate_season,
            k_rate_career=k_rate_career,
            k_per_9=(
                _safe_ratio(
                    season_so,
                    innings,
                    9,
                )
                if innings is not None
                else None
            ),
            k_per_bf=(
                season_so / season_bf
                if (
                    season_so is not None
                    and season_bf
                )
                else None
            ),
            k_per_start=(
                season_so / starts
                if (
                    season_so is not None
                    and starts > 0
                )
                else None
            ),
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
            missing_fields=list(dict.fromkeys(missing)),
        )

    def _get_handedness_splits(
        self,
        pitcher_id: int,
        season: int,
    ) -> Optional[dict]:
        """
        Request sitCodes-based splits directly.

        Returns None, rather than a fabricated split, if the endpoint does
        not return usable data.
        """
        url = (
            f"{self.provider.base}/people/"
            f"{pitcher_id}/stats"
        )

        params = {
            "stats": "statSplits",
            "group": "pitching",
            "season": season,
            "sitCodes": "vr,vl",
        }

        response = http_client.get_json(
            url,
            params=params,
            cache_category="pitcher_splits",
            cache_ttl_seconds=(
                settings.cache_ttl_player_stats_hours
                * 3600
            ),
        )

        if response is None:
            return None

        stats = response.json_body.get(
            "stats",
            [],
        )

        output: dict[str, dict] = {}

        for block in stats:
            for split in block.get(
                "splits",
                [],
            ):
                code = (
                    split.get("split", {}) or {}
                ).get("code")

                stat = split.get(
                    "stat",
                    {},
                )

                if code == "vr":
                    output["vs_rhb"] = stat
                elif code == "vl":
                    output["vs_lhb"] = stat

        return output or None


def _to_int(
    value: object,
) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_innings(
    innings_value: Optional[str],
) -> Optional[float]:
    """
    Convert MLB innings notation into decimal innings.

    MLB represents partial innings as:
        "123.1" = 123 and 1/3 innings
        "123.2" = 123 and 2/3 innings
    """
    if innings_value is None:
        return None

    try:
        whole, _, fraction = str(
            innings_value
        ).partition(".")

        whole_innings = int(whole)
        fractional_outs = (
            int(fraction)
            if fraction
            else 0
        )

        if fractional_outs not in {0, 1, 2}:
            return None

        return (
            whole_innings
            + fractional_outs / 3.0
        )

    except (ValueError, TypeError):
        return None


def _safe_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    multiplier: float = 1,
) -> Optional[float]:
    if (
        numerator is None
        or denominator is None
        or denominator == 0
    ):
        return None

    return round(
        (numerator / denominator)
        * multiplier,
        2,
    )