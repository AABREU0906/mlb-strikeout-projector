"""Team-level offensive stat builder (supporting context; lineup-level
batter data is weighted more heavily than these full-season team averages
whenever a confirmed lineup is available -- see projections/stage2_batter_probability.py)."""
from __future__ import annotations

from typing import Optional

from app.config.settings import settings
from app.data_sources.base import utc_now_iso
from app.schemas.player import TeamProfile
from app.utilities.http_client import http_client


class TeamFeatureBuilder:
    def __init__(self):
        self.base = settings.mlb_stats_api_base_url

    def build(self, team_id: int, team_name: str, season: int) -> TeamProfile:
        url = f"{self.base}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "hitting", "season": season}
        resp = http_client.get_json(
            url, params=params, cache_category="team_stats", cache_ttl_seconds=6 * 3600
        )
        if resp is None:
            return TeamProfile(team_id=team_id, team_name=team_name)

        stat = {}
        for block in resp.json_body.get("stats", []):
            splits = block.get("splits", [])
            if splits:
                stat = splits[0].get("stat", {})
                break

        pa = _to_float(stat.get("plateAppearances"))
        so = _to_float(stat.get("strikeOuts"))
        bb = _to_float(stat.get("baseOnBalls"))
        runs = _to_float(stat.get("runs"))
        games = _to_float(stat.get("gamesPlayed"))

        return TeamProfile(
            team_id=team_id,
            team_name=team_name,
            k_rate_overall=(so / pa) if (so and pa) else None,
            bb_rate=(bb / pa) if (bb and pa) else None,
            pa_per_game=(pa / games) if (pa and games) else None,
            runs_per_game=(runs / games) if (runs and games) else None,
        )


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
