"""
First-Inning Threat Score (0-100).

Per spec: "The Threat Score must support the probability model but must not
replace the calibrated probability." This is enforced structurally -- the
Threat Score is computed independently here and is never fed back into
compute_half_inning_scoring_probability(); the two are displayed side by
side so a person can see whether they agree or diverge, rather than the
Threat Score silently overriding the probability.

Method: each component is expressed as a z-score-like deviation from
league average, weighted per its documented importance, summed, and mapped
onto 0-100 via a bounded logistic transform (so extreme inputs can't
produce a nonsensical score below 0 or above 100).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from app.schemas.nrfi import BvPProfile

COMPONENT_WEIGHTS = {
    "obp_quality": 0.22,
    "slg_quality": 0.18,
    "k_rate_suppression": 0.12,
    "bb_rate": 0.10,
    "hr_rate": 0.13,
    "top_order_bvp": 0.15,
    "recent_form": 0.10,
}


@dataclass
class ThreatScoreResult:
    score: float
    components: dict = field(default_factory=dict)
    top_contributors: list[str] = field(default_factory=list)


def _z_like(value: Optional[float], league_avg: float, spread: float) -> float:
    if value is None:
        return 0.0
    z = (value - league_avg) / spread
    return max(min(z, 3.0), -3.0)


def compute_threat_score(
    team_slash_line,
    league_obp: float,
    league_slg: float,
    league_k_pct: float,
    league_bb_pct: float,
    league_hr_rate: float,
    top_order_bvp: Optional[list[BvPProfile]] = None,
    league_bvp_obp: float = 0.318,
    recent_form_rate: Optional[float] = None,
    season_form_rate: Optional[float] = None,
) -> ThreatScoreResult:
    components = {}

    components["obp_quality"] = _z_like(team_slash_line.obp, league_obp, league_obp * 0.15)
    components["slg_quality"] = _z_like(team_slash_line.slg, league_slg, league_slg * 0.18)
    components["k_rate_suppression"] = -_z_like(team_slash_line.k_pct, league_k_pct, league_k_pct * 0.20)
    components["bb_rate"] = _z_like(team_slash_line.bb_pct, league_bb_pct, league_bb_pct * 0.25)
    components["hr_rate"] = _z_like(team_slash_line.hr_rate, league_hr_rate, max(league_hr_rate * 0.5, 0.005))

    if top_order_bvp:
        obp_values = [b.obp.final_adjusted_value for b in top_order_bvp if b.obp is not None]
        components["top_order_bvp"] = _z_like(
            sum(obp_values) / len(obp_values) if obp_values else None, league_bvp_obp, league_bvp_obp * 0.15
        )
    else:
        components["top_order_bvp"] = 0.0

    if recent_form_rate is not None and season_form_rate is not None and season_form_rate > 0:
        form_dev = (recent_form_rate - season_form_rate) / max(season_form_rate, 0.05)
        components["recent_form"] = max(min(form_dev * 3.0, 3.0), -3.0)
    else:
        components["recent_form"] = 0.0

    weighted_sum = sum(components[k] * COMPONENT_WEIGHTS[k] for k in COMPONENT_WEIGHTS)
    score = 100.0 / (1.0 + math.exp(-1.1 * weighted_sum))

    ranked = sorted(components.items(), key=lambda kv: abs(kv[1]) * COMPONENT_WEIGHTS[kv[0]], reverse=True)
    top_contributors = [name for name, _ in ranked[:3] if abs(components[name]) > 0.15]

    return ThreatScoreResult(
        score=round(score, 1),
        components={k: round(v, 3) for k, v in components.items()},
        top_contributors=top_contributors,
    )
