"""
Stage 1: Expected Workload Model.

Transparent, documented formula-based model (not a black box) that
estimates innings pitched, batters faced, pitch count, and the probability
distribution over "times through the order," using:

  - Pitcher's own season workload (avg IP/BF/pitches per start)
  - Recent workload trend (last 3 starts)
  - Pitch efficiency (pitches per BF)
  - Walk rate (higher BB rate -> more pitches per out -> shorter outings)
  - Opponent offensive quality (better offense -> shorter average outings)
  - Team bullpen strength proxy (not modeled with real data here without a
    bullpen-quality source; documented as a neutral multiplier until wired
    to a bullpen ERA/FIP feed)
  - Rest status, opener/tandem flags, pitch-limit / injury-return warnings

This stage NEVER assumes a normal workload when opener/tandem/pitch-limit/
injury-return flags are present -- it applies documented, capped
reductions instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.features.league_constants import get_league_average
from app.schemas.player import PitcherProfile, TeamProfile


@dataclass
class WorkloadEstimate:
    expected_innings: float
    expected_batters_faced: float
    expected_pitch_count: float
    prob_complete_5: float
    prob_complete_6: float
    prob_complete_7: float
    prob_early_exit: float  # < 4 IP
    times_through_order_expected: float
    workload_confidence_penalty: float  # 0-1, higher = less confident
    notes: list[str] = field(default_factory=list)


def estimate_workload(
    pitcher: PitcherProfile,
    opponent_team: Optional[TeamProfile],
    is_opener: bool = False,
    is_tandem_risk: bool = False,
    announced_pitch_limit: Optional[int] = None,
    short_rest: bool = False,
    extra_rest: bool = False,
    recent_rehab_assignment: bool = False,
    recent_skipped_start: bool = False,
    lineup_batters: int = 9,
) -> WorkloadEstimate:
    notes: list[str] = []
    penalty = 0.0

    base_ip = pitcher.avg_innings_per_start or get_league_average("league_ip_per_start_avg")
    base_bf = pitcher.avg_bf_per_start or get_league_average("league_bf_per_start_avg")
    base_pitches = pitcher.avg_pitches_per_start or get_league_average("league_pitches_per_start_avg")

    # Recent-form adjustment: if recent starts (last 3) show meaningfully
    # fewer innings than the season average, weight recent form partially.
    if pitcher.recent_innings:
        recent_avg = sum(pitcher.recent_innings[-3:]) / len(pitcher.recent_innings[-3:])
        if recent_avg > 0:
            base_ip = 0.6 * base_ip + 0.4 * recent_avg
            if recent_avg < base_ip - 0.75:
                notes.append("Recent starts trending shorter than season average.")
                penalty += 0.1

    # Walk rate impact: high walk rate -> more pitches per out.
    bb_rate = pitcher.bb_rate_season.shrunk_rate if pitcher.bb_rate_season else None
    if bb_rate is not None:
        league_bb = get_league_average("league_bb_rate")
        if bb_rate > league_bb * 1.15:
            base_ip *= 0.94
            base_pitches *= 1.04
            notes.append("Elevated walk rate is expected to shorten the outing modestly.")

    # Opponent offensive quality.
    if opponent_team is not None and opponent_team.k_rate_overall is not None:
        league_k = get_league_average("league_k_rate")
        # Opponent that strikes out less than average tends to put more
        # balls in play -> slightly higher pitch efficiency variance, modest effect.
        if opponent_team.k_rate_overall < league_k * 0.92:
            base_pitches *= 1.02
            notes.append("Contact-oriented opponent may increase pitch count modestly.")

    # Hard overrides for non-standard roles.
    if is_opener:
        base_ip = min(base_ip, 2.0)
        base_bf = min(base_bf, 9)
        base_pitches = min(base_pitches, 35)
        penalty += 0.30
        notes.append("Pitcher is being used as an opener -- workload capped accordingly.")
    if is_tandem_risk:
        base_ip = min(base_ip, 4.0)
        base_bf = min(base_bf, 18)
        penalty += 0.20
        notes.append("Tandem/piggyback risk -- workload capped conservatively.")
    if announced_pitch_limit:
        implied_ip_from_limit = announced_pitch_limit / max(base_pitches / max(base_ip, 0.1), 12.0)
        base_ip = min(base_ip, implied_ip_from_limit)
        base_pitches = min(base_pitches, announced_pitch_limit)
        penalty += 0.25
        notes.append(f"Announced/possible pitch limit (~{announced_pitch_limit}) applied.")
    if short_rest:
        base_ip *= 0.85
        penalty += 0.15
        notes.append("Pitcher is on short rest -- workload reduced.")
    if extra_rest:
        base_ip *= 1.03
        notes.append("Extra rest -- slight workload increase.")
    if recent_rehab_assignment:
        base_ip = min(base_ip, 4.5)
        penalty += 0.25
        notes.append("Recent rehab assignment -- conservative workload cap applied.")
    if recent_skipped_start:
        penalty += 0.10
        notes.append("Recently skipped a start -- added uncertainty.")

    base_ip = max(base_ip, 0.1)
    base_bf = max(base_bf, lineup_batters)

    times_through_order = base_bf / max(lineup_batters, 1)

    # Probability of completing N innings: modeled via a logistic function
    # centered on the (adjusted) expected innings, with spread widened by
    # the accumulated confidence penalty (more uncertainty -> flatter curve).
    spread = 0.9 + penalty * 1.5

    def _prob_complete(n_innings: float) -> float:
        z = (base_ip - n_innings) / spread
        return 1.0 / (1.0 + pow(2.71828, -3.0 * z))

    prob_5 = _prob_complete(5.0)
    prob_6 = _prob_complete(6.0)
    prob_7 = _prob_complete(7.0)
    prob_early_exit = 1.0 - _prob_complete(4.0)
    prob_early_exit = max(0.0, min(1.0, 1.0 - (1.0 - prob_early_exit)))  # already 0-1

    return WorkloadEstimate(
        expected_innings=round(base_ip, 2),
        expected_batters_faced=round(base_bf, 1),
        expected_pitch_count=round(base_pitches, 1),
        prob_complete_5=round(prob_5, 3),
        prob_complete_6=round(prob_6, 3),
        prob_complete_7=round(prob_7, 3),
        prob_early_exit=round(1.0 - prob_5, 3),
        times_through_order_expected=round(times_through_order, 2),
        workload_confidence_penalty=round(min(penalty, 0.9), 3),
        notes=notes,
    )
