"""
Stage 1: Expected Workload Model.

Transparent, documented formula-based model that estimates innings pitched,
batters faced, pitch count, and the probability distribution over times
through the order.

Inputs include:

- Pitcher's season workload
- Recent workload trend
- Pitch efficiency
- Walk rate
- Opponent offensive quality
- Rest status
- Opener or tandem role
- Pitch limits
- Injury-return and rehab warnings

This stage does not assume a normal workload when non-standard role or
workload warnings are present.
"""

from __future__ import annotations

import math
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
    prob_early_exit: float
    times_through_order_expected: float
    workload_confidence_penalty: float
    notes: list[str] = field(default_factory=list)


def _valid_number(
    value: object,
    *,
    minimum: float,
    maximum: float,
) -> bool:
    """
    Return True when a value is a finite number within the allowed range.
    """
    if value is None:
        return False

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(numeric_value)
        and minimum <= numeric_value <= maximum
    )


def _league_workload_defaults() -> tuple[float, float, float]:
    """
    Return safe league-average workload defaults.
    """
    league_ip = float(
        get_league_average("league_ip_per_start_avg")
    )
    league_bf = float(
        get_league_average("league_bf_per_start_avg")
    )
    league_pitches = float(
        get_league_average("league_pitches_per_start_avg")
    )

    return league_ip, league_bf, league_pitches


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

    league_ip, league_bf, league_pitches = (
        _league_workload_defaults()
    )

    raw_ip = pitcher.avg_innings_per_start
    raw_bf = pitcher.avg_bf_per_start
    raw_pitches = pitcher.avg_pitches_per_start

    if _valid_number(
        raw_ip,
        minimum=0.5,
        maximum=9.0,
    ):
        base_ip = float(raw_ip)
    else:
        base_ip = league_ip

        if raw_ip is not None:
            notes.append(
                "Invalid pitcher innings-per-start value was "
                "replaced with the league average."
            )
            penalty += 0.15

    if _valid_number(
        raw_bf,
        minimum=3.0,
        maximum=45.0,
    ):
        base_bf = float(raw_bf)
    else:
        base_bf = league_bf

        if raw_bf is not None:
            notes.append(
                "Invalid pitcher batters-faced-per-start value was "
                "replaced with the league average."
            )
            penalty += 0.15

    if _valid_number(
        raw_pitches,
        minimum=10.0,
        maximum=130.0,
    ):
        base_pitches = float(raw_pitches)
    else:
        base_pitches = league_pitches

        if raw_pitches is not None:
            notes.append(
                "Invalid pitcher pitches-per-start value was "
                "replaced with the league average."
            )
            penalty += 0.15

    # Recent-form adjustment.
    recent_innings = [
        float(value)
        for value in (pitcher.recent_innings or [])
        if _valid_number(
            value,
            minimum=0.0,
            maximum=9.0,
        )
    ]

    if recent_innings:
        recent_sample = recent_innings[-3:]
        recent_avg = sum(recent_sample) / len(recent_sample)

        season_base_before_blend = base_ip
        base_ip = (
            0.60 * season_base_before_blend
            + 0.40 * recent_avg
        )

        if recent_avg < season_base_before_blend - 0.75:
            notes.append(
                "Recent starts are trending shorter than the "
                "season workload estimate."
            )
            penalty += 0.10

    # Walk-rate adjustment.
    bb_rate = (
        pitcher.bb_rate_season.shrunk_rate
        if pitcher.bb_rate_season
        else None
    )

    if _valid_number(
        bb_rate,
        minimum=0.0,
        maximum=1.0,
    ):
        league_bb = float(
            get_league_average("league_bb_rate")
        )

        if float(bb_rate) > league_bb * 1.15:
            base_ip *= 0.94
            base_pitches *= 1.04

            notes.append(
                "Elevated walk rate is expected to shorten the "
                "outing modestly."
            )

    # Opponent contact adjustment.
    if (
        opponent_team is not None
        and _valid_number(
            opponent_team.k_rate_overall,
            minimum=0.0,
            maximum=1.0,
        )
    ):
        league_k = float(
            get_league_average("league_k_rate")
        )

        if (
            float(opponent_team.k_rate_overall)
            < league_k * 0.92
        ):
            base_pitches *= 1.02

            notes.append(
                "Contact-oriented opponent may increase pitch "
                "count modestly."
            )

    # Non-standard roles and workload warnings.
    if is_opener:
        base_ip = min(base_ip, 2.0)
        base_bf = min(base_bf, 9.0)
        base_pitches = min(base_pitches, 35.0)
        penalty += 0.30

        notes.append(
            "Pitcher is being used as an opener; workload was "
            "capped accordingly."
        )

    if is_tandem_risk:
        base_ip = min(base_ip, 4.0)
        base_bf = min(base_bf, 18.0)
        base_pitches = min(base_pitches, 75.0)
        penalty += 0.20

        notes.append(
            "Tandem or piggyback risk detected; workload was "
            "capped conservatively."
        )

    if announced_pitch_limit is not None:
        if announced_pitch_limit <= 0:
            notes.append(
                "Invalid pitch-limit value was ignored."
            )
            penalty += 0.10
        else:
            pitch_limit = min(
                float(announced_pitch_limit),
                130.0,
            )

            pitches_per_inning = (
                base_pitches / max(base_ip, 0.1)
            )

            pitches_per_inning = max(
                pitches_per_inning,
                12.0,
            )

            implied_ip_from_limit = (
                pitch_limit / pitches_per_inning
            )

            base_ip = min(
                base_ip,
                implied_ip_from_limit,
            )

            base_pitches = min(
                base_pitches,
                pitch_limit,
            )

            penalty += 0.25

            notes.append(
                f"Announced or possible pitch limit "
                f"(approximately {int(pitch_limit)}) was applied."
            )

    if short_rest:
        base_ip *= 0.85
        base_bf *= 0.88
        base_pitches *= 0.90
        penalty += 0.15

        notes.append(
            "Pitcher is on short rest; workload was reduced."
        )

    if extra_rest:
        base_ip *= 1.03
        base_bf *= 1.02

        notes.append(
            "Extra rest produced a slight workload increase."
        )

    if recent_rehab_assignment:
        base_ip = min(base_ip, 4.5)
        base_bf = min(base_bf, 20.0)
        base_pitches = min(base_pitches, 80.0)
        penalty += 0.25

        notes.append(
            "Recent rehab assignment detected; a conservative "
            "workload cap was applied."
        )

    if recent_skipped_start:
        penalty += 0.10

        notes.append(
            "Pitcher recently skipped a start; additional "
            "workload uncertainty was applied."
        )

    # Final workload guardrails.
    if not _valid_number(
        base_ip,
        minimum=0.1,
        maximum=9.0,
    ):
        notes.append(
            f"Invalid final innings estimate ({base_ip!r}) was "
            "replaced with the league average."
        )
        base_ip = league_ip
        penalty += 0.25

    if not _valid_number(
        base_bf,
        minimum=3.0,
        maximum=45.0,
    ):
        notes.append(
            f"Invalid final batters-faced estimate ({base_bf!r}) "
            "was replaced with the league average."
        )
        base_bf = league_bf
        penalty += 0.25

    if not _valid_number(
        base_pitches,
        minimum=10.0,
        maximum=130.0,
    ):
        notes.append(
            f"Invalid final pitch-count estimate "
            f"({base_pitches!r}) was replaced with the league "
            "average."
        )
        base_pitches = league_pitches
        penalty += 0.25

    # Maintain a reasonable relationship among workload values.
    base_ip = max(
        0.1,
        min(base_ip, 9.0),
    )

    base_bf = max(
        float(max(lineup_batters, 1)),
        min(base_bf, 45.0),
    )

    base_pitches = max(
        10.0,
        min(base_pitches, 130.0),
    )

    # Guard against unrealistic relationships between BF, IP, and pitches.
    minimum_bf_for_ip = base_ip * 3.0
    maximum_bf_for_ip = base_ip * 6.5

    if base_bf < minimum_bf_for_ip:
        base_bf = minimum_bf_for_ip
        notes.append(
            "Batters-faced estimate was raised to remain "
            "consistent with expected innings."
        )

    if base_bf > maximum_bf_for_ip:
        base_bf = maximum_bf_for_ip
        notes.append(
            "Batters-faced estimate was lowered to remain "
            "consistent with expected innings."
        )

    estimated_pitches_per_bf = (
        base_pitches / max(base_bf, 1.0)
    )

    if estimated_pitches_per_bf < 2.5:
        base_pitches = base_bf * 2.5
        notes.append(
            "Pitch-count estimate was raised to remain "
            "consistent with expected batters faced."
        )

    elif estimated_pitches_per_bf > 6.0:
        base_pitches = base_bf * 6.0
        notes.append(
            "Pitch-count estimate was lowered to remain "
            "consistent with expected batters faced."
        )

    base_pitches = min(
        max(base_pitches, 10.0),
        130.0,
    )

    times_through_order = (
        base_bf / max(lineup_batters, 1)
    )

    # Probability model.
    spread = 0.9 + min(penalty, 0.9) * 1.5

    def _prob_complete(
        innings_threshold: float,
    ) -> float:
        z_score = (
            base_ip - innings_threshold
        ) / max(spread, 0.1)

        probability = (
            1.0
            / (
                1.0
                + math.exp(-3.0 * z_score)
            )
        )

        return max(
            0.0,
            min(1.0, probability),
        )

    prob_5 = _prob_complete(5.0)
    prob_6 = _prob_complete(6.0)
    prob_7 = _prob_complete(7.0)
    prob_early_exit = 1.0 - _prob_complete(4.0)

    return WorkloadEstimate(
        expected_innings=round(base_ip, 2),
        expected_batters_faced=round(base_bf, 1),
        expected_pitch_count=round(base_pitches, 1),
        prob_complete_5=round(prob_5, 3),
        prob_complete_6=round(prob_6, 3),
        prob_complete_7=round(prob_7, 3),
        prob_early_exit=round(prob_early_exit, 3),
        times_through_order_expected=round(
            times_through_order,
            2,
        ),
        workload_confidence_penalty=round(
            min(penalty, 0.9),
            3,
        ),
        notes=notes,
    )