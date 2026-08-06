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
from app.validation.bounds import (
    MAX_BATTERS_FACED_PER_INNING,
    MAX_PITCHES_PER_BATTER_FACED,
    MAX_PLAUSIBLE_AVG_BF_PER_START,
    MAX_PLAUSIBLE_AVG_INNINGS_PER_START,
    MAX_PLAUSIBLE_AVG_PITCHES_PER_START,
    MIN_BATTERS_FACED_PER_INNING,
    MIN_PITCHES_PER_BATTER_FACED,
    MIN_PLAUSIBLE_AVG_BF_PER_START,
    MIN_PLAUSIBLE_AVG_INNINGS_PER_START,
    MIN_PLAUSIBLE_AVG_PITCHES_PER_START,
)


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
    # Structured workload-validity fields (replace fragile note-text
    # string matching, per audit requirement). These are the single
    # source of truth for "did this projection actually use
    # pitcher-specific workload data" -- every downstream layer
    # (ProjectionResult, the validator, betting-edge confidence, display)
    # reads these fields directly instead of parsing `notes`.
    workload_data_valid: bool = True          # True only if ALL THREE metrics used real pitcher-specific data
    workload_fallback_used: bool = False       # True if ANY of the three metrics fell back to league average
    workload_fallback_count: int = 0           # how many of the 3 metrics fell back (0-3)
    workload_all_metrics_fallback: bool = False  # True only if ALL THREE fell back
    # Role-aware metadata (pass-through from PitcherProfile -- see
    # app/features/pitcher_role_workload.py). Exposed here too so display
    # and confidence logic can read everything off WorkloadEstimate
    # without reaching back into the pitcher profile separately.
    workload_role: Optional[str] = None
    workload_source: Optional[str] = None
    workload_source_level: Optional[str] = None
    start_specific_sample_size: int = 0
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
    effective_pitch_limit: Optional[float] = None

    league_ip, league_bf, league_pitches = (
        _league_workload_defaults()
    )

    raw_ip = pitcher.avg_innings_per_start
    raw_bf = pitcher.avg_bf_per_start
    raw_pitches = pitcher.avg_pitches_per_start

    # Structured fallback tracking (per-metric). These flags are set
    # whenever a metric falls back to the league average, REGARDLESS of
    # whether the raw value arrived as None (already rejected upstream by
    # PitcherFeatureBuilder's own guardrail) or as a present-but-invalid
    # number caught right here. Previously, the note/penalty below only
    # fired when `raw_x is not None` -- so a value already nulled out by
    # PitcherFeatureBuilder (exactly what happens when the API returns an
    # implausible per-start average) silently skipped both the note AND
    # the confidence penalty, because this function assumed "someone else
    # already warned about it." Nobody had. This is the root cause of
    # confidence displaying HIGH after a workload fallback.
    ip_is_fallback = False
    bf_is_fallback = False
    pitches_is_fallback = False

    if _valid_number(
        raw_ip,
        minimum=MIN_PLAUSIBLE_AVG_INNINGS_PER_START,
        maximum=MAX_PLAUSIBLE_AVG_INNINGS_PER_START,
    ):
        base_ip = float(raw_ip)
    else:
        base_ip = league_ip
        ip_is_fallback = True

        if raw_ip is not None:
            notes.append(
                "Invalid pitcher innings-per-start value was "
                "replaced with the league average."
            )
        else:
            notes.append(
                "Pitcher innings-per-start data was unavailable "
                "(rejected upstream or missing); using the league average."
            )
        penalty += 0.15

    if _valid_number(
        raw_bf,
        minimum=MIN_PLAUSIBLE_AVG_BF_PER_START,
        maximum=MAX_PLAUSIBLE_AVG_BF_PER_START,
    ):
        base_bf = float(raw_bf)
    else:
        base_bf = league_bf
        bf_is_fallback = True

        if raw_bf is not None:
            notes.append(
                "Invalid pitcher batters-faced-per-start value was "
                "replaced with the league average."
            )
        else:
            notes.append(
                "Pitcher batters-faced-per-start data was unavailable "
                "(rejected upstream or missing); using the league average."
            )
        penalty += 0.15

    if _valid_number(
        raw_pitches,
        minimum=MIN_PLAUSIBLE_AVG_PITCHES_PER_START,
        maximum=MAX_PLAUSIBLE_AVG_PITCHES_PER_START,
    ):
        base_pitches = float(raw_pitches)
    else:
        base_pitches = league_pitches
        pitches_is_fallback = True

        if raw_pitches is not None:
            notes.append(
                "Invalid pitcher pitches-per-start value was "
                "replaced with the league average."
            )
        else:
            notes.append(
                "Pitcher pitches-per-start data was unavailable "
                "(rejected upstream or missing); using the league average."
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
            effective_pitch_limit = pitch_limit

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
        ip_is_fallback = True
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
        bf_is_fallback = True
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
        pitches_is_fallback = True
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
    minimum_bf_for_ip = base_ip * MIN_BATTERS_FACED_PER_INNING
    maximum_bf_for_ip = base_ip * MAX_BATTERS_FACED_PER_INNING

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

    if estimated_pitches_per_bf < MIN_PITCHES_PER_BATTER_FACED:
        adjusted_pitch_count = (
            base_bf * MIN_PITCHES_PER_BATTER_FACED
        )

        # An announced pitch limit is a hard ceiling. Do not let the
        # consistency adjustment raise the estimate above that limit.
        if effective_pitch_limit is not None:
            adjusted_pitch_count = min(
                adjusted_pitch_count,
                effective_pitch_limit,
            )

        if adjusted_pitch_count > base_pitches:
            base_pitches = adjusted_pitch_count
            notes.append(
                "Pitch-count estimate was raised to remain "
                "consistent with expected batters faced without "
                "exceeding the announced pitch limit."
            )

    elif estimated_pitches_per_bf > MAX_PITCHES_PER_BATTER_FACED:
        base_pitches = base_bf * MAX_PITCHES_PER_BATTER_FACED
        notes.append(
            "Pitch-count estimate was lowered to remain "
            "consistent with expected batters faced."
        )

    base_pitches = min(
        max(base_pitches, 10.0),
        130.0,
    )

    # Reapply the announced limit after every later adjustment so it
    # remains the final hard ceiling on expected pitch count.
    if effective_pitch_limit is not None:
        base_pitches = min(
            base_pitches,
            effective_pitch_limit,
        )

    times_through_order = (
        base_bf / max(lineup_batters, 1)
    )

    # Probability model.
    # Small-sample penalty: relying on fewer than 3 real starts (whether
    # recent-starts, season-starts-only, or previous-season tiers) is
    # meaningfully less certain than a full season average or a healthy
    # recent sample, even though the value itself is real MLB data (not a
    # league-average substitution) -- see pitcher_role_workload.py's
    # MIN_STARTS_FOR_RECENT_WINDOW.
    start_specific_sample_size = getattr(pitcher, "start_specific_sample_size", 0) or 0
    workload_source = getattr(pitcher, "workload_source", None)
    if workload_source in ("mlb_recent_starts", "mlb_season_starts_only", "mlb_previous_season_starts"):
        if 0 < start_specific_sample_size < 3:
            notes.append(
                f"Workload is based on only {start_specific_sample_size} MLB start(s); "
                f"confidence is reduced for this thin a sample."
            )
            penalty += 0.10

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

    workload_source_level = getattr(pitcher, "workload_source_level", None)
    # Per requirement: "If no MLB start-specific workload exists, confidence
    # may not exceed MEDIUM." Reuses the SAME tested confidence-cap
    # mechanism as the existing all-three-metrics-fallback case (see
    # app/markets/edge_analysis.py's determine_confidence/determine_edge_grade)
    # rather than adding a second, parallel cap -- both conditions mean the
    # same thing downstream: "don't trust this workload enough for HIGH
    # confidence or an Elite/Strong edge label."
    no_mlb_source_available = workload_source_level not in (None, "MLB")

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
        workload_data_valid=not (ip_is_fallback or bf_is_fallback or pitches_is_fallback),
        workload_fallback_used=(ip_is_fallback or bf_is_fallback or pitches_is_fallback),
        workload_fallback_count=sum([ip_is_fallback, bf_is_fallback, pitches_is_fallback]),
        workload_all_metrics_fallback=(
            (ip_is_fallback and bf_is_fallback and pitches_is_fallback)
            or no_mlb_source_available
        ),
        workload_role=getattr(pitcher, "workload_role", None),
        workload_source=workload_source,
        workload_source_level=workload_source_level,
        start_specific_sample_size=start_specific_sample_size,
        notes=notes,
    )