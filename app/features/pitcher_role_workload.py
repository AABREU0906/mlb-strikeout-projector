"""
Role-aware pitcher workload resolution.

ROOT ISSUE THIS MODULE FIXES: `season_innings / gamesStarted` (and the
batters-faced/pitches equivalents) is only a meaningful "per start"
average when the season total is actually dominated by starts. For a
pitcher whose season total is mostly relief work -- Mason Barnett's case:
1 start against 14 relief appearances -- dividing the full-season total by
`gamesStarted` produces a number that is mathematically correct division
but conceptually meaningless (it mixes relief innings into a "per start"
figure). The previous fix (plausibility guardrails) could only catch the
resulting nonsense NUMBER after the fact; this module detects the ROLE
MISMATCH before the division ever happens, and prefers real start-specific
data instead.

Fallback hierarchy (documented, in priority order):
  A. Recent MLB start-specific game logs (last 3-5 starts)
  B. Current-season MLB start-specific average (all starts this season,
     computed from game log, NOT from season totals / gamesStarted)
  C. Previous-season MLB start-specific average (same method, prior year)
  D. Recent Triple-A/highest-level MiLB start-specific workload -- NOT
     INTEGRATED in this codebase (no MiLB data source exists yet). This
     tier is honestly reported as unavailable rather than faked; a
     pitcher who reaches this point falls through to tier E/F.
  E. Team role context (opener / announced pitch limit / rehab assignment)
     -- these signals are already collected as explicit CLI/user inputs
     and applied downstream in app/projections/stage1_workload.py, not
     re-implemented here.
  F. Conservative league-average starter workload -- the existing final
     fallback in stage1_workload.py.

Every function here is pure (no I/O), so the whole hierarchy's LOGIC is
unit-testable without a live API call; only the game-log FETCH itself
(in pitcher_features.py) touches the network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

START_RATIO_MISMATCH_THRESHOLD = 0.5
MIN_STARTS_FOR_RECENT_WINDOW = 3
RECENT_STARTS_WINDOW = 5


@dataclass
class StartAppearance:
    date: Optional[str]
    innings_pitched: Optional[float]
    batters_faced: Optional[int]
    pitches: Optional[int]
    strikeouts: Optional[int]
    walks: Optional[int]


@dataclass
class WorkloadRoleAssessment:
    games_pitched: Optional[int]
    games_started: Optional[int]
    start_ratio: Optional[float]
    workload_role: str
    role_mismatch: bool
    reasons: list[str] = field(default_factory=list)


@dataclass
class StartSpecificAverages:
    avg_innings_per_start: Optional[float]
    avg_bf_per_start: Optional[float]
    avg_pitches_per_start: Optional[float]
    sample_size: int
    source_dates: list[str] = field(default_factory=list)


@dataclass
class WorkloadResolution:
    avg_innings_per_start: Optional[float]
    avg_bf_per_start: Optional[float]
    avg_pitches_per_start: Optional[float]
    workload_source: str
    workload_source_level: str
    start_specific_sample_size: int
    workload_data_valid: bool
    workload_fallback_used: bool
    reasons: list[str] = field(default_factory=list)


def assess_workload_role(
    games_started: Optional[int],
    games_pitched: Optional[int],
) -> WorkloadRoleAssessment:
    reasons: list[str] = []

    if games_started is None or games_started <= 0:
        return WorkloadRoleAssessment(
            games_pitched=games_pitched,
            games_started=games_started,
            start_ratio=None,
            workload_role="reliever" if (games_pitched or 0) > 0 else "unknown",
            role_mismatch=True,
            reasons=["gamesStarted is zero or missing; season-total division is unsafe."],
        )

    if games_pitched is None or games_pitched <= 0:
        return WorkloadRoleAssessment(
            games_pitched=games_pitched,
            games_started=games_started,
            start_ratio=None,
            workload_role="unknown",
            role_mismatch=True,
            reasons=["gamesPitched is missing; cannot verify start ratio, treating as unsafe."],
        )

    start_ratio = games_started / games_pitched

    if start_ratio < START_RATIO_MISMATCH_THRESHOLD:
        role = "swingman" if start_ratio >= 0.15 else "reliever"
        reasons.append(
            f"start_ratio={start_ratio:.2f} is below the {START_RATIO_MISMATCH_THRESHOLD} "
            f"threshold ({games_started} starts / {games_pitched} appearances); "
            f"season totals include substantial relief work."
        )
        return WorkloadRoleAssessment(
            games_pitched=games_pitched,
            games_started=games_started,
            start_ratio=start_ratio,
            workload_role=role,
            role_mismatch=True,
            reasons=reasons,
        )

    return WorkloadRoleAssessment(
        games_pitched=games_pitched,
        games_started=games_started,
        start_ratio=start_ratio,
        workload_role="starter",
        role_mismatch=False,
        reasons=["start_ratio at or above threshold; season-total division is safe."],
    )


def extract_start_appearances(gamelog_splits: list[dict]) -> list[StartAppearance]:
    starts: list[StartAppearance] = []
    for split in gamelog_splits:
        stat = split.get("stat", {}) or {}
        games_started_flag = stat.get("gamesStarted")
        try:
            is_start = int(games_started_flag) == 1
        except (TypeError, ValueError):
            is_start = False
        if not is_start:
            continue

        starts.append(
            StartAppearance(
                date=split.get("date"),
                innings_pitched=_parse_innings_local(stat.get("inningsPitched")),
                batters_faced=_to_int_local(stat.get("battersFaced")),
                pitches=_to_int_local(stat.get("numberOfPitches")),
                strikeouts=_to_int_local(stat.get("strikeOuts")),
                walks=_to_int_local(stat.get("baseOnBalls")),
            )
        )
    return starts


def compute_start_specific_averages(
    starts: list[StartAppearance],
    limit: Optional[int] = None,
) -> StartSpecificAverages:
    usable = starts[:limit] if limit is not None else starts

    ip_values = [s.innings_pitched for s in usable if s.innings_pitched is not None]
    bf_values = [s.batters_faced for s in usable if s.batters_faced is not None]
    pitch_values = [s.pitches for s in usable if s.pitches is not None]
    dates = [s.date for s in usable if s.date]

    return StartSpecificAverages(
        avg_innings_per_start=round(sum(ip_values) / len(ip_values), 2) if ip_values else None,
        avg_bf_per_start=round(sum(bf_values) / len(bf_values), 1) if bf_values else None,
        avg_pitches_per_start=round(sum(pitch_values) / len(pitch_values), 1) if pitch_values else None,
        sample_size=len(usable),
        source_dates=dates,
    )


def resolve_workload(
    *,
    role_assessment: WorkloadRoleAssessment,
    current_season_starts: list[StartAppearance],
    previous_season_starts: Optional[list[StartAppearance]],
    season_total_avg_ip: Optional[float],
    season_total_avg_bf: Optional[float],
    season_total_avg_pitches: Optional[float],
) -> WorkloadResolution:
    reasons = list(role_assessment.reasons)

    if not role_assessment.role_mismatch:
        if season_total_avg_ip is not None:
            reasons.append("Role is consistent with starting; using season-total/gamesStarted averages.")
            return WorkloadResolution(
                avg_innings_per_start=season_total_avg_ip,
                avg_bf_per_start=season_total_avg_bf,
                avg_pitches_per_start=season_total_avg_pitches,
                workload_source="mlb_season_totals",
                workload_source_level="MLB",
                start_specific_sample_size=role_assessment.games_started or 0,
                workload_data_valid=True,
                workload_fallback_used=False,
                reasons=reasons,
            )
        reasons.append("Season-total averages were unavailable despite a consistent starter role; trying start-specific game logs.")

    sorted_current = sorted(
        [s for s in current_season_starts if s.date],
        key=lambda s: s.date,
        reverse=True,
    ) + [s for s in current_season_starts if not s.date]

    if len(sorted_current) >= 1:
        recent = compute_start_specific_averages(sorted_current, limit=RECENT_STARTS_WINDOW)
        if recent.avg_innings_per_start is not None and recent.sample_size >= 1:
            valid_confidence_sample = recent.sample_size >= MIN_STARTS_FOR_RECENT_WINDOW
            if not valid_confidence_sample:
                reasons.append(
                    f"Only {recent.sample_size} recent MLB start(s) available "
                    f"(fewer than {MIN_STARTS_FOR_RECENT_WINDOW}); workload confidence should be reduced."
                )
            else:
                reasons.append(f"Using the last {recent.sample_size} MLB start(s) as the workload source.")
            return WorkloadResolution(
                avg_innings_per_start=recent.avg_innings_per_start,
                avg_bf_per_start=recent.avg_bf_per_start,
                avg_pitches_per_start=recent.avg_pitches_per_start,
                workload_source="mlb_recent_starts",
                workload_source_level="MLB",
                start_specific_sample_size=recent.sample_size,
                workload_data_valid=True,
                workload_fallback_used=role_assessment.role_mismatch,
                reasons=reasons,
            )

    season_only = compute_start_specific_averages(current_season_starts, limit=None)
    if season_only.avg_innings_per_start is not None:
        reasons.append(f"Using all {season_only.sample_size} current-season MLB start(s) as the workload source.")
        return WorkloadResolution(
            avg_innings_per_start=season_only.avg_innings_per_start,
            avg_bf_per_start=season_only.avg_bf_per_start,
            avg_pitches_per_start=season_only.avg_pitches_per_start,
            workload_source="mlb_season_starts_only",
            workload_source_level="MLB",
            start_specific_sample_size=season_only.sample_size,
            workload_data_valid=True,
            workload_fallback_used=True,
            reasons=reasons,
        )

    if previous_season_starts:
        prev = compute_start_specific_averages(previous_season_starts, limit=None)
        if prev.avg_innings_per_start is not None:
            reasons.append(f"No current-season MLB starts available; using {prev.sample_size} previous-season MLB start(s).")
            return WorkloadResolution(
                avg_innings_per_start=prev.avg_innings_per_start,
                avg_bf_per_start=prev.avg_bf_per_start,
                avg_pitches_per_start=prev.avg_pitches_per_start,
                workload_source="mlb_previous_season_starts",
                workload_source_level="MLB",
                start_specific_sample_size=prev.sample_size,
                workload_data_valid=True,
                workload_fallback_used=True,
                reasons=reasons,
            )

    reasons.append(
        "No MLB start-specific workload data (current or previous season) is available. "
        "Triple-A/MiLB start-specific data is not integrated in this system. "
        "Falling through to team-role-context and league-average handling."
    )
    return WorkloadResolution(
        avg_innings_per_start=None,
        avg_bf_per_start=None,
        avg_pitches_per_start=None,
        workload_source="unresolved",
        workload_source_level="unavailable",
        start_specific_sample_size=0,
        workload_data_valid=False,
        workload_fallback_used=True,
        reasons=reasons,
    )


def _to_int_local(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_innings_local(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        whole, _, frac = str(value).partition(".")
        whole_i = int(whole)
        frac_i = int(frac) if frac else 0
        if frac_i not in (0, 1, 2):
            return None
        return whole_i + frac_i / 3.0
    except (ValueError, TypeError):
        return None
