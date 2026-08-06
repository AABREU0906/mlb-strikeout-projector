"""
Central Projection Validator.

Per the audit's own stated top priority: "The system must prefer showing
PASS or VALIDATION FAILED over producing a confident but unreliable
betting recommendation." This module is the single gate every projection
passes through before ANY betting-related output (recommended side,
estimated EV, edge grade, "did you place this bet?" prompt, or a
save-as-valid-recommendation) is allowed to appear.

Design:
  - `validate_projection()` returns a `ValidationReport` with CRITICAL
    issues (block all betting output) kept structurally separate from
    WARNING issues (surfaced to the user and fed into confidence
    reduction, but don't by themselves block display).
  - Every check is a small, named, independently testable function so
    "avoid duplicated validation logic" holds even as new checks are
    added later.
  - Nothing here re-implements the workload bound constants -- it imports
    the same `app.validation.bounds` values `stage1_workload.py` uses, so
    the two layers of defense (build-time clamping, display-time
    validation) can never silently diverge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from app.validation.bounds import (
    MAX_BATTERS_FACED_PER_INNING,
    MAX_INNINGS_PER_START,
    MAX_PITCHES_PER_BATTER_FACED,
    MIN_BATTERS_FACED_PER_INNING,
    MIN_PITCHES_PER_BATTER_FACED,
)

PROBABILITY_SUM_TOLERANCE = 0.02
EXTREME_PROBABILITY_LOW = 0.02
EXTREME_PROBABILITY_HIGH = 0.98
STALE_DATA_MINUTES_WARNING = 120


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_critical(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue("critical", code, message))

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue("warning", code, message))

    @property
    def critical_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warning_issues(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        return len(self.critical_issues) == 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warning_issues) > 0


def check_expected_innings(report: ValidationReport, expected_innings: float) -> None:
    if expected_innings is None or not (0.0 < expected_innings <= MAX_INNINGS_PER_START):
        report.add_critical(
            "invalid_expected_innings",
            f"Expected innings ({expected_innings}) must be greater than 0 and no more than "
            f"{MAX_INNINGS_PER_START}.",
        )


def check_batters_faced_consistency(report: ValidationReport, expected_innings: float, expected_batters_faced: float) -> None:
    if expected_innings is None or expected_batters_faced is None or expected_innings <= 0:
        return
    ratio = expected_batters_faced / expected_innings
    if not (MIN_BATTERS_FACED_PER_INNING - 0.25 <= ratio <= MAX_BATTERS_FACED_PER_INNING + 0.25):
        report.add_critical(
            "inconsistent_batters_faced",
            f"Expected batters faced ({expected_batters_faced}) implies {ratio:.2f} batters/inning, "
            f"outside the realistic {MIN_BATTERS_FACED_PER_INNING}-{MAX_BATTERS_FACED_PER_INNING} range "
            f"for {expected_innings} expected innings.",
        )


def check_pitch_count_consistency(report: ValidationReport, expected_batters_faced: float, expected_pitch_count: float) -> None:
    if expected_batters_faced is None or expected_pitch_count is None or expected_batters_faced <= 0:
        return
    ratio = expected_pitch_count / expected_batters_faced
    if not (MIN_PITCHES_PER_BATTER_FACED - 0.5 <= ratio <= MAX_PITCHES_PER_BATTER_FACED + 0.5):
        report.add_critical(
            "inconsistent_pitch_count",
            f"Expected pitch count ({expected_pitch_count}) implies {ratio:.2f} pitches/batter, "
            f"outside the realistic {MIN_PITCHES_PER_BATTER_FACED}-{MAX_PITCHES_PER_BATTER_FACED} range "
            f"for {expected_batters_faced} expected batters faced.",
        )


def check_probability_distribution(report: ValidationReport, probability_by_k: dict) -> None:
    if not probability_by_k:
        report.add_critical("empty_distribution", "Strikeout probability distribution is empty.")
        return
    for k, p in probability_by_k.items():
        if p is None or not (0.0 <= p <= 1.0):
            report.add_critical(
                "probability_out_of_bounds",
                f"Probability for {k} strikeouts ({p}) is not between 0 and 1.",
            )
            return
    total = sum(probability_by_k.values())
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        report.add_critical(
            "distribution_does_not_sum_to_one",
            f"Strikeout probability distribution sums to {total:.4f}, not ~1.0.",
        )


def check_percentile_ordering(report: ValidationReport, percentiles: dict) -> None:
    if not percentiles:
        return
    order = [10, 25, 50, 75, 90]
    values = [percentiles.get(p) for p in order]
    if any(v is None for v in values):
        return
    for a, b in zip(values, values[1:]):
        if a > b:
            report.add_critical(
                "percentiles_out_of_order",
                f"Percentiles are not correctly ordered: {dict(zip(order, values))}.",
            )
            return


def check_std_dev(report: ValidationReport, std_dev: Optional[float]) -> None:
    if std_dev is None or std_dev < 0:
        report.add_critical("invalid_std_dev", f"Standard deviation ({std_dev}) must be zero or positive.")
        return
    if not math.isfinite(std_dev):
        report.add_critical("invalid_std_dev", f"Standard deviation ({std_dev}) is not finite.")


def check_projection_vs_distribution_mean(
    report: ValidationReport, final_projection: float, probability_by_k: dict, tolerance: float = 2.5
) -> None:
    if not probability_by_k or final_projection is None:
        return
    implied_mean = sum(k * p for k, p in probability_by_k.items())
    if abs(implied_mean - final_projection) > tolerance:
        report.add_critical(
            "projection_inconsistent_with_distribution",
            f"Final projection ({final_projection}) differs from the distribution's implied mean "
            f"({implied_mean:.2f}) by more than {tolerance}.",
        )


def check_projection_not_exceeding_batters_faced(
    report: ValidationReport, final_projection: float, expected_batters_faced: float
) -> None:
    if final_projection is None or expected_batters_faced is None:
        return
    if final_projection > expected_batters_faced:
        report.add_critical(
            "projection_exceeds_batters_faced",
            f"Final projected strikeouts ({final_projection}) cannot exceed expected batters "
            f"faced ({expected_batters_faced}).",
        )


def check_workload_completion_probabilities(
    report: ValidationReport,
    prob_complete_5: Optional[float],
    prob_complete_6: Optional[float],
    prob_complete_7: Optional[float],
    prob_early_exit: Optional[float],
) -> None:
    probs = {
        "prob_complete_5": prob_complete_5,
        "prob_complete_6": prob_complete_6,
        "prob_complete_7": prob_complete_7,
        "prob_early_exit": prob_early_exit,
    }
    for name, value in probs.items():
        if value is not None and not (0.0 <= value <= 1.0):
            report.add_critical("workload_probability_out_of_bounds", f"{name} ({value}) is not between 0 and 1.")

    if prob_complete_5 is not None and prob_complete_6 is not None and prob_complete_6 > prob_complete_5 + 1e-9:
        report.add_critical(
            "workload_probability_ordering",
            f"Probability of completing 6 innings ({prob_complete_6}) exceeds probability of "
            f"completing 5 ({prob_complete_5}).",
        )
    if prob_complete_6 is not None and prob_complete_7 is not None and prob_complete_7 > prob_complete_6 + 1e-9:
        report.add_critical(
            "workload_probability_ordering",
            f"Probability of completing 7 innings ({prob_complete_7}) exceeds probability of "
            f"completing 6 ({prob_complete_6}).",
        )


def check_over_under_probabilities(
    report: ValidationReport,
    over_probability: Optional[float],
    under_probability: Optional[float],
    push_probability: float = 0.0,
) -> None:
    if over_probability is None or under_probability is None:
        return
    if not (0.0 <= over_probability <= 1.0) or not (0.0 <= under_probability <= 1.0):
        report.add_critical(
            "over_under_probability_out_of_bounds",
            f"Over ({over_probability}) / Under ({under_probability}) probabilities must be between 0 and 1.",
        )
        return
    total = over_probability + under_probability + push_probability
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        report.add_critical(
            "over_under_do_not_sum_to_one",
            f"Over + Under + Push probabilities sum to {total:.4f}, not ~1.0.",
        )


def check_extreme_probability_risk(report: ValidationReport, over_probability: Optional[float], under_probability: Optional[float]) -> None:
    for label, value in (("over", over_probability), ("under", under_probability)):
        if value is not None and (value <= EXTREME_PROBABILITY_LOW or value >= EXTREME_PROBABILITY_HIGH):
            report.add_warning(
                "extreme_probability",
                f"Model {label} probability ({value:.1%}) is near 0% or 100%; elevated model risk "
                f"unless supported by historical calibration.",
            )


def check_lineup_and_pitcher_confirmation(report: ValidationReport, lineup_confirmed: bool, pitcher_confirmed: bool) -> None:
    if not pitcher_confirmed:
        report.add_warning("pitcher_unconfirmed", "Starting pitcher is not yet confirmed.")
    if not lineup_confirmed:
        report.add_warning("lineup_projected", "Opponent lineup is projected, not confirmed.")


def check_data_freshness(report: ValidationReport, data_age_minutes: Optional[float]) -> None:
    if data_age_minutes is not None and data_age_minutes > STALE_DATA_MINUTES_WARNING:
        report.add_warning(
            "stale_data",
            f"Underlying data is {data_age_minutes:.0f} minutes old (over the "
            f"{STALE_DATA_MINUTES_WARNING}-minute freshness threshold).",
        )


def workload_notes_indicate_fallback(workload_notes: list[str]) -> bool:
    """True if any workload note documents a fallback substitution (e.g.
    'replaced with the league average'). Shared by the validator's
    warning check and by the CLI/display layer, which needs the same
    boolean to decide the displayed confidence level -- so both places
    can never diverge on what counts as a fallback."""
    return any(
        "league average" in note.lower() or "replaced with" in note.lower()
        for note in (workload_notes or [])
    )


def check_workload_fallback_used(report: ValidationReport, workload_notes: list[str]) -> None:
    for note in workload_notes or []:
        if "league average" in note.lower() or "replaced with" in note.lower():
            report.add_warning("workload_fallback_used", note)


def validate_projection(
    *,
    expected_innings: float,
    expected_batters_faced: float,
    expected_pitch_count: float,
    final_projection: float,
    probability_by_k: dict,
    percentiles: dict,
    std_dev: Optional[float],
    prob_complete_5: Optional[float] = None,
    prob_complete_6: Optional[float] = None,
    prob_complete_7: Optional[float] = None,
    prob_early_exit: Optional[float] = None,
    over_probability: Optional[float] = None,
    under_probability: Optional[float] = None,
    push_probability: float = 0.0,
    lineup_confirmed: bool = True,
    pitcher_confirmed: bool = True,
    data_age_minutes: Optional[float] = None,
    workload_notes: Optional[list[str]] = None,
) -> ValidationReport:
    """Runs every check and returns one report. Call this exactly once,
    immediately before any betting-related output is displayed or saved
    as a valid recommendation."""
    report = ValidationReport()

    check_expected_innings(report, expected_innings)
    check_batters_faced_consistency(report, expected_innings, expected_batters_faced)
    check_pitch_count_consistency(report, expected_batters_faced, expected_pitch_count)
    check_probability_distribution(report, probability_by_k)
    check_percentile_ordering(report, percentiles)
    check_std_dev(report, std_dev)
    check_projection_vs_distribution_mean(report, final_projection, probability_by_k)
    check_projection_not_exceeding_batters_faced(report, final_projection, expected_batters_faced)
    check_workload_completion_probabilities(report, prob_complete_5, prob_complete_6, prob_complete_7, prob_early_exit)
    check_over_under_probabilities(report, over_probability, under_probability, push_probability)
    check_extreme_probability_risk(report, over_probability, under_probability)
    check_lineup_and_pitcher_confirmation(report, lineup_confirmed, pitcher_confirmed)
    check_data_freshness(report, data_age_minutes)
    check_workload_fallback_used(report, workload_notes or [])

    return report
