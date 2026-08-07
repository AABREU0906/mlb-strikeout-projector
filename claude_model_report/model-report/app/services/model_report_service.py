"""
Orchestration + extraction layer for `python main.py model-report`.

Turns Projection (+ its ActualResult) ORM rows into the plain dicts
app.evaluation.model_report_metrics operates on, then calls those pure
functions to build the full report. Kept separate from the metrics module
itself so the math stays testable without a database.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.evaluation.model_report_metrics import (
    compute_bias_by_group,
    compute_calibration,
    compute_directional_metrics,
    compute_error_metrics,
    line_bucket_for,
    summarize_error_decomposition,
)


def _extract_row(projection) -> dict:
    actual = projection.actual_result
    workload = projection.workload_inputs_json or {}
    market = projection.market_snapshot_json or {}

    strikeout_line = market.get("strikeout_line")

    return {
        "projection_id": projection.id,
        "pitcher_name": projection.pitcher_name,
        "opponent_team": projection.opponent_team,
        "game_date": projection.game_date,
        "statistics_only_projection": projection.statistics_only_projection,
        "market_informed_projection": projection.market_informed_projection,
        "final_blended_projection": projection.final_blended_projection,
        "actual_strikeouts": actual.actual_strikeouts if actual else None,
        "expected_batters_faced": projection.expected_batters_faced,
        "actual_batters_faced": actual.actual_batters_faced if actual else None,
        "strikeout_line": strikeout_line,
        "line_bucket": line_bucket_for(strikeout_line),
        "recommended_side": projection.recommended_side,
        "edge_grade": projection.edge_grade,
        "betting_confidence": projection.betting_confidence,
        "model_over_probability": projection.model_over_probability,
        "estimated_ev": projection.estimated_ev,
        "workload_role": workload.get("workload_role"),
        "workload_source": workload.get("workload_source"),
        "workload_fallback_used": workload.get("workload_fallback_used"),
        "lineup_status": projection.lineup_status,
    }


@dataclass
class ModelReport:
    n_total_graded: int
    core_metrics: dict = field(default_factory=dict)
    core_metrics_statistics_only: dict = field(default_factory=dict)
    core_metrics_market_informed: dict = field(default_factory=dict)
    directional: dict = field(default_factory=dict)
    calibration: list = field(default_factory=list)
    bias_by_pitcher: list = field(default_factory=list)
    bias_by_opponent: list = field(default_factory=list)
    bias_by_line_bucket: list = field(default_factory=list)
    bias_by_confidence: list = field(default_factory=list)
    bias_by_edge_grade: list = field(default_factory=list)
    bias_by_workload_source: list = field(default_factory=list)
    bias_by_workload_role: list = field(default_factory=list)
    bias_by_lineup_status: list = field(default_factory=list)
    bias_by_fallback_used: list = field(default_factory=list)
    error_decomposition: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


def generate_model_report(projections: list) -> ModelReport:
    rows = [_extract_row(p) for p in projections]
    graded_rows = [r for r in rows if r["actual_strikeouts"] is not None]
    n = len(graded_rows)

    report = ModelReport(n_total_graded=n)

    if n == 0:
        report.warnings.append("No graded projections found for the given filters.")
        return report

    if n < 20:
        report.warnings.append(
            f"Sample size is small (n={n}). Treat every metric below as directional, not conclusive."
        )

    report.core_metrics = compute_error_metrics(graded_rows, "final_blended_projection")
    report.core_metrics_statistics_only = compute_error_metrics(graded_rows, "statistics_only_projection")
    report.core_metrics_market_informed = compute_error_metrics(graded_rows, "market_informed_projection")

    report.directional = compute_directional_metrics(graded_rows)
    report.calibration = compute_calibration(graded_rows)

    report.bias_by_pitcher = compute_bias_by_group(graded_rows, "pitcher_name")
    report.bias_by_opponent = compute_bias_by_group(graded_rows, "opponent_team")
    report.bias_by_line_bucket = compute_bias_by_group(graded_rows, "line_bucket")
    report.bias_by_confidence = compute_bias_by_group(graded_rows, "betting_confidence")
    report.bias_by_edge_grade = compute_bias_by_group(graded_rows, "edge_grade")
    report.bias_by_workload_source = compute_bias_by_group(graded_rows, "workload_source")
    report.bias_by_workload_role = compute_bias_by_group(graded_rows, "workload_role")
    report.bias_by_lineup_status = compute_bias_by_group(graded_rows, "lineup_status")
    report.bias_by_fallback_used = compute_bias_by_group(
        [{**r, "workload_fallback_used": "fallback_used" if r["workload_fallback_used"] else "pitcher_specific"} for r in graded_rows],
        "workload_fallback_used",
    )

    report.error_decomposition = summarize_error_decomposition(graded_rows)

    if report.directional.get("n", 0) == 0:
        report.warnings.append("No graded projections had a sportsbook line; directional metrics unavailable.")

    return report
