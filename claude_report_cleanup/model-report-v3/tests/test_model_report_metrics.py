"""
Tests for app/evaluation/model_report_metrics.py -- pure functions, no
database or pydantic required.
"""
from app.evaluation.model_report_metrics import (
    compute_bias_by_group,
    compute_calibration,
    compute_directional_metrics,
    compute_error_metrics,
    decompose_error,
    line_bucket_for,
    summarize_error_decomposition,
)


def test_mae_calculation():
    rows = [
        {"final_blended_projection": 5.0, "actual_strikeouts": 6},
        {"final_blended_projection": 6.0, "actual_strikeouts": 6},
        {"final_blended_projection": 7.0, "actual_strikeouts": 5},
    ]
    m = compute_error_metrics(rows)
    assert m["n"] == 3
    assert abs(m["mae"] - 1.0) < 0.01


def test_rmse_calculation():
    rows = [
        {"final_blended_projection": 5.0, "actual_strikeouts": 6},
        {"final_blended_projection": 6.0, "actual_strikeouts": 6},
        {"final_blended_projection": 7.0, "actual_strikeouts": 5},
    ]
    m = compute_error_metrics(rows)
    expected_rmse = ((1 + 0 + 4) / 3) ** 0.5
    assert abs(m["rmse"] - expected_rmse) < 0.01


def test_signed_bias_calculation():
    rows = [
        {"final_blended_projection": 5.0, "actual_strikeouts": 6},
        {"final_blended_projection": 6.0, "actual_strikeouts": 6},
        {"final_blended_projection": 7.0, "actual_strikeouts": 5},
    ]
    m = compute_error_metrics(rows)
    expected_bias = (-1 + 0 + 2) / 3
    assert abs(m["bias"] - expected_bias) < 0.01


def test_percentage_within_thresholds():
    rows = [
        {"final_blended_projection": 5.0, "actual_strikeouts": 5.4},
        {"final_blended_projection": 5.0, "actual_strikeouts": 6.0},
        {"final_blended_projection": 5.0, "actual_strikeouts": 8.0},
    ]
    m = compute_error_metrics(rows)
    assert abs(m["pct_within_0_5"] - (1 / 3 * 100)) < 0.1
    assert abs(m["pct_within_1_0"] - (2 / 3 * 100)) < 0.1
    assert abs(m["pct_within_2_0"] - (2 / 3 * 100)) < 0.1


def test_error_metrics_empty_input_no_crash():
    m = compute_error_metrics([])
    assert m["n"] == 0
    assert m["mae"] is None


def test_error_metrics_skips_rows_missing_data():
    rows = [
        {"final_blended_projection": 5.0, "actual_strikeouts": None},
        {"final_blended_projection": None, "actual_strikeouts": 6},
        {"final_blended_projection": 5.0, "actual_strikeouts": 5},
    ]
    m = compute_error_metrics(rows)
    assert m["n"] == 1


def test_directional_accuracy_basic():
    rows = [
        {"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": "OVER"},
        {"strikeout_line": 5.5, "actual_strikeouts": 5, "recommended_side": "OVER"},
        {"strikeout_line": 5.5, "actual_strikeouts": 5, "recommended_side": "UNDER"},
        {"strikeout_line": 5.5, "actual_strikeouts": 4, "recommended_side": "PASS"},
    ]
    d = compute_directional_metrics(rows)
    assert d["n"] == 4
    assert d["projected_pass"] == 1
    assert abs(d["recommendation_win_rate"] - (2 / 3 * 100)) < 0.5
    assert d["over_results"]["wins"] == 1
    assert d["over_results"]["losses"] == 1
    assert d["under_results"]["wins"] == 1


def test_directional_pass_never_counted_as_bet():
    rows = [{"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": "PASS"}]
    d = compute_directional_metrics(rows)
    assert d["over_results"]["wins"] == 0
    assert d["under_results"]["wins"] == 0
    assert d["projected_pass"] == 1


def test_directional_no_line_returns_zero_n():
    rows = [{"strikeout_line": None, "actual_strikeouts": 6, "recommended_side": "OVER"}]
    d = compute_directional_metrics(rows)
    assert d["n"] == 0


def test_directional_push_handled():
    rows = [{"strikeout_line": 6.0, "actual_strikeouts": 6, "recommended_side": "OVER"}]
    d = compute_directional_metrics(rows)
    assert d["over_results"]["pushes"] == 1
    assert d["over_results"]["wins"] == 0
    assert d["over_results"]["losses"] == 0


def test_calibration_bucket_counts_and_gap():
    rows = []
    for i in range(20):
        over = i < 11
        rows.append({"model_over_probability": 0.52, "strikeout_line": 5.5, "actual_strikeouts": 6 if over else 5})
    buckets = compute_calibration(rows)
    bucket = next(b for b in buckets if b["bucket"] == "50-55%")
    assert bucket["n"] == 20
    assert bucket["reliable"] is True
    assert abs(bucket["actual_over_rate"] - 55.0) < 0.1
    assert abs(bucket["calibration_gap"] - 3.0) < 0.1


def test_calibration_small_bucket_flagged_unreliable():
    rows = [{"model_over_probability": 0.72, "strikeout_line": 5.5, "actual_strikeouts": 6}]
    buckets = compute_calibration(rows)
    bucket = next(b for b in buckets if b["bucket"] == "70%+")
    assert bucket["n"] == 1
    assert bucket["reliable"] is False


def test_calibration_empty_bucket_shows_zero_not_crash():
    buckets = compute_calibration([])
    assert all(b["n"] == 0 for b in buckets)
    assert len(buckets) == 7


def test_calibration_brier_score_perfect_predictions():
    rows = [{"model_over_probability": 0.99, "strikeout_line": 5.5, "actual_strikeouts": 6} for _ in range(10)]
    buckets = compute_calibration(rows)
    bucket = next(b for b in buckets if b["bucket"] == "70%+")
    assert bucket["brier"] < 0.01


def test_bias_by_group():
    rows = [
        {"pitcher_name": "A", "final_blended_projection": 5.0, "actual_strikeouts": 6},
        {"pitcher_name": "A", "final_blended_projection": 5.0, "actual_strikeouts": 4},
        {"pitcher_name": "B", "final_blended_projection": 6.0, "actual_strikeouts": 6},
    ]
    groups = compute_bias_by_group(rows, "pitcher_name")
    a = next(g for g in groups if g["group"] == "A")
    assert abs(a["avg_bias"]) < 0.01
    assert a["n"] == 2


def test_line_bucket_boundaries():
    assert line_bucket_for(3.5) == "<4.0"
    assert line_bucket_for(4.5) == "4.0-4.5"
    assert line_bucket_for(5.5) == "5.0-5.5"
    assert line_bucket_for(6.5) == "6.0-6.5"
    assert line_bucket_for(7.5) == "7.0-7.5"
    assert line_bucket_for(8.5) == "8.0+"
    assert line_bucket_for(None) is None


def test_workload_error_decomposition_dylan_cease_example():
    row = {
        "pitcher_name": "Dylan Cease",
        "final_blended_projection": 7.30,
        "actual_strikeouts": 10,
        "expected_batters_faced": 24.1,
        "actual_batters_faced": 24,
    }
    d = decompose_error(row)
    assert abs(d["total_miss"] - 2.70) < 0.01
    assert abs(d["workload_contribution"]) < 0.1
    assert abs(d["rate_contribution"] - 2.70) < 0.1


def test_k_rate_error_decomposition_dominant_case():
    row = {
        "final_blended_projection": 4.0,
        "actual_strikeouts": 6.0,
        "expected_batters_faced": 20,
        "actual_batters_faced": 30,
    }
    d = decompose_error(row)
    assert abs(d["rate_contribution"]) < 0.1
    assert abs(d["workload_contribution"] - 2.0) < 0.2


def test_component_reconciliation():
    test_cases = [
        {"final_blended_projection": 7.30, "actual_strikeouts": 10, "expected_batters_faced": 24.1, "actual_batters_faced": 24},
        {"final_blended_projection": 5.0, "actual_strikeouts": 3, "expected_batters_faced": 22.0, "actual_batters_faced": 15},
        {"final_blended_projection": 6.5, "actual_strikeouts": 6, "expected_batters_faced": 25.0, "actual_batters_faced": 25},
        {"final_blended_projection": 4.2, "actual_strikeouts": 9, "expected_batters_faced": 18.5, "actual_batters_faced": 27},
    ]
    for row in test_cases:
        d = decompose_error(row)
        assert abs(d["reconciliation_error"]) < 0.5, f"Reconciliation failed for {row}: {d}"


def test_decompose_error_missing_data_returns_none():
    assert decompose_error({"final_blended_projection": 5.0, "actual_strikeouts": None,
                             "expected_batters_faced": 20, "actual_batters_faced": 20}) is None
    assert decompose_error({"final_blended_projection": 5.0, "actual_strikeouts": 5,
                             "expected_batters_faced": 0, "actual_batters_faced": 20}) is None


def test_summarize_error_decomposition_biggest_misses():
    rows = [
        {"pitcher_name": "Big Under", "final_blended_projection": 4.0, "actual_strikeouts": 10, "expected_batters_faced": 20, "actual_batters_faced": 20},
        {"pitcher_name": "Small Miss", "final_blended_projection": 5.0, "actual_strikeouts": 5.5, "expected_batters_faced": 20, "actual_batters_faced": 20},
        {"pitcher_name": "Big Over", "final_blended_projection": 9.0, "actual_strikeouts": 2, "expected_batters_faced": 20, "actual_batters_faced": 20},
    ]
    summary = summarize_error_decomposition(rows)
    assert summary["n"] == 3
    assert summary["biggest_underprojections"][0]["pitcher_name"] == "Big Under"
    assert summary["biggest_overprojections"][0]["pitcher_name"] == "Big Over"


def test_summarize_error_decomposition_empty_no_crash():
    summary = summarize_error_decomposition([])
    assert summary["n"] == 0
    assert summary["biggest_underprojections"] == []


# --- Problem 3: NULL recommendation must never become PASS ---

def test_null_recommendation_not_counted_as_pass():
    """Reproduces the exact reported bug: 27 legacy rows with
    recommended_side=None must show as 0 Over / 0 Under / 0 PASS (with 27
    unrecorded), not '27 PASS'."""
    rows = [{"strikeout_line": 5.5, "actual_strikeouts": 5 + (i % 3), "recommended_side": None} for i in range(27)]
    d = compute_directional_metrics(rows)
    assert d["projected_pass"] == 0
    assert d["n_unknown_recommendation"] == 27
    assert d["projected_over_pct"] is None


def test_genuine_pass_still_counts_as_pass():
    rows = [
        {"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": "PASS"},
        {"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": "PASS"},
        {"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": None},
        {"strikeout_line": 5.5, "actual_strikeouts": 7, "recommended_side": "OVER"},
    ]
    d = compute_directional_metrics(rows)
    assert d["projected_pass"] == 2
    assert d["n_unknown_recommendation"] == 1
    assert d["projected_over"] == 1
    # Percentages are relative to the KNOWN subset (3), not all 4 rows.
    assert abs(d["projected_pass_pct"] - (2 / 3 * 100)) < 0.1


def test_unknown_recommendation_excluded_from_win_rate():
    rows = [{"strikeout_line": 5.5, "actual_strikeouts": 6, "recommended_side": None} for _ in range(10)]
    d = compute_directional_metrics(rows)
    assert d["recommendation_win_rate"] is None
    assert d["over_results"]["n"] == 0
    assert d["under_results"]["n"] == 0


# --- Problem 4: calibration only uses rows with a real stored probability ---

def test_calibration_excludes_rows_missing_probability():
    rows = [
        {"model_over_probability": 0.55, "strikeout_line": 5.5, "actual_strikeouts": 6},
        {"model_over_probability": None, "strikeout_line": 5.5, "actual_strikeouts": 6},  # legacy row, no stored probability
    ]
    buckets = compute_calibration(rows)
    total_n = sum(b["n"] for b in buckets)
    assert total_n == 1  # only the row with a real probability is counted anywhere


def test_calibration_never_substitutes_zero_for_missing_probability():
    """A missing probability must be excluded entirely, never treated as
    0.0 (which would incorrectly land it in the lowest bucket)."""
    rows = [{"model_over_probability": None, "strikeout_line": 5.5, "actual_strikeouts": 6}]
    buckets = compute_calibration(rows)
    assert all(b["n"] == 0 for b in buckets)


# --- Bug 1: bias sign must be consistent (projection - actual) everywhere ---

def test_bias_by_group_dylan_cease_exact_example():
    """The exact reported case: projected 7.30, actual 10. Documented
    convention (final projection minus actual) gives -2.70. The old code
    used actual-proj and displayed +2.70 instead."""
    rows = [{"pitcher_name": "Dylan Cease", "final_blended_projection": 7.30, "actual_strikeouts": 10}]
    groups = compute_bias_by_group(rows, "pitcher_name")
    assert abs(groups[0]["avg_bias"] - (-2.70)) < 0.01


def test_bias_by_group_miles_mikolas_exact_example():
    """The exact reported case: projected 3.63, actual 1. Correct bias is
    +2.63. The old code displayed -2.63."""
    rows = [{"pitcher_name": "Miles Mikolas", "final_blended_projection": 3.63, "actual_strikeouts": 1}]
    groups = compute_bias_by_group(rows, "pitcher_name")
    assert abs(groups[0]["avg_bias"] - 2.63) < 0.01


def test_bias_by_group_sign_matches_overall_core_metrics_convention():
    """Bias by any group must use the exact same sign convention as the
    overall Core Projection Accuracy bias (compute_error_metrics), for
    every group category the report displays."""
    rows = [
        {"pitcher_name": "A", "opponent_team": "NYY", "line_bucket": "5.0-5.5",
         "betting_confidence": "HIGH", "edge_grade": "Moderate estimated edge",
         "workload_source": "mlb_season_totals", "workload_role": "starter",
         "lineup_status": "confirmed", "final_blended_projection": 7.30, "actual_strikeouts": 10},
    ]
    overall = compute_error_metrics(rows)
    for group_field in ("pitcher_name", "opponent_team", "line_bucket", "betting_confidence",
                         "edge_grade", "workload_source", "workload_role", "lineup_status"):
        groups = compute_bias_by_group(rows, group_field)
        assert abs(groups[0]["avg_bias"] - overall["bias"]) < 0.01, f"{group_field} bias sign mismatch"


def test_error_decomposition_miss_convention_unchanged():
    """The PROJECTION ERROR REVIEW section intentionally keeps actual -
    projected under the label 'miss', which is a different metric and
    must NOT be affected by the bias sign fix."""
    row = {"pitcher_name": "Dylan Cease", "final_blended_projection": 7.30, "actual_strikeouts": 10,
           "expected_batters_faced": 24.0, "actual_batters_faced": 24}
    d = decompose_error(row)
    assert abs(d["total_miss"] - 2.70) < 0.01  # actual - projected, unchanged
