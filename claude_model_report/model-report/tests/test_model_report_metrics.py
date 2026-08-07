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
