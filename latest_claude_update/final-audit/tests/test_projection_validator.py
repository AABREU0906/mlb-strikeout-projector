"""
Tests for the central projection validator -- the gate that enforces
"PASS or VALIDATION FAILED over a confident but unreliable recommendation."
"""
from app.validation.projection_validator import validate_projection

VALID_PROBS = {
    0: 0.02, 1: 0.06, 2: 0.12, 3: 0.18, 4: 0.20, 5: 0.18, 6: 0.12, 7: 0.07,
    8: 0.03, 9: 0.01, 10: 0.005, 11: 0.002, 12: 0.001, 13: 0.001, 14: 0.0005, 15: 0.0005,
}
VALID_MEAN = round(sum(k * p for k, p in VALID_PROBS.items()), 2)


def _valid_kwargs(**overrides):
    kwargs = dict(
        expected_innings=5.8,
        expected_batters_faced=24.0,
        expected_pitch_count=95.0,
        final_projection=VALID_MEAN,
        probability_by_k=dict(VALID_PROBS),
        percentiles={10: 2, 25: 3, 50: 4, 75: 6, 90: 7},
        std_dev=2.0,
        prob_complete_5=0.75, prob_complete_6=0.55, prob_complete_7=0.30, prob_early_exit=0.15,
        lineup_confirmed=True, pitcher_confirmed=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_projection_passes():
    report = validate_projection(**_valid_kwargs())
    assert report.is_valid
    assert not report.critical_issues


def test_reported_bug_numbers_fail_validation():
    report = validate_projection(**_valid_kwargs(
        expected_innings=27.9, expected_batters_faced=126, expected_pitch_count=531,
    ))
    assert not report.is_valid
    assert any(i.code == "invalid_expected_innings" for i in report.critical_issues)


def test_zero_innings_is_invalid():
    report = validate_projection(**_valid_kwargs(expected_innings=0.0))
    assert not report.is_valid


def test_negative_innings_is_invalid():
    report = validate_projection(**_valid_kwargs(expected_innings=-1.0))
    assert not report.is_valid


def test_nine_point_one_innings_is_invalid():
    report = validate_projection(**_valid_kwargs(expected_innings=9.1))
    assert not report.is_valid


def test_exactly_nine_innings_is_valid_boundary():
    report = validate_projection(**_valid_kwargs(expected_innings=9.0, expected_batters_faced=40.0, expected_pitch_count=160.0))
    assert not any(i.code == "invalid_expected_innings" for i in report.critical_issues)


def test_batters_faced_wildly_inconsistent_with_innings_fails():
    report = validate_projection(**_valid_kwargs(expected_innings=6.0, expected_batters_faced=126.0))
    assert not report.is_valid
    assert any(i.code == "inconsistent_batters_faced" for i in report.critical_issues)


def test_pitch_count_wildly_inconsistent_with_batters_faced_fails():
    report = validate_projection(**_valid_kwargs(expected_batters_faced=24.0, expected_pitch_count=531.0))
    assert not report.is_valid
    assert any(i.code == "inconsistent_pitch_count" for i in report.critical_issues)


def test_distribution_not_summing_to_one_fails():
    report = validate_projection(**_valid_kwargs(probability_by_k={0: 0.3, 1: 0.3, 2: 0.3}))
    assert not report.is_valid
    assert any(i.code == "distribution_does_not_sum_to_one" for i in report.critical_issues)


def test_probability_out_of_bounds_fails():
    report = validate_projection(**_valid_kwargs(probability_by_k={0: 1.5, 1: -0.5}))
    assert not report.is_valid
    assert any(i.code == "probability_out_of_bounds" for i in report.critical_issues)


def test_empty_distribution_fails():
    report = validate_projection(**_valid_kwargs(probability_by_k={}))
    assert not report.is_valid
    assert any(i.code == "empty_distribution" for i in report.critical_issues)


def test_percentiles_out_of_order_fails():
    report = validate_projection(**_valid_kwargs(percentiles={10: 8, 25: 6, 50: 4, 75: 7, 90: 9}))
    assert not report.is_valid
    assert any(i.code == "percentiles_out_of_order" for i in report.critical_issues)


def test_incomplete_percentiles_does_not_fail():
    report = validate_projection(**_valid_kwargs(percentiles={10: 2, 50: 4}))
    assert not any(i.code == "percentiles_out_of_order" for i in report.critical_issues)


def test_negative_std_dev_fails():
    report = validate_projection(**_valid_kwargs(std_dev=-1.0))
    assert not report.is_valid
    assert any(i.code == "invalid_std_dev" for i in report.critical_issues)


def test_infinite_std_dev_fails():
    report = validate_projection(**_valid_kwargs(std_dev=float("inf")))
    assert not report.is_valid


def test_projection_inconsistent_with_distribution_mean_fails():
    report = validate_projection(**_valid_kwargs(final_projection=VALID_MEAN + 10))
    assert not report.is_valid
    assert any(i.code == "projection_inconsistent_with_distribution" for i in report.critical_issues)


def test_projection_exceeding_batters_faced_fails():
    report = validate_projection(**_valid_kwargs(
        final_projection=30.0,
        probability_by_k={30: 1.0},
        expected_batters_faced=24.0,
    ))
    assert not report.is_valid
    assert any(i.code == "projection_exceeds_batters_faced" for i in report.critical_issues)


def test_completion_probability_6_exceeding_5_fails():
    report = validate_projection(**_valid_kwargs(prob_complete_5=0.40, prob_complete_6=0.60))
    assert not report.is_valid
    assert any(i.code == "workload_probability_ordering" for i in report.critical_issues)


def test_completion_probability_7_exceeding_6_fails():
    report = validate_projection(**_valid_kwargs(prob_complete_6=0.30, prob_complete_7=0.50))
    assert not report.is_valid


def test_correct_completion_probability_ordering_passes():
    report = validate_projection(**_valid_kwargs(prob_complete_5=0.75, prob_complete_6=0.55, prob_complete_7=0.30))
    assert not any(i.code == "workload_probability_ordering" for i in report.critical_issues)


def test_over_under_summing_correctly_passes():
    report = validate_projection(**_valid_kwargs(over_probability=0.48, under_probability=0.52))
    assert not any(i.code == "over_under_do_not_sum_to_one" for i in report.critical_issues)


def test_over_under_not_summing_to_one_fails():
    report = validate_projection(**_valid_kwargs(over_probability=0.60, under_probability=0.60))
    assert not report.is_valid
    assert any(i.code == "over_under_do_not_sum_to_one" for i in report.critical_issues)


def test_whole_number_line_push_probability_accounted_for():
    report = validate_projection(**_valid_kwargs(over_probability=0.40, under_probability=0.45, push_probability=0.15))
    assert not any(i.code == "over_under_do_not_sum_to_one" for i in report.critical_issues)


def test_extreme_probability_warns_but_does_not_fail():
    report = validate_projection(**_valid_kwargs(over_probability=0.99, under_probability=0.01))
    assert report.is_valid
    assert any(i.code == "extreme_probability" for i in report.warning_issues)


def test_zero_percent_probability_boundary_warns():
    report = validate_projection(**_valid_kwargs(over_probability=0.0, under_probability=1.0))
    assert report.is_valid
    assert any(i.code == "extreme_probability" for i in report.warning_issues)


def test_hundred_percent_probability_boundary_warns():
    report = validate_projection(**_valid_kwargs(over_probability=1.0, under_probability=0.0))
    assert any(i.code == "extreme_probability" for i in report.warning_issues)


def test_unconfirmed_pitcher_warns_not_fails():
    report = validate_projection(**_valid_kwargs(pitcher_confirmed=False))
    assert report.is_valid
    assert any(i.code == "pitcher_unconfirmed" for i in report.warning_issues)


def test_projected_lineup_warns_not_fails():
    report = validate_projection(**_valid_kwargs(lineup_confirmed=False))
    assert report.is_valid
    assert any(i.code == "lineup_projected" for i in report.warning_issues)


def test_stale_data_warns():
    report = validate_projection(**_valid_kwargs(data_age_minutes=200))
    assert report.is_valid
    assert any(i.code == "stale_data" for i in report.warning_issues)


def test_fresh_data_does_not_warn():
    report = validate_projection(**_valid_kwargs(data_age_minutes=10))
    assert not any(i.code == "stale_data" for i in report.warning_issues)


def test_workload_fallback_note_produces_warning():
    report = validate_projection(**_valid_kwargs(
        workload_fallback_used=True,
        workload_fallback_count=1,
        workload_all_metrics_fallback=False,
    ))
    assert report.is_valid
    assert any(i.code == "workload_fallback_used" for i in report.warning_issues)


def test_workload_all_metrics_fallback_produces_warning():
    report = validate_projection(**_valid_kwargs(
        workload_fallback_used=True,
        workload_fallback_count=3,
        workload_all_metrics_fallback=True,
    ))
    assert report.is_valid
    warning = next(i for i in report.warning_issues if i.code == "workload_fallback_used")
    assert "all three" in warning.message.lower()


def test_multiple_critical_issues_all_recorded():
    report = validate_projection(**_valid_kwargs(
        expected_innings=27.9, expected_batters_faced=126, expected_pitch_count=531,
        probability_by_k={0: 0.3, 1: 0.3},
    ))
    assert not report.is_valid
    assert len(report.critical_issues) >= 2
