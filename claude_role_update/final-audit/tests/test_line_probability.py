from app.markets.line_probability import compute_line_probabilities

PROBS = {
    0: 0.02, 1: 0.06, 2: 0.12, 3: 0.18, 4: 0.20, 5: 0.18, 6: 0.12, 7: 0.07,
    8: 0.03, 9: 0.01, 10: 0.005, 11: 0.002, 12: 0.001, 13: 0.001, 14: 0.0005, 15: 0.0005,
}


def test_half_point_line_has_no_push():
    result = compute_line_probabilities(PROBS, 4.5)
    assert result.push_probability == 0.0
    assert not result.is_whole_number_line


def test_half_point_line_sums_to_one():
    result = compute_line_probabilities(PROBS, 4.5)
    assert abs(result.total - 1.0) < 1e-9


def test_half_point_line_over_excludes_at_or_below():
    result = compute_line_probabilities(PROBS, 4.5)
    expected_over = sum(p for k, p in PROBS.items() if k > 4)
    assert abs(result.over_probability - expected_over) < 1e-9


def test_whole_number_line_detected():
    result = compute_line_probabilities(PROBS, 5.0)
    assert result.is_whole_number_line


def test_whole_number_line_push_equals_exact_probability():
    result = compute_line_probabilities(PROBS, 5.0)
    assert result.push_probability == PROBS[5]


def test_whole_number_line_under_excludes_push():
    result = compute_line_probabilities(PROBS, 5.0)
    expected_under_excluding_push = sum(p for k, p in PROBS.items() if k < 5)
    assert abs(result.under_probability - expected_under_excluding_push) < 1e-9
    buggy_under_would_have_been = 1.0 - result.over_probability
    assert abs(result.under_probability - buggy_under_would_have_been) > 1e-6


def test_whole_number_line_over_under_push_sum_to_one():
    result = compute_line_probabilities(PROBS, 5.0)
    assert abs(result.total - 1.0) < 1e-9


def test_whole_number_line_at_boundary_zero():
    result = compute_line_probabilities(PROBS, 0.0)
    assert result.is_whole_number_line
    assert result.under_probability == 0.0
    assert result.push_probability == PROBS[0]


def test_empty_distribution_returns_zeros_not_crash():
    result = compute_line_probabilities({}, 4.5)
    assert result.over_probability == 0.0
    assert result.under_probability == 0.0
