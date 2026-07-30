import pytest

from app.markets.odds_math import (
    american_odds_to_implied_probability,
    classify_edge,
    expected_value_per_dollar,
    implied_probability_to_fair_american_odds,
    remove_vig_two_way,
)


def test_positive_odds_implied_probability():
    assert american_odds_to_implied_probability(150) == pytest.approx(0.4, abs=1e-6)


def test_negative_odds_implied_probability():
    assert american_odds_to_implied_probability(-150) == pytest.approx(0.6, abs=1e-6)


def test_even_money():
    assert american_odds_to_implied_probability(100) == pytest.approx(0.5, abs=1e-6)


def test_zero_odds_raises():
    with pytest.raises(ValueError):
        american_odds_to_implied_probability(0)


def test_vig_removal_sums_to_one():
    result = remove_vig_two_way(-115, -105)
    assert result.vig_free_over_prob + result.vig_free_under_prob == pytest.approx(1.0, abs=1e-9)


def test_vig_removal_overround_positive():
    result = remove_vig_two_way(-110, -110)
    assert result.overround > 0


def test_fair_odds_roundtrip():
    prob = 0.6
    odds = implied_probability_to_fair_american_odds(prob)
    recovered = american_odds_to_implied_probability(odds)
    assert recovered == pytest.approx(prob, abs=0.01)


def test_negative_and_positive_odds_supported_in_devig():
    result = remove_vig_two_way(120, -140)
    assert 0 < result.vig_free_over_prob < 1
    assert 0 < result.vig_free_under_prob < 1


def test_expected_value_breakeven_at_fair_odds():
    # At true fair odds for a given probability, EV should be ~0.
    prob = 0.55
    odds = implied_probability_to_fair_american_odds(prob)
    ev = expected_value_per_dollar(prob, odds)
    assert ev == pytest.approx(0.0, abs=0.02)


def test_classify_edge_labels():
    assert classify_edge(0.50, 0.505) == "No meaningful edge"
    assert classify_edge(0.50, 0.45) == "Small estimated edge"
    assert classify_edge(0.55, 0.45) == "Moderate estimated edge"
    assert classify_edge(0.70, 0.45) == "Large estimated edge with elevated uncertainty"


def test_classify_edge_never_promises_certainty():
    for label in [classify_edge(0.9, 0.1), classify_edge(0.5, 0.5)]:
        assert "guarantee" not in label.lower()
        assert "certain" not in label.lower()
        assert "safe" not in label.lower()
