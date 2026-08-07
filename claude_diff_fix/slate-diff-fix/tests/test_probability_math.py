"""
Tests for app/features/probability_math.py -- the log5 hardening and the
shared 4-tier rate-fallback resolver that fixes the Cristopher Sanchez
crash (TypeError: '>' not supported between instances of 'float' and
'NoneType', from a None reaching log5() with no input validation).
"""
import math

import pytest

from app.features.probability_math import is_valid_rate, log5, resolve_rate_with_fallback


def test_is_valid_rate_accepts_normal_values():
    assert is_valid_rate(0.25)
    assert is_valid_rate(0.5)


def test_is_valid_rate_accepts_zero_and_one_boundaries():
    """0.0 must never be treated as missing -- this is the exact class of
    truthiness bug the fix must avoid."""
    assert is_valid_rate(0.0)
    assert is_valid_rate(1.0)


def test_is_valid_rate_rejects_none():
    assert not is_valid_rate(None)


def test_is_valid_rate_rejects_nan():
    assert not is_valid_rate(float("nan"))


def test_is_valid_rate_rejects_infinity():
    assert not is_valid_rate(float("inf"))
    assert not is_valid_rate(float("-inf"))


def test_is_valid_rate_rejects_out_of_range():
    assert not is_valid_rate(-0.01)
    assert not is_valid_rate(1.01)


def test_is_valid_rate_rejects_non_numeric():
    assert not is_valid_rate("0.25")
    assert not is_valid_rate([0.25])
    assert not is_valid_rate({})


def test_is_valid_rate_rejects_bool():
    assert not is_valid_rate(True)
    assert not is_valid_rate(False)


def test_log5_raises_clear_value_error_on_none_pitcher_rate():
    with pytest.raises(ValueError, match="invalid pitcher or batter probability"):
        log5(None, 0.22, 0.22)


def test_log5_raises_clear_value_error_on_none_batter_rate():
    with pytest.raises(ValueError, match="invalid pitcher or batter probability"):
        log5(0.25, None, 0.22)


def test_log5_raises_on_none_league_rate():
    with pytest.raises(ValueError):
        log5(0.25, 0.22, None)


def test_log5_raises_on_nan():
    with pytest.raises(ValueError):
        log5(float("nan"), 0.22, 0.22)


def test_log5_raises_on_out_of_range():
    with pytest.raises(ValueError):
        log5(1.5, 0.22, 0.22)


def test_log5_accepts_zero_rate_without_crashing():
    result = log5(0.25, 0.0, 0.22)
    assert 0.0 <= result <= 1.0


def test_log5_valid_case_produces_expected_range():
    result = log5(0.25, 0.22, 0.222)
    assert 0.0 < result < 1.0


def test_log5_is_not_a_simple_average():
    result = log5(0.30, 0.30, 0.222)
    assert abs(result - 0.30) > 1e-6  # must differ meaningfully from a naive average


def test_log5_symmetric_at_league_average_returns_league_average():
    result = log5(0.222, 0.222, 0.222)
    assert abs(result - 0.222) < 1e-6


def test_resolve_rate_with_fallback_uses_split_when_valid():
    res = resolve_rate_with_fallback(
        [("split", 0.28), ("season", 0.24), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "split"
    assert not res.fallback_used
    assert res.value == 0.28


def test_resolve_rate_with_fallback_falls_to_season():
    res = resolve_rate_with_fallback(
        [("split", None), ("season", 0.24), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "season"
    assert res.fallback_used


def test_resolve_rate_with_fallback_falls_to_career():
    res = resolve_rate_with_fallback(
        [("split", None), ("season", None), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "career"
    assert res.fallback_used


def test_resolve_rate_with_fallback_falls_to_league_average():
    res = resolve_rate_with_fallback(
        [("split", None), ("season", None), ("career", None), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "league_average"
    assert res.fallback_used


def test_resolve_rate_with_fallback_zero_split_is_used_not_skipped():
    res = resolve_rate_with_fallback(
        [("split", 0.0), ("season", 0.24), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "split"
    assert res.value == 0.0
    assert not res.fallback_used


def test_resolve_rate_with_fallback_raises_when_nothing_valid():
    with pytest.raises(ValueError):
        resolve_rate_with_fallback(
            [("split", None), ("season", None), ("career", None), ("league_average", None)], "pitcher"
        )


def test_resolve_rate_with_fallback_produces_visible_note_on_fallback():
    res = resolve_rate_with_fallback(
        [("split", None), ("season", 0.24), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert len(res.notes) == 1
    assert "season" in res.notes[0]


def test_resolve_rate_with_fallback_no_note_when_split_available():
    res = resolve_rate_with_fallback(
        [("split", 0.28), ("season", 0.24), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.notes == []


def test_resolve_rate_with_fallback_rejects_invalid_intermediate_values():
    """NaN or out-of-range intermediate candidates must be skipped just
    like None, not accidentally accepted."""
    res = resolve_rate_with_fallback(
        [("split", float("nan")), ("season", 1.5), ("career", 0.23), ("league_average", 0.22)], "pitcher"
    )
    assert res.source == "career"
