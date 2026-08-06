from app.projections.stage2_batter_probability import _log5


def test_log5_returns_league_average_when_both_at_average():
    assert abs(_log5(0.224, 0.224, 0.224) - 0.224) < 1e-6


def test_log5_amplifies_favorable_matchup():
    result = _log5(0.30, 0.32, 0.224)
    assert result > 0.30
    assert result > 0.32


def test_log5_amplifies_unfavorable_matchup():
    result = _log5(0.15, 0.15, 0.224)
    assert result < 0.15


def test_log5_is_not_simple_average():
    # Explicit project requirement: log5 must not equal (p+b)/2
    p, b, lg = 0.30, 0.32, 0.224
    naive_avg = (p + b) / 2
    assert abs(_log5(p, b, lg) - naive_avg) > 0.01


def test_log5_bounded_between_zero_and_one():
    for p in (0.01, 0.5, 0.99):
        for b in (0.01, 0.5, 0.99):
            result = _log5(p, b, 0.224)
            assert 0.0 < result < 1.0
