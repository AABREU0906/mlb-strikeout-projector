import types

import pytest

from app.features.probability_math import combine_independent_no_score_probabilities, log5


def test_log5_league_average_identity():
    assert abs(log5(0.224, 0.224, 0.224) - 0.224) < 1e-6


def test_log5_not_simple_average():
    p, b, lg = 0.30, 0.32, 0.224
    naive_avg = (p + b) / 2
    assert abs(log5(p, b, lg) - naive_avg) > 0.01


def test_combine_nrfi_yrfi_sums_to_one():
    nrfi, yrfi = combine_independent_no_score_probabilities(0.75, 0.70)
    assert abs(nrfi + yrfi - 1.0) < 1e-9


def test_combine_is_multiplicative_not_average():
    nrfi, _ = combine_independent_no_score_probabilities(0.75, 0.70)
    naive_avg = (0.75 + 0.70) / 2
    assert abs(nrfi - naive_avg) > 0.05
    assert abs(nrfi - 0.525) < 1e-9


def test_combine_edge_cases():
    n1, y1 = combine_independent_no_score_probabilities(1.0, 1.0)
    assert n1 == 1.0 and y1 == 0.0
    n2, y2 = combine_independent_no_score_probabilities(0.0, 0.9)
    assert n2 == 0.0 and y2 == 1.0


def _team(scoring_rate):
    return types.SimpleNamespace(season_scoring_rate=types.SimpleNamespace(shrunk_rate=scoring_rate))


def _pitcher(scoreless_rate):
    return types.SimpleNamespace(season_scoreless_rate=types.SimpleNamespace(shrunk_rate=scoreless_rate))


@pytest.fixture
def half_inning_model():
    from app.projections.nrfi_half_inning_model import compute_half_inning_scoring_probability
    return compute_half_inning_scoring_probability


def test_half_inning_average_matchup_returns_league_rate(half_inning_model):
    result = half_inning_model(_team(0.285), _pitcher(0.715), league_scoring_rate=0.285)
    assert abs(result.scoring_probability - 0.285) < 0.01


def test_half_inning_weak_team_vs_elite_pitcher(half_inning_model):
    result = half_inning_model(_team(0.15), _pitcher(0.85), league_scoring_rate=0.285)
    assert result.scoring_probability < 0.15


def test_half_inning_strong_team_vs_poor_pitcher(half_inning_model):
    result = half_inning_model(_team(0.42), _pitcher(0.55), league_scoring_rate=0.285)
    assert result.scoring_probability > 0.35


def test_half_inning_park_weather_adjustments_capped(half_inning_model):
    result = half_inning_model(
        _team(0.285), _pitcher(0.715), league_scoring_rate=0.285,
        ballpark_run_factor=1.5, weather_run_multiplier=1.5,
    )
    assert result.adjustments_applied["ballpark"] <= 1.08 + 1e-9
    assert result.adjustments_applied["weather"] <= 1.05 + 1e-9


def test_full_nrfi_game_matches_independent_math():
    from app.projections.nrfi_half_inning_model import compute_nrfi_probability

    game = compute_nrfi_probability(
        away_offense=_team(0.285), home_pitcher=_pitcher(0.715),
        home_offense=_team(0.285), away_pitcher=_pitcher(0.715),
        league_scoring_rate=0.285,
    )
    expected_nrfi = (1 - 0.285) * (1 - 0.285)
    assert abs(game.nrfi_probability - expected_nrfi) < 0.01
    assert abs(game.nrfi_probability + game.yrfi_probability - 1.0) < 1e-9


def test_two_elite_pitchers_produce_high_nrfi_probability():
    from app.projections.nrfi_half_inning_model import compute_nrfi_probability

    game = compute_nrfi_probability(
        away_offense=_team(0.15), home_pitcher=_pitcher(0.85),
        home_offense=_team(0.15), away_pitcher=_pitcher(0.85),
        league_scoring_rate=0.285,
    )
    assert game.nrfi_probability > 0.70
