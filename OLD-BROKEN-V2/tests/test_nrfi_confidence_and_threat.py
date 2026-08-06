import types

import pytest

from app.projections.nrfi_confidence import compute_nrfi_confidence


def test_confidence_perfect_data_scores_100():
    r = compute_nrfi_confidence(True, True, 0.9, 0.9, 0.9, 0.9, 0, True, False, False)
    assert r.score == 100.0


def test_confidence_bad_data_scores_low_and_floors_at_zero():
    r = compute_nrfi_confidence(
        False, False, 0.1, 0.1, 0.1, 0.1, 5, False, True, True,
        data_freshness_minutes=200, recent_calibration_penalty=1.0,
    )
    assert 0 <= r.score < 20


def test_confidence_structurally_independent_of_prediction():
    import inspect
    sig = inspect.signature(compute_nrfi_confidence)
    params = str(sig.parameters).lower()
    assert "probability" not in params
    assert "prediction" not in params


def test_confidence_unconfirmed_pitcher_reduces_score():
    confirmed = compute_nrfi_confidence(True, True, 0.9, 0.9, 0.9, 0.9, 0, True, False, False)
    unconfirmed = compute_nrfi_confidence(False, True, 0.9, 0.9, 0.9, 0.9, 0, True, False, False)
    assert unconfirmed.score < confirmed.score


@pytest.fixture
def threat_score_fn():
    from app.projections.nrfi_threat_score import compute_threat_score
    return compute_threat_score


def _run_threat(fn, slash, obp, slg, k, bb, hr):
    result = fn(slash, obp, slg, k, bb, hr)
    return result.score, result.components


def test_threat_score_league_average_is_fifty(threat_score_fn):
    league = types.SimpleNamespace(obp=0.318, slg=0.405, k_pct=0.225, bb_pct=0.085, hr_rate=0.028)
    score, _ = _run_threat(threat_score_fn, league, 0.318, 0.405, 0.225, 0.085, 0.028)
    assert abs(score - 50.0) < 0.5


def test_threat_score_elite_offense_scores_higher(threat_score_fn):
    elite = types.SimpleNamespace(obp=0.370, slg=0.480, k_pct=0.170, bb_pct=0.110, hr_rate=0.045)
    score, _ = _run_threat(threat_score_fn, elite, 0.318, 0.405, 0.225, 0.085, 0.028)
    assert score > 65


def test_threat_score_weak_offense_scores_lower(threat_score_fn):
    weak = types.SimpleNamespace(obp=0.270, slg=0.330, k_pct=0.290, bb_pct=0.055, hr_rate=0.015)
    score, _ = _run_threat(threat_score_fn, weak, 0.318, 0.405, 0.225, 0.085, 0.028)
    assert score < 35


def test_threat_score_extreme_inputs_stay_bounded(threat_score_fn):
    extreme = types.SimpleNamespace(obp=0.999, slg=0.999, k_pct=0.001, bb_pct=0.999, hr_rate=0.999)
    score, _ = _run_threat(threat_score_fn, extreme, 0.318, 0.405, 0.225, 0.085, 0.028)
    assert 0 < score < 100
