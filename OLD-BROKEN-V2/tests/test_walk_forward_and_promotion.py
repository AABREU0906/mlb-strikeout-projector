import numpy as np
import pytest

from app.training.model_promotion import (
    MIN_VALIDATION_OBSERVATIONS,
    evaluate_promotion,
)
from app.training.walk_forward import walk_forward_validate


def _synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=5.5, scale=1.0, size=(n, 3))
    y = np.clip(rng.poisson(lam=5.5, size=n), 0, 15).astype(float)
    dates = [f"2025-{1 + (i % 9):02d}-01" for i in range(n)]
    return X, y, dates


def test_walk_forward_raises_on_too_little_data():
    X, y, dates = _synthetic(n=20)
    with pytest.raises(ValueError):
        walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)


def test_walk_forward_produces_expected_fold_count():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    assert report.n_folds == 5
    assert report.overall_mae > 0


def test_walk_forward_folds_are_expanding_not_shrinking():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    train_sizes = [f.train_n for f in report.folds]
    assert train_sizes == sorted(train_sizes)  # strictly non-decreasing


def test_promotion_rejected_below_min_observations():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    decision = evaluate_promotion(report, current_active_mae=None, total_validation_n=MIN_VALIDATION_OBSERVATIONS - 1)
    assert decision.promoted is False


def test_promotion_accepted_as_first_model():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    total_n = sum(f.val_n for f in report.folds)
    decision = evaluate_promotion(report, current_active_mae=None, total_validation_n=total_n)
    assert decision.promoted is True


def test_promotion_rejected_when_worse_than_active():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    total_n = sum(f.val_n for f in report.folds)
    decision = evaluate_promotion(report, current_active_mae=report.overall_mae - 1.0, total_validation_n=total_n)
    assert decision.promoted is False


def test_promotion_accepted_when_strictly_better():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    total_n = sum(f.val_n for f in report.folds)
    decision = evaluate_promotion(report, current_active_mae=report.overall_mae + 1.0, total_validation_n=total_n)
    assert decision.promoted is True


def test_promotion_decision_always_has_reasons():
    X, y, dates = _synthetic(n=200)
    report = walk_forward_validate(X, y, dates, n_folds=5, min_train_size=40)
    decision = evaluate_promotion(report, current_active_mae=None, total_validation_n=5)
    assert len(decision.reasons) > 0
