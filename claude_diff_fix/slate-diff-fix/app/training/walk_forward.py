"""
Walk-forward validation: strictly time-ordered expanding-window splits so a
model is never validated on data that precedes its training window in a way
that would leak future information. This is the only validation scheme used
for model promotion decisions (never a random shuffle-split, which would
leak across time for a sports time series).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import PoissonRegressor

from app.evaluation.metrics import mae, medae, rmse


@dataclass
class FoldResult:
    train_n: int
    val_n: int
    mae: float
    rmse: float
    medae: float


@dataclass
class WalkForwardReport:
    folds: list[FoldResult]
    overall_mae: float
    overall_rmse: float
    overall_medae: float
    n_folds: int


def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    n_folds: int = 5,
    min_train_size: int = 40,
) -> WalkForwardReport:
    """Assumes X, y, dates are already sorted ascending by date. Splits the
    remaining (post min_train_size) data into n_folds expanding-window
    folds: fold i trains on [0, split_i) and validates on [split_i, split_{i+1})."""
    n = len(y)
    if n < min_train_size + n_folds:
        raise ValueError(
            f"Not enough graded observations ({n}) for walk-forward validation "
            f"with min_train_size={min_train_size} and n_folds={n_folds}."
        )

    remaining = n - min_train_size
    fold_size = max(remaining // n_folds, 1)

    folds: list[FoldResult] = []
    all_errors: list[float] = []

    split = min_train_size
    for _ in range(n_folds):
        val_end = min(split + fold_size, n)
        if val_end <= split:
            break

        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:val_end], y[split:val_end]

        model = PoissonRegressor(alpha=1.0, max_iter=500)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        errors = list(preds - y_val)
        all_errors.extend(errors)

        folds.append(
            FoldResult(
                train_n=len(y_train),
                val_n=len(y_val),
                mae=round(mae(errors), 4),
                rmse=round(rmse(errors), 4),
                medae=round(medae(errors), 4),
            )
        )
        split = val_end

    return WalkForwardReport(
        folds=folds,
        overall_mae=round(mae(all_errors), 4) if all_errors else float("inf"),
        overall_rmse=round(rmse(all_errors), 4) if all_errors else float("inf"),
        overall_medae=round(medae(all_errors), 4) if all_errors else float("inf"),
        n_folds=len(folds),
    )
