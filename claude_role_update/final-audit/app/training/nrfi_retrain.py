"""
`python main.py nrfi-train`

Below settings.min_projections_for_ml_retrain graded NRFI/YRFI projections,
this declines to train and the transparent baseline (log5 half-inning
model) stays active -- same rule as the strikeout model, same reasoning
(don't train on a tiny personal dataset).

Above threshold: logistic regression (spec's stated preference for "the
first transparent version"), features built from FirstInningGameResult
history via the SAME leakage-safe repository queries the pitcher/team
feature builders already use, walk-forward validated (never a random
shuffle-split), and only promoted if it clears documented bars -- reusing
app.training.model_promotion's evaluate_promotion() and ModelVersion table
rather than building a parallel promotion system.
"""
from __future__ import annotations

import datetime as dt
import uuid

import joblib
import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression

from app.config.settings import PROJECT_ROOT, settings
from app.database.models import ModelVersion
from app.database.repositories import FirstInningGameResultRepository, ModelVersionRepository
from app.database.session import session_scope
from app.evaluation.metrics import brier_score, log_loss
from app.features.nrfi_league_constants import get_nrfi_league_average
from app.training.model_promotion import evaluate_promotion
from app.training.walk_forward import FoldResult, WalkForwardReport

console = Console()
MODEL_TYPE = "nrfi_logistic"


def _models_dir():
    d = PROJECT_ROOT / "saved_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_training_matrix(rows_sorted) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Leakage-safe by construction: each row's features come only from
    aggregate FirstInningGameResult history strictly BEFORE that row's own
    date, computed the same way the live pitcher/team builders do (a
    trailing rolling count as of that game), not from the row's own
    outcome."""
    feature_names = ["home_pitcher_scoreless_rate", "away_pitcher_scoreless_rate",
                      "home_team_scoring_rate", "away_team_scoring_rate"]
    X, y = [], []

    pitcher_history: dict[int, list[bool]] = {}
    team_history: dict[int, list[bool]] = {}
    league_scoreless = get_nrfi_league_average("league_scoreless_half_inning_rate")

    for row in rows_sorted:
        if row.is_nrfi is None:
            continue

        def _rate(history: list[bool], prior: float) -> float:
            if not history:
                return prior
            return sum(history) / len(history)

        home_p_hist = pitcher_history.get(row.home_starting_pitcher_id, [])
        away_p_hist = pitcher_history.get(row.away_starting_pitcher_id, [])
        home_t_hist = team_history.get(row.home_team_id, [])
        away_t_hist = team_history.get(row.away_team_id, [])

        feats = [
            _rate(home_p_hist, league_scoreless),
            _rate(away_p_hist, league_scoreless),
            _rate(home_t_hist, 1 - league_scoreless),
            _rate(away_t_hist, 1 - league_scoreless),
        ]
        X.append(feats)
        y.append(1 if row.is_nrfi else 0)

        if row.home_pitcher_scoreless_first is not None:
            pitcher_history.setdefault(row.home_starting_pitcher_id, []).append(row.home_pitcher_scoreless_first)
        if row.away_pitcher_scoreless_first is not None:
            pitcher_history.setdefault(row.away_starting_pitcher_id, []).append(row.away_pitcher_scoreless_first)
        if row.home_first_inning_runs is not None:
            team_history.setdefault(row.home_team_id, []).append(row.home_first_inning_runs > 0)
        if row.away_first_inning_runs is not None:
            team_history.setdefault(row.away_team_id, []).append(row.away_first_inning_runs > 0)

    return np.array(X), np.array(y), feature_names


def _classification_walk_forward(X, y, n_folds: int = 5, min_train_size: int = 100) -> WalkForwardReport:
    n = len(y)
    if n < min_train_size + n_folds:
        raise ValueError(f"Not enough graded observations ({n}) for walk-forward validation.")

    remaining = n - min_train_size
    fold_size = max(remaining // n_folds, 1)
    folds = []
    all_probs, all_y = [], []
    split = min_train_size
    for _ in range(n_folds):
        val_end = min(split + fold_size, n)
        if val_end <= split:
            break
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:val_end], y[split:val_end]
        if len(set(y_train)) < 2:
            split = val_end
            continue
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_val)[:, 1]
        all_probs.extend(probs.tolist())
        all_y.extend(y_val.tolist())
        fold_ll = log_loss(probs.tolist(), y_val.tolist())
        folds.append(FoldResult(train_n=len(y_train), val_n=len(y_val), mae=round(fold_ll, 4), rmse=0.0, medae=0.0))
        split = val_end

    overall_ll = log_loss(all_probs, all_y) if all_probs else float("inf")
    return WalkForwardReport(folds=folds, overall_mae=round(overall_ll, 4), overall_rmse=0.0, overall_medae=0.0, n_folds=len(folds))


def run_nrfi_retraining() -> None:
    with session_scope() as session:
        rows = FirstInningGameResultRepository.list_for_training(session)

    n = len(rows)
    console.print(f"[bold]Found {n} graded historical first-inning game(s).[/bold]")

    if n < settings.min_projections_for_ml_retrain:
        console.print(
            f"[yellow]Below the minimum ({settings.min_projections_for_ml_retrain}) required for ML retraining. "
            f"The transparent baseline (log5 half-inning model) remains active. "
            f"Run `nrfi-backfill` to accumulate more history.[/yellow]"
        )
        return

    rows_sorted = sorted(rows, key=lambda r: r.game_date)
    X, y, feature_names = _build_training_matrix(rows_sorted)

    if len(y) < 100:
        console.print(f"[yellow]Only {len(y)} usable rows after feature construction; skipping.[/yellow]")
        return

    try:
        report = _classification_walk_forward(X, y, n_folds=5, min_train_size=100)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    with session_scope() as session:
        current_active = next(
            (mv for mv in ModelVersionRepository.list_all(session) if mv.model_type == MODEL_TYPE and mv.is_active),
            None,
        )
        current_active_ll = (current_active.validation_metrics or {}).get("overall_mae") if current_active else None
        total_val_n = sum(f.val_n for f in report.folds)

        decision = evaluate_promotion(report, current_active_ll, total_val_n)

        final_model = LogisticRegression(max_iter=500)
        final_model.fit(X, y)
        final_probs = final_model.predict_proba(X)[:, 1]

        version_label = f"{MODEL_TYPE}-{dt.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
        artifact_path = _models_dir() / f"{version_label}.joblib"
        joblib.dump({"model": final_model, "feature_names": feature_names}, artifact_path)

        mv = ModelVersion(
            version_label=version_label,
            model_type=MODEL_TYPE,
            algorithm="LogisticRegression(max_iter=500)",
            training_window_start=rows_sorted[0].game_date,
            training_window_end=rows_sorted[-1].game_date,
            feature_list=feature_names,
            hyperparameters={"max_iter": 500},
            validation_metrics={
                "overall_mae": report.overall_mae,
                "n_folds": report.n_folds,
                "fold_details": [f.__dict__ for f in report.folds],
                "in_sample_brier": round(brier_score(final_probs.tolist(), y.tolist()), 4),
            },
            promoted=decision.promoted,
            promotion_decision_notes="; ".join(decision.reasons),
            artifact_path=str(artifact_path),
            is_active=decision.promoted,
            n_training_observations=len(y),
        )

        if decision.promoted:
            for existing in ModelVersionRepository.list_all(session):
                if existing.model_type == MODEL_TYPE:
                    existing.is_active = False

        ModelVersionRepository.save(session, mv)

    table = Table(title="NRFI/YRFI Retrain Result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Version", version_label)
    table.add_row("N observations", str(len(y)))
    table.add_row("Walk-forward log loss", str(report.overall_mae))
    table.add_row("Promoted", "YES" if decision.promoted else "NO")
    for reason in decision.reasons:
        table.add_row("Reason", reason)
    console.print(table)
