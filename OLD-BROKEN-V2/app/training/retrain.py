"""
`python main.py retrain`

1. Loads all completed (graded) historical projections.
2. Refuses to train an ML model below settings.min_projections_for_ml_retrain
   -- below that threshold, the transparent documented baseline (Stages
   1-5 formulas) remains the active approach, per project rules against
   training complex models on tiny personal datasets.
3. Above threshold: builds statistics-only and market-informed training
   matrices (leakage-safe: only pregame-snapshotted fields), sorts by
   game_date, walk-forward validates a PoissonRegressor candidate for each
   path, and applies the promotion rules.
4. Persists every trained candidate (promoted or not) as a versioned
   artifact with full metadata, so rollback and audit are always possible.
"""
from __future__ import annotations

import datetime as dt
import uuid

import joblib
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import PoissonRegressor

from app.config.logging_config import get_logger
from app.config.settings import settings
from app.database.models import ModelVersion
from app.database.repositories import ModelVersionRepository, ProjectionRepository
from app.database.session import session_scope
from app.training.feature_extraction import build_training_matrix
from app.training.model_promotion import evaluate_promotion
from app.training.walk_forward import walk_forward_validate

logger = get_logger(__name__)
console = Console()


def _models_dir():
    from app.config.settings import PROJECT_ROOT
    d = PROJECT_ROOT / "saved_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_retraining() -> None:
    with session_scope() as session:
        graded = ProjectionRepository.list_all_with_results(session)

    n = len(graded)
    console.print(f"[bold]Found {n} graded historical projection(s).[/bold]")

    if n < settings.min_projections_for_ml_retrain:
        console.print(
            f"[yellow]Below the minimum ({settings.min_projections_for_ml_retrain}) required for ML "
            f"retraining. The transparent baseline model (documented Stage 1-5 formulas) remains active. "
            f"Keep running `update-results` after games complete to accumulate graded history.[/yellow]"
        )
        return

    graded_sorted = sorted(graded, key=lambda p: p.game_date)

    _train_and_maybe_promote(graded_sorted, model_type="stats_ml", include_market=False)
    _train_and_maybe_promote(graded_sorted, model_type="market_ml", include_market=True)


def _train_and_maybe_promote(graded_sorted, model_type: str, include_market: bool) -> None:
    X, y, feature_names = build_training_matrix(graded_sorted, include_market=include_market)
    if len(y) < 40:
        console.print(
            f"[yellow]Not enough complete-feature observations ({len(y)}) for {model_type} after "
            f"filtering rows with missing fields; skipping this path.[/yellow]"
        )
        return

    try:
        report = walk_forward_validate(X, y, dates=[p.game_date for p in graded_sorted], n_folds=5, min_train_size=40)
    except ValueError as e:
        console.print(f"[yellow]{model_type}: {e}[/yellow]")
        return

    with session_scope() as session:
        current_active = None
        for mv in ModelVersionRepository.list_all(session):
            if mv.model_type == model_type and mv.is_active:
                current_active = mv
                break
        current_active_mae = (current_active.validation_metrics or {}).get("overall_mae") if current_active else None
        total_val_n = sum(f.val_n for f in report.folds)

        decision = evaluate_promotion(report, current_active_mae, total_val_n)

        # Train the final candidate model on ALL available data for
        # deployment (the walk-forward folds were for honest validation
        # only; the deployed artifact uses everything once validated).
        final_model = PoissonRegressor(alpha=1.0, max_iter=500)
        final_model.fit(X, y)

        version_label = f"{model_type}-{dt.date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
        artifact_path = _models_dir() / f"{version_label}.joblib"
        joblib.dump({"model": final_model, "feature_names": feature_names}, artifact_path)

        mv = ModelVersion(
            version_label=version_label,
            model_type=model_type,
            algorithm="PoissonRegressor(alpha=1.0)",
            training_window_start=graded_sorted[0].game_date,
            training_window_end=graded_sorted[-1].game_date,
            feature_list=feature_names,
            hyperparameters={"alpha": 1.0, "max_iter": 500},
            validation_metrics={
                "overall_mae": report.overall_mae,
                "overall_rmse": report.overall_rmse,
                "overall_medae": report.overall_medae,
                "n_folds": report.n_folds,
                "fold_details": [f.__dict__ for f in report.folds],
            },
            promoted=decision.promoted,
            promotion_decision_notes="; ".join(decision.reasons),
            artifact_path=str(artifact_path),
            is_active=decision.promoted,
            n_training_observations=len(y),
        )

        if decision.promoted:
            for existing in ModelVersionRepository.list_all(session):
                if existing.model_type == model_type:
                    existing.is_active = False

        ModelVersionRepository.save(session, mv)

    table = Table(title=f"Retrain result: {model_type}")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Version", version_label)
    table.add_row("N observations", str(len(y)))
    table.add_row("Walk-forward MAE", str(report.overall_mae))
    table.add_row("Promoted", "YES" if decision.promoted else "NO")
    for reason in decision.reasons:
        table.add_row("Reason", reason)
    console.print(table)
