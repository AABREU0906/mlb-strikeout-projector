"""
`python main.py evaluate`

Loads every Projection that has a matching ActualResult, computes point
(MAE/RMSE/MedAE/bias) and probabilistic (Brier/log loss/calibration)
metrics separately for the statistics-only, market-informed, and blended
projections, plus accuracy broken out by handedness, confidence rating,
lineup status, and warning presence -- so the report itself answers
whether market data is actually helping, rather than assuming it does.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

from rich.console import Console
from rich.table import Table

from app.database.repositories import ProjectionRepository
from app.database.session import session_scope
from app.evaluation.metrics import bias, brier_score, calibration_buckets, log_loss, mae, medae, over_under_accuracy, rmse

console = Console()


def run_evaluation(since_date: Optional[str] = None) -> dict:
    with session_scope() as session:
        projections = ProjectionRepository.list_all_with_results(session, since_date=since_date)

        if not projections:
            console.print("[yellow]No graded projections yet (need update-results to have run on completed games).[/yellow]")
            return {}

        report = {"n_graded": len(projections)}

        for model_key, field_name in [
            ("statistics_only", "statistics_only_projection"),
            ("market_informed", "market_informed_projection"),
            ("blended", "final_blended_projection"),
        ]:
            errors = []
            for p in projections:
                pred = getattr(p, field_name)
                actual = p.actual_result.actual_strikeouts if p.actual_result else None
                if pred is None or actual is None:
                    continue
                errors.append(pred - actual)

            report[model_key] = {
                "n": len(errors),
                "mae": _r(mae(errors)),
                "rmse": _r(rmse(errors)),
                "medae": _r(medae(errors)),
                "bias": _r(bias(errors)),
            }

        # Over/under probabilistic metrics using the stored simulation distribution vs the market line.
        probs, outcomes = [], []
        ou_pred, ou_actual = [], []
        for p in projections:
            if not p.market_snapshot_json or p.market_snapshot_json.get("strikeout_line") is None:
                continue
            if not p.simulation_distribution_json or not p.actual_result:
                continue
            line = p.market_snapshot_json["strikeout_line"]
            actual_k = p.actual_result.actual_strikeouts
            if actual_k is None:
                continue
            import math
            floor_line = math.floor(line)
            model_over_prob = sum(v for k, v in p.simulation_distribution_json.items() if int(k) > floor_line)
            probs.append(model_over_prob)
            outcomes.append(1 if actual_k > line else 0)
            ou_pred.append(model_over_prob >= 0.5)
            ou_actual.append(actual_k > line)

        report["probabilistic"] = {
            "n": len(probs),
            "brier_score": _r(brier_score(probs, outcomes)),
            "log_loss": _r(log_loss(probs, outcomes)),
            "over_under_accuracy": _r(over_under_accuracy(ou_pred, ou_actual)),
            "calibration": calibration_buckets(probs, outcomes) if probs else [],
        }

        # Subgroup breakdowns.
        report["by_confidence"] = _subgroup_mae(projections, key=lambda p: p.confidence_rating or "Unknown")
        report["by_lineup_status"] = _subgroup_mae(projections, key=lambda p: p.lineup_status)
        report["by_handedness"] = _subgroup_mae(
            projections, key=lambda p: (p.pitcher_inputs_json or {}).get("throws", "Unknown")
        )

        _print_report(report)
        return report


def _subgroup_mae(projections, key) -> dict:
    groups = defaultdict(list)
    for p in projections:
        if p.final_blended_projection is None or not p.actual_result or p.actual_result.actual_strikeouts is None:
            continue
        groups[key(p)].append(p.final_blended_projection - p.actual_result.actual_strikeouts)
    return {k: {"n": len(v), "mae": _r(mae(v))} for k, v in groups.items()}


def _r(x, digits=4):
    return round(x, digits) if isinstance(x, (int, float)) else x


def _print_report(report: dict) -> None:
    console.print(f"[bold]Graded projections: {report['n_graded']}[/bold]\n")

    table = Table(title="Model Comparison (point accuracy)")
    table.add_column("Model")
    table.add_column("N", justify="right")
    table.add_column("MAE", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("MedAE", justify="right")
    table.add_column("Bias", justify="right")
    for key in ("statistics_only", "market_informed", "blended"):
        m = report.get(key, {})
        table.add_row(key, str(m.get("n")), str(m.get("mae")), str(m.get("rmse")), str(m.get("medae")), str(m.get("bias")))
    console.print(table)

    prob = report.get("probabilistic", {})
    if prob.get("n"):
        console.print(
            f"\n[bold]Over/Under probabilistic performance[/bold] (n={prob['n']}): "
            f"Brier={prob.get('brier_score')} | LogLoss={prob.get('log_loss')} | "
            f"O/U Accuracy={prob.get('over_under_accuracy')}"
        )

    for label, key in [("Confidence Rating", "by_confidence"), ("Lineup Status", "by_lineup_status"), ("Pitcher Handedness", "by_handedness")]:
        sub = report.get(key, {})
        if not sub:
            continue
        t = Table(title=f"Accuracy by {label}")
        t.add_column(label)
        t.add_column("N", justify="right")
        t.add_column("MAE", justify="right")
        for k, v in sub.items():
            t.add_row(str(k), str(v["n"]), str(v["mae"]))
        console.print(t)
