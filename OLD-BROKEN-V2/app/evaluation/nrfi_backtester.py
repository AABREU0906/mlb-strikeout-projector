"""
`python main.py nrfi-backtest --start-date ... --end-date ...`

Reuses app.evaluation.metrics (brier_score, log_loss, calibration_buckets)
for the probabilistic side, and adds NRFI-specific confusion-matrix /
precision-recall since those are classification-specific and don't exist
in the strikeout model's regression-oriented metrics module.
"""
from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from app.database.repositories import NrfiProjectionRepository
from app.database.session import session_scope
from app.evaluation.metrics import brier_score, calibration_buckets, log_loss

console = Console()


def _confusion_and_prf(preds_nrfi: list[bool], actual_nrfi: list[bool]) -> dict:
    tp = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if p and a)
    fp = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if p and not a)
    fn = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if not p and a)
    tn = sum(1 for p, a in zip(preds_nrfi, actual_nrfi) if not p and not a)

    nrfi_precision = tp / (tp + fp) if (tp + fp) else None
    nrfi_recall = tp / (tp + fn) if (tp + fn) else None
    yrfi_precision = tn / (tn + fn) if (tn + fn) else None
    yrfi_recall = tn / (tn + fp) if (tn + fp) else None
    accuracy = (tp + tn) / len(preds_nrfi) if preds_nrfi else None

    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "nrfi_precision": round(nrfi_precision, 4) if nrfi_precision is not None else None,
        "nrfi_recall": round(nrfi_recall, 4) if nrfi_recall is not None else None,
        "yrfi_precision": round(yrfi_precision, 4) if yrfi_precision is not None else None,
        "yrfi_recall": round(yrfi_recall, 4) if yrfi_recall is not None else None,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
    }


def run_nrfi_backtest(start_date: str, end_date: str) -> dict:
    with session_scope() as session:
        graded = NrfiProjectionRepository.list_all_with_results(session, since_date=start_date)
        projections = [p for p in graded if start_date <= p.game_date <= end_date]

    if not projections:
        console.print(f"[yellow]No graded NRFI/YRFI projections found between {start_date} and {end_date}.[/yellow]")
        return {}

    probs, outcomes, preds_nrfi, actual_nrfi = [], [], [], []
    by_month = defaultdict(list)
    by_lineup_status = defaultdict(list)
    league_average_correct = 0

    for p in projections:
        if p.nrfi_probability is None or p.actual_result is None or p.actual_result.is_nrfi is None:
            continue
        was_nrfi = p.actual_result.is_nrfi
        probs.append(p.nrfi_probability)
        outcomes.append(1 if was_nrfi else 0)
        predicted_nrfi = p.nrfi_probability >= 0.5
        preds_nrfi.append(predicted_nrfi)
        actual_nrfi.append(was_nrfi)

        correct = int(predicted_nrfi == was_nrfi)
        by_month[p.game_date[:7]].append(correct)
        by_lineup_status[p.lineup_status].append(correct)

        if was_nrfi:
            league_average_correct += 1

    n = len(probs)
    prf = _confusion_and_prf(preds_nrfi, actual_nrfi)

    report = {
        "n": n,
        "accuracy": prf["accuracy"],
        "brier_score": round(brier_score(probs, outcomes), 4) if probs else None,
        "log_loss": round(log_loss(probs, outcomes), 4) if probs else None,
        "calibration": calibration_buckets(probs, outcomes),
        "confusion_matrix": prf["confusion_matrix"],
        "nrfi_precision": prf["nrfi_precision"],
        "nrfi_recall": prf["nrfi_recall"],
        "yrfi_precision": prf["yrfi_precision"],
        "yrfi_recall": prf["yrfi_recall"],
        "by_month": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in sorted(by_month.items())},
        "by_lineup_status": {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)} for k, v in by_lineup_status.items()},
        "league_average_baseline_accuracy": round(league_average_correct / n, 4) if n else None,
    }

    _print_report(start_date, end_date, report)
    return report


def _print_report(start_date: str, end_date: str, report: dict) -> None:
    console.print(f"[bold]NRFI/YRFI Backtest {start_date} to {end_date}[/bold] -- {report['n']} graded projection(s)")
    console.print(
        f"Accuracy: {report['accuracy']} | Brier: {report['brier_score']} | Log Loss: {report['log_loss']}"
    )
    console.print(
        f"Model accuracy vs. always-predict-NRFI baseline: "
        f"{report['accuracy']} vs {report['league_average_baseline_accuracy']}"
    )

    cm = report["confusion_matrix"]
    t = Table(title="Confusion Matrix (rows=predicted, cols=actual)")
    t.add_column("")
    t.add_column("Actual NRFI", justify="right")
    t.add_column("Actual YRFI", justify="right")
    t.add_row("Predicted NRFI", str(cm["tp"]), str(cm["fp"]))
    t.add_row("Predicted YRFI", str(cm["fn"]), str(cm["tn"]))
    console.print(t)

    console.print(
        f"NRFI precision/recall: {report['nrfi_precision']}/{report['nrfi_recall']} | "
        f"YRFI precision/recall: {report['yrfi_precision']}/{report['yrfi_recall']}"
    )

    for label, key in [("Month", "by_month"), ("Lineup Status", "by_lineup_status")]:
        sub = report.get(key, {})
        if not sub:
            continue
        t2 = Table(title=f"Accuracy by {label}")
        t2.add_column(label)
        t2.add_column("N", justify="right")
        t2.add_column("Accuracy", justify="right")
        for k, v in sub.items():
            t2.add_row(str(k), str(v["n"]), str(v["accuracy"]))
        console.print(t2)
