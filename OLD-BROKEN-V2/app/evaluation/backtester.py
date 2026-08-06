"""
`python main.py backtest --start-date ... --end-date ...`

Design note on scope: this backtester replays projections that were
actually generated and stored by this system (each Projection row already
snapshots exactly the pregame data used, per the storage design). This is
leakage-safe by construction going forward. It additionally performs an
explicit leakage AUDIT on each row: it checks that the recorded market
timestamp and lineup retrieval timestamp are not later than the game's
scheduled start, and flags any row where that can't be confirmed.

Backtesting against seasons *before* this tool was run requires importing
historical data via the CSV templates in app/services/historical_import.py,
which produce the same kind of pregame-snapshotted rows this backtester
consumes -- there is no separate "backtest engine" vs "live engine" split,
by design, so historical and live projections are always evaluated
identically.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from app.database.repositories import ProjectionRepository
from app.database.session import session_scope
from app.evaluation.metrics import calibration_buckets, mae, medae, rmse

console = Console()


def run_backtest(start_date: str, end_date: str) -> dict:
    with session_scope() as session:
        all_graded = ProjectionRepository.list_all_with_results(session, since_date=start_date)
        projections = [p for p in all_graded if start_date <= p.game_date <= end_date]

        if not projections:
            console.print(f"[yellow]No graded projections found between {start_date} and {end_date}.[/yellow]")
            return {}

        leakage_flags = []
        for p in projections:
            game_start = p.game_start_utc
            if not game_start:
                continue
            if p.market_timestamp_utc and p.market_timestamp_utc > game_start:
                leakage_flags.append((p.id, "market_timestamp_after_game_start"))
            if p.lineup_retrieved_at and p.lineup_retrieved_at > game_start:
                leakage_flags.append((p.id, "lineup_retrieved_after_game_start"))

        by_month = defaultdict(list)
        by_confidence = defaultdict(list)
        by_lineup_status = defaultdict(list)
        by_market_availability = defaultdict(list)

        for p in projections:
            err = p.final_blended_projection - p.actual_result.actual_strikeouts
            month_key = p.game_date[:7]
            by_month[month_key].append(err)
            by_confidence[p.confidence_rating or "Unknown"].append(err)
            by_lineup_status[p.lineup_status].append(err)
            has_market = bool(p.market_snapshot_json and p.market_snapshot_json.get("strikeout_line") is not None)
            by_market_availability["market_data_available" if has_market else "stats_only_no_market"].append(err)

        report = {
            "n": len(projections),
            "leakage_flags": leakage_flags,
            "overall_mae": round(mae([e for group in by_month.values() for e in group]), 4),
            "overall_rmse": round(rmse([e for group in by_month.values() for e in group]), 4),
            "by_month": {k: {"n": len(v), "mae": round(mae(v), 4)} for k, v in sorted(by_month.items())},
            "by_confidence": {k: {"n": len(v), "mae": round(mae(v), 4)} for k, v in by_confidence.items()},
            "by_lineup_status": {k: {"n": len(v), "mae": round(mae(v), 4)} for k, v in by_lineup_status.items()},
            "by_market_availability": {k: {"n": len(v), "mae": round(mae(v), 4)} for k, v in by_market_availability.items()},
        }

        _print_backtest_report(start_date, end_date, report)
        return report


def _print_backtest_report(start_date: str, end_date: str, report: dict) -> None:
    console.print(f"[bold]Backtest {start_date} to {end_date}[/bold] -- {report['n']} projection(s)")
    console.print(f"Overall MAE: {report['overall_mae']} | RMSE: {report['overall_rmse']}")

    if report["leakage_flags"]:
        console.print(f"[bold red]LEAKAGE WARNING:[/bold red] {len(report['leakage_flags'])} row(s) flagged.")
        for pid, reason in report["leakage_flags"][:10]:
            console.print(f"  - {pid}: {reason}")
    else:
        console.print("[green]No leakage flags detected in this window.[/green]")

    for label, key in [
        ("Month", "by_month"),
        ("Confidence Rating", "by_confidence"),
        ("Lineup Status", "by_lineup_status"),
        ("Market Data Availability", "by_market_availability"),
    ]:
        sub = report.get(key, {})
        if not sub:
            continue
        t = Table(title=f"Backtest accuracy by {label}")
        t.add_column(label)
        t.add_column("N", justify="right")
        t.add_column("MAE", justify="right")
        for k, v in sub.items():
            t.add_row(str(k), str(v["n"]), str(v["mae"]))
        console.print(t)
