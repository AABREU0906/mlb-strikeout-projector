"""Terminal display for `python main.py model-report`."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

from app.services.model_report_service import ModelReport

console = Console()


def _metrics_row(table: Table, label: str, m: dict) -> None:
    if m.get("n", 0) == 0:
        table.add_row(label, "0", "-", "-", "-", "-", "-", "-", "-")
        return
    table.add_row(
        label, str(m["n"]), f"{m['mae']:.2f}", f"{m['rmse']:.2f}", f"{m['bias']:+.2f}",
        f"{m['median_ae']:.2f}", f"{m['pct_within_0_5']:.0f}%", f"{m['pct_within_1_0']:.0f}%", f"{m['pct_within_2_0']:.0f}%",
    )


def _print_footer(report: ModelReport) -> None:
    """
    BUG FIX: this previously printed a hardcoded literal "-" character
    directly in front of n_excluded_invalid/n_excluded_reruns (e.g.
    f"-{report.n_excluded_invalid}"), intended only as a visual
    "subtracted from the total" cue. Both counts are, and always were,
    correctly non-negative integers (they come from `len()` of a
    filtered list, which cannot be negative) -- the "-" prefix was
    purely a display artifact that made a correct positive count LOOK
    like a negative number. Removed, and replaced with an explicit
    reconciliation line so the raw/independent relationship is shown as
    an actual equation instead of relying on a sign convention.

    Pipeline definition (sequential, disjoint, as specified):
        raw_count            = every raw row before filtering
        invalid_removed      = rows removed by the validity filter
        rows_after_invalid   = raw_count - invalid_removed
        reruns_removed       = rows removed by dedup, from rows_after_invalid
        independent_count    = rows_after_invalid - reruns_removed
        graded_count         = independent rows with an actual result
    """
    console.print()
    console.print("[bold]Sample composition[/bold]")
    console.print(f"  Raw projections:               {report.n_raw_projections}")
    console.print(f"  Excluded invalid:               {report.n_excluded_invalid}")
    console.print(f"  Excluded reruns:                {report.n_excluded_reruns}")
    console.print(f"  Independent projections:        {report.n_independent_projections}")
    console.print(f"  Graded (has actual result):     {report.n_total_graded}")
    console.print(
        f"  Calibration eligible:           {report.n_calibration_eligible} / {report.n_total_graded}"
    )
    console.print(
        f"  [dim]{report.n_raw_projections} raw - {report.n_excluded_invalid} invalid - "
        f"{report.n_excluded_reruns} reruns = {report.n_independent_projections} independent[/dim]"
    )


def print_model_report(report: ModelReport) -> None:
    console.print(f"\n[bold]MODEL HEALTH REPORT[/bold]  (n = {report.n_total_graded} graded projections)\n")

    for w in report.warnings:
        console.print(f"[yellow]\u26a0 {w}[/yellow]")

    if report.n_total_graded == 0:
        _print_footer(report)
        return

    console.print()
    table = Table(title="Core Projection Accuracy")
    for col in ("Model", "n", "MAE", "RMSE", "Bias", "Median AE", "\u00b10.5K", "\u00b11.0K", "\u00b12.0K"):
        table.add_column(col)
    _metrics_row(table, "Statistics-only", report.core_metrics_statistics_only)
    _metrics_row(table, "Market-informed", report.core_metrics_market_informed)
    _metrics_row(table, "Final blended", report.core_metrics)
    console.print(table)
    console.print("[dim]Bias = final projection minus actual strikeouts (positive = model runs high).[/dim]\n")

    d = report.directional
    if d.get("n", 0) > 0:
        console.print("[bold]Directional[/bold]")
        console.print(
            f"  Projected: {d['projected_over']} Over ({d['projected_over_pct']}%), "
            f"{d['projected_under']} Under ({d['projected_under_pct']}%), "
            f"{d['projected_pass']} PASS ({d['projected_pass_pct']}%)"
        )
        console.print(
            f"  Actual:    {d['actual_over']} Over ({d['actual_over_pct']}%), "
            f"{d['actual_under']} Under ({d['actual_under_pct']}%), {d['actual_push']} push"
        )
        wr = f"{d['recommendation_win_rate']}%" if d["recommendation_win_rate"] is not None else "n/a (no decided bets)"
        console.print(f"  Recommendation win rate (Over+Under, excl. PASS/push): {wr}")
        over_r, under_r = d["over_results"], d["under_results"]
        if over_r["win_rate"] is not None:
            console.print(f"    Over:  {over_r['wins']}-{over_r['losses']} ({over_r['win_rate']}%)")
        else:
            console.print("    Over: no decided bets")
        if under_r["win_rate"] is not None:
            console.print(f"    Under: {under_r['wins']}-{under_r['losses']} ({under_r['win_rate']}%)")
        else:
            console.print("    Under: no decided bets")
        console.print()

    if report.calibration:
        console.print("[bold]Calibration (Model Over Probability)[/bold]")
        cal_table = Table()
        for col in ("Bucket", "n", "Avg Predicted", "Actual Over Rate", "Gap", "Brier"):
            cal_table.add_column(col)
        for b in report.calibration:
            if b["n"] == 0:
                cal_table.add_row(b["bucket"], "0", "-", "-", "-", "-")
                continue
            note = "" if b["reliable"] else " (small n)"
            cal_table.add_row(
                b["bucket"] + note, str(b["n"]), f"{b['avg_predicted']}%",
                f"{b['actual_over_rate']}%", f"{b['calibration_gap']:+.1f}pp", f"{b['brier']:.3f}",
            )
        console.print(cal_table)
        console.print()

    def _print_bias_group(title: str, groups: list, top_n: int = 8) -> None:
        if not groups:
            return
        console.print(f"[bold]Bias by {title}[/bold]")
        for g in groups[:top_n]:
            note = "" if g["reliable"] else " [dim](small n)[/dim]"
            console.print(f"  {g['group']}: {g['avg_bias']:+.2f}K (n={g['n']}){note}")
        console.print()

    _print_bias_group("Pitcher", report.bias_by_pitcher)
    _print_bias_group("Opponent", report.bias_by_opponent)
    _print_bias_group("Line bucket", report.bias_by_line_bucket)
    _print_bias_group("Confidence", report.bias_by_confidence)
    _print_bias_group("Edge grade", report.bias_by_edge_grade)
    _print_bias_group("Workload source", report.bias_by_workload_source)
    _print_bias_group("Workload role", report.bias_by_workload_role)
    _print_bias_group("Lineup status", report.bias_by_lineup_status)
    _print_bias_group("Workload data type", report.bias_by_fallback_used)

    ed = report.error_decomposition
    if ed.get("n", 0) > 0:
        console.print("[bold]PROJECTION ERROR REVIEW[/bold]")
        console.print(
            f"  n={ed['n']} | Avg workload contribution: {ed['avg_workload_contribution']:+.2f}K | "
            f"Avg strikeout-rate contribution: {ed['avg_rate_contribution']:+.2f}K\n"
        )

        def _print_miss_list(title, misses):
            console.print(f"  {title}:")
            for m in misses:
                console.print(
                    f"    {m['pitcher_name']}: projected {m['projected_ks']}, actual {m['actual_ks']} "
                    f"(miss {m['total_miss']:+.2f}) | workload {m['workload_contribution']:+.2f}K, "
                    f"rate {m['rate_contribution']:+.2f}K"
                )

        _print_miss_list("Biggest underprojections", ed["biggest_underprojections"])
        _print_miss_list("Biggest overprojections", ed["biggest_overprojections"])
        _print_miss_list("Largest workload misses", ed["largest_workload_misses"])
        _print_miss_list("Largest K-rate misses", ed["largest_rate_misses"])

    _print_footer(report)
