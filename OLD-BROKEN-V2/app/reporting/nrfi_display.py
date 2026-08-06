from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.markets.nrfi_edge_analysis import NrfiEdgeAnalysis
from app.projections.nrfi_engine import NrfiEngineResult

console = Console()


def print_nrfi_summary(
    matchup: str, game_time: str, home_pitcher: str, away_pitcher: str,
    home_hand: Optional[str], away_hand: Optional[str], lineup_status: str,
    result: NrfiEngineResult,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()

    gr = result.game_result
    rows = [
        ("Matchup", matchup),
        ("Game Time", game_time),
        ("Away Starter", f"{away_pitcher} ({away_hand or '?'})"),
        ("Home Starter", f"{home_pitcher} ({home_hand or '?'})"),
        ("Lineup Status", lineup_status.upper()),
        ("", ""),
        ("Away Scoring Probability", f"{gr.away_half.scoring_probability*100:.1f}%"),
        ("Home Scoring Probability", f"{gr.home_half.scoring_probability*100:.1f}%"),
        ("Away Threat Score", f"{result.away_threat.score:.0f}/100"),
        ("Home Threat Score", f"{result.home_threat.score:.0f}/100"),
        ("NRFI Probability", f"{gr.nrfi_probability*100:.1f}%"),
        ("YRFI Probability", f"{gr.yrfi_probability*100:.1f}%"),
        ("Expected 1st-Inning Runs", f"{gr.expected_first_inning_runs:.2f}"),
        ("Model Confidence", f"{result.confidence.score:.0f}/100"),
    ]
    for label, value in rows:
        table.add_row(label + ":" if label else "", str(value))

    style = "bold green" if gr.nrfi_probability >= 0.5 else "bold yellow"
    console.print(Panel(table, title="NRFI / YRFI PROJECTION", border_style=style))


def print_nrfi_explanation(result: NrfiEngineResult) -> None:
    text = Text()
    text.append("NRFI FACTORS\n", style="bold green")
    for line in result.explanation.get("nrfi_factors", []):
        text.append(f"  + {line}\n", style="green")
    text.append("\nYRFI FACTORS\n", style="bold red")
    for line in result.explanation.get("yrfi_factors", []):
        text.append(f"  - {line}\n", style="red")
    console.print(Panel(text, title="Projection Explanation"))


def print_nrfi_edge(analysis: Optional[NrfiEdgeAnalysis]) -> None:
    if analysis is None:
        console.print("[dim]No NRFI/YRFI odds available for comparison.[/dim]")
        return

    table = Table(title="NRFI/YRFI Betting Edge Analysis", show_lines=True)
    table.add_column("Metric")
    table.add_column("NRFI", justify="right")
    table.add_column("YRFI", justify="right")

    def fmt_odds(o: int) -> str:
        return f"+{o}" if o > 0 else str(o)

    table.add_row("Sportsbook odds", fmt_odds(analysis.nrfi.sportsbook_odds), fmt_odds(analysis.yrfi.sportsbook_odds))
    table.add_row("Break-even probability", f"{analysis.nrfi.break_even_probability:.1%}", f"{analysis.yrfi.break_even_probability:.1%}")
    table.add_row("Vig-free market probability", f"{analysis.nrfi.vig_free_market_probability:.1%}", f"{analysis.yrfi.vig_free_market_probability:.1%}")
    table.add_row("Model probability", f"{analysis.nrfi.model_probability:.1%}", f"{analysis.yrfi.model_probability:.1%}")
    table.add_row("Fair model odds", fmt_odds(analysis.nrfi.fair_model_odds), fmt_odds(analysis.yrfi.fair_model_odds))
    table.add_row("Expected value", f"{analysis.nrfi.expected_value:+.1%}", f"{analysis.yrfi.expected_value:+.1%}")

    console.print(table)
    console.print(f"[bold]Recommendation:[/bold] {analysis.recommended_side}  ({analysis.grade}, {analysis.stars}\u2605)")
    console.print(f"[dim]{analysis.reason}[/dim]")
    console.print(f"[bold]Data confidence:[/bold] {analysis.confidence}")
    console.print(
        "[dim]Projections involve uncertainty and are not financial guarantees. "
        "No outcome is ever guaranteed or certain.[/dim]"
    )


def print_daily_nrfi_comparison(rows: list[dict]) -> None:
    if not rows:
        console.print("[yellow]No qualifying NRFI/YRFI edges found for this date.[/yellow]")
        return
    sorted_rows = sorted(rows, key=lambda r: abs(r.get("edge_value", 0)), reverse=True)
    table = Table(title="Daily NRFI/YRFI Comparison (sorted by largest qualifying edge)")
    table.add_column("Matchup")
    table.add_column("NRFI %", justify="right")
    table.add_column("Recommendation")
    table.add_column("Grade")
    for r in sorted_rows:
        table.add_row(
            r["matchup"], f"{r['nrfi_probability']*100:.1f}%", r.get("recommended_side", "-"), r.get("edge_grade", "-"),
        )
    console.print(table)
