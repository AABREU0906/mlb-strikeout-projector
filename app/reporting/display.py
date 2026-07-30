from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.markets.odds_math import classify_edge
from app.projections.engine import ProjectionResult
from app.schemas.market import MarketSnapshot

console = Console()


def print_lineup_status(status: str, source: str, retrieved_at: str) -> None:
    if status == "confirmed":
        console.print(Panel("CONFIRMED LINEUP", style="bold green", expand=False))
    else:
        console.print(
            Panel(
                "WARNING: PROJECTED LINEUP — LOWER CONFIDENCE",
                style="bold yellow",
                expand=False,
            )
        )
    console.print(f"  [dim]Lineup source: {source} | Retrieved: {retrieved_at}[/dim]")


def print_warnings(warnings: list[dict], title: str) -> None:
    if not warnings:
        return
    table = Table(title=title, show_lines=True)
    for col in ["Player", "Issue", "Source", "Published", "Confidence", "Effect"]:
        table.add_column(col)
    for w in warnings:
        published = " ".join(filter(None, [w.get("published_date"), w.get("published_time")]))
        table.add_row(
            w.get("player", ""),
            w.get("issue", ""),
            w.get("source", ""),
            published,
            w.get("confidence", ""),
            w.get("expected_effect", ""),
        )
    console.print(table)


def print_main_summary(
    game_label: str,
    pitcher_name: str,
    opponent: str,
    game_time: str,
    lineup_status: str,
    last_updated: str,
    result: ProjectionResult,
) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()

    rows = [
        ("Game", game_label),
        ("Pitcher", pitcher_name),
        ("Opponent", opponent),
        ("Game Time", game_time),
        ("Lineup Status", lineup_status.upper()),
        ("Last Updated", last_updated),
        ("", ""),
        ("Statistics-Only Projection", f"{result.statistics_only_projection:.2f} K"),
        ("Market-Informed Projection", f"{result.market_informed_projection:.2f} K"),
        ("Final Blended Projection", f"{result.final_blended_projection:.2f} K"),
        ("Median Strikeouts", f"{result.median_strikeouts:.1f}"),
        ("Expected Innings", f"{result.expected_innings:.1f}"),
        ("Expected Batters Faced", f"{result.expected_batters_faced:.1f}"),
        ("Expected Pitch Count", f"{result.expected_pitch_count:.0f}"),
        ("Confidence", result.confidence_rating),
    ]
    for label, value in rows:
        table.add_row(label + ":" if label else "", str(value))

    console.print(Panel(table, title="MLB PITCHER STRIKEOUT PROJECTION", border_style="bold blue"))


def print_distribution(result: ProjectionResult) -> None:
    table = Table(title="Strikeout Probability Distribution")
    table.add_column("K's", justify="right")
    table.add_column("Probability", justify="right")
    table.add_column("", width=30)

    for k in range(0, 16):
        label = "15 or more" if k == 15 else str(k)
        p = result.probability_by_k.get(k, 0.0)
        bar = "█" * int(round(p * 60))
        table.add_row(label, f"{p*100:5.1f}%", bar)
    console.print(table)

    pct_table = Table(title="Percentiles / Range")
    for col in ["10th", "25th", "50th", "75th", "90th", "Most Likely", "Std Dev"]:
        pct_table.add_column(col, justify="right")
    p = result.percentiles
    pct_table.add_row(
        f"{p.get(10,'-')}", f"{p.get(25,'-')}", f"{p.get(50,'-')}", f"{p.get(75,'-')}", f"{p.get(90,'-')}",
        str(result.most_likely_k), f"{result.std_dev:.2f}",
    )
    console.print(pct_table)


def print_batter_matchup_table(result: ProjectionResult) -> None:
    table = Table(title="Batter Matchup Table", show_lines=False)
    for col in ["Spot", "Batter", "Hand", "Status", "Batter K% vs Hand", "Pitcher K% vs Hand", "K Prob", "Warnings"]:
        table.add_column(col)

    for b in result.batter_results:
        warn = "SMALL SAMPLE" if b.sample_size_warning else ""
        table.add_row(
            str(b.batting_order or "-"),
            b.name,
            "-",
            "",
            "",
            "",
            f"{b.adjusted_probability*100:.1f}%",
            warn,
        )
    console.print(table)


def print_explanation(result: ProjectionResult) -> None:
    pos = result.explanation.get("positive_factors", [])
    neg = result.explanation.get("negative_factors", [])
    text = Text()
    text.append("POSITIVE FACTORS\n", style="bold green")
    for f in pos:
        text.append(f"  + {f['description']} ({f['estimated_effect']})\n", style="green")
    text.append("\nNEGATIVE FACTORS\n", style="bold red")
    for f in neg:
        text.append(f"  - {f['description']} ({f['estimated_effect']})\n", style="red")
    console.print(Panel(text, title="Projection Explanation"))


def print_market_comparison(market: Optional[MarketSnapshot], result: ProjectionResult) -> None:
    if market is None or market.strikeout_line is None:
        console.print("[dim]No sportsbook strikeout line available for comparison.[/dim]")
        return

    table = Table(title="Sportsbook Comparison", show_lines=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row("Sportsbook line", f"{market.strikeout_line}")
    table.add_row("Over odds", str(market.over_odds))
    table.add_row("Under odds", str(market.under_odds))
    if market.raw_over_prob is not None:
        table.add_row("Raw implied over probability", f"{market.raw_over_prob*100:.1f}%")
        table.add_row("Raw implied under probability", f"{market.raw_under_prob*100:.1f}%")
        table.add_row("Vig-free over probability", f"{market.vig_free_over_prob*100:.1f}%")
        table.add_row("Vig-free under probability", f"{market.vig_free_under_prob*100:.1f}%")
        table.add_row("Fair model over odds", str(market.fair_over_odds))
        table.add_row("Fair model under odds", str(market.fair_under_odds))

    # Model-implied over/under probability from the simulated distribution.
    line = market.strikeout_line
    import math
    floor_line = math.floor(line)
    model_over_prob = sum(p for k, p in result.probability_by_k.items() if k > floor_line)
    if line == floor_line:  # whole-number line has a push at that value; approximate using > line
        pass
    model_under_prob = 1 - model_over_prob

    table.add_row("Model over probability", f"{model_over_prob*100:.1f}%")
    table.add_row("Model under probability", f"{model_under_prob*100:.1f}%")
    table.add_row("Difference vs. line", f"{result.final_blended_projection - line:+.2f}")

    if market.vig_free_over_prob is not None:
        edge_label = classify_edge(model_over_prob, market.vig_free_over_prob)
        table.add_row("Probability edge vs vig-free market", edge_label)

    console.print(table)
    console.print(
        "[dim]Projections involve uncertainty and are not financial guarantees. "
        "No outcome is ever guaranteed or certain.[/dim]"
    )
