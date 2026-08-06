from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.markets.edge_analysis import EdgeAnalysis


def format_odds(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def display_edge_analysis(
    analysis: EdgeAnalysis,
    *,
    sportsbook_line: float,
    projected_strikeouts: float | None = None,
) -> None:
    console = Console()

    table = Table(
        title="Betting Edge Analysis",
        show_header=True,
        header_style="bold",
    )

    table.add_column("Metric")
    table.add_column("Over", justify="right")
    table.add_column("Under", justify="right")

    table.add_row(
        "Sportsbook odds",
        format_odds(analysis.over.sportsbook_odds),
        format_odds(analysis.under.sportsbook_odds),
    )

    table.add_row(
        "Break-even probability",
        f"{analysis.over.break_even_probability:.1%}",
        f"{analysis.under.break_even_probability:.1%}",
    )

    table.add_row(
        "Vig-free market probability",
        f"{analysis.over.vig_free_market_probability:.1%}",
        f"{analysis.under.vig_free_market_probability:.1%}",
    )

    table.add_row(
        "Model probability",
        f"{analysis.over.model_probability:.1%}",
        f"{analysis.under.model_probability:.1%}",
    )

    table.add_row(
        "Model fair odds",
        format_odds(analysis.over.fair_model_odds),
        format_odds(analysis.under.fair_model_odds),
    )

    table.add_row(
        "Edge vs. offered price",
        f"{analysis.over.probability_edge_vs_price:+.1%}",
        f"{analysis.under.probability_edge_vs_price:+.1%}",
    )

    table.add_row(
        "Edge vs. vig-free market",
        f"{analysis.over.probability_edge_vs_market:+.1%}",
        f"{analysis.under.probability_edge_vs_market:+.1%}",
    )

    table.add_row(
        "Estimated EV per unit",
        f"{analysis.over.expected_value:+.1%}",
        f"{analysis.under.expected_value:+.1%}",
    )

    console.print(table)

    star_display = "★" * analysis.stars + "☆" * (
        5 - analysis.stars
    )

    if analysis.selected is None:
        recommendation = Text("PASS", style="bold yellow")
        selected_probability = "N/A"
        selected_odds = "N/A"
        selected_ev = "N/A"
        fair_odds = "N/A"
    else:
        recommendation = Text(
            f"{analysis.recommended_side} {sportsbook_line}",
            style="bold green",
        )
        selected_probability = (
            f"{analysis.selected.model_probability:.1%}"
        )
        selected_odds = format_odds(
            analysis.selected.sportsbook_odds
        )
        selected_ev = (
            f"{analysis.selected.expected_value:+.1%}"
        )
        fair_odds = format_odds(
            analysis.selected.fair_model_odds
        )

    summary_lines = [
        f"Recommended side: {recommendation.plain}",
        f"Sportsbook line: {sportsbook_line}",
        f"Offered odds: {selected_odds}",
        f"Model probability: {selected_probability}",
        f"Model fair odds: {fair_odds}",
        f"Estimated EV: {selected_ev}",
        f"Edge grade: {star_display} {analysis.grade}",
        f"Input confidence: {analysis.confidence}",
    ]

    if projected_strikeouts is not None:
        difference = projected_strikeouts - sportsbook_line
        summary_lines.insert(
            2,
            (
                f"Strikeout projection: "
                f"{projected_strikeouts:.2f} "
                f"({difference:+.2f} vs. line)"
            ),
        )

    summary_lines.append("")
    summary_lines.append(analysis.reason)

    console.print(
        Panel(
            "\n".join(summary_lines),
            title="Model Recommendation",
            border_style=(
                "green"
                if analysis.selected is not None
                else "yellow"
            ),
        )
    )

    console.print(
        "[dim]Estimated edge depends on the model being properly "
        "calibrated. It is not a guarantee of profit.[/dim]"
    )