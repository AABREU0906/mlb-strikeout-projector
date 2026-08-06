from __future__ import annotations

import math
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.markets.edge_analysis import analyze_betting_edge
from app.markets.odds_math import classify_edge
from app.projections.engine import ProjectionResult
from app.reporting.edge_report import display_edge_analysis
from app.schemas.market import MarketSnapshot


console = Console()


def print_lineup_status(
    status: str,
    source: str,
    retrieved_at: str,
) -> None:
    if status.lower() == "confirmed":
        console.print(
            Panel(
                "CONFIRMED LINEUP",
                style="bold green",
                expand=False,
            )
        )
    else:
        console.print(
            Panel(
                "WARNING: PROJECTED LINEUP — LOWER CONFIDENCE",
                style="bold yellow",
                expand=False,
            )
        )

    console.print(
        f"  [dim]Lineup source: {source} | "
        f"Retrieved: {retrieved_at}[/dim]"
    )


def print_warnings(
    warnings: list[dict],
    title: str,
) -> None:
    if not warnings:
        return

    table = Table(
        title=title,
        show_lines=True,
    )

    for column in [
        "Player",
        "Issue",
        "Source",
        "Published",
        "Confidence",
        "Effect",
    ]:
        table.add_column(column)

    for warning in warnings:
        published = " ".join(
            filter(
                None,
                [
                    warning.get("published_date"),
                    warning.get("published_time"),
                ],
            )
        )

        table.add_row(
            warning.get("player", ""),
            warning.get("issue", ""),
            warning.get("source", ""),
            published,
            warning.get("confidence", ""),
            warning.get("expected_effect", ""),
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
    table = Table.grid(
        padding=(0, 2),
    )

    table.add_column(
        style="bold cyan",
        justify="right",
    )
    table.add_column()

    rows = [
        ("Game", game_label),
        ("Pitcher", pitcher_name),
        ("Opponent", opponent),
        ("Game Time", game_time),
        ("Lineup Status", lineup_status.upper()),
        ("Last Updated", last_updated),
        ("", ""),
        (
            "Statistics-Only Projection",
            f"{result.statistics_only_projection:.2f} K",
        ),
        (
            "Market-Informed Projection",
            f"{result.market_informed_projection:.2f} K",
        ),
        (
            "Final Blended Projection",
            f"{result.final_blended_projection:.2f} K",
        ),
        (
            "Median Strikeouts",
            f"{result.median_strikeouts:.1f}",
        ),
        (
            "Expected Innings",
            f"{result.expected_innings:.1f}",
        ),
        (
            "Expected Batters Faced",
            f"{result.expected_batters_faced:.1f}",
        ),
        (
            "Expected Pitch Count",
            f"{result.expected_pitch_count:.0f}",
        ),
        (
            "Confidence",
            result.confidence_rating,
        ),
    ]

    for label, value in rows:
        table.add_row(
            label + ":" if label else "",
            str(value),
        )

    console.print(
        Panel(
            table,
            title="MLB PITCHER STRIKEOUT PROJECTION",
            border_style="bold blue",
        )
    )


def print_distribution(
    result: ProjectionResult,
) -> None:
    table = Table(
        title="Strikeout Probability Distribution",
    )

    table.add_column(
        "K's",
        justify="right",
    )
    table.add_column(
        "Probability",
        justify="right",
    )
    table.add_column(
        "",
        width=30,
    )

    for strikeouts in range(0, 16):
        label = (
            "15 or more"
            if strikeouts == 15
            else str(strikeouts)
        )

        probability = result.probability_by_k.get(
            strikeouts,
            0.0,
        )

        bar = "█" * int(
            round(probability * 60)
        )

        table.add_row(
            label,
            f"{probability * 100:5.1f}%",
            bar,
        )

    console.print(table)

    percentile_table = Table(
        title="Percentiles / Range",
    )

    for column in [
        "10th",
        "25th",
        "50th",
        "75th",
        "90th",
        "Most Likely",
        "Std Dev",
    ]:
        percentile_table.add_column(
            column,
            justify="right",
        )

    percentiles = result.percentiles

    percentile_table.add_row(
        f"{percentiles.get(10, '-')}",
        f"{percentiles.get(25, '-')}",
        f"{percentiles.get(50, '-')}",
        f"{percentiles.get(75, '-')}",
        f"{percentiles.get(90, '-')}",
        str(result.most_likely_k),
        f"{result.std_dev:.2f}",
    )

    console.print(percentile_table)


def print_batter_matchup_table(
    result: ProjectionResult,
) -> None:
    table = Table(
        title="Batter Matchup Table",
        show_lines=False,
    )

    for column in [
        "Spot",
        "Batter",
        "Hand",
        "Status",
        "Batter K% vs Hand",
        "Pitcher K% vs Hand",
        "K Prob",
        "Warnings",
    ]:
        table.add_column(column)

    for batter in result.batter_results:
        warning = (
            "SMALL SAMPLE"
            if batter.sample_size_warning
            else ""
        )

        table.add_row(
            str(batter.batting_order or "-"),
            batter.name,
            "-",
            "",
            "",
            "",
            f"{batter.adjusted_probability * 100:.1f}%",
            warning,
        )

    console.print(table)


def print_explanation(
    result: ProjectionResult,
) -> None:
    positive_factors = result.explanation.get(
        "positive_factors",
        [],
    )

    negative_factors = result.explanation.get(
        "negative_factors",
        [],
    )

    text = Text()

    text.append(
        "POSITIVE FACTORS\n",
        style="bold green",
    )

    for factor in positive_factors:
        text.append(
            (
                f"  + {factor['description']} "
                f"({factor['estimated_effect']})\n"
            ),
            style="green",
        )

    text.append(
        "\nNEGATIVE FACTORS\n",
        style="bold red",
    )

    for factor in negative_factors:
        text.append(
            (
                f"  - {factor['description']} "
                f"({factor['estimated_effect']})\n"
            ),
            style="red",
        )

    console.print(
        Panel(
            text,
            title="Projection Explanation",
        )
    )


def print_market_comparison(
    market: Optional[MarketSnapshot],
    result: ProjectionResult,
) -> None:
    if market is None or market.strikeout_line is None:
        console.print(
            "[dim]No sportsbook strikeout line available "
            "for comparison.[/dim]"
        )
        return

    line = float(market.strikeout_line)

    table = Table(
        title="Sportsbook Comparison",
        show_lines=True,
    )

    table.add_column("Metric")
    table.add_column(
        "Value",
        justify="right",
    )

    table.add_row(
        "Sportsbook line",
        f"{line}",
    )

    table.add_row(
        "Over odds",
        str(market.over_odds),
    )

    table.add_row(
        "Under odds",
        str(market.under_odds),
    )

    if market.raw_over_prob is not None:
        table.add_row(
            "Raw implied over probability",
            f"{market.raw_over_prob * 100:.1f}%",
        )

    if market.raw_under_prob is not None:
        table.add_row(
            "Raw implied under probability",
            f"{market.raw_under_prob * 100:.1f}%",
        )

    if market.vig_free_over_prob is not None:
        table.add_row(
            "Vig-free over probability",
            f"{market.vig_free_over_prob * 100:.1f}%",
        )

    if market.vig_free_under_prob is not None:
        table.add_row(
            "Vig-free under probability",
            f"{market.vig_free_under_prob * 100:.1f}%",
        )

    # These are sportsbook market fair odds after removing vig,
    # not the model's calculated fair odds.
    if market.fair_over_odds is not None:
        table.add_row(
            "Fair market over odds",
            str(market.fair_over_odds),
        )

    if market.fair_under_odds is not None:
        table.add_row(
            "Fair market under odds",
            str(market.fair_under_odds),
        )

    floor_line = math.floor(line)

    # This works normally for half-point strikeout lines,
    # such as 4.5, 5.5, and 6.5.
    model_over_probability = sum(
        probability
        for strikeouts, probability
        in result.probability_by_k.items()
        if strikeouts > floor_line
    )

    model_under_probability = (
        1.0 - model_over_probability
    )

    table.add_row(
        "Model over probability",
        f"{model_over_probability * 100:.1f}%",
    )

    table.add_row(
        "Model under probability",
        f"{model_under_probability * 100:.1f}%",
    )

    projection_difference = (
        result.final_blended_projection - line
    )

    table.add_row(
        "Difference vs. line",
        f"{projection_difference:+.2f}",
    )

    if market.vig_free_over_prob is not None:
        edge_label = classify_edge(
            model_over_probability,
            market.vig_free_over_prob,
        )

        table.add_row(
            "Probability edge vs vig-free market",
            edge_label,
        )

    console.print(table)

    # Whole-number lines can result in a push.
    # The current edge-analysis module is designed for half-point lines.
    is_whole_number_line = line.is_integer()

    if is_whole_number_line:
        console.print(
            Panel(
                (
                    "This is a whole-number strikeout line. "
                    "A pitcher finishing exactly on the line "
                    "would produce a push.\n\n"
                    "The recommendation module currently supports "
                    "half-point lines such as 4.5, 5.5, and 6.5."
                ),
                title="Push Warning",
                border_style="yellow",
            )
        )

        console.print(
            "[dim]Projections involve uncertainty and are not "
            "financial guarantees. No outcome is guaranteed.[/dim]"
        )
        return

    if (
        market.over_odds is None
        or market.under_odds is None
    ):
        console.print(
            "[yellow]Both over and under odds are required "
            "to calculate the betting edge.[/yellow]"
        )

        console.print(
            "[dim]Projections involve uncertainty and are not "
            "financial guarantees. No outcome is guaranteed.[/dim]"
        )
        return

    # Try to determine lineup status from ProjectionResult.
    # If your ProjectionResult does not contain this field,
    # it defaults to confirmed so the code continues to work.
    result_lineup_status = getattr(
        result,
        "lineup_status",
        "confirmed",
    )

    lineup_confirmed = (
        str(result_lineup_status).lower()
        == "confirmed"
    )

    pitcher_confirmed = bool(
        getattr(
            result,
            "pitcher_confirmed",
            True,
        )
    )

    workload_warning = bool(
        getattr(
            result,
            "workload_warning",
            False,
        )
    )

    injury_warning = bool(
        getattr(
            result,
            "injury_warning",
            False,
        )
    )

    weather_warning = bool(
        getattr(
            result,
            "weather_warning",
            False,
        )
    )

    stale_data = bool(
        getattr(
            result,
            "stale_data",
            False,
        )
    )

    model_sample_size = getattr(
        result,
        "model_sample_size",
        None,
    )

    edge_analysis = analyze_betting_edge(
        over_odds=int(market.over_odds),
        under_odds=int(market.under_odds),
        model_over_probability=(
            model_over_probability
        ),
        model_under_probability=(
            model_under_probability
        ),
        lineup_confirmed=lineup_confirmed,
        pitcher_confirmed=pitcher_confirmed,
        workload_warning=workload_warning,
        injury_warning=injury_warning,
        weather_warning=weather_warning,
        stale_data=stale_data,
        model_sample_size=model_sample_size,
    )

    display_edge_analysis(
        edge_analysis,
        sportsbook_line=line,
        projected_strikeouts=(
            result.final_blended_projection
        ),
    )