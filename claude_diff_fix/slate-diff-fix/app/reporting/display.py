from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app.markets.edge_analysis import EdgeAnalysis, analyze_betting_edge
from app.markets.line_probability import compute_line_probabilities
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


def print_workload_data_panel(result: ProjectionResult) -> None:
    """Displays the role-aware workload resolution transparently, per
    requirement: never let a fallback-heavy or thin-sample workload
    appear deceptively precise. Reads structured fields directly off
    WorkloadEstimate -- no string parsing."""
    workload = result.workload

    role_display = {
        "starter": "Starting pitcher",
        "reliever": "Reliever",
        "swingman": "Swingman (mixed starter/reliever usage)",
        "unknown": "Role undetermined",
    }.get(workload.workload_role, workload.workload_role or "Unknown")

    source_display = {
        "mlb_season_totals": "MLB season totals (consistent starter role)",
        "mlb_recent_starts": "Recent MLB start game log",
        "mlb_season_starts_only": "Current-season MLB starts only (game log)",
        "mlb_previous_season_starts": "Previous-season MLB starts only (game log)",
        "unresolved": "League-average fallback (no MLB start-specific data found)",
    }.get(workload.workload_source, workload.workload_source or "Unknown")

    text = Text()
    text.append(f"Role today: {role_display}\n")
    text.append(f"Workload source: {source_display}\n")
    if workload.start_specific_sample_size:
        text.append(f"Start-specific sample: {workload.start_specific_sample_size} MLB start(s)\n")
    text.append(
        f"Fallback used: {'Yes' if workload.workload_fallback_used else 'No'}\n",
        style="yellow" if workload.workload_fallback_used else "green",
    )
    if workload.workload_all_metrics_fallback:
        text.append(
            "Confidence impact: Capped at Medium (no reliable MLB "
            "start-specific workload data)\n",
            style="bold yellow",
        )
    elif workload.workload_fallback_used:
        text.append("Confidence impact: Reduced\n", style="yellow")

    console.print(Panel(text, title="WORKLOAD DATA", border_style="cyan"))


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
    *,
    lineup_confirmed: bool = True,
    pitcher_confirmed: bool = True,
    injury_warning_present: bool = False,
    weather_warning_present: bool = False,
    stale_data: bool = False,
    model_sample_size: Optional[int] = None,
) -> Optional[EdgeAnalysis]:
    """
    BUG FIX (audit item #11): this function previously derived
    lineup_confirmed/pitcher_confirmed/workload_warning/injury_warning/
    weather_warning/stale_data via `getattr(result, "<name>", <safe
    default>)`. `ProjectionResult` never actually carries any of those
    attributes, so every one of those lookups silently fell through to
    its "everything is fine" default on every single call -- meaning the
    betting-edge confidence calculation always assumed a confirmed
    lineup, a confirmed pitcher, and zero warnings, regardless of what
    actually happened in the pipeline. That is why HIGH confidence could
    display even when league-average workload fallbacks were in use.

    The caller (app/cli/main_app.py) now passes the REAL values it
    already has from the pipeline run. `workload_warning` and the
    all-metrics-fallback hard cap are read directly from
    WorkloadEstimate's own structured fields
    (`result.workload.workload_fallback_used` /
    `result.workload.workload_all_metrics_fallback`) -- NOT from parsing
    note text -- so this can never silently miss a fallback that
    originated upstream in PitcherFeatureBuilder (the root cause of a
    real bug: a pitcher whose per-start data was already rejected and
    nulled out before reaching the workload model previously skipped the
    fallback warning entirely, because the old detection only fired on a
    present-but-invalid number, not on an already-None value).
    """
    if market is None or market.strikeout_line is None:
        console.print(
            "[dim]No sportsbook strikeout line available "
            "for comparison.[/dim]"
        )
        return None

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

    line_probs = compute_line_probabilities(result.probability_by_k, line)
    model_over_probability = line_probs.over_probability
    model_under_probability = line_probs.under_probability

    table.add_row(
        "Model over probability",
        f"{model_over_probability * 100:.1f}%",
    )

    table.add_row(
        "Model under probability",
        f"{model_under_probability * 100:.1f}%",
    )

    if line_probs.is_whole_number_line:
        table.add_row(
            "Model push probability",
            f"{line_probs.push_probability * 100:.1f}%",
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
        return None

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
        return None

    # BUG FIX (audit item #11): these now reflect what actually happened
    # in the pipeline, passed explicitly by the caller, instead of
    # getattr() lookups against attributes that never existed on
    # ProjectionResult and therefore always silently defaulted to "fine."
    workload_warning = result.workload.workload_fallback_used
    workload_all_metrics_fallback = result.workload.workload_all_metrics_fallback
    # BUG FIX: a thin-but-real MLB sample (e.g. one actual start for a
    # reliever making a spot start) is NOT a fallback substitution at all
    # -- the value is real, valid, pitcher-specific data -- so
    # workload_fallback_used correctly stays False for it. That means
    # betting confidence never saw the thin-sample risk at all. This
    # reads the sample size directly off WorkloadEstimate's structured
    # field (no string parsing) and passes it as its own, independent
    # signal into the confidence/grade calculation.
    workload_sample_size = result.workload.start_specific_sample_size
    injury_warning = injury_warning_present
    weather_warning = weather_warning_present

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
        workload_all_metrics_fallback=workload_all_metrics_fallback,
        workload_sample_size=workload_sample_size,
        injury_warning=injury_warning,
        weather_warning=weather_warning,
        stale_data=stale_data,
        model_sample_size=model_sample_size,
        # Projection-quality signals for the Elite/Strong separation gate
        # (see determine_edge_grade's docstring for the root cause this
        # fixes).
        #
        # DATA-CONSISTENCY FIX: this previously used
        # statistics_only_projection specifically to avoid a mild
        # circularity concern (the market-informed blend already leans
        # partly toward the market price, which could understate the
        # measured separation for a market-informed number). That
        # reasoning was sound in isolation, but it meant this gate was
        # silently evaluating a DIFFERENT number than the one shown to
        # the user as "the projection" everywhere else (display, CSV,
        # database) -- exactly the kind of one-value-computed /
        # another-value-shown inconsistency reported as a bug. Per the
        # explicit requirement that "Projection column, Diff, and
        # recommendation quality/separation logic... all refer to the
        # intended same final projection value," this now uses
        # final_blended_projection like everything else. The original
        # circularity concern still exists in principle but is secondary
        # to guaranteeing one consistent number throughout the system.
        projection_value=result.final_blended_projection,
        line_value=line,
        projection_std_dev=result.std_dev,
    )

    display_edge_analysis(
        edge_analysis,
        sportsbook_line=line,
        projected_strikeouts=(
            result.final_blended_projection
        ),
    )

    return edge_analysis


def print_validation_report(report) -> None:
    """Displays the central validator's findings. Per the audit's top
    priority rule, a CRITICAL failure gets a loud, unmissable panel and
    the caller must not proceed to display any betting recommendation --
    this function only renders the panel; the caller (main_app.py) is
    responsible for actually gating what runs afterward."""
    if not report.is_valid:
        text = Text()
        text.append(
            "This projection failed validation and no betting "
            "recommendation will be shown.\n\n",
            style="bold",
        )
        text.append("Reasons:\n", style="bold")
        for issue in report.critical_issues:
            text.append(f"  \u2717 {issue.message}\n", style="red")
        if report.warning_issues:
            text.append("\nAdditional warnings:\n", style="bold yellow")
            for issue in report.warning_issues:
                text.append(f"  \u26a0 {issue.message}\n", style="yellow")
        console.print(Panel(text, title="VALIDATION FAILED", border_style="bold red"))
        return

    if report.has_warnings:
        text = Text()
        for issue in report.warning_issues:
            text.append(f"  \u26a0 {issue.message}\n", style="yellow")
        console.print(Panel(text, title="Data Quality Warnings", border_style="yellow"))