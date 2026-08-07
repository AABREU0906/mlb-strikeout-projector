from __future__ import annotations

import datetime as dt
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from app.cli.interactive import prompt_game_date, prompt_manual_market, prompt_warnings, select_game, select_pitcher
from app.cli.nrfi_interactive import prompt_nrfi_odds
from app.config.logging_config import configure_logging
from app.database.session import init_db
from app.markets.line_probability import compute_line_probabilities
from app.reporting.display import (
    print_batter_matchup_table,
    print_workload_data_panel,
    print_distribution,
    print_explanation,
    print_lineup_status,
    print_main_summary,
    print_market_comparison,
    print_validation_report,
    print_warnings,
)
from app.services.pipeline import ProjectionPipeline
from app.validation.projection_validator import validate_projection

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    configure_logging()
    init_db()

    if ctx.invoked_subcommand is not None:
        return

    console.print(
        "\n[bold]MLB Projection System[/bold]\n\n"
        "1. Strikeout projection\n"
        "2. NRFI/YRFI projection\n"
        "3. Run both\n"
        "4. Settle open bets\n"
        "5. Update completed projection results\n"
        "6. Model health report\n"
        "7. Betting history\n"
        "8. Show full command menu\n"
        "9. Exit\n"
    )

    choice = Prompt.ask(
        "Select an option",
        choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        default="1",
    )

    if choice == "1":
        _run_interactive_projection()
    elif choice == "2":
        nrfi_project(date=None)
    elif choice == "3":
        both_command(date=None)
    elif choice == "4":
        settle_bets_command()
    elif choice == "5":
        update_results(force=False)
    elif choice == "6":
        model_report(last=None, season=None, pitcher=None, confidence=None, edge_grade=None, since=None)
    elif choice == "7":
        bet_history(limit=100)
    elif choice == "8":
        menu_command()
    else:
        raise typer.Exit()


@app.command()
def project(
    date: Optional[str] = typer.Option(None, help="Game date YYYY-MM-DD (defaults to today)"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    simulations: Optional[int] = typer.Option(None, help="Monte Carlo iteration count"),
):
    """Run the full interactive daily projection workflow."""
    _run_interactive_projection(date=date, seed=seed, simulations=simulations)


def _run_interactive_projection(date: Optional[str] = None, seed: Optional[int] = None, simulations: Optional[int] = None):
    pipeline = ProjectionPipeline()

    _review_unsettled_bets_before_games()

    game_date = date or prompt_game_date()
    console.print(f"[bold]Fetching schedule for {game_date}...[/bold]")
    games = pipeline.get_schedule(game_date)

    game = select_game(games)
    if game is None:
        raise typer.Exit(code=1)

    selection = select_pitcher(game)
    if selection is None:
        raise typer.Exit(code=1)
    pitcher_id, is_home = selection

    season = dt.date.fromisoformat(game_date).year

    console.print("[bold]Checking for confirmed lineup...[/bold]")

    manual_market = prompt_manual_market()
    warning_log = prompt_warnings()

    console.print("[bold]Gathering data and running simulation (this may take a few seconds)...[/bold]")

    result, projection_id, lineup_status, lineup_source, pitcher_name = pipeline.run(
        game=game,
        pitcher_id=pitcher_id,
        pitcher_is_home=is_home,
        season=season,
        manual_market=manual_market,
        warning_log=warning_log,
        seed=seed,
        n_simulations=simulations,
    )

    opponent = game["away_team"] if is_home else game["home_team"]
    print_lineup_status(
        status=lineup_status,
        source=lineup_source,
        retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
    print_main_summary(
        game_label=f"{game['away_team']} @ {game['home_team']}",
        pitcher_name=pitcher_name,
        opponent=opponent,
        game_time=game.get("scheduled_start_utc", "?"),
        lineup_status=lineup_status,
        last_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        result=result,
    )
    print_distribution(result)
    print_workload_data_panel(result)
    print_batter_matchup_table(result)
    print_explanation(result)

    # --- Central validation gate (audit item #5, the top-priority rule:
    # "prefer PASS or VALIDATION FAILED over a confident but unreliable
    # betting recommendation"). Every value fed in here is the REAL value
    # from this run, not a fabricated or defaulted one. ---
    market_snapshot = None
    if result.market_used.get("snapshot"):
        from app.schemas.market import MarketSnapshot
        market_snapshot = MarketSnapshot(**result.market_used["snapshot"])

    line_probs = None
    if market_snapshot is not None and market_snapshot.strikeout_line is not None:
        line_probs = compute_line_probabilities(result.probability_by_k, float(market_snapshot.strikeout_line))

    injury_warning_present = any(
        "injur" in w.issue.lower() or "il " in w.issue.lower() or "pitch limit" in w.issue.lower()
        for w in warning_log.all()
    )

    validation_report = validate_projection(
        expected_innings=result.expected_innings,
        expected_batters_faced=result.expected_batters_faced,
        expected_pitch_count=result.expected_pitch_count,
        # Compared against statistics_only_projection, not
        # final_blended_projection: probability_by_k IS the statistics-only
        # simulation's distribution, so that's the exact value it should be
        # consistent with. final_blended_projection legitimately differs
        # once market data is blended in -- that's by design, not a bug,
        # and checking against it here would risk false "VALIDATION
        # FAILED" panels on ordinary, expected market-blend differences.
        final_projection=result.statistics_only_projection,
        probability_by_k=result.probability_by_k,
        percentiles=result.percentiles,
        std_dev=result.std_dev,
        prob_complete_5=result.workload.prob_complete_5,
        prob_complete_6=result.workload.prob_complete_6,
        prob_complete_7=result.workload.prob_complete_7,
        prob_early_exit=result.workload.prob_early_exit,
        over_probability=line_probs.over_probability if line_probs else None,
        under_probability=line_probs.under_probability if line_probs else None,
        push_probability=line_probs.push_probability if line_probs else 0.0,
        lineup_confirmed=(lineup_status == "confirmed"),
        pitcher_confirmed=True,
        workload_fallback_used=result.workload.workload_fallback_used,
        workload_fallback_count=result.workload.workload_fallback_count,
        workload_all_metrics_fallback=result.workload.workload_all_metrics_fallback,
    )
    print_validation_report(validation_report)

    from app.services.projection_persistence import update_projection_edge_outcome

    home_or_away = "home" if is_home else "away"
    validation_status = "valid" if validation_report.is_valid else "invalid"
    if validation_report.is_valid and validation_report.has_warnings:
        validation_status = "valid_with_warnings"

    if not validation_report.is_valid:
        # Per the audit's top rule: no recommended side, no EV, no edge
        # grade, no "did you place this bet?" prompt, and this run is
        # never offered as a valid betting recommendation. The
        # statistics-only summary above is still shown (it's diagnostic
        # information, not a betting recommendation), but nothing further
        # happens here.
        update_projection_edge_outcome(
            projection_id,
            validation_status=validation_status,
            pitcher_confirmed=True,
            home_or_away=home_or_away,
        )
        console.print(f"\n[dim]Projection saved (marked invalid for betting purposes). ID: {projection_id}[/dim]")
        return

    edge_analysis = print_market_comparison(
        market_snapshot,
        result,
        lineup_confirmed=(lineup_status == "confirmed"),
        pitcher_confirmed=True,
        injury_warning_present=injury_warning_present,
        stale_data=False,
    )

    if edge_analysis is not None:
        update_projection_edge_outcome(
            projection_id,
            validation_status=validation_status,
            pitcher_confirmed=True,
            home_or_away=home_or_away,
            recommended_side=edge_analysis.recommended_side,
            edge_grade=edge_analysis.grade,
            betting_confidence=edge_analysis.confidence,
            estimated_ev=(edge_analysis.selected.expected_value if edge_analysis.selected else None),
            model_over_probability=edge_analysis.over.model_probability,
            model_under_probability=edge_analysis.under.model_probability,
            projection_minus_line=(
                result.statistics_only_projection - float(market_snapshot.strikeout_line)
                if market_snapshot and market_snapshot.strikeout_line is not None
                else None
            ),
        )
    else:
        # No sportsbook line, whole-number-line unsupported case, or
        # missing odds -- still record that this projection was valid and
        # what lineup/confirmation state it had, even with no market to
        # grade against (Feature 1: every projection is saved, whether or
        # not a market line exists).
        update_projection_edge_outcome(
            projection_id,
            validation_status=validation_status,
            pitcher_confirmed=True,
            home_or_away=home_or_away,
        )

    console.print(f"\n[dim]Projection saved. ID: {projection_id}[/dim]")

    _prompt_to_record_bet(
        projection_id=projection_id,
        game=game,
        game_date=game_date,
        pitcher_id=pitcher_id,
        pitcher_name=pitcher_name,
        opponent=opponent,
        result=result,
        market_snapshot=market_snapshot,
    )


@app.command("update-results")
def update_results(
    force: bool = typer.Option(False, "--force", help="Re-fetch and overwrite already-recorded results too."),
):
    """Fetch actual results for completed games with saved projections."""
    from app.services.results_updater import update_all_pending_results

    console.print("[bold]Updating results for completed games...[/bold]")
    summary = update_all_pending_results(force=force)

    console.print(f"[green]Updated: {len(summary.updated)}[/green]")
    for label in summary.updated:
        console.print(f"  [green]\u2713[/green] {label}")

    if summary.skipped_not_final:
        console.print(f"[yellow]Still in progress / not started: {len(summary.skipped_not_final)}[/yellow]")
        for label in summary.skipped_not_final:
            console.print(f"  [dim]{label}[/dim]")

    if summary.postponed_or_suspended:
        console.print(f"[yellow]Postponed / suspended / cancelled: {len(summary.postponed_or_suspended)}[/yellow]")
        for label in summary.postponed_or_suspended:
            console.print(f"  [dim]{label}[/dim]")

    if summary.unavailable:
        console.print(f"[red]Unavailable (fetch failed or no pitching line found): {len(summary.unavailable)}[/red]")
        for label in summary.unavailable:
            console.print(f"  [dim]{label}[/dim]")

    if summary.already_settled_skipped:
        console.print(
            f"[dim]Already had a result, skipped ({len(summary.already_settled_skipped)}). "
            f"Use --force to overwrite.[/dim]"
        )

    console.print(f"\n[bold]{len(summary.updated)} of {summary.total_pending} pending projection(s) updated.[/bold]")


@app.command()
def evaluate(
    since: Optional[str] = typer.Option(None, help="Only evaluate games on/after this date (YYYY-MM-DD)"),
):
    """Produce model evaluation reports (MAE, RMSE, calibration, etc.)."""
    from app.evaluation.evaluator import run_evaluation

    run_evaluation(since_date=since)


@app.command("model-report")
def model_report(
    last: Optional[int] = typer.Option(None, "--last", help="Only the N most recently graded projections."),
    season: Optional[int] = typer.Option(None, "--season", help="Only projections from this season (e.g. 2026)."),
    pitcher: Optional[str] = typer.Option(None, "--pitcher", help="Only projections for a pitcher matching this name (partial match)."),
    confidence: Optional[str] = typer.Option(None, "--confidence", help="Only projections with this betting confidence (HIGH/MEDIUM/LOW/AVOID/PASS)."),
    edge_grade: Optional[str] = typer.Option(None, "--edge-grade", help="Only projections matching this edge grade (partial match, e.g. 'Moderate')."),
    since: Optional[str] = typer.Option(None, "--since", help="Only projections on/after this date (YYYY-MM-DD)."),
):
    """
    Model health report: evaluates ALL graded strikeout projections (not
    just placed bets) -- accuracy (MAE/RMSE/bias), directional accuracy,
    probability calibration, bias breakdowns, and error decomposition.
    Filters are optional and combinable.
    """
    from app.database.repositories import ProjectionRepository
    from app.database.session import init_db, session_scope
    from app.reporting.model_report_display import print_model_report
    from app.services.model_report_service import generate_model_report

    init_db()
    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(
            session,
            since_date=since,
            season=season,
            pitcher_name=pitcher,
            confidence=confidence,
            edge_grade=edge_grade,
            last_n=last,
        )
        report = generate_model_report(projections)

    print_model_report(report)


@app.command()
def retrain():
    """Retrain the model using stored historical projections and results."""
    from app.training.retrain import run_retraining

    run_retraining()


@app.command()
def history(
    date: Optional[str] = typer.Option(None),
    pitcher: Optional[str] = typer.Option(None),
    team: Optional[str] = typer.Option(None),
    confidence: Optional[str] = typer.Option(None),
    model_version: Optional[str] = typer.Option(None),
    limit: int = typer.Option(50),
):
    """Browse historical projections and outcomes."""
    from app.services.history_service import show_history

    show_history(date=date, pitcher=pitcher, team=team, confidence=confidence, model_version=model_version, limit=limit)


@app.command()
def backtest(
    start_date: str = typer.Option(..., help="Backtest start date YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="Backtest end date YYYY-MM-DD"),
):
    """Recreate historical projections using only data available before each game."""
    from app.evaluation.backtester import run_backtest

    run_backtest(start_date=start_date, end_date=end_date)


@app.command("import-data")
def import_data(
    entity: str = typer.Option(..., help="One of: games, actual_results (others: write templates only)"),
    csv_path: Optional[str] = typer.Option(None, help="Path to CSV file to import"),
    write_templates_only: bool = typer.Option(False, "--write-templates", help="Just write empty CSV templates to data/imports/templates/"),
):
    """Import historical data from CSV, or write documented CSV templates."""
    from app.services.historical_import import import_actual_results, import_games, write_templates

    if write_templates_only or not csv_path:
        paths = write_templates()
        console.print(f"[green]Wrote {len(paths)} template(s) to data/imports/templates/[/green]")
        for p in paths:
            console.print(f"  - {p}")
        return

    from pathlib import Path
    if entity == "games":
        n = import_games(Path(csv_path))
    elif entity == "actual_results":
        n = import_actual_results(Path(csv_path))
    else:
        console.print(f"[red]No loader wired for '{entity}' yet; template is available via --write-templates.[/red]")
        raise typer.Exit(code=1)
    console.print(f"[green]Imported {n} row(s) into {entity}.[/green]")


@app.command()
def models():
    """List all trained model versions and which is active."""
    from app.training.rollback import list_versions
    from rich.table import Table

    versions = list_versions()
    table = Table(title="Model Versions")
    for col in ["Version", "Type", "Trained", "Promoted", "Active", "Val MAE", "N Obs"]:
        table.add_column(col)
    for v in versions:
        table.add_row(
            v.version_label, v.model_type, str(v.trained_at)[:19],
            "yes" if v.promoted else "no", "ACTIVE" if v.is_active else "",
            str((v.validation_metrics or {}).get("overall_mae", "-")),
            str(v.n_training_observations or "-"),
        )
    console.print(table)


@app.command()
def rollback(version: str = typer.Option(..., help="Model version label to reactivate")):
    """Roll back to a previously trained model version."""
    from app.training.rollback import rollback_to_version

    rollback_to_version(version)


def _review_unsettled_bets_before_games() -> None:
    """Prompt to settle prior bets before displaying today's schedule."""
    from app.services.bet_ledger import list_unsettled, settle_bet, summarize_bets

    yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
    pending = list_unsettled(through_date=yesterday)
    if not pending:
        return

    console.print(f"\n[bold yellow]You have {len(pending)} unsettled bet(s) from {yesterday} or earlier.[/bold yellow]")
    if not Confirm.ask("Review and settle them now?", default=True):
        return

    for bet in pending:
        console.print(
            f"\n[bold]{bet.pitcher_name}[/bold] — {bet.side} {bet.strikeout_line} "
            f"at {bet.american_odds:+d} | Risked ${bet.amount_risked:.2f} | Date {bet.game_date}"
        )
        if not Confirm.ask("Settle this bet now?", default=True):
            continue
        actual = IntPrompt.ask("Actual strikeouts", default=0)
        settled = settle_bet(bet.id, actual)
        style = "green" if settled.result == "WIN" else "red" if settled.result == "LOSS" else "yellow"
        console.print(
            f"[{style}]{settled.result}[/{style}] | Profit/Loss: ${settled.profit_loss:+.2f}"
        )

    summary = summarize_bets()
    console.print(
        f"\n[bold]Betting record:[/bold] {summary.wins}-{summary.losses}-{summary.pushes} "
        f"| Profit/Loss: ${summary.profit_loss:+.2f} | ROI: {summary.roi:+.1%}"
    )


def _prompt_to_record_bet(*, projection_id: str, game: dict, game_date: str, pitcher_id: int,
                          pitcher_name: str, opponent: str, result, market_snapshot) -> None:
    """Record only wagers the user actually placed."""
    from app.services.bet_ledger import record_bet

    if not Confirm.ask("Did you place a bet on this pitcher prop?", default=False):
        return

    default_line = None
    default_over = None
    default_under = None
    if market_snapshot is not None:
        default_line = market_snapshot.strikeout_line
        default_over = market_snapshot.over_odds
        default_under = market_snapshot.under_odds

    while True:
        line = FloatPrompt.ask("Strikeout line", default=float(default_line or 5.5))
        if line > 0:
            break
        console.print("[red]Strikeout line must be positive, such as 5.5 or 6.5.[/red]")

    side = Prompt.ask("Bet side", choices=["OVER", "UNDER"], default="UNDER").upper()
    default_odds = default_over if side == "OVER" else default_under
    odds = IntPrompt.ask("American odds", default=int(default_odds or -110))
    amount = FloatPrompt.ask("Amount risked", default=10.0)
    sportsbook = Prompt.ask("Sportsbook", default="")
    notes = Prompt.ask("Optional note", default="")

    floor_line = int(line // 1)
    over_probability = sum(
        probability for strikeouts, probability in result.probability_by_k.items()
        if strikeouts > floor_line
    )
    model_probability = over_probability if side == "OVER" else 1.0 - over_probability

    bet = record_bet(
        projection_id=projection_id,
        game_id=str(game.get("game_id")) if game.get("game_id") is not None else None,
        game_date=game_date,
        pitcher_id=pitcher_id,
        pitcher_name=pitcher_name,
        opponent_team=opponent,
        side=side,
        strikeout_line=line,
        american_odds=odds,
        amount_risked=amount,
        sportsbook=sportsbook or None,
        model_probability=model_probability,
        model_projection=result.final_blended_projection,
        confidence_rating=result.confidence_rating,
        notes=notes or None,
    )
    console.print(f"[green]Bet saved to SQLite. Bet ID: {bet.id}[/green]")
    console.print("[dim]CSV backup updated at data/exports/bets.csv[/dim]")


@app.command("settle-bets")
def settle_bets_command():
    """Review and settle all unresolved bets (both strikeout and NRFI/YRFI)."""
    from app.services.bet_ledger import list_unsettled, settle_bet, settle_nrfi_bet, summarize_bets

    pending = list_unsettled()
    if not pending:
        console.print("[green]No unresolved bets.[/green]")
        return

    for bet in pending:
        if bet.market_type == "nrfi_yrfi":
            # BUG FIX: this branch previously didn't exist at all -- every
            # unsettled bet, regardless of market, was asked for "Actual
            # strikeouts". NRFI/YRFI bets never had a strikeout total to
            # begin with, so this now asks the question that actually
            # applies to this market.
            console.print(f"\n[bold]Settling {bet.side} bet[/bold]")
            console.print(f"  Game: {bet.pitcher_name or bet.game_id} ({bet.game_date})")
            console.print(f"  Side: {bet.side} at {bet.american_odds:+d} | Risked ${bet.amount_risked:.2f}")
            if Confirm.ask("Settle this bet?", default=True):
                run_occurred = Confirm.ask("Did either team score in the first inning?", default=False)
                settled = settle_nrfi_bet(bet.id, run_occurred=run_occurred)
                console.print(f"Result: {settled.result} | Profit/Loss: ${settled.profit_loss:+.2f}")
        else:
            console.print(
                f"\n[bold]{bet.pitcher_name}[/bold] — {bet.side} {bet.strikeout_line} "
                f"at {bet.american_odds:+d} | Risked ${bet.amount_risked:.2f} | {bet.game_date}"
            )
            if Confirm.ask("Settle this bet?", default=True):
                actual = IntPrompt.ask("Actual strikeouts", default=0)
                settled = settle_bet(bet.id, actual)
                console.print(f"Result: {settled.result} | Profit/Loss: ${settled.profit_loss:+.2f}")

    summary = summarize_bets()
    console.print(
        f"Record: {summary.wins}-{summary.losses}-{summary.pushes} | "
        f"Profit/Loss: ${summary.profit_loss:+.2f} | ROI: {summary.roi:+.1%}"
    )


@app.command("bet-history")
def bet_history(limit: int = typer.Option(100, help="Maximum number of bets to display")):
    """Display the betting ledger and running totals."""
    from app.services.bet_ledger import list_bets, summarize_bets

    bets = list_bets(limit=limit)
    table = Table(title="Bet History")
    for col in ["Date", "Pitcher", "Bet", "Odds", "Risked", "Actual K", "Result", "P/L"]:
        table.add_column(col)
    for bet in bets:
        table.add_row(
            bet.game_date, bet.pitcher_name, f"{bet.side} {bet.strikeout_line}",
            f"{bet.american_odds:+d}", f"${bet.amount_risked:.2f}",
            "-" if bet.actual_strikeouts is None else str(bet.actual_strikeouts),
            bet.result or "UNSETTLED",
            "-" if bet.profit_loss is None else f"${bet.profit_loss:+.2f}",
        )
    console.print(table)
    summary = summarize_bets(bets)
    console.print(
        f"[bold]Record:[/bold] {summary.wins}-{summary.losses}-{summary.pushes} | "
        f"Unsettled: {summary.unresolved} | Risked: ${summary.total_risked:.2f} | "
        f"P/L: ${summary.profit_loss:+.2f} | ROI: {summary.roi:+.1%}"
    )


@app.command("export-bets")
def export_bets_command(path: Optional[str] = typer.Option(None, help="Optional CSV output path")):
    """Export the SQLite betting ledger to CSV."""
    from pathlib import Path
    from app.services.bet_ledger import export_bets_csv

    exported = export_bets_csv(Path(path) if path else None)
    console.print(f"[green]Exported bets to {exported}[/green]")


# ============================================================================
# NRFI / YRFI commands
# ============================================================================

@app.command("nrfi-project")
def nrfi_project(
    date: Optional[str] = typer.Option(None, help="Game date YYYY-MM-DD (defaults to today)"),
):
    """Run an NRFI/YRFI projection for a selected game."""
    from app.cli.nrfi_interactive import confirm_pitchers_known
    from app.markets.nrfi_edge_analysis import analyze_nrfi_edge
    from app.reporting.nrfi_display import print_daily_nrfi_comparison, print_nrfi_edge, print_nrfi_explanation, print_nrfi_summary
    from app.services.bet_ledger import record_nrfi_bet
    from app.services.nrfi_pipeline import NrfiPipeline

    pipeline = NrfiPipeline()
    game_date = date or prompt_game_date()
    console.print(f"[bold]Fetching schedule for {game_date}...[/bold]")
    games = pipeline.get_schedule(game_date)

    game = select_game(games)
    if game is None:
        raise typer.Exit(code=1)

    if not confirm_pitchers_known(game):
        console.print("[yellow]Both starting pitchers must be announced for an NRFI/YRFI projection. Try again closer to game time.[/yellow]")
        raise typer.Exit(code=1)

    season = dt.date.fromisoformat(game_date).year
    console.print("[bold]Gathering data and running the NRFI/YRFI model...[/bold]")

    result, projection_id = pipeline.run(game=game, season=season, pitchers_confirmed=True)

    matchup = f"{game['away_team']} @ {game['home_team']}"
    print_nrfi_summary(
        matchup=matchup, game_time=game.get("scheduled_start_utc", "?"),
        home_pitcher=game.get("probable_home_pitcher_name", "?"), away_pitcher=game.get("probable_away_pitcher_name", "?"),
        home_hand=None, away_hand=None,
        lineup_status="confirmed" if result.confidence.score >= 78 else "projected",
        result=result,
    )
    print_nrfi_explanation(result)

    odds = prompt_nrfi_odds()
    edge_analysis = None
    if odds is not None:
        nrfi_odds, yrfi_odds = odds
        edge_analysis = analyze_nrfi_edge(
            nrfi_odds=nrfi_odds, yrfi_odds=yrfi_odds,
            model_nrfi_probability=result.game_result.nrfi_probability,
            model_yrfi_probability=result.game_result.yrfi_probability,
            lineup_confirmed=True,
        )
        print_nrfi_edge(edge_analysis)

        if Confirm.ask("Did you place a bet on this NRFI/YRFI line?", default=False):
            side = Prompt.ask("Bet side", choices=["NRFI", "YRFI"], default=edge_analysis.recommended_side if edge_analysis.recommended_side != "PASS" else "NRFI")
            default_odds = nrfi_odds if side == "NRFI" else yrfi_odds
            bet_odds = IntPrompt.ask("American odds", default=default_odds)
            amount = FloatPrompt.ask("Amount risked", default=10.0)
            sportsbook = Prompt.ask("Sportsbook", default="")
            model_prob = result.game_result.nrfi_probability if side == "NRFI" else result.game_result.yrfi_probability
            bet = record_nrfi_bet(
                game_date=game_date, side=side, american_odds=bet_odds, amount_risked=amount,
                nrfi_projection_id=projection_id, game_id=game.get("game_id"), matchup_label=matchup,
                sportsbook=sportsbook or None, model_probability=model_prob,
                model_projection=result.game_result.expected_first_inning_runs,
                confidence_rating=str(result.confidence.score),
            )
            console.print(f"[green]NRFI/YRFI bet saved. Bet ID: {bet.id}[/green]")

    console.print(f"\n[dim]NRFI/YRFI projection saved. ID: {projection_id}[/dim]")


@app.command("nrfi-backfill")
def nrfi_backfill_command(
    start_date: Optional[str] = typer.Option(None, help="Backfill start date YYYY-MM-DD"),
    end_date: Optional[str] = typer.Option(None, help="Backfill end date YYYY-MM-DD"),
    season: Optional[int] = typer.Option(None, help="Backfill an entire season instead of a date range"),
    recent_days: Optional[int] = typer.Option(None, help="Backfill/update just the last N days"),
):
    """Backfill historical first-inning results (date range, season, or recent)."""
    from app.services.nrfi_backfill import NrfiBackfillService

    service = NrfiBackfillService()
    if recent_days is not None:
        service.update_recent(days_back=recent_days)
    elif season is not None:
        service.backfill_season(season)
    elif start_date and end_date:
        service.backfill_date_range(start_date, end_date, season=dt.date.fromisoformat(start_date).year)
    else:
        console.print("[red]Provide --start-date/--end-date, --season, or --recent-days.[/red]")
        raise typer.Exit(code=1)


@app.command("nrfi-update-results")
def nrfi_update_results_command():
    """Grade completed games against pending NRFI/YRFI projections and
    automatically settle any matching unsettled NRFI/YRFI bets."""
    from app.services.nrfi_results_updater import update_all_pending_nrfi_results

    console.print("[bold]Updating NRFI/YRFI results...[/bold]")
    n_projections, n_bets = update_all_pending_nrfi_results()
    console.print(f"[green]Updated {n_projections} NRFI/YRFI projection(s) with actual results.[/green]")
    if n_bets:
        console.print(f"[green]Automatically settled {n_bets} NRFI/YRFI bet(s).[/green]")


@app.command("nrfi-history")
def nrfi_history_command(
    date: Optional[str] = typer.Option(None),
    team: Optional[str] = typer.Option(None),
    limit: int = typer.Option(50),
):
    """Browse historical NRFI/YRFI projections and outcomes."""
    from app.database.repositories import NrfiProjectionRepository
    from app.database.session import session_scope

    with session_scope() as session:
        projections = NrfiProjectionRepository.list_filtered(session, date=date, team=team, limit=limit)

        table = Table(title="NRFI/YRFI Projection History")
        for col in ["Date", "Matchup", "Lineup", "NRFI %", "Threat (A/H)", "Confidence", "Actual"]:
            table.add_column(col)
        for p in projections:
            actual = p.actual_result
            actual_str = "-"
            if actual is not None and actual.is_nrfi is not None:
                actual_str = "NRFI" if actual.is_nrfi else "YRFI"
            table.add_row(
                p.game_date, f"{p.away_team} @ {p.home_team}", p.lineup_status,
                f"{p.nrfi_probability*100:.1f}%" if p.nrfi_probability is not None else "-",
                f"{p.away_threat_score:.0f}/{p.home_threat_score:.0f}" if p.away_threat_score is not None else "-",
                f"{p.confidence_score:.0f}" if p.confidence_score is not None else "-",
                actual_str,
            )
        console.print(table)
        console.print(f"[dim]{len(projections)} NRFI/YRFI projection(s) shown.[/dim]")


@app.command("nrfi-backtest")
def nrfi_backtest_command(
    start_date: str = typer.Option(..., help="Backtest start date YYYY-MM-DD"),
    end_date: str = typer.Option(..., help="Backtest end date YYYY-MM-DD"),
):
    """Backtest the NRFI/YRFI model against stored, graded projections."""
    from app.evaluation.nrfi_backtester import run_nrfi_backtest

    run_nrfi_backtest(start_date, end_date)


@app.command("nrfi-train")
def nrfi_train_command():
    """Train/validate the NRFI/YRFI model on accumulated backfilled history."""
    from app.training.nrfi_retrain import run_nrfi_retraining

    run_nrfi_retraining()


@app.command("nrfi-bet-history")
def nrfi_bet_history_command(limit: int = typer.Option(100)):
    """Display NRFI/YRFI-specific betting history and totals."""
    from app.services.bet_ledger import list_bets_by_market, summarize_bets

    bets = list_bets_by_market(market_type="nrfi_yrfi", limit=limit)
    table = Table(title="NRFI/YRFI Bet History")
    for col in ["Date", "Matchup", "Side", "Odds", "Risked", "Actual", "Result", "P/L"]:
        table.add_column(col)
    for bet in bets:
        table.add_row(
            bet.game_date, bet.pitcher_name or "-", bet.side, f"{bet.american_odds:+d}",
            f"${bet.amount_risked:.2f}", bet.actual_nrfi_result or "-", bet.result or "UNSETTLED",
            "-" if bet.profit_loss is None else f"${bet.profit_loss:+.2f}",
        )
    console.print(table)
    summary = summarize_bets(bets)
    console.print(
        f"[bold]Record:[/bold] {summary.wins}-{summary.losses}-{summary.pushes} | "
        f"Unsettled: {summary.unresolved} | Risked: ${summary.total_risked:.2f} | "
        f"P/L: ${summary.profit_loss:+.2f} | ROI: {summary.roi:+.1%}"
    )


@app.command("menu")
def menu_command():
    """Show the full command menu (both strikeout and NRFI/YRFI workflows)."""
    console.print(
        "\n[bold]MLB Strikeout + NRFI/YRFI Projection System[/bold]\n\n"
        "  1. python main.py project              Run strikeout projections\n"
        "  2. python main.py nrfi-project          Run NRFI/YRFI projections\n"
        "  3. python main.py both                  Run both projection types for one game\n"
        "  4. python main.py settle-bets           Enter unresolved strikeout bet results\n"
        "     python main.py nrfi-update-results    Grade unresolved NRFI/YRFI results\n"
        "  5. python main.py bet-history           View strikeout betting history\n"
        "     python main.py nrfi-bet-history       View NRFI/YRFI betting history\n"
        "  6. python main.py update-results        Update historical strikeout database\n"
        "     python main.py nrfi-backfill          Update historical first-inning database\n"
        "  6b. python main.py model-report          Model health report (accuracy, calibration, bias)\n"
        "  7. python main.py retrain               Train the strikeout model\n"
        "     python main.py nrfi-train             Train the NRFI/YRFI model\n"
        "  8. python main.py backtest              Run strikeout backtest\n"
        "     python main.py nrfi-backtest          Run NRFI/YRFI backtest\n"
        "  9. Exit (Ctrl+C or close the terminal)\n"
    )


@app.command("both")
def both_command(date: Optional[str] = typer.Option(None)):
    """Run both the strikeout and NRFI/YRFI projections for one game in sequence."""
    console.print("[bold]Running strikeout projection first...[/bold]\n")
    project(date=date, seed=None, simulations=None)
    console.print("\n[bold]Now running NRFI/YRFI projection for the same date...[/bold]\n")
    nrfi_project(date=date)
