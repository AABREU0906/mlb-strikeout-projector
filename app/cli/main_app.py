from __future__ import annotations

import datetime as dt
from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from app.cli.interactive import prompt_game_date, prompt_manual_market, prompt_warnings, select_game, select_pitcher
from app.config.logging_config import configure_logging
from app.database.session import init_db
from app.reporting.display import (
    print_batter_matchup_table,
    print_distribution,
    print_explanation,
    print_lineup_status,
    print_main_summary,
    print_market_comparison,
    print_warnings,
)
from app.services.pipeline import ProjectionPipeline

app = typer.Typer(add_completion=False, no_args_is_help=False)
console = Console()


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    configure_logging()
    init_db()
    if ctx.invoked_subcommand is None:
        _run_interactive_projection()


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
    print_batter_matchup_table(result)
    print_explanation(result)
    market_snapshot = None
    if result.market_used.get("snapshot"):
        from app.schemas.market import MarketSnapshot
        market_snapshot = MarketSnapshot(**result.market_used["snapshot"])
    print_market_comparison(market_snapshot, result)
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
def update_results():
    """Fetch actual results for completed games with saved projections."""
    from app.services.results_updater import update_all_pending_results

    console.print("[bold]Updating results for completed games...[/bold]")
    n_updated = update_all_pending_results()
    console.print(f"[green]Updated {n_updated} projection(s) with actual results.[/green]")


@app.command()
def evaluate(
    since: Optional[str] = typer.Option(None, help="Only evaluate games on/after this date (YYYY-MM-DD)"),
):
    """Produce model evaluation reports (MAE, RMSE, calibration, etc.)."""
    from app.evaluation.evaluator import run_evaluation

    run_evaluation(since_date=since)


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
    """Review and settle all unresolved bets."""
    from app.services.bet_ledger import list_unsettled, settle_bet, summarize_bets

    pending = list_unsettled()
    if not pending:
        console.print("[green]No unresolved bets.[/green]")
        return
    for bet in pending:
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
