"""Terminal display for `python main.py project-confirmed-slate`."""
from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()

_GRADE_RANK = {
    "Elite estimated edge": 5, "Strong estimated edge": 4, "Moderate estimated edge": 3,
    "Small estimated edge": 2, "No positive estimated edge": 1, None: 0,
}


def _sort_key(row):
    """Actionable recommendations ranked ahead of PASS, and among
    actionable rows, sorted by the EXISTING edge grade/EV -- never a new
    recommendation formula, just a display ordering over already-computed
    values."""
    is_actionable = row.recommended_side in ("OVER", "UNDER")
    grade_rank = _GRADE_RANK.get(row.edge_grade, 0)
    ev = row.estimated_ev if row.estimated_ev is not None else -999
    return (0 if is_actionable else 1, -grade_rank, -ev)


def print_confirmed_slate(result) -> None:
    console.print("\n[bold]TODAY'S CONFIRMED STRIKEOUT SLATE[/bold]\n")

    if result.skipped_games:
        console.print("[dim]Skipped games:[/dim]")
        for g in result.skipped_games:
            console.print(f"  {g.matchup}\n  [yellow]SKIPPED[/yellow] \u2014 {g.reason}")
        console.print()

    if result.failed_items:
        console.print("[red]Failed (isolated, rest of slate unaffected):[/red]")
        for f in result.failed_items:
            console.print(f"  {f.label}: {f.reason}")
        console.print()

    if result.rows:
        table = Table(show_lines=True)
        for col in (
            "Pitcher", "Opponent", "Line", "Projection", "Diff", "Over", "Under",
            "Mdl Over%", "Mdl Under%", "Rec", "Edge Grade", "Confidence", "EV",
        ):
            table.add_column(col)

        for row in sorted(result.rows, key=_sort_key):
            line = f"{row.strikeout_line}" if row.strikeout_line is not None else "-"
            diff = f"{row.projection_minus_line:+.2f}" if row.projection_minus_line is not None else "-"
            over_odds = f"{row.over_odds:+d}" if row.over_odds is not None else "-"
            under_odds = f"{row.under_odds:+d}" if row.under_odds is not None else "-"
            mo = f"{row.model_over_probability*100:.0f}%" if row.model_over_probability is not None else "-"
            mu = f"{row.model_under_probability*100:.0f}%" if row.model_under_probability is not None else "-"
            rec = row.recommended_side or ("INVALID" if row.validation_status == "invalid" else "-")
            grade = row.edge_grade or "-"
            conf = row.confidence or "-"
            ev = f"{row.estimated_ev*100:+.1f}%" if row.estimated_ev is not None else "-"

            table.add_row(
                row.pitcher_name, row.opponent, line, f"{row.final_blended_projection:.2f}",
                diff, over_odds, under_odds, mo, mu, rec, grade, conf, ev,
            )
        console.print(table)
    else:
        console.print("[dim]No pitchers were projected this run.[/dim]")

    s = result.summary
    console.print()
    console.print("[bold]Summary[/bold]")
    console.print(f"  MLB games today:              {s.games_today}")
    console.print(f"  Games with confirmed lineups:  {s.games_confirmed}")
    console.print(f"  Already projected today:       {s.already_projected_today}")
    console.print(f"  New games projected:           {s.new_games_projected}")
    console.print(f"  Games skipped:                 {s.games_skipped}")
    console.print(f"  Pitchers projected:            {s.pitchers_projected}")
    console.print(f"  FanDuel markets found:         {s.fanduel_markets_found}")
    console.print(f"  FanDuel markets unavailable:   {s.fanduel_markets_unavailable}")
    console.print(f"  Actionable recommendations:    {s.actionable_recommendations}")
    console.print(f"  PASS:                          {s.pass_count}")
    console.print(f"  Projections saved:             {s.projections_saved}")

    if not s.odds_api_configured:
        console.print("  [yellow]ODDS_API_KEY not set \u2014 ran without odds.[/yellow]")
    else:
        console.print(f"  Odds API events-list calls:    {s.events_list_calls}")
        console.print(f"  Odds API player-prop calls:    {s.event_odds_calls}")
        credits_used = s.credits_used_this_run if s.credits_used_this_run is not None else "unavailable"
        credits_remaining = s.credits_remaining if s.credits_remaining is not None else "unavailable"
        console.print(f"  Credits used this run:         {credits_used}")
        console.print(f"  Credits remaining:             {credits_remaining}")
