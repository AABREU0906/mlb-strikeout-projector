from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.table import Table

from app.database.repositories import ProjectionRepository
from app.database.session import session_scope

console = Console()


def show_history(
    date: Optional[str] = None,
    pitcher: Optional[str] = None,
    team: Optional[str] = None,
    confidence: Optional[str] = None,
    model_version: Optional[str] = None,
    limit: int = 50,
) -> None:
    with session_scope() as session:
        projections = ProjectionRepository.list_filtered(
            session,
            date=date,
            pitcher_name=pitcher,
            team=team,
            confidence=confidence,
            model_version_label=model_version,
            limit=limit,
        )

        table = Table(title="Projection History")
        for col in [
            "Date", "Pitcher", "Opponent", "Lineup", "Stats-Only", "Market-Informed",
            "Blended", "Line", "Actual K", "Result vs Line", "Confidence",
        ]:
            table.add_column(col)

        for p in projections:
            actual = p.actual_result
            line = None
            if p.market_snapshot_json:
                line = p.market_snapshot_json.get("strikeout_line")
            actual_k = actual.actual_strikeouts if actual else None
            vs_line = ""
            if actual_k is not None and line is not None:
                vs_line = "OVER" if actual_k > line else ("UNDER" if actual_k < line else "PUSH")

            table.add_row(
                p.game_date,
                p.pitcher_name,
                p.opponent_team or "-",
                p.lineup_status,
                f"{p.statistics_only_projection:.1f}" if p.statistics_only_projection is not None else "-",
                f"{p.market_informed_projection:.1f}" if p.market_informed_projection is not None else "-",
                f"{p.final_blended_projection:.1f}" if p.final_blended_projection is not None else "-",
                f"{line}" if line is not None else "-",
                str(actual_k) if actual_k is not None else "-",
                vs_line,
                p.confidence_rating or "-",
            )

        console.print(table)
        console.print(f"[dim]{len(projections)} projection(s) shown.[/dim]")
