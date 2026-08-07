"""CSV export for `python main.py project-confirmed-slate`."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from app.config.settings import settings

FIELDNAMES = [
    "game_date", "game_time", "game_id", "pitcher_id", "pitcher", "team", "opponent",
    "home_or_away", "lineup_status", "sportsbook", "strikeout_line", "over_odds", "under_odds",
    "statistics_only_projection", "market_informed_projection", "final_blended_projection",
    "median_strikeouts", "std_dev", "model_over_probability", "model_under_probability",
    "projection_minus_line", "recommendation", "edge_grade", "confidence", "estimated_ev",
    "expected_innings", "expected_batters_faced", "expected_pitch_count", "workload_source",
    "workload_role", "validation_status", "odds_timestamp", "projection_timestamp", "projection_id",
]


def _exports_dir() -> Path:
    base = Path(settings.database_full_path).resolve().parent.parent
    return base / "exports"


def _next_available_path(game_date: str) -> Path:
    """Never overwrites an existing file: date_confirmed_slate.csv, then
    _2, _3, etc."""
    exports_dir = _exports_dir()
    exports_dir.mkdir(parents=True, exist_ok=True)

    base_name = f"{game_date}_confirmed_slate"
    candidate = exports_dir / f"{base_name}.csv"
    if not candidate.exists():
        return candidate

    n = 2
    while True:
        candidate = exports_dir / f"{base_name}_{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def export_confirmed_slate_csv(rows: list, game_date: str, destination: Optional[Path] = None) -> Path:
    """Writes one row per projected pitcher, using the exact stored
    values already computed by the pipeline/edge-analysis/validator --
    nothing here recomputes a different number for the CSV. Raises on
    failure; the caller (CLI command) is responsible for catching this so
    a CSV write error never causes already-saved projections to be lost
    or the run to abort."""
    path = destination or _next_available_path(game_date)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "game_date": row.game_date,
                "game_time": row.game_time,
                "game_id": row.game_id,
                "pitcher_id": row.pitcher_id,
                "pitcher": row.pitcher_name,
                "team": row.team,
                "opponent": row.opponent,
                "home_or_away": row.home_or_away,
                "lineup_status": row.lineup_status,
                "sportsbook": row.sportsbook,
                "strikeout_line": row.strikeout_line,
                "over_odds": row.over_odds,
                "under_odds": row.under_odds,
                "statistics_only_projection": row.statistics_only_projection,
                "market_informed_projection": row.market_informed_projection,
                "final_blended_projection": row.final_blended_projection,
                "median_strikeouts": row.median_strikeouts,
                "std_dev": row.std_dev,
                "model_over_probability": row.model_over_probability,
                "model_under_probability": row.model_under_probability,
                "projection_minus_line": row.projection_minus_line,
                "recommendation": row.recommended_side,
                "edge_grade": row.edge_grade,
                "confidence": row.confidence,
                "estimated_ev": row.estimated_ev,
                "expected_innings": row.expected_innings,
                "expected_batters_faced": row.expected_batters_faced,
                "expected_pitch_count": row.expected_pitch_count,
                "workload_source": row.workload_source,
                "workload_role": row.workload_role,
                "validation_status": row.validation_status,
                "odds_timestamp": row.odds_timestamp,
                "projection_timestamp": row.projection_timestamp,
                "projection_id": row.projection_id,
            })

    return path
