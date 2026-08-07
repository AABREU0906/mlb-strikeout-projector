"""
Historical data import from CSV.

Each entity has a documented required-column template. `write_templates()`
generates empty template CSVs into data/imports/templates/ so the user has
an exact schema to fill in. `import_games()` / `import_actual_results()`
are the two loaders wired to the database now (games and results are the
two entities the rest of the system directly consumes); the remaining
templates (pitchers, batters, lineups, weather, markets, projection
results) are provided as documented schemas for staging historical data
that feeds into a full historical projection replay via the same
pipeline/backtester code path -- validated the same way, using
`validate_columns()`.

All imports validate required columns up front and raise a clear,
specific error naming the missing/incorrect column rather than partially
importing malformed rows.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from app.config.settings import PROJECT_ROOT
from app.database.models import ActualResult, Game
from app.database.session import session_scope

TEMPLATES_DIR = PROJECT_ROOT / "data" / "imports" / "templates"

TEMPLATES: dict[str, list[str]] = {
    "games": [
        "game_id", "game_date", "scheduled_start_utc", "home_team", "away_team",
        "home_team_id", "away_team_id", "ballpark", "venue_id", "status",
        "probable_home_pitcher_id", "probable_away_pitcher_id",
    ],
    "pitchers": [
        "pitcher_id", "name", "throws", "season", "season_bf", "season_strikeouts",
        "season_walks", "games_started", "innings_pitched", "pitches_thrown",
        "bf_vs_rhb", "k_vs_rhb", "bf_vs_lhb", "k_vs_lhb",
    ],
    "batters": [
        "batter_id", "name", "bat_side", "season", "season_pa", "season_strikeouts",
        "season_walks", "pa_vs_rhp", "k_vs_rhp", "pa_vs_lhp", "k_vs_lhp",
    ],
    "lineups": [
        "game_id", "team_id", "batting_order", "player_id", "bat_side", "position", "status",
    ],
    "weather": [
        "game_id", "temperature_f", "humidity_pct", "wind_speed_mph", "wind_direction_deg",
        "precipitation_probability_pct", "roof_status", "retrieved_at",
    ],
    "sportsbook_markets": [
        "game_id", "pitcher_id", "sportsbook", "strikeout_line", "over_odds", "under_odds",
        "opening_strikeout_line", "opening_over_odds", "opening_under_odds",
        "game_total", "team_moneyline", "opponent_moneyline", "timestamp_utc",
    ],
    "projection_results": [
        "projection_id", "game_id", "pitcher_id", "game_date", "statistics_only_projection",
        "market_informed_projection", "final_blended_projection", "confidence_rating",
        "model_version_label", "created_at_utc",
    ],
    "actual_results": [
        "projection_id", "game_id", "pitcher_id", "actual_strikeouts", "actual_innings_pitched",
        "actual_batters_faced", "actual_pitch_count", "removed_with_injury", "game_delayed",
        "closing_line", "closing_over_odds", "closing_under_odds",
    ],
}


class ImportValidationError(Exception):
    pass


def write_templates() -> list[Path]:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, columns in TEMPLATES.items():
        path = TEMPLATES_DIR / f"{name}_template.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        written.append(path)
    return written


def _read_rows(csv_path: Path) -> Iterable[dict]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def validate_columns(csv_path: Path, entity: str) -> None:
    required = set(TEMPLATES[entity])
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        actual = set(reader.fieldnames or [])
    missing = required - actual
    if missing:
        raise ImportValidationError(
            f"{csv_path.name}: missing required column(s) for '{entity}': {sorted(missing)}"
        )


def import_games(csv_path: Path) -> int:
    validate_columns(csv_path, "games")
    n = 0
    with session_scope() as session:
        for row in _read_rows(csv_path):
            if not row.get("game_id"):
                raise ImportValidationError(f"Row missing game_id: {row}")
            existing = session.get(Game, row["game_id"])
            if existing:
                continue
            session.add(
                Game(
                    id=row["game_id"],
                    game_date=row["game_date"],
                    scheduled_start_utc=row["scheduled_start_utc"],
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    home_team_id=int(row["home_team_id"]),
                    away_team_id=int(row["away_team_id"]),
                    ballpark=row.get("ballpark") or None,
                    venue_id=int(row["venue_id"]) if row.get("venue_id") else None,
                    status=row.get("status") or None,
                    probable_home_pitcher_id=int(row["probable_home_pitcher_id"]) if row.get("probable_home_pitcher_id") else None,
                    probable_away_pitcher_id=int(row["probable_away_pitcher_id"]) if row.get("probable_away_pitcher_id") else None,
                    source="csv_import",
                )
            )
            n += 1
    return n


def import_actual_results(csv_path: Path) -> int:
    validate_columns(csv_path, "actual_results")
    n = 0
    with session_scope() as session:
        for row in _read_rows(csv_path):
            if not row.get("projection_id"):
                raise ImportValidationError(f"Row missing projection_id: {row}")
            existing = (
                session.query(ActualResult)
                .filter(ActualResult.projection_id == row["projection_id"])
                .one_or_none()
            )
            if existing:
                continue
            session.add(
                ActualResult(
                    projection_id=row["projection_id"],
                    actual_strikeouts=int(row["actual_strikeouts"]) if row.get("actual_strikeouts") else None,
                    actual_innings_pitched=float(row["actual_innings_pitched"]) if row.get("actual_innings_pitched") else None,
                    actual_batters_faced=int(row["actual_batters_faced"]) if row.get("actual_batters_faced") else None,
                    actual_pitch_count=int(row["actual_pitch_count"]) if row.get("actual_pitch_count") else None,
                    removed_with_injury=(row.get("removed_with_injury") or "").lower() in ("1", "true", "yes"),
                    game_delayed=(row.get("game_delayed") or "").lower() in ("1", "true", "yes"),
                    closing_line=float(row["closing_line"]) if row.get("closing_line") else None,
                    closing_over_odds=int(row["closing_over_odds"]) if row.get("closing_over_odds") else None,
                    closing_under_odds=int(row["closing_under_odds"]) if row.get("closing_under_odds") else None,
                    source="csv_import",
                )
            )
            n += 1
    return n
