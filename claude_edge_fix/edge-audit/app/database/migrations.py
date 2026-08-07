"""
Lightweight SQLite migrations.

`Base.metadata.create_all()` (called from init_db) only creates tables that
don't exist yet -- it never alters an existing table's columns. The `bets`
table predates this NRFI/YRFI addition and has real user data in it, so
adding market_type/nrfi columns and loosening strikeout_line's NOT NULL
constraint requires an actual rebuild (SQLite's ALTER TABLE cannot drop or
loosen a column constraint in place).

This migration is idempotent (checks column existence before acting) and
runs inside a single transaction, so it's safe to call on every startup and
never leaves the database half-migrated. Existing rows are carried forward
unchanged with market_type defaulted to 'strikeouts'.
"""
from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.config.logging_config import get_logger

logger = get_logger(__name__)


def _existing_columns(conn, table: str) -> set[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return {row[1] for row in rows}  # row[1] = column name


def migrate_bets_table_for_nrfi(engine: Engine) -> None:
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "bets" not in tables:
            # Fresh database -- create_all() will build the up-to-date
            # schema directly; nothing to migrate.
            return

        columns = _existing_columns(conn, "bets")
        if "market_type" in columns and "actual_nrfi_result" in columns:
            logger.debug("bets table already migrated for NRFI/YRFI support; skipping.")
            return

        logger.info("Migrating 'bets' table to support NRFI/YRFI bets (rebuilding table, preserving all rows)...")

        conn.execute(text("ALTER TABLE bets RENAME TO bets_pre_nrfi_migration"))

        conn.execute(
            text(
                """
                CREATE TABLE bets (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    created_at_utc DATETIME,
                    settled_at_utc DATETIME,
                    market_type VARCHAR,
                    projection_id VARCHAR,
                    nrfi_projection_id VARCHAR,
                    game_id VARCHAR,
                    game_date VARCHAR,
                    pitcher_id INTEGER,
                    pitcher_name VARCHAR,
                    opponent_team VARCHAR,
                    side VARCHAR,
                    strikeout_line FLOAT,
                    american_odds INTEGER,
                    amount_risked FLOAT,
                    sportsbook VARCHAR,
                    model_probability FLOAT,
                    model_projection FLOAT,
                    confidence_rating VARCHAR,
                    edge_grade VARCHAR,
                    actual_strikeouts INTEGER,
                    actual_nrfi_result VARCHAR,
                    result VARCHAR,
                    profit_loss FLOAT,
                    notes TEXT,
                    FOREIGN KEY(projection_id) REFERENCES projections (id),
                    FOREIGN KEY(nrfi_projection_id) REFERENCES nrfi_projections (id)
                )
                """
            )
        )

        old_columns = _existing_columns(conn, "bets_pre_nrfi_migration")
        # Only copy columns that existed on the old table; anything new
        # (market_type, nrfi_projection_id, actual_nrfi_result) gets its
        # column default / NULL for old rows, which is exactly correct --
        # those rows are strikeout bets and have no NRFI data.
        copyable = [
            c for c in [
                "id", "created_at_utc", "settled_at_utc", "projection_id", "game_id",
                "game_date", "pitcher_id", "pitcher_name", "opponent_team", "side",
                "strikeout_line", "american_odds", "amount_risked", "sportsbook",
                "model_probability", "model_projection", "confidence_rating", "edge_grade",
                "actual_strikeouts", "result", "profit_loss", "notes",
            ]
            if c in old_columns
        ]
        col_list = ", ".join(copyable)
        conn.execute(
            text(
                f"INSERT INTO bets (market_type, {col_list}) "
                f"SELECT 'strikeouts', {col_list} FROM bets_pre_nrfi_migration"
            )
        )

        n_migrated = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()
        n_original = conn.execute(text("SELECT COUNT(*) FROM bets_pre_nrfi_migration")).scalar()
        if n_migrated != n_original:
            raise RuntimeError(
                f"Bet migration row-count mismatch (original={n_original}, migrated={n_migrated}); "
                f"aborting so the transaction rolls back and no data is lost."
            )

        conn.execute(text("DROP TABLE bets_pre_nrfi_migration"))
        logger.info("bets table migration complete: %s row(s) preserved.", n_migrated)


def migrate_bets_table_add_first_inning_runs(engine: Engine) -> None:
    """Simple additive column migration -- unlike the market_type migration
    above (which required a full table rebuild to loosen a NOT NULL
    constraint), this just adds one new nullable column, which SQLite
    supports directly via ALTER TABLE ADD COLUMN. Idempotent: checks
    column existence first."""
    with engine.begin() as conn:
        tables = {
            row[0]
            for row in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        }
        if "bets" not in tables:
            return

        columns = _existing_columns(conn, "bets")
        if "first_inning_runs" in columns:
            logger.debug("bets.first_inning_runs already present; skipping.")
            return

        logger.info("Adding 'first_inning_runs' column to bets table...")
        conn.execute(text("ALTER TABLE bets ADD COLUMN first_inning_runs INTEGER"))
        logger.info("bets.first_inning_runs column added.")


def run_all_migrations(engine: Engine) -> None:
    migrate_bets_table_for_nrfi(engine)
    migrate_bets_table_add_first_inning_runs(engine)
