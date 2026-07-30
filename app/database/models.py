"""
ORM models.

Design notes:
- Every "point of data" that must record provenance (source + retrieved_at)
  is stored either as its own row in `data_source_log`, or as a JSON blob
  that includes `source` and `retrieved_at` keys per field. Bulk structured
  data (batter inputs, pitcher inputs, weather, warnings, market snapshot)
  is stored as JSON on the Projection row so a historical projection can be
  fully reproduced/audited without re-querying live sources.
- Pregame and postgame data are kept in separate tables
  (Projection vs ActualResult) and are never merged in place.
"""
from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Game(Base):
    __tablename__ = "games"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # MLB gamePk as string
    game_date: Mapped[str] = mapped_column(String, index=True)  # YYYY-MM-DD
    scheduled_start_utc: Mapped[str] = mapped_column(String)
    home_team: Mapped[str] = mapped_column(String)
    away_team: Mapped[str] = mapped_column(String)
    home_team_id: Mapped[int] = mapped_column(Integer)
    away_team_id: Mapped[int] = mapped_column(Integer)
    ballpark: Mapped[str] = mapped_column(String, nullable=True)
    venue_id: Mapped[int] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=True)
    doubleheader: Mapped[str] = mapped_column(String, nullable=True)  # 'N','Y','S' (split)
    game_number: Mapped[int] = mapped_column(Integer, default=1)
    postponed: Mapped[bool] = mapped_column(Boolean, default=False)
    probable_home_pitcher_id: Mapped[int] = mapped_column(Integer, nullable=True)
    probable_away_pitcher_id: Mapped[int] = mapped_column(Integer, nullable=True)
    raw_source_payload: Mapped[dict] = mapped_column(JSON, nullable=True)
    source: Mapped[str] = mapped_column(String, default="mlb_stats_api")
    retrieved_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    projections: Mapped[list["Projection"]] = relationship(back_populates="game")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    version_label: Mapped[str] = mapped_column(String, unique=True)
    model_type: Mapped[str] = mapped_column(String)  # baseline | stats_ml | market_ml | ensemble
    algorithm: Mapped[str] = mapped_column(String, nullable=True)
    trained_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    training_window_start: Mapped[str] = mapped_column(String, nullable=True)
    training_window_end: Mapped[str] = mapped_column(String, nullable=True)
    feature_list: Mapped[dict] = mapped_column(JSON, nullable=True)
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=True)
    validation_metrics: Mapped[dict] = mapped_column(JSON, nullable=True)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False)
    promotion_decision_notes: Mapped[str] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    n_training_observations: Mapped[int] = mapped_column(Integer, nullable=True)


class Projection(Base):
    __tablename__ = "projections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    created_at_utc: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    game_id: Mapped[str] = mapped_column(String, ForeignKey("games.id"))
    game_date: Mapped[str] = mapped_column(String, index=True)
    game_start_utc: Mapped[str] = mapped_column(String, nullable=True)

    pitcher_id: Mapped[int] = mapped_column(Integer, index=True)
    pitcher_name: Mapped[str] = mapped_column(String)
    pitcher_team: Mapped[str] = mapped_column(String, nullable=True)
    opponent_team: Mapped[str] = mapped_column(String, nullable=True)
    ballpark: Mapped[str] = mapped_column(String, nullable=True)

    lineup_status: Mapped[str] = mapped_column(String)  # "confirmed" | "projected"
    lineup_source: Mapped[str] = mapped_column(String, nullable=True)
    lineup_retrieved_at: Mapped[str] = mapped_column(String, nullable=True)
    lineup_json: Mapped[list] = mapped_column(JSON)  # full batter-by-batter lineup

    pitcher_inputs_json: Mapped[dict] = mapped_column(JSON)
    batter_inputs_json: Mapped[list] = mapped_column(JSON)
    team_inputs_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    weather_inputs_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    umpire_inputs_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    workload_inputs_json: Mapped[dict] = mapped_column(JSON, nullable=True)

    news_warnings_json: Mapped[list] = mapped_column(JSON, nullable=True)
    injury_warnings_json: Mapped[list] = mapped_column(JSON, nullable=True)
    workload_warnings_json: Mapped[list] = mapped_column(JSON, nullable=True)

    market_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    market_timestamp_utc: Mapped[str] = mapped_column(String, nullable=True)
    market_source: Mapped[str] = mapped_column(String, nullable=True)
    manual_line_override: Mapped[bool] = mapped_column(Boolean, default=False)

    statistics_only_projection: Mapped[float] = mapped_column(Float, nullable=True)
    market_informed_projection: Mapped[float] = mapped_column(Float, nullable=True)
    final_blended_projection: Mapped[float] = mapped_column(Float, nullable=True)
    median_strikeouts: Mapped[float] = mapped_column(Float, nullable=True)
    std_dev: Mapped[float] = mapped_column(Float, nullable=True)

    simulation_distribution_json: Mapped[dict] = mapped_column(JSON, nullable=True)  # {k: prob}
    percentiles_json: Mapped[dict] = mapped_column(JSON, nullable=True)

    expected_innings: Mapped[float] = mapped_column(Float, nullable=True)
    expected_batters_faced: Mapped[float] = mapped_column(Float, nullable=True)
    expected_pitch_count: Mapped[float] = mapped_column(Float, nullable=True)

    confidence_rating: Mapped[str] = mapped_column(String, nullable=True)  # High/Medium/Low/Avoid
    confidence_factors_json: Mapped[dict] = mapped_column(JSON, nullable=True)

    explanation_json: Mapped[dict] = mapped_column(JSON, nullable=True)  # positive/negative factors

    model_version_id: Mapped[str] = mapped_column(String, ForeignKey("model_versions.id"), nullable=True)
    model_version_label: Mapped[str] = mapped_column(String, nullable=True)
    data_source_versions_json: Mapped[dict] = mapped_column(JSON, nullable=True)
    random_seed: Mapped[int] = mapped_column(Integer, nullable=True)
    n_simulations: Mapped[int] = mapped_column(Integer, nullable=True)

    game: Mapped["Game"] = relationship(back_populates="projections")
    actual_result: Mapped["ActualResult"] = relationship(back_populates="projection", uselist=False)


class ActualResult(Base):
    """Postgame data. Never overwrites Projection; stored separately by design."""

    __tablename__ = "actual_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    projection_id: Mapped[str] = mapped_column(String, ForeignKey("projections.id"), unique=True)
    recorded_at_utc: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)

    actual_strikeouts: Mapped[int] = mapped_column(Integer, nullable=True)
    actual_innings_pitched: Mapped[float] = mapped_column(Float, nullable=True)
    actual_batters_faced: Mapped[int] = mapped_column(Integer, nullable=True)
    actual_pitch_count: Mapped[int] = mapped_column(Integer, nullable=True)
    actual_lineup_json: Mapped[list] = mapped_column(JSON, nullable=True)
    game_result_json: Mapped[dict] = mapped_column(JSON, nullable=True)

    removed_with_injury: Mapped[bool] = mapped_column(Boolean, nullable=True)
    game_delayed: Mapped[bool] = mapped_column(Boolean, nullable=True)

    closing_line: Mapped[float] = mapped_column(Float, nullable=True)
    closing_over_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    closing_under_odds: Mapped[int] = mapped_column(Integer, nullable=True)
    closing_line_source: Mapped[str] = mapped_column(String, nullable=True)
    closing_line_retrieved_at: Mapped[str] = mapped_column(String, nullable=True)

    source: Mapped[str] = mapped_column(String, default="mlb_stats_api")

    projection: Mapped["Projection"] = relationship(back_populates="actual_result")


class DataSourceLog(Base):
    """Generic provenance log: which source supplied a data point and when."""

    __tablename__ = "data_source_log"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    entity_type: Mapped[str] = mapped_column(String)  # e.g. "pitcher_stats", "lineup", "odds"
    entity_key: Mapped[str] = mapped_column(String)   # e.g. pitcher_id, game_id
    source_name: Mapped[str] = mapped_column(String)
    retrieved_at_utc: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    detail: Mapped[str] = mapped_column(Text, nullable=True)


class CacheEntry(Base):
    """Fallback DB-backed cache record (the primary cache is file-based; this
    table lets `history`/debugging inspect what was cached and when)."""

    __tablename__ = "cache_entries"

    cache_key: Mapped[str] = mapped_column(String, primary_key=True)
    category: Mapped[str] = mapped_column(String)
    stored_at_utc: Mapped[dt.datetime] = mapped_column(DateTime, default=_now)
    expires_at_utc: Mapped[dt.datetime] = mapped_column(DateTime)
