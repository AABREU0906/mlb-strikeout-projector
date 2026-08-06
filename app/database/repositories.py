"""
Repository layer: all reads/writes to the database go through here so the
CLI and services never construct raw SQL/ORM queries directly.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import ActualResult, Bet, DataSourceLog, Game, ModelVersion, Projection


class GameRepository:
    @staticmethod
    def upsert(session: Session, game_dict: dict) -> Game:
        existing = session.get(Game, game_dict["game_id"])
        if existing:
            existing.status = game_dict.get("status")
            existing.probable_home_pitcher_id = game_dict.get("probable_home_pitcher_id")
            existing.probable_away_pitcher_id = game_dict.get("probable_away_pitcher_id")
            existing.raw_source_payload = game_dict.get("raw")
            existing.retrieved_at = dt.datetime.now(dt.timezone.utc)
            return existing

        game = Game(
            id=game_dict["game_id"],
            game_date=game_dict["game_date"],
            scheduled_start_utc=game_dict["scheduled_start_utc"],
            home_team=game_dict["home_team"],
            away_team=game_dict["away_team"],
            home_team_id=game_dict["home_team_id"],
            away_team_id=game_dict["away_team_id"],
            ballpark=game_dict.get("ballpark"),
            venue_id=game_dict.get("venue_id"),
            status=game_dict.get("status"),
            doubleheader=game_dict.get("doubleheader"),
            game_number=game_dict.get("game_number", 1),
            probable_home_pitcher_id=game_dict.get("probable_home_pitcher_id"),
            probable_away_pitcher_id=game_dict.get("probable_away_pitcher_id"),
            raw_source_payload=game_dict.get("raw"),
        )
        session.add(game)
        return game


class ProjectionRepository:
    @staticmethod
    def save(session: Session, projection: Projection) -> Projection:
        session.add(projection)
        session.flush()
        return projection

    @staticmethod
    def get(session: Session, projection_id: str) -> Optional[Projection]:
        return session.get(Projection, projection_id)

    @staticmethod
    def list_without_results(session: Session, before_date: Optional[str] = None) -> list[Projection]:
        stmt = select(Projection).outerjoin(ActualResult).where(ActualResult.id.is_(None))
        if before_date:
            stmt = stmt.where(Projection.game_date <= before_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_filtered(
        session: Session,
        date: Optional[str] = None,
        pitcher_name: Optional[str] = None,
        team: Optional[str] = None,
        confidence: Optional[str] = None,
        model_version_label: Optional[str] = None,
        limit: int = 100,
    ) -> list[Projection]:
        stmt = select(Projection).order_by(Projection.created_at_utc.desc()).limit(limit)
        if date:
            stmt = stmt.where(Projection.game_date == date)
        if pitcher_name:
            stmt = stmt.where(Projection.pitcher_name.ilike(f"%{pitcher_name}%"))
        if team:
            stmt = stmt.where(
                (Projection.pitcher_team.ilike(f"%{team}%")) | (Projection.opponent_team.ilike(f"%{team}%"))
            )
        if confidence:
            stmt = stmt.where(Projection.confidence_rating == confidence)
        if model_version_label:
            stmt = stmt.where(Projection.model_version_label == model_version_label)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_all_with_results(session: Session, since_date: Optional[str] = None) -> list[Projection]:
        stmt = select(Projection).join(ActualResult)
        if since_date:
            stmt = stmt.where(Projection.game_date >= since_date)
        return list(session.execute(stmt).scalars())


class ActualResultRepository:
    @staticmethod
    def save(session: Session, result: ActualResult) -> ActualResult:
        session.add(result)
        session.flush()
        return result

    @staticmethod
    def exists_for_projection(session: Session, projection_id: str) -> bool:
        stmt = select(ActualResult).where(ActualResult.projection_id == projection_id)
        return session.execute(stmt).scalar_one_or_none() is not None


class ModelVersionRepository:
    @staticmethod
    def get_active(session: Session) -> Optional[ModelVersion]:
        stmt = select(ModelVersion).where(ModelVersion.is_active.is_(True))
        return session.execute(stmt).scalar_one_or_none()

    @staticmethod
    def save(session: Session, model_version: ModelVersion) -> ModelVersion:
        session.add(model_version)
        session.flush()
        return model_version

    @staticmethod
    def list_all(session: Session) -> list[ModelVersion]:
        stmt = select(ModelVersion).order_by(ModelVersion.trained_at.desc())
        return list(session.execute(stmt).scalars())

    @staticmethod
    def deactivate_all(session: Session) -> None:
        for mv in ModelVersionRepository.list_all(session):
            mv.is_active = False


class DataSourceLogRepository:
    @staticmethod
    def log(session: Session, entity_type: str, entity_key: str, source_name: str, success: bool = True, detail: Optional[str] = None) -> None:
        session.add(
            DataSourceLog(
                entity_type=entity_type,
                entity_key=entity_key,
                source_name=source_name,
                success=success,
                detail=detail,
            )
        )


class BetRepository:
    @staticmethod
    def save(session: Session, bet: Bet) -> Bet:
        session.add(bet)
        session.flush()
        return bet

    @staticmethod
    def get(session: Session, bet_id: str) -> Optional[Bet]:
        return session.get(Bet, bet_id)

    @staticmethod
    def list_unsettled(session: Session, through_date: Optional[str] = None) -> list[Bet]:
        stmt = select(Bet).where(Bet.result.is_(None)).order_by(Bet.game_date, Bet.created_at_utc)
        if through_date:
            stmt = stmt.where(Bet.game_date <= through_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_all(session: Session, limit: Optional[int] = None) -> list[Bet]:
        stmt = select(Bet).order_by(Bet.created_at_utc.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars())
