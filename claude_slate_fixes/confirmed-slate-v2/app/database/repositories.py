"""
Repository layer: all reads/writes to the database go through here so the
CLI and services never construct raw SQL/ORM queries directly.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import (
    ActualResult,
    Bet,
    DataSourceLog,
    FirstInningGameResult,
    Game,
    ModelVersion,
    NrfiActualResult,
    NrfiProjection,
    Projection,
)


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
    def exists_for_game_pitcher_date(session: Session, game_id: str, pitcher_id: int, game_date: str) -> bool:
        """Used by project-confirmed-slate's incremental-run logic: any
        existing projection for this (game, pitcher, date) combination --
        regardless of how it was created (manual or automated) -- counts
        as 'already processed today,' the more conservative choice for
        credit conservation."""
        stmt = (
            select(Projection)
            .where(Projection.game_id == game_id)
            .where(Projection.pitcher_id == pitcher_id)
            .where(Projection.game_date == game_date)
        )
        return session.execute(stmt).first() is not None

    @staticmethod
    def update_edge_outcome(session: Session, projection_id: str, **fields) -> Optional[Projection]:
        """Updates only the betting-edge outcome fields on an already-saved
        projection (never touches the pregame inputs/model-output fields
        saved at creation time). Used because edge analysis is computed
        one layer up (in the CLI/display layer) after the projection row
        already exists, not inside pipeline.run() itself."""
        projection = session.get(Projection, projection_id)
        if projection is None:
            return None
        for key, value in fields.items():
            if hasattr(projection, key):
                setattr(projection, key, value)
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

    @staticmethod
    def list_all(session: Session) -> list[Projection]:
        return list(session.execute(select(Projection)).scalars())

    @staticmethod
    def list_graded_filtered(
        session: Session,
        *,
        since_date: Optional[str] = None,
        season: Optional[int] = None,
        pitcher_name: Optional[str] = None,
        confidence: Optional[str] = None,
        edge_grade: Optional[str] = None,
        last_n: Optional[int] = None,
    ) -> list[Projection]:
        """Graded projections (i.e. an ActualResult exists) for the model
        health report, combinable with the same filter set as
        `model-report`'s CLI options. Ordered most-recent-first so
        `last_n` means 'the N most recently graded projections,' not an
        arbitrary DB-order slice."""
        stmt = select(Projection).join(ActualResult).order_by(Projection.game_date.desc(), Projection.created_at_utc.desc())
        if since_date:
            stmt = stmt.where(Projection.game_date >= since_date)
        if season:
            stmt = stmt.where(Projection.game_date.like(f"{season}-%"))
        if pitcher_name:
            stmt = stmt.where(Projection.pitcher_name.ilike(f"%{pitcher_name}%"))
        if confidence:
            stmt = stmt.where(Projection.betting_confidence == confidence.upper())
        if edge_grade:
            stmt = stmt.where(Projection.edge_grade.ilike(f"%{edge_grade}%"))
        if last_n:
            stmt = stmt.limit(last_n)
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

    @staticmethod
    def delete_for_projection(session: Session, projection_id: str) -> bool:
        """Only ever called when --force is explicitly passed to
        update-results; a finalized result is never silently overwritten
        otherwise."""
        stmt = select(ActualResult).where(ActualResult.projection_id == projection_id)
        existing = session.execute(stmt).scalar_one_or_none()
        if existing is None:
            return False
        session.delete(existing)
        session.flush()
        return True


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
    def list_unsettled_for_game(session: Session, game_id: str, market_type: Optional[str] = None) -> list[Bet]:
        stmt = select(Bet).where(Bet.game_id == game_id).where(Bet.result.is_(None))
        if market_type:
            stmt = stmt.where(Bet.market_type == market_type)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_all(session: Session, limit: Optional[int] = None) -> list[Bet]:
        stmt = select(Bet).order_by(Bet.created_at_utc.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_unsettled_by_market(session: Session, market_type: Optional[str] = None, through_date: Optional[str] = None) -> list[Bet]:
        stmt = select(Bet).where(Bet.result.is_(None)).order_by(Bet.game_date, Bet.created_at_utc)
        if market_type:
            stmt = stmt.where(Bet.market_type == market_type)
        if through_date:
            stmt = stmt.where(Bet.game_date <= through_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_all_by_market(session: Session, market_type: Optional[str] = None, limit: Optional[int] = None) -> list[Bet]:
        stmt = select(Bet).order_by(Bet.created_at_utc.desc())
        if market_type:
            stmt = stmt.where(Bet.market_type == market_type)
        if limit is not None:
            stmt = stmt.limit(limit)
        return list(session.execute(stmt).scalars())


class FirstInningGameResultRepository:
    @staticmethod
    def upsert(session: Session, result: FirstInningGameResult) -> FirstInningGameResult:
        existing = session.execute(
            select(FirstInningGameResult).where(FirstInningGameResult.game_id == result.game_id)
        ).scalar_one_or_none()
        if existing is not None:
            for attr in (
                "away_first_inning_runs", "home_first_inning_runs", "is_nrfi",
                "away_pitcher_scoreless_first", "home_pitcher_scoreless_first",
                "home_starting_pitcher_id", "away_starting_pitcher_id",
                "home_starting_pitcher_name", "away_starting_pitcher_name",
                "game_status", "day_night", "venue_id",
                "away_plate_appearances", "away_at_bats", "away_hits", "away_walks",
                "away_strikeouts", "away_home_runs", "away_total_bases",
                "home_plate_appearances", "home_at_bats", "home_hits", "home_walks",
                "home_strikeouts", "home_home_runs", "home_total_bases",
                "away_pitcher_first_inning_pitches", "home_pitcher_first_inning_pitches",
            ):
                setattr(existing, attr, getattr(result, attr))
            existing.retrieved_at_utc = dt.datetime.now(dt.timezone.utc)
            return existing
        session.add(result)
        session.flush()
        return result

    @staticmethod
    def exists_complete(session: Session, game_id: str) -> bool:
        row = session.execute(
            select(FirstInningGameResult).where(FirstInningGameResult.game_id == game_id)
        ).scalar_one_or_none()
        return row is not None and row.is_nrfi is not None

    @staticmethod
    def get_by_game(session: Session, game_id: str) -> Optional[FirstInningGameResult]:
        return session.execute(
            select(FirstInningGameResult).where(FirstInningGameResult.game_id == game_id)
        ).scalar_one_or_none()

    @staticmethod
    def list_for_training(
        session: Session, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> list[FirstInningGameResult]:
        stmt = select(FirstInningGameResult).where(FirstInningGameResult.is_nrfi.is_not(None))
        if start_date:
            stmt = stmt.where(FirstInningGameResult.game_date >= start_date)
        if end_date:
            stmt = stmt.where(FirstInningGameResult.game_date <= end_date)
        stmt = stmt.order_by(FirstInningGameResult.game_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_pitcher_history(
        session: Session, pitcher_id: int, before_date: str, limit: int = 20
    ) -> list[FirstInningGameResult]:
        """All prior completed starts for a pitcher (either home or away)
        strictly before before_date -- the caller is responsible for using
        this only in leakage-safe contexts (features for a game must only
        use starts before that game's date)."""
        stmt = (
            select(FirstInningGameResult)
            .where(
                (FirstInningGameResult.home_starting_pitcher_id == pitcher_id)
                | (FirstInningGameResult.away_starting_pitcher_id == pitcher_id)
            )
            .where(FirstInningGameResult.game_date < before_date)
            .where(FirstInningGameResult.is_nrfi.is_not(None))
            .order_by(FirstInningGameResult.game_date.desc())
            .limit(limit)
        )
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_team_history(
        session: Session, team_id: int, before_date: str, limit: int = 30
    ) -> list[FirstInningGameResult]:
        stmt = (
            select(FirstInningGameResult)
            .where(
                (FirstInningGameResult.home_team_id == team_id)
                | (FirstInningGameResult.away_team_id == team_id)
            )
            .where(FirstInningGameResult.game_date < before_date)
            .where(FirstInningGameResult.is_nrfi.is_not(None))
            .order_by(FirstInningGameResult.game_date.desc())
            .limit(limit)
        )
        return list(session.execute(stmt).scalars())


class NrfiProjectionRepository:
    @staticmethod
    def save(session: Session, projection: NrfiProjection) -> NrfiProjection:
        session.add(projection)
        session.flush()
        return projection

    @staticmethod
    def get(session: Session, projection_id: str) -> Optional[NrfiProjection]:
        return session.get(NrfiProjection, projection_id)

    @staticmethod
    def list_without_results(session: Session, before_date: Optional[str] = None) -> list[NrfiProjection]:
        stmt = select(NrfiProjection).outerjoin(NrfiActualResult).where(NrfiActualResult.id.is_(None))
        if before_date:
            stmt = stmt.where(NrfiProjection.game_date <= before_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_all_with_results(session: Session, since_date: Optional[str] = None) -> list[NrfiProjection]:
        stmt = select(NrfiProjection).join(NrfiActualResult)
        if since_date:
            stmt = stmt.where(NrfiProjection.game_date >= since_date)
        return list(session.execute(stmt).scalars())

    @staticmethod
    def list_filtered(
        session: Session,
        date: Optional[str] = None,
        team: Optional[str] = None,
        limit: int = 100,
    ) -> list[NrfiProjection]:
        stmt = select(NrfiProjection).order_by(NrfiProjection.created_at_utc.desc()).limit(limit)
        if date:
            stmt = stmt.where(NrfiProjection.game_date == date)
        if team:
            stmt = stmt.where(
                (NrfiProjection.home_team.ilike(f"%{team}%")) | (NrfiProjection.away_team.ilike(f"%{team}%"))
            )
        return list(session.execute(stmt).scalars())


class NrfiActualResultRepository:
    @staticmethod
    def save(session: Session, result: NrfiActualResult) -> NrfiActualResult:
        session.add(result)
        session.flush()
        return result

    @staticmethod
    def exists_for_projection(session: Session, nrfi_projection_id: str) -> bool:
        stmt = select(NrfiActualResult).where(NrfiActualResult.nrfi_projection_id == nrfi_projection_id)
        return session.execute(stmt).scalar_one_or_none() is not None
