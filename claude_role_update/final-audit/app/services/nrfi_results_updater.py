"""
Grades pending NrfiProjection rows against completed games, using the same
FirstInningGameResult data the backfill/training pipeline already collects
-- so `update-results` and the historical database share one source of
truth for what actually happened in the 1st inning, rather than two
separate result-fetching code paths.
"""
from __future__ import annotations

from app.config.logging_config import get_logger
from app.data_sources.mlb_stats_api import MlbStatsApiProvider
from app.database.models import NrfiActualResult
from app.database.repositories import (
    FirstInningGameResultRepository,
    NrfiActualResultRepository,
    NrfiProjectionRepository,
)
from app.database.session import session_scope

logger = get_logger(__name__)


def update_all_pending_nrfi_results() -> int:
    provider = MlbStatsApiProvider()
    updated = 0

    with session_scope() as session:
        pending = NrfiProjectionRepository.list_without_results(session)

        for proj in pending:
            existing = FirstInningGameResultRepository.get_by_game(session, proj.game_id)
            if existing is None or existing.is_nrfi is None:
                fetched = provider.get_first_inning_result(proj.game_id)
                if fetched is None or fetched.get("is_nrfi") is None:
                    continue
                is_nrfi = fetched["is_nrfi"]
                away_runs = fetched["away_first_inning_runs"]
                home_runs = fetched["home_first_inning_runs"]
            else:
                is_nrfi = existing.is_nrfi
                away_runs = existing.away_first_inning_runs
                home_runs = existing.home_first_inning_runs

            if NrfiActualResultRepository.exists_for_projection(session, proj.id):
                continue

            result = NrfiActualResult(
                nrfi_projection_id=proj.id,
                away_first_inning_runs=away_runs,
                home_first_inning_runs=home_runs,
                is_nrfi=is_nrfi,
                source="mlb_stats_api",
            )
            NrfiActualResultRepository.save(session, result)
            updated += 1

    return updated
