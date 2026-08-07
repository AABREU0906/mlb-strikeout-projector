"""
Grades pending NrfiProjection rows against completed games, using the same
FirstInningGameResult data the backfill/training pipeline already collects
-- so `update-results` and the historical database share one source of
truth for what actually happened in the 1st inning, rather than two
separate result-fetching code paths.

Also automatically settles any unsettled NRFI/YRFI bets on the same game,
right after that game's official first-inning result is known -- see
app.services.bet_ledger.settle_nrfi_bets_for_game.
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


def update_all_pending_nrfi_results() -> tuple[int, int]:
    """Returns (projections_updated, bets_settled)."""
    provider = MlbStatsApiProvider()
    updated = 0
    bets_settled = 0

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

            if not NrfiActualResultRepository.exists_for_projection(session, proj.id):
                result = NrfiActualResult(
                    nrfi_projection_id=proj.id,
                    away_first_inning_runs=away_runs,
                    home_first_inning_runs=home_runs,
                    is_nrfi=is_nrfi,
                    source="mlb_stats_api",
                )
                NrfiActualResultRepository.save(session, result)
                updated += 1

            # Settle any unsettled bets on this game now that the official
            # first-inning result is known -- done every pass (not just
            # when a new NrfiActualResult was just created) so a bet
            # placed AFTER the projection was already graded still gets
            # settled on the next run.
            from app.services.bet_ledger import settle_nrfi_bets_for_game

            bets_settled += settle_nrfi_bets_for_game(
                game_id=proj.game_id,
                is_nrfi=is_nrfi,
                away_first_inning_runs=away_runs,
                home_first_inning_runs=home_runs,
            )

    return updated, bets_settled
