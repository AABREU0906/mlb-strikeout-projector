from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.config.settings import settings
from app.database.models import Bet
from app.database.repositories import BetRepository
from app.database.session import session_scope


@dataclass(frozen=True)
class BetSummary:
    total_bets: int
    wins: int
    losses: int
    pushes: int
    unresolved: int
    total_risked: float
    profit_loss: float
    roi: float


def american_profit(amount_risked: float, american_odds: int) -> float:
    if amount_risked <= 0:
        raise ValueError("Amount risked must be greater than zero.")
    if american_odds == 0:
        raise ValueError("American odds cannot be zero.")
    if american_odds > 0:
        return round(amount_risked * american_odds / 100.0, 2)
    return round(amount_risked * 100.0 / abs(american_odds), 2)


def grade_bet(side: str, line: float, actual_strikeouts: int) -> str:
    normalized = side.strip().upper()
    if normalized not in {"OVER", "UNDER"}:
        raise ValueError("Bet side must be OVER or UNDER.")
    if line <= 0:
        raise ValueError("Strikeout line must be positive.")
    if actual_strikeouts < 0:
        raise ValueError("Actual strikeouts cannot be negative.")

    if actual_strikeouts == line:
        return "PUSH"
    if normalized == "OVER":
        return "WIN" if actual_strikeouts > line else "LOSS"
    return "WIN" if actual_strikeouts < line else "LOSS"


def grade_nrfi_bet(side: str, actual_is_nrfi: bool) -> str:
    """NRFI/YRFI is a strict binary outcome -- no push is possible (unlike
    a strikeout total, which can land exactly on a whole-number line)."""
    normalized = side.strip().upper()
    if normalized not in {"NRFI", "YRFI"}:
        raise ValueError("NRFI bet side must be NRFI or YRFI.")
    actual_side = "NRFI" if actual_is_nrfi else "YRFI"
    return "WIN" if normalized == actual_side else "LOSS"


def settle_profit_loss(result: str, amount_risked: float, american_odds: int) -> float:
    normalized = result.upper()
    if normalized == "WIN":
        return american_profit(amount_risked, american_odds)
    if normalized == "LOSS":
        return round(-amount_risked, 2)
    if normalized == "PUSH":
        return 0.0
    raise ValueError("Result must be WIN, LOSS, or PUSH.")


def record_bet(
    *,
    game_date: str,
    pitcher_name: str,
    side: str,
    strikeout_line: float,
    american_odds: int,
    amount_risked: float,
    projection_id: Optional[str] = None,
    game_id: Optional[str] = None,
    pitcher_id: Optional[int] = None,
    opponent_team: Optional[str] = None,
    sportsbook: Optional[str] = None,
    model_probability: Optional[float] = None,
    model_projection: Optional[float] = None,
    confidence_rating: Optional[str] = None,
    edge_grade: Optional[str] = None,
    notes: Optional[str] = None,
) -> Bet:
    normalized_side = side.strip().upper()
    if normalized_side not in {"OVER", "UNDER"}:
        raise ValueError("Bet side must be OVER or UNDER.")
    if strikeout_line <= 0:
        raise ValueError("Strikeout line must be positive.")
    if american_odds == 0:
        raise ValueError("American odds cannot be zero.")
    if amount_risked <= 0:
        raise ValueError("Amount risked must be greater than zero.")

    bet = Bet(
        projection_id=projection_id,
        game_id=game_id,
        game_date=game_date,
        pitcher_id=pitcher_id,
        pitcher_name=pitcher_name,
        opponent_team=opponent_team,
        side=normalized_side,
        strikeout_line=float(strikeout_line),
        american_odds=int(american_odds),
        amount_risked=round(float(amount_risked), 2),
        sportsbook=sportsbook,
        model_probability=model_probability,
        model_projection=model_projection,
        confidence_rating=confidence_rating,
        edge_grade=edge_grade,
        notes=notes,
    )
    with session_scope() as session:
        BetRepository.save(session, bet)
    export_bets_csv()
    return bet


def settle_bet(bet_id: str, actual_strikeouts: int) -> Bet:
    with session_scope() as session:
        bet = BetRepository.get(session, bet_id)
        if bet is None:
            raise ValueError(f"Bet not found: {bet_id}")
        if bet.result is not None:
            return bet
        result = grade_bet(bet.side, bet.strikeout_line, actual_strikeouts)
        bet.actual_strikeouts = actual_strikeouts
        bet.result = result
        bet.profit_loss = settle_profit_loss(result, bet.amount_risked, bet.american_odds)
        bet.settled_at_utc = dt.datetime.now(dt.timezone.utc)
        session.flush()
    export_bets_csv()
    return bet


def record_nrfi_bet(
    *,
    game_date: str,
    side: str,  # "NRFI" | "YRFI"
    american_odds: int,
    amount_risked: float,
    nrfi_projection_id: Optional[str] = None,
    game_id: Optional[str] = None,
    matchup_label: Optional[str] = None,  # e.g. "Away @ Home" -- stored in pitcher_name for a readable ledger row
    opponent_team: Optional[str] = None,
    sportsbook: Optional[str] = None,
    model_probability: Optional[float] = None,
    model_projection: Optional[float] = None,  # expected first-inning runs
    confidence_rating: Optional[str] = None,
    edge_grade: Optional[str] = None,
    notes: Optional[str] = None,
) -> Bet:
    normalized_side = side.strip().upper()
    if normalized_side not in {"NRFI", "YRFI"}:
        raise ValueError("NRFI bet side must be NRFI or YRFI.")
    if american_odds == 0:
        raise ValueError("American odds cannot be zero.")
    if amount_risked <= 0:
        raise ValueError("Amount risked must be greater than zero.")

    bet = Bet(
        market_type="nrfi_yrfi",
        nrfi_projection_id=nrfi_projection_id,
        game_id=game_id,
        game_date=game_date,
        pitcher_name=matchup_label,
        opponent_team=opponent_team,
        side=normalized_side,
        strikeout_line=None,
        american_odds=int(american_odds),
        amount_risked=round(float(amount_risked), 2),
        sportsbook=sportsbook,
        model_probability=model_probability,
        model_projection=model_projection,
        confidence_rating=confidence_rating,
        edge_grade=edge_grade,
        notes=notes,
    )
    with session_scope() as session:
        BetRepository.save(session, bet)
    export_bets_csv()
    return bet


def settle_nrfi_bet(
    bet_id: str,
    first_inning_runs: Optional[int] = None,
    run_occurred: Optional[bool] = None,
) -> Bet:
    """Grades an NRFI/YRFI bet from EITHER the raw combined first-inning
    run count OR a plain yes/no "did a run occur" answer -- exactly one
    must be provided. The raw count (when known) is stored in its own
    `first_inning_runs` column, never overloaded into `actual_strikeouts`,
    which stays exclusively a strikeout-prop field."""
    if first_inning_runs is None and run_occurred is None:
        raise ValueError("Provide either first_inning_runs or run_occurred.")
    if first_inning_runs is not None and first_inning_runs < 0:
        raise ValueError("first_inning_runs cannot be negative.")

    if first_inning_runs is not None:
        actual_is_nrfi = first_inning_runs == 0
    else:
        actual_is_nrfi = not run_occurred

    with session_scope() as session:
        bet = BetRepository.get(session, bet_id)
        if bet is None:
            raise ValueError(f"Bet not found: {bet_id}")
        if bet.result is not None:
            return bet
        result = grade_nrfi_bet(bet.side, actual_is_nrfi)
        bet.actual_nrfi_result = "NRFI" if actual_is_nrfi else "YRFI"
        bet.first_inning_runs = first_inning_runs
        bet.result = result
        bet.profit_loss = settle_profit_loss(result, bet.amount_risked, bet.american_odds)
        bet.settled_at_utc = dt.datetime.now(dt.timezone.utc)
        session.flush()
    export_bets_csv()
    return bet


def settle_nrfi_bets_for_game(
    game_id: str,
    is_nrfi: bool,
    away_first_inning_runs: Optional[int] = None,
    home_first_inning_runs: Optional[int] = None,
) -> int:
    """Automatically grades every unsettled NRFI/YRFI bet on a given game,
    once its official first-inning result is known. Used by
    app/services/nrfi_results_updater.py right after it grades that same
    game's NrfiProjection -- so 'fetch the official first-inning score and
    grade NRFI/YRFI bets automatically' happens for bets too, not just
    projections. Returns the number of bets settled."""
    combined_runs: Optional[int] = None
    if away_first_inning_runs is not None and home_first_inning_runs is not None:
        combined_runs = away_first_inning_runs + home_first_inning_runs

    with session_scope() as session:
        pending = BetRepository.list_unsettled_for_game(session, game_id, market_type="nrfi_yrfi")
        pending_ids = [bet.id for bet in pending]

    settled = 0
    for bet_id in pending_ids:
        if combined_runs is not None:
            settle_nrfi_bet(bet_id, first_inning_runs=combined_runs)
        else:
            settle_nrfi_bet(bet_id, run_occurred=not is_nrfi)
        settled += 1

    return settled


def list_unsettled_by_market(market_type: Optional[str] = None, through_date: Optional[str] = None) -> list[Bet]:
    with session_scope() as session:
        return BetRepository.list_unsettled_by_market(session, market_type=market_type, through_date=through_date)


def list_bets_by_market(market_type: Optional[str] = None, limit: Optional[int] = None) -> list[Bet]:
    with session_scope() as session:
        return BetRepository.list_all_by_market(session, market_type=market_type, limit=limit)


def list_unsettled(through_date: Optional[str] = None) -> list[Bet]:
    with session_scope() as session:
        return BetRepository.list_unsettled(session, through_date=through_date)


def list_bets(limit: Optional[int] = None) -> list[Bet]:
    with session_scope() as session:
        return BetRepository.list_all(session, limit=limit)


def summarize_bets(bets: Optional[Iterable[Bet]] = None) -> BetSummary:
    rows = list(bets) if bets is not None else list_bets()
    settled = [b for b in rows if b.result is not None]
    total_risked = round(sum(float(b.amount_risked or 0.0) for b in settled), 2)
    profit_loss = round(sum(float(b.profit_loss or 0.0) for b in settled), 2)
    roi = (profit_loss / total_risked) if total_risked else 0.0
    return BetSummary(
        total_bets=len(rows),
        wins=sum(1 for b in settled if b.result == "WIN"),
        losses=sum(1 for b in settled if b.result == "LOSS"),
        pushes=sum(1 for b in settled if b.result == "PUSH"),
        unresolved=sum(1 for b in rows if b.result is None),
        total_risked=total_risked,
        profit_loss=profit_loss,
        roi=roi,
    )


def _default_export_path() -> Path:
    base = Path(settings.database_full_path).resolve().parent.parent
    return base / "exports" / "bets.csv"


def export_bets_csv(path: Optional[Path] = None) -> Path:
    destination = path or _default_export_path()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list_bets()
    fieldnames = [
        "id", "created_at_utc", "settled_at_utc", "market_type", "projection_id",
        "nrfi_projection_id", "game_id", "game_date", "pitcher_id", "pitcher_name",
        "opponent_team", "side", "strikeout_line", "american_odds", "amount_risked",
        "sportsbook", "model_probability", "model_projection", "confidence_rating",
        "edge_grade", "actual_strikeouts", "actual_nrfi_result", "first_inning_runs", "result", "profit_loss", "notes",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bet in rows:
            writer.writerow({name: getattr(bet, name) for name in fieldnames})
    return destination
