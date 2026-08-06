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
        "id", "created_at_utc", "settled_at_utc", "projection_id", "game_id",
        "game_date", "pitcher_id", "pitcher_name", "opponent_team", "side",
        "strikeout_line", "american_odds", "amount_risked", "sportsbook",
        "model_probability", "model_projection", "confidence_rating", "edge_grade",
        "actual_strikeouts", "result", "profit_loss", "notes",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bet in rows:
            writer.writerow({name: getattr(bet, name) for name in fieldnames})
    return destination
