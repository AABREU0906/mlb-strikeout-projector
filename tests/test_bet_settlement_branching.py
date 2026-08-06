"""
Tests for the settlement-branching bug fix.

BUG: settle-bets asked every unresolved bet for "Actual strikeouts,"
including NRFI/YRFI bets, which have no strikeout total at all.

These tests exercise the real bet_ledger functions against an isolated
temp SQLite database (see conftest.py), not mocks -- so they cover the
actual production code path the CLI uses.
"""
import pytest

from app.services.bet_ledger import (
    grade_bet,
    grade_nrfi_bet,
    list_unsettled,
    list_unsettled_by_market,
    record_bet,
    record_nrfi_bet,
    settle_bet,
    settle_nrfi_bet,
    settle_nrfi_bets_for_game,
)


def _record_strikeout_bet(**overrides):
    kwargs = dict(
        game_date="2026-07-30", pitcher_name="Test Pitcher", side="OVER",
        strikeout_line=5.5, american_odds=-115, amount_risked=25.0,
        game_id="strikeout-game-1",
    )
    kwargs.update(overrides)
    return record_bet(**kwargs)


def _record_nrfi_bet(**overrides):
    kwargs = dict(
        game_date="2026-07-30", side="NRFI", american_odds=-130,
        amount_risked=20.0, game_id="nrfi-game-1", matchup_label="Away @ Home",
    )
    kwargs.update(overrides)
    return record_nrfi_bet(**kwargs)


def test_grade_nrfi_bet_yrfi_win():
    assert grade_nrfi_bet("YRFI", actual_is_nrfi=False) == "WIN"


def test_grade_nrfi_bet_yrfi_loss():
    assert grade_nrfi_bet("YRFI", actual_is_nrfi=True) == "LOSS"


def test_grade_nrfi_bet_nrfi_win():
    assert grade_nrfi_bet("NRFI", actual_is_nrfi=True) == "WIN"


def test_grade_nrfi_bet_nrfi_loss():
    assert grade_nrfi_bet("NRFI", actual_is_nrfi=False) == "LOSS"


def test_yrfi_bet_wins_when_run_scored():
    bet = _record_nrfi_bet(side="YRFI", game_id="yrfi-win-game")
    settled = settle_nrfi_bet(bet.id, run_occurred=True)
    assert settled.result == "WIN"
    assert settled.actual_nrfi_result == "YRFI"
    assert settled.profit_loss is not None and settled.profit_loss > 0


def test_yrfi_bet_loses_when_no_run_scored():
    bet = _record_nrfi_bet(side="YRFI", game_id="yrfi-loss-game")
    settled = settle_nrfi_bet(bet.id, run_occurred=False)
    assert settled.result == "LOSS"
    assert settled.actual_nrfi_result == "NRFI"
    assert settled.profit_loss == -bet.amount_risked


def test_nrfi_bet_wins_when_no_run_scored():
    bet = _record_nrfi_bet(side="NRFI", game_id="nrfi-win-game")
    settled = settle_nrfi_bet(bet.id, run_occurred=False)
    assert settled.result == "WIN"
    assert settled.actual_nrfi_result == "NRFI"


def test_nrfi_bet_loses_when_run_scored():
    bet = _record_nrfi_bet(side="NRFI", game_id="nrfi-loss-game")
    settled = settle_nrfi_bet(bet.id, run_occurred=True)
    assert settled.result == "LOSS"
    assert settled.actual_nrfi_result == "YRFI"


def test_yrfi_settlement_via_raw_run_count():
    bet = _record_nrfi_bet(side="YRFI", game_id="yrfi-raw-count-game")
    settled = settle_nrfi_bet(bet.id, first_inning_runs=2)
    assert settled.result == "WIN"
    assert settled.first_inning_runs == 2
    assert settled.actual_strikeouts is None


def test_settle_nrfi_bet_requires_one_input():
    bet = _record_nrfi_bet(side="NRFI", game_id="nrfi-missing-input-game")
    with pytest.raises(ValueError):
        settle_nrfi_bet(bet.id)


def test_strikeout_bet_settlement_unchanged():
    bet = _record_strikeout_bet(side="OVER", strikeout_line=5.5, game_id="strikeout-unchanged-game")
    settled = settle_bet(bet.id, actual_strikeouts=7)
    assert settled.result == "WIN"
    assert settled.actual_strikeouts == 7
    assert settled.actual_nrfi_result is None
    assert settled.first_inning_runs is None


def test_strikeout_bet_push_unchanged():
    bet = _record_strikeout_bet(side="OVER", strikeout_line=6.0, game_id="strikeout-push-game")
    settled = settle_bet(bet.id, actual_strikeouts=6)
    assert settled.result == "PUSH"
    assert settled.profit_loss == 0.0


def test_grade_bet_function_unchanged():
    assert grade_bet("OVER", 5.5, 6) == "WIN"
    assert grade_bet("UNDER", 5.5, 6) == "LOSS"
    assert grade_bet("OVER", 6.0, 6) == "PUSH"


def test_mixed_unsettled_ledger_contains_both_market_types():
    _record_strikeout_bet(side="OVER", game_id="mixed-strikeout-game")
    _record_nrfi_bet(side="YRFI", game_id="mixed-nrfi-game")

    pending = list_unsettled()
    market_types = {bet.market_type for bet in pending}
    assert "strikeouts" in market_types
    assert "nrfi_yrfi" in market_types


def test_market_scoped_listing_separates_types():
    _record_strikeout_bet(side="UNDER", game_id="scoped-strikeout-game")
    _record_nrfi_bet(side="NRFI", game_id="scoped-nrfi-game")

    strikeout_only = list_unsettled_by_market("strikeouts")
    nrfi_only = list_unsettled_by_market("nrfi_yrfi")

    assert all(b.market_type == "strikeouts" for b in strikeout_only)
    assert all(b.market_type == "nrfi_yrfi" for b in nrfi_only)
    assert any(b.game_id == "scoped-strikeout-game" for b in strikeout_only)
    assert any(b.game_id == "scoped-nrfi-game" for b in nrfi_only)


def test_settling_nrfi_bet_never_prompts_or_requires_strikeout_field():
    import inspect

    sig = inspect.signature(settle_nrfi_bet)
    assert "actual_strikeouts" not in sig.parameters
    assert "strikeout" not in " ".join(sig.parameters.keys()).lower()


def test_automatic_settlement_grades_yrfi_bet():
    _record_nrfi_bet(side="YRFI", game_id="auto-settle-yrfi-game")
    n_settled = settle_nrfi_bets_for_game(
        game_id="auto-settle-yrfi-game", is_nrfi=False,
        away_first_inning_runs=1, home_first_inning_runs=0,
    )
    assert n_settled == 1

    remaining = list_unsettled_by_market("nrfi_yrfi")
    assert not any(b.game_id == "auto-settle-yrfi-game" for b in remaining)


def test_automatic_settlement_grades_nrfi_bet():
    _record_nrfi_bet(side="NRFI", game_id="auto-settle-nrfi-game")
    n_settled = settle_nrfi_bets_for_game(
        game_id="auto-settle-nrfi-game", is_nrfi=True,
        away_first_inning_runs=0, home_first_inning_runs=0,
    )
    assert n_settled == 1


def test_automatic_settlement_stores_combined_run_count():
    bet = _record_nrfi_bet(side="YRFI", game_id="auto-settle-runs-game")
    settle_nrfi_bets_for_game(
        game_id="auto-settle-runs-game", is_nrfi=False,
        away_first_inning_runs=2, home_first_inning_runs=1,
    )
    from app.services.bet_ledger import list_bets_by_market
    settled_bets = {b.id: b for b in list_bets_by_market("nrfi_yrfi")}
    assert settled_bets[bet.id].first_inning_runs == 3
    assert settled_bets[bet.id].result == "WIN"


def test_automatic_settlement_ignores_other_games():
    _record_nrfi_bet(side="NRFI", game_id="auto-settle-untouched-game")
    n_settled = settle_nrfi_bets_for_game(
        game_id="a-completely-different-game", is_nrfi=True,
        away_first_inning_runs=0, home_first_inning_runs=0,
    )
    assert n_settled == 0
    remaining = list_unsettled_by_market("nrfi_yrfi")
    assert any(b.game_id == "auto-settle-untouched-game" for b in remaining)


def test_automatic_settlement_does_not_touch_strikeout_bets():
    _record_strikeout_bet(side="OVER", game_id="auto-settle-strikeout-safety-game")
    n_settled = settle_nrfi_bets_for_game(
        game_id="auto-settle-strikeout-safety-game", is_nrfi=True,
        away_first_inning_runs=0, home_first_inning_runs=0,
    )
    assert n_settled == 0
    remaining = list_unsettled_by_market("strikeouts")
    assert any(b.game_id == "auto-settle-strikeout-safety-game" for b in remaining)
