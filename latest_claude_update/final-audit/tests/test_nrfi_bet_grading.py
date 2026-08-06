import pytest

from app.services.bet_ledger import american_profit, grade_nrfi_bet, settle_profit_loss


def test_grade_nrfi_bet_win_on_nrfi():
    assert grade_nrfi_bet("NRFI", actual_is_nrfi=True) == "WIN"


def test_grade_nrfi_bet_loss_on_nrfi():
    assert grade_nrfi_bet("NRFI", actual_is_nrfi=False) == "LOSS"


def test_grade_nrfi_bet_win_on_yrfi():
    assert grade_nrfi_bet("YRFI", actual_is_nrfi=False) == "WIN"


def test_grade_nrfi_bet_loss_on_yrfi():
    assert grade_nrfi_bet("YRFI", actual_is_nrfi=True) == "LOSS"


def test_grade_nrfi_bet_never_pushes():
    for side in ("NRFI", "YRFI"):
        for actual in (True, False):
            assert grade_nrfi_bet(side, actual) in ("WIN", "LOSS")


def test_grade_nrfi_bet_invalid_side_raises():
    with pytest.raises(ValueError):
        grade_nrfi_bet("OVER", actual_is_nrfi=True)


def test_grade_nrfi_bet_case_insensitive():
    assert grade_nrfi_bet("nrfi", actual_is_nrfi=True) == "WIN"


def test_settle_profit_loss_win_uses_american_profit():
    assert settle_profit_loss("WIN", 10.0, -110) == american_profit(10.0, -110)


def test_settle_profit_loss_loss_is_negative_stake():
    assert settle_profit_loss("LOSS", 10.0, -110) == -10.0


def test_american_profit_negative_odds():
    assert american_profit(110, -110) == 100.0


def test_american_profit_positive_odds():
    assert american_profit(100, 150) == 150.0
