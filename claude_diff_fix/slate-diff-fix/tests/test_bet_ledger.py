from app.services.bet_ledger import american_profit, grade_bet, settle_profit_loss


def test_grade_half_point_bets():
    assert grade_bet("OVER", 5.5, 6) == "WIN"
    assert grade_bet("OVER", 5.5, 5) == "LOSS"
    assert grade_bet("UNDER", 5.5, 5) == "WIN"
    assert grade_bet("UNDER", 5.5, 6) == "LOSS"


def test_grade_whole_number_push():
    assert grade_bet("OVER", 5.0, 5) == "PUSH"
    assert grade_bet("UNDER", 5.0, 5) == "PUSH"


def test_profit_calculation():
    assert american_profit(20.0, 120) == 24.0
    assert american_profit(20.0, -130) == 15.38
    assert settle_profit_loss("LOSS", 20.0, -110) == -20.0
    assert settle_profit_loss("PUSH", 20.0, -110) == 0.0
