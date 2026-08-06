from app.projections.confidence_rating import compute_confidence
from app.schemas.market import ManualMarketEntry


def test_manual_market_entry_fields_roundtrip():
    entry = ManualMarketEntry(strikeout_line=6.5, over_odds=-115, under_odds=-105)
    assert entry.strikeout_line == 6.5
    assert entry.over_odds == -115
    assert entry.sportsbook_name == "manual_entry"


def test_confidence_high_with_clean_inputs():
    result = compute_confidence(
        lineup_is_confirmed=True,
        pitcher_confirmed=True,
        pitcher_data_completeness=1.0,
        batter_avg_data_completeness=1.0,
        workload_confidence_penalty=0.0,
        news_confidence_penalty=0.0,
        weather_delay_risk=0.0,
        market_disagreement_flag=False,
        stats_vs_market_disagreement_pct=0.0,
        simulation_std_dev=1.0,
        simulation_mean=5.5,
    )
    assert result.rating == "High"


def test_confidence_avoid_with_many_problems():
    result = compute_confidence(
        lineup_is_confirmed=False,
        pitcher_confirmed=False,
        pitcher_data_completeness=0.3,
        batter_avg_data_completeness=0.3,
        workload_confidence_penalty=0.8,
        news_confidence_penalty=0.8,
        weather_delay_risk=0.9,
        market_disagreement_flag=True,
        stats_vs_market_disagreement_pct=0.3,
        simulation_std_dev=4.0,
        simulation_mean=5.5,
    )
    assert result.rating == "Avoid"


def test_confidence_degrades_monotonically_with_projected_lineup():
    confirmed = compute_confidence(
        True, True, 1.0, 1.0, 0.0, 0.0, 0.0, False, 0.0, 1.0, 5.5
    )
    projected = compute_confidence(
        False, True, 1.0, 1.0, 0.0, 0.0, 0.0, False, 0.0, 1.0, 5.5
    )
    assert projected.total_penalty > confirmed.total_penalty


def test_confidence_factors_are_all_recorded():
    result = compute_confidence(True, True, 1.0, 1.0, 0.0, 0.0, 0.0, False, 0.0, 1.0, 5.5)
    for key in ["projected_lineup", "unconfirmed_pitcher", "workload_uncertainty", "simulation_variance"]:
        assert key in result.factors
