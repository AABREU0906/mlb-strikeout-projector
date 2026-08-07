"""
Regression tests for the edge-grading audit fix.

BUG: determine_edge_grade() awarded "Elite estimated edge" based purely on
EV and probability_edge_vs_price -- both PRICE-derived signals -- with no
regard for how far the model's own projection actually sits from the
betting line. A model probability of 54% (barely above a coinflip; the
projection is essentially ON the line) could earn Elite purely from a
generous offered price.

FIX: Elite/Strong now additionally require a minimum "quality separation"
-- Monte Carlo standard deviations between the projection and the line.
"""
import random

from app.markets.edge_analysis import (
    SideAnalysis,
    analyze_betting_edge,
    analyze_side,
    american_odds_to_probability,
    determine_edge_grade,
    expected_value_per_unit,
    probability_to_american_odds,
    remove_vig,
)


def test_reported_bug_scenario_no_longer_elite():
    result = analyze_betting_edge(
        over_odds=-140, under_odds=114,
        model_over_probability=0.46, model_under_probability=0.54,
        lineup_confirmed=True, pitcher_confirmed=True,
        projection_value=5.45, line_value=5.5, projection_std_dev=2.1,
    )
    assert result.grade != "Elite estimated edge"


def test_reported_bug_scenario_prefers_pass():
    result = analyze_betting_edge(
        over_odds=-140, under_odds=114,
        model_over_probability=0.46, model_under_probability=0.54,
        lineup_confirmed=True, pitcher_confirmed=True,
        projection_value=5.45, line_value=5.5, projection_std_dev=2.1,
    )
    assert result.recommended_side == "PASS"
    assert result.selected is None


def test_desired_elite_scenario_still_achieves_elite():
    result = analyze_betting_edge(
        over_odds=-115, under_odds=-105,
        model_over_probability=0.62, model_under_probability=0.38,
        lineup_confirmed=True, pitcher_confirmed=True,
        workload_sample_size=15,
        projection_value=7.9, line_value=6.5, projection_std_dev=1.8,
    )
    assert result.grade == "Elite estimated edge"
    assert result.stars == 5


def test_tiny_separation_with_large_ev_capped_at_moderate_not_forced_pass():
    result = analyze_betting_edge(
        over_odds=-140, under_odds=180,
        model_over_probability=0.46, model_under_probability=0.54,
        lineup_confirmed=True, pitcher_confirmed=True,
        projection_value=5.45, line_value=5.5, projection_std_dev=2.1,
    )
    assert result.selected is not None
    assert result.grade in ("Moderate estimated edge", "Strong estimated edge")
    assert result.grade != "Elite estimated edge"


def test_tiny_projection_difference_cannot_produce_elite_across_many_prices():
    for odds in range(-300, 400, 25):
        if odds == 0:
            continue
        result = analyze_betting_edge(
            over_odds=-110, under_odds=odds,
            model_over_probability=0.47, model_under_probability=0.53,
            lineup_confirmed=True, pitcher_confirmed=True,
            projection_value=5.48, line_value=5.5, projection_std_dev=2.0,
        )
        assert result.grade != "Elite estimated edge", f"Elite awarded at odds={odds} with near-zero separation"


def test_large_separation_with_full_confidence_can_reach_elite():
    result = analyze_betting_edge(
        over_odds=-120, under_odds=-110,
        model_over_probability=0.65, model_under_probability=0.35,
        lineup_confirmed=True, pitcher_confirmed=True,
        workload_sample_size=20,
        projection_value=8.5, line_value=6.0, projection_std_dev=2.0,
    )
    assert result.grade == "Elite estimated edge"


def test_elite_is_uncommon_across_random_realistic_scenarios():
    random.seed(42)
    grades = []
    for _ in range(500):
        projection = round(random.uniform(3.0, 9.0), 2)
        line = round(projection + random.uniform(-1.5, 1.5), 1)
        std_dev = round(random.uniform(1.5, 2.5), 2)
        model_prob = min(max(random.gauss(0.52, 0.08), 0.05), 0.95)
        odds = random.choice([-150, -130, -110, 100, 120, 140, 160])

        result = analyze_betting_edge(
            over_odds=-110, under_odds=odds,
            model_over_probability=1 - model_prob, model_under_probability=model_prob,
            lineup_confirmed=random.random() > 0.15,
            pitcher_confirmed=True,
            projection_value=projection, line_value=line, projection_std_dev=std_dev,
        )
        grades.append(result.grade)

    elite_count = sum(1 for g in grades if g == "Elite estimated edge")
    elite_fraction = elite_count / len(grades)
    assert elite_fraction < 0.15, f"Elite awarded too often: {elite_fraction:.1%} of scenarios"


def test_no_quality_data_preserves_prior_behavior_for_other_callers():
    result = analyze_betting_edge(
        over_odds=-140, under_odds=114,
        model_over_probability=0.46, model_under_probability=0.54,
        lineup_confirmed=True, pitcher_confirmed=True,
    )
    assert result.grade == "Elite estimated edge"


def test_unconfirmed_lineup_caps_below_elite():
    result = analyze_betting_edge(
        over_odds=-115, under_odds=-105,
        model_over_probability=0.62, model_under_probability=0.38,
        lineup_confirmed=False, pitcher_confirmed=True,
        projection_value=7.9, line_value=6.5, projection_std_dev=1.8,
    )
    assert result.grade != "Elite estimated edge"


def test_quality_separation_thresholds_directly():
    strong_analysis = SideAnalysis(
        side="OVER", sportsbook_odds=-110, model_probability=0.65,
        break_even_probability=0.524, vig_free_market_probability=0.50,
        probability_edge_vs_price=0.126, probability_edge_vs_market=0.15,
        expected_value=0.15, fair_model_odds=-186,
    )

    # z = |8.0 - 6.0| / 2.0 = 1.0 -- >= 0.75 (Elite band)
    grade, stars = determine_edge_grade(
        strong_analysis, projection_value=8.0, line_value=6.0, projection_std_dev=2.0
    )
    assert grade == "Elite estimated edge"

    # z = |7.3 - 6.5| / 2.0 = 0.4 -- in [0.35, 0.75) (Strong band)
    grade, stars = determine_edge_grade(
        strong_analysis, projection_value=7.3, line_value=6.5, projection_std_dev=2.0
    )
    assert grade == "Strong estimated edge"

    # z = |6.05 - 6.0| / 2.0 = 0.025 -- < 0.35 (Moderate band)
    grade, stars = determine_edge_grade(
        strong_analysis, projection_value=6.05, line_value=6.0, projection_std_dev=2.0
    )
    assert grade == "Moderate estimated edge"


def test_american_odds_to_probability_unchanged():
    assert abs(american_odds_to_probability(-130) - (130 / 230)) < 1e-9
    assert abs(american_odds_to_probability(120) - (100 / 220)) < 1e-9


def test_probability_to_american_odds_unchanged():
    assert probability_to_american_odds(0.5) in (-100, 100)


def test_remove_vig_sums_to_one():
    over, under = remove_vig(-110, -110)
    assert abs((over + under) - 1.0) < 1e-9
    assert abs(over - 0.5) < 1e-9


def test_expected_value_per_unit_unchanged():
    ev = expected_value_per_unit(0.55, -110)
    profit_if_win = 100 / 110
    expected = 0.55 * profit_if_win - 0.45
    assert abs(ev - expected) < 1e-9


def test_analyze_side_math_unchanged():
    analysis = analyze_side(
        side="OVER", sportsbook_odds=-120, model_probability=0.58,
        vig_free_market_probability=0.52,
    )
    assert abs(analysis.break_even_probability - (120 / 220)) < 1e-9
    assert abs(analysis.probability_edge_vs_price - (0.58 - 120 / 220)) < 1e-9
    assert abs(analysis.probability_edge_vs_market - (0.58 - 0.52)) < 1e-9


def test_ev_and_price_edge_thresholds_still_gate_natural_tier():
    weak_analysis = SideAnalysis(
        side="OVER", sportsbook_odds=-110, model_probability=0.51,
        break_even_probability=0.524, vig_free_market_probability=0.50,
        probability_edge_vs_price=-0.014, probability_edge_vs_market=0.01,
        expected_value=-0.02, fair_model_odds=-104,
    )
    grade, stars = determine_edge_grade(weak_analysis)
    assert grade == "No positive estimated edge"
