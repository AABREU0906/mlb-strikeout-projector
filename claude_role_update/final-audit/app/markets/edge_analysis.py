from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BetSide = Literal["OVER", "UNDER", "PASS"]


@dataclass(frozen=True)
class SideAnalysis:
    side: Literal["OVER", "UNDER"]
    sportsbook_odds: int
    model_probability: float
    break_even_probability: float
    vig_free_market_probability: float
    probability_edge_vs_price: float
    probability_edge_vs_market: float
    expected_value: float
    fair_model_odds: int


@dataclass(frozen=True)
class EdgeAnalysis:
    recommended_side: BetSide
    selected: SideAnalysis | None
    over: SideAnalysis
    under: SideAnalysis
    grade: str
    stars: int
    confidence: str
    reason: str


def american_odds_to_probability(odds: int) -> float:
    """
    Convert American odds into raw implied probability.

    Examples:
        -130 -> 0.5652
        +120 -> 0.4545
    """
    if odds == 0:
        raise ValueError("American odds cannot be 0.")

    if odds < 0:
        return abs(odds) / (abs(odds) + 100)

    return 100 / (odds + 100)


def probability_to_american_odds(probability: float) -> int:
    """
    Convert a model probability into fair American odds.

    Examples:
        0.592 -> approximately -145
        0.469 -> approximately +113
    """
    if not 0 < probability < 1:
        raise ValueError("Probability must be between 0 and 1.")

    if probability >= 0.5:
        odds = -(probability / (1 - probability)) * 100
    else:
        odds = ((1 - probability) / probability) * 100

    return round(odds)


def remove_vig(
    over_odds: int,
    under_odds: int,
) -> tuple[float, float]:
    """
    Remove vig by normalizing both raw implied probabilities to 100%.
    """
    raw_over = american_odds_to_probability(over_odds)
    raw_under = american_odds_to_probability(under_odds)

    total = raw_over + raw_under

    if total <= 0:
        raise ValueError("Invalid sportsbook probabilities.")

    return raw_over / total, raw_under / total


def expected_value_per_unit(
    model_probability: float,
    american_odds: int,
) -> float:
    """
    Calculate expected profit per 1 unit risked.

    A result of 0.05 means +5% expected return per unit risked.
    A result of -0.03 means -3% expected return per unit risked.
    """
    if not 0 <= model_probability <= 1:
        raise ValueError("Model probability must be between 0 and 1.")

    if american_odds > 0:
        profit_if_win = american_odds / 100
    elif american_odds < 0:
        profit_if_win = 100 / abs(american_odds)
    else:
        raise ValueError("American odds cannot be 0.")

    loss_probability = 1 - model_probability

    return (
        model_probability * profit_if_win
        - loss_probability
    )


def analyze_side(
    *,
    side: Literal["OVER", "UNDER"],
    sportsbook_odds: int,
    model_probability: float,
    vig_free_market_probability: float,
) -> SideAnalysis:
    break_even = american_odds_to_probability(sportsbook_odds)

    return SideAnalysis(
        side=side,
        sportsbook_odds=sportsbook_odds,
        model_probability=model_probability,
        break_even_probability=break_even,
        vig_free_market_probability=vig_free_market_probability,
        probability_edge_vs_price=model_probability - break_even,
        probability_edge_vs_market=(
            model_probability - vig_free_market_probability
        ),
        expected_value=expected_value_per_unit(
            model_probability,
            sportsbook_odds,
        ),
        fair_model_odds=probability_to_american_odds(
            model_probability
        ),
    )


def determine_edge_grade(
    analysis: SideAnalysis,
    *,
    workload_all_metrics_fallback: bool = False,
) -> tuple[str, int]:
    """
    Grade the selected side using both expected value and probability edge.

    The thresholds are deliberately conservative.

    `workload_all_metrics_fallback=True` means no pitcher-specific workload
    data survived at all -- per requirement, Elite and Strong grades must
    never display in that case, regardless of how favorable the price
    looks, since the underlying projection itself is running on
    league-average substitutions rather than this pitcher's own data.
    """
    ev = analysis.expected_value
    price_edge = analysis.probability_edge_vs_price

    if ev >= 0.10 and price_edge >= 0.06:
        if workload_all_metrics_fallback:
            return "Moderate estimated edge", 3
        return "Elite estimated edge", 5

    if ev >= 0.06 and price_edge >= 0.04:
        if workload_all_metrics_fallback:
            return "Moderate estimated edge", 3
        return "Strong estimated edge", 4

    if ev >= 0.03 and price_edge >= 0.02:
        return "Moderate estimated edge", 3

    if ev > 0 and price_edge > 0:
        return "Small estimated edge", 2

    return "No positive estimated edge", 1


def determine_confidence(
    *,
    selected: SideAnalysis | None,
    lineup_confirmed: bool,
    pitcher_confirmed: bool = True,
    workload_warning: bool = False,
    workload_all_metrics_fallback: bool = False,
    injury_warning: bool = False,
    weather_warning: bool = False,
    stale_data: bool = False,
    model_sample_size: int | None = None,
) -> str:
    """
    Confidence is different from edge.

    Edge measures price value.
    Confidence measures trust in the inputs and model.

    `workload_all_metrics_fallback=True` means innings, batters faced, AND
    pitch count are ALL league-average substitutions (no pitcher-specific
    workload data survived at all) -- per requirement, confidence may not
    exceed MEDIUM in that case, enforced as a hard cap below rather than
    just another point deduction (deductions alone could still reach HIGH
    if every other factor were perfect).
    """
    if selected is None:
        return "PASS"

    score = 100

    if not lineup_confirmed:
        score -= 20

    if not pitcher_confirmed:
        score -= 25

    if workload_warning:
        score -= 20

    if injury_warning:
        score -= 20

    if weather_warning:
        score -= 10

    if stale_data:
        score -= 20

    if model_sample_size is not None:
        if model_sample_size < 100:
            score -= 20
        elif model_sample_size < 300:
            score -= 10

    if abs(selected.probability_edge_vs_price) < 0.02:
        score -= 10

    if workload_all_metrics_fallback:
        # Hard cap, not just a deduction: no combination of other-factor
        # bonuses can push a fully-fallback-workload projection to HIGH.
        score = min(score, 79)

    if score >= 80:
        return "HIGH"

    if score >= 60:
        return "MEDIUM"

    if score >= 40:
        return "LOW"

    return "AVOID"


def analyze_betting_edge(
    *,
    over_odds: int,
    under_odds: int,
    model_over_probability: float,
    model_under_probability: float | None = None,
    lineup_confirmed: bool,
    pitcher_confirmed: bool = True,
    workload_warning: bool = False,
    workload_all_metrics_fallback: bool = False,
    injury_warning: bool = False,
    weather_warning: bool = False,
    stale_data: bool = False,
    model_sample_size: int | None = None,
    minimum_ev_to_recommend: float = 0.01,
    minimum_price_edge_to_recommend: float = 0.01,
) -> EdgeAnalysis:
    """
    Compare the model probabilities against the offered sportsbook prices.

    Probabilities should be decimals:
        59.2% -> 0.592
    """
    if not 0 <= model_over_probability <= 1:
        raise ValueError(
            "model_over_probability must be between 0 and 1."
        )

    if model_under_probability is None:
        model_under_probability = 1 - model_over_probability

    probability_total = (
        model_over_probability + model_under_probability
    )

    if abs(probability_total - 1.0) > 0.01:
        raise ValueError(
            "Model over and under probabilities must add to "
            "approximately 1.0."
        )

    vig_free_over, vig_free_under = remove_vig(
        over_odds,
        under_odds,
    )

    over_analysis = analyze_side(
        side="OVER",
        sportsbook_odds=over_odds,
        model_probability=model_over_probability,
        vig_free_market_probability=vig_free_over,
    )

    under_analysis = analyze_side(
        side="UNDER",
        sportsbook_odds=under_odds,
        model_probability=model_under_probability,
        vig_free_market_probability=vig_free_under,
    )

    best_side = max(
        [over_analysis, under_analysis],
        key=lambda item: item.expected_value,
    )

    qualifies = (
        best_side.expected_value >= minimum_ev_to_recommend
        and best_side.probability_edge_vs_price
        >= minimum_price_edge_to_recommend
    )

    selected = best_side if qualifies else None
    recommended_side: BetSide = (
        best_side.side if qualifies else "PASS"
    )

    if selected is None:
        grade = "No meaningful estimated edge"
        stars = 1
        reason = (
            "Neither side meets the minimum expected-value and "
            "probability-edge requirements."
        )
    else:
        grade, stars = determine_edge_grade(
            selected,
            workload_all_metrics_fallback=workload_all_metrics_fallback,
        )
        reason = (
            f"The model gives the {selected.side.lower()} a "
            f"{selected.probability_edge_vs_price:.1%} probability "
            f"edge over the offered price and an estimated "
            f"{selected.expected_value:.1%} return per unit risked."
        )

    confidence = determine_confidence(
        selected=selected,
        lineup_confirmed=lineup_confirmed,
        pitcher_confirmed=pitcher_confirmed,
        workload_warning=workload_warning,
        workload_all_metrics_fallback=workload_all_metrics_fallback,
        injury_warning=injury_warning,
        weather_warning=weather_warning,
        stale_data=stale_data,
        model_sample_size=model_sample_size,
    )

    # Requirement: "If confidence becomes too low because of workload
    # uncertainty, the program should prefer PASS instead of giving a
    # betting recommendation." Scoped specifically to the workload-driven
    # case (not a general low-confidence-always-PASS rule, which would be
    # a broader behavior change outside this fix's scope): when ALL THREE
    # workload metrics are fallback values AND the resulting confidence
    # has dropped to AVOID, override any selected side with PASS.
    if (
        workload_all_metrics_fallback
        and confidence == "AVOID"
        and selected is not None
    ):
        selected = None
        recommended_side = "PASS"
        grade = "No meaningful estimated edge"
        stars = 1
        reason = (
            "Pitcher-specific workload data (innings, batters faced, and "
            "pitch count) was entirely unavailable, and the resulting "
            "confidence is too low to recommend a side."
        )

    return EdgeAnalysis(
        recommended_side=recommended_side,
        selected=selected,
        over=over_analysis,
        under=under_analysis,
        grade=grade,
        stars=stars,
        confidence=confidence,
        reason=reason,
    )