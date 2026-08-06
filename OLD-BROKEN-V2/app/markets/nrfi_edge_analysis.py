"""
NRFI/YRFI betting-edge analysis.

Reuses app.markets.edge_analysis's core math directly (american_odds_to_probability,
remove_vig, analyze_side, determine_edge_grade, determine_confidence) rather
than reimplementing vig removal / EV math a second time. The only new code
here is presenting the two-sided result with NRFI/YRFI labels instead of
OVER/UNDER, since edge_analysis.py's dataclasses don't enforce their
Literal["OVER","UNDER"] type hints at runtime (Python doesn't check
dataclass field types), so passing "NRFI"/"YRFI" strings works correctly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from app.markets.edge_analysis import (
    SideAnalysis,
    analyze_side,
    determine_confidence,
    determine_edge_grade,
    remove_vig,
)

NrfiSide = Literal["NRFI", "YRFI", "PASS"]


@dataclass(frozen=True)
class NrfiEdgeAnalysis:
    recommended_side: NrfiSide
    selected: Optional[SideAnalysis]
    nrfi: SideAnalysis
    yrfi: SideAnalysis
    grade: str
    stars: int
    confidence: str
    reason: str


def analyze_nrfi_edge(
    *,
    nrfi_odds: int,
    yrfi_odds: int,
    model_nrfi_probability: float,
    model_yrfi_probability: float,
    lineup_confirmed: bool,
    pitcher_confirmed: bool = True,
    workload_warning: bool = False,
    injury_warning: bool = False,
    weather_warning: bool = False,
    stale_data: bool = False,
    model_sample_size: Optional[int] = None,
    minimum_ev_to_recommend: float = 0.0,
    minimum_price_edge_to_recommend: float = 0.0,
) -> NrfiEdgeAnalysis:
    probability_total = model_nrfi_probability + model_yrfi_probability
    if abs(probability_total - 1.0) > 0.01:
        raise ValueError("Model NRFI and YRFI probabilities must add to approximately 1.0.")

    vig_free_nrfi, vig_free_yrfi = remove_vig(nrfi_odds, yrfi_odds)

    nrfi_analysis = analyze_side(
        side="NRFI",  # type: ignore[arg-type]  -- see module docstring
        sportsbook_odds=nrfi_odds,
        model_probability=model_nrfi_probability,
        vig_free_market_probability=vig_free_nrfi,
    )
    yrfi_analysis = analyze_side(
        side="YRFI",  # type: ignore[arg-type]
        sportsbook_odds=yrfi_odds,
        model_probability=model_yrfi_probability,
        vig_free_market_probability=vig_free_yrfi,
    )

    best_side = max([nrfi_analysis, yrfi_analysis], key=lambda item: item.expected_value)
    qualifies = (
        best_side.expected_value >= minimum_ev_to_recommend
        and best_side.probability_edge_vs_price >= minimum_price_edge_to_recommend
    )
    selected = best_side if qualifies else None
    recommended_side: NrfiSide = best_side.side if qualifies else "PASS"  # type: ignore[assignment]

    if selected is None:
        grade = "No meaningful estimated edge"
        stars = 1
        reason = "Neither side meets the minimum expected-value and probability-edge requirements."
    else:
        grade, stars = determine_edge_grade(selected)
        reason = (
            f"The model gives {selected.side} a {selected.probability_edge_vs_price:.1%} "
            f"probability edge over the offered price and an estimated "
            f"{selected.expected_value:.1%} return per unit risked."
        )

    confidence = determine_confidence(
        selected=selected,
        lineup_confirmed=lineup_confirmed,
        pitcher_confirmed=pitcher_confirmed,
        workload_warning=workload_warning,
        injury_warning=injury_warning,
        weather_warning=weather_warning,
        stale_data=stale_data,
        model_sample_size=model_sample_size,
    )

    return NrfiEdgeAnalysis(
        recommended_side=recommended_side,
        selected=selected,
        nrfi=nrfi_analysis,
        yrfi=yrfi_analysis,
        grade=grade,
        stars=stars,
        confidence=confidence,
        reason=reason,
    )
