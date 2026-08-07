"""
NRFI/YRFI Projection Explanation.

Every line is generated from an actual computed feature value (a rate, a
sample size, a threat-score component) -- never a templated sentence
unsupported by the underlying data, per spec ("Do not use generic
explanation templates unsupported by the data").
"""
from __future__ import annotations

from dataclasses import dataclass

from app.projections.nrfi_half_inning_model import HalfInningResult
from app.projections.nrfi_threat_score import ThreatScoreResult
from app.schemas.nrfi import PitcherFirstInningProfile, TeamFirstInningProfile


@dataclass
class ExplanationLine:
    direction: str
    text: str


def build_nrfi_explanation(
    away_pitcher: PitcherFirstInningProfile,
    home_pitcher: PitcherFirstInningProfile,
    away_offense: TeamFirstInningProfile,
    home_offense: TeamFirstInningProfile,
    away_half: HalfInningResult,
    home_half: HalfInningResult,
    away_threat: ThreatScoreResult,
    home_threat: ThreatScoreResult,
    league_scoreless_rate: float,
) -> dict:
    nrfi_lines: list[ExplanationLine] = []
    yrfi_lines: list[ExplanationLine] = []

    for pitcher, label in [(home_pitcher, "Home starter"), (away_pitcher, "Away starter")]:
        rate = pitcher.season_scoreless_rate.shrunk_rate
        n = pitcher.season_scoreless_rate.observed_n
        if rate is None:
            continue
        if rate >= league_scoreless_rate * 1.08 and n >= 3:
            nrfi_lines.append(ExplanationLine(
                "nrfi",
                f"{label} {pitcher.name} has a shrinkage-adjusted scoreless-first-inning "
                f"rate of {rate*100:.0f}% over {n} start(s) with data.",
            ))
        elif rate <= league_scoreless_rate * 0.90 and n >= 3:
            yrfi_lines.append(ExplanationLine(
                "yrfi",
                f"{label} {pitcher.name} has an elevated shrinkage-adjusted first-inning "
                f"run-allowed rate this sample ({(1-rate)*100:.0f}%, over {n} start(s)).",
            ))

    for offense, threat, label in [(away_offense, away_threat, "Away lineup"), (home_offense, home_threat, "Home lineup")]:
        if threat.score >= 62:
            top = ", ".join(threat.top_contributors[:2]) if threat.top_contributors else "overall offensive profile"
            yrfi_lines.append(ExplanationLine(
                "yrfi", f"{label} carries an elevated First-Inning Threat Score ({threat.score:.0f}/100), driven by {top}."
            ))
        elif threat.score <= 38:
            nrfi_lines.append(ExplanationLine(
                "nrfi", f"{label} carries a suppressed First-Inning Threat Score ({threat.score:.0f}/100)."
            ))

    both_below_avg = (
        away_offense.season_scoring_rate.shrunk_rate is not None
        and home_offense.season_scoring_rate.shrunk_rate is not None
        and away_offense.season_scoring_rate.shrunk_rate < (1 - league_scoreless_rate)
        and home_offense.season_scoring_rate.shrunk_rate < (1 - league_scoreless_rate)
    )
    if both_below_avg:
        nrfi_lines.append(ExplanationLine("nrfi", "Both teams are below league average in first-inning scoring rate this season."))

    for half, label in [(away_half, "away half-inning"), (home_half, "home half-inning")]:
        for note in half.notes:
            direction = "yrfi" if "above" in note else "nrfi"
            (yrfi_lines if direction == "yrfi" else nrfi_lines).append(
                ExplanationLine(direction, f"{label.capitalize()}: {note}")
            )

    return {
        "nrfi_factors": [l.text for l in nrfi_lines],
        "yrfi_factors": [l.text for l in yrfi_lines],
    }
