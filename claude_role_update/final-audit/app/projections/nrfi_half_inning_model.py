"""
Half-Inning Scoring Probability Model.

Core method: log5 combines the opposing team's first-inning scoring rate
with the starting pitcher's first-inning run-allowed rate against a league
baseline -- the same log5 combination already used (and tested) for the
strikeout model's batter/pitcher matchups, imported from the shared
app/features/probability_math module rather than reimplemented.

  p_team_scores = log5(team_scoring_rate, pitcher_run_allowed_rate, league_scoring_rate)

On top of the log5 core, small CAPPED adjustments are layered for:
  - Top-of-lineup (batting spots 1-5) BvP-informed OBP/SLG quality --
    weighted highest per spec ("give the highest importance to hitters
    batting first through fifth").
  - Ballpark run factor (reused from the strikeout model's ballpark
    reference table -- not a second table).
  - Weather (very light touch, consistent with the strikeout model's
    "modest, explainable, validated" rule against overweighting weather).
  - Umpire tendency (reused from the strikeout model's umpire module).

Every adjustment is capped so it cannot overpower the core pitcher/team
matchup, matching the explicit spec requirement that "weather and park
effects should adjust the projection without overpowering the pitcher and
lineup matchup."

Finally: P(NRFI) = P(away doesn't score) * P(home doesn't score), via
combine_independent_no_score_probabilities -- explicitly NOT an average.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.features.probability_math import combine_independent_no_score_probabilities, log5
from app.schemas.nrfi import BvPProfile, PitcherFirstInningProfile, TeamFirstInningProfile

MAX_LINEUP_ADJUSTMENT = 0.15
MAX_PARK_ADJUSTMENT = 0.08
MAX_WEATHER_ADJUSTMENT = 0.05
MAX_UMPIRE_ADJUSTMENT = 0.05


@dataclass
class HalfInningResult:
    scoring_probability: float
    no_score_probability: float
    log5_base_probability: float
    adjustments_applied: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def compute_half_inning_scoring_probability(
    offense: TeamFirstInningProfile,
    defense_pitcher: PitcherFirstInningProfile,
    league_scoring_rate: float,
    top_order_bvp: Optional[list[BvPProfile]] = None,
    league_obp: float = 0.318,
    league_slg: float = 0.405,
    ballpark_run_factor: float = 1.0,
    weather_run_multiplier: float = 1.0,
    umpire_run_multiplier: float = 1.0,
) -> HalfInningResult:
    team_rate = offense.season_scoring_rate.shrunk_rate
    if team_rate is None:
        team_rate = league_scoring_rate

    pitcher_scoreless = defense_pitcher.season_scoreless_rate.shrunk_rate
    if pitcher_scoreless is None:
        pitcher_scoreless = 1.0 - league_scoring_rate
    pitcher_run_allowed_rate = 1.0 - pitcher_scoreless

    base_prob = log5(team_rate, pitcher_run_allowed_rate, league_scoring_rate)

    adjustments: dict[str, float] = {}
    notes: list[str] = []
    running_mult = 1.0

    if top_order_bvp:
        obp_values = [b.obp.final_adjusted_value for b in top_order_bvp if b.obp is not None]
        slg_values = [b.slg.final_adjusted_value for b in top_order_bvp if b.slg is not None]
        if obp_values and slg_values:
            avg_obp = sum(obp_values) / len(obp_values)
            avg_slg = sum(slg_values) / len(slg_values)
            obp_dev = (avg_obp - league_obp) / league_obp
            slg_dev = (avg_slg - league_slg) / league_slg
            combined_dev = (obp_dev + slg_dev) / 2
            lineup_mult = 1.0 + max(min(combined_dev * 0.5, MAX_LINEUP_ADJUSTMENT), -MAX_LINEUP_ADJUSTMENT)
            running_mult *= lineup_mult
            adjustments["top_order_bvp_quality"] = round(lineup_mult, 4)
            if abs(combined_dev) > 0.15:
                notes.append(
                    "Top-of-order (1-5) BvP-informed OBP/SLG is meaningfully "
                    f"{'above' if combined_dev > 0 else 'below'} league average."
                )

    park_mult = 1.0 + max(min((ballpark_run_factor - 1.0), MAX_PARK_ADJUSTMENT), -MAX_PARK_ADJUSTMENT)
    running_mult *= park_mult
    adjustments["ballpark"] = round(park_mult, 4)

    weather_mult = min(max(weather_run_multiplier, 1 - MAX_WEATHER_ADJUSTMENT), 1 + MAX_WEATHER_ADJUSTMENT)
    running_mult *= weather_mult
    adjustments["weather"] = round(weather_mult, 4)

    umpire_mult = min(max(umpire_run_multiplier, 1 - MAX_UMPIRE_ADJUSTMENT), 1 + MAX_UMPIRE_ADJUSTMENT)
    running_mult *= umpire_mult
    adjustments["umpire"] = round(umpire_mult, 4)

    final_prob = min(max(base_prob * running_mult, 0.02), 0.85)

    return HalfInningResult(
        scoring_probability=round(final_prob, 4),
        no_score_probability=round(1.0 - final_prob, 4),
        log5_base_probability=round(base_prob, 4),
        adjustments_applied=adjustments,
        notes=notes,
    )


@dataclass
class NrfiGameResult:
    away_half: HalfInningResult
    home_half: HalfInningResult
    nrfi_probability: float
    yrfi_probability: float
    expected_first_inning_runs: float


def compute_nrfi_probability(
    away_offense: TeamFirstInningProfile,
    home_pitcher: PitcherFirstInningProfile,
    home_offense: TeamFirstInningProfile,
    away_pitcher: PitcherFirstInningProfile,
    league_scoring_rate: float,
    away_top_order_bvp: Optional[list[BvPProfile]] = None,
    home_top_order_bvp: Optional[list[BvPProfile]] = None,
    ballpark_run_factor: float = 1.0,
    weather_run_multiplier: float = 1.0,
    umpire_run_multiplier: float = 1.0,
) -> NrfiGameResult:
    """Models the two half-innings SEPARATELY (never averaged) then
    combines via P(NRFI) = P(away no score) * P(home no score)."""

    away_half = compute_half_inning_scoring_probability(
        offense=away_offense,
        defense_pitcher=home_pitcher,
        league_scoring_rate=league_scoring_rate,
        top_order_bvp=away_top_order_bvp,
        ballpark_run_factor=ballpark_run_factor,
        weather_run_multiplier=weather_run_multiplier,
        umpire_run_multiplier=umpire_run_multiplier,
    )
    home_half = compute_half_inning_scoring_probability(
        offense=home_offense,
        defense_pitcher=away_pitcher,
        league_scoring_rate=league_scoring_rate,
        top_order_bvp=home_top_order_bvp,
        ballpark_run_factor=ballpark_run_factor,
        weather_run_multiplier=weather_run_multiplier,
        umpire_run_multiplier=umpire_run_multiplier,
    )

    nrfi_prob, yrfi_prob = combine_independent_no_score_probabilities(
        away_half.no_score_probability, home_half.no_score_probability
    )

    expected_runs = (away_half.scoring_probability + home_half.scoring_probability) * 0.62
    # 0.62 is a documented approximation of average runs-given-scored in a
    # single inning (most scoring innings produce 1 run, some produce 2+).

    return NrfiGameResult(
        away_half=away_half,
        home_half=home_half,
        nrfi_probability=round(nrfi_prob, 4),
        yrfi_probability=round(yrfi_prob, 4),
        expected_first_inning_runs=round(expected_runs, 3),
    )
