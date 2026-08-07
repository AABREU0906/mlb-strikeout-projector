"""
Stage 2: Batter-Level Strikeout Probability.

Core method: Log5 (Bill James), the standard sabermetric method for
combining two independent rates against a league baseline, applied to
(shrunk) pitcher-vs-hand and batter-vs-hand strikeout rates:

    log5(pK, bK, lgK) =  (pK*bK/lgK) / ( (pK*bK/lgK) + ((1-pK)*(1-bK)/(1-lgK)) )

We deliberately do NOT simply average pitcher and batter rates (explicitly
disallowed) -- log5 is a multiplicative combination that properly accounts
for the league baseline.

On top of the log5 core, small, capped, documented multiplicative
adjustments are applied for: recent form (partial weight), pitch-type
matchup signal (partial weight, capped so it cannot dominate on small
samples), ballpark factor, weather, and umpire effect. Every adjustment is
capped in magnitude so no single modest-sample signal can swing the
probability unrealistically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.features.league_constants import get_league_average
from app.features.probability_math import log5 as _log5
from app.features.probability_math import resolve_rate_with_fallback
from app.schemas.player import BatterProfile, PitcherProfile

MAX_TOTAL_ADJUSTMENT = 0.35  # multiplicative cap: final prob within [base*(1-0.35), base*(1+0.35)]


@dataclass
class BatterMatchupResult:
    player_id: int
    name: str
    batting_order: Optional[int]
    log5_base_probability: float
    adjusted_probability: float
    adjustments_applied: dict = field(default_factory=dict)
    sample_size_warning: bool = False
    notes: list[str] = field(default_factory=list)
    # Structured fallback tracking (replaces the previous fragile ternary
    # chain that could silently pass None into log5() -- see the root
    # cause writeup in compute_batter_matchup_probability's docstring).
    pitcher_rate_source: str = "split"  # "split" | "season" | "career" | "league_average"
    batter_rate_source: str = "split"
    pitcher_rate_fallback_used: bool = False
    batter_rate_fallback_used: bool = False
    fallback_notes: list[str] = field(default_factory=list)


def compute_batter_matchup_probability(
    pitcher: PitcherProfile,
    batter: BatterProfile,
    ballpark_k_factor: float = 1.0,
    weather_k_multiplier: float = 1.0,
    umpire_k_multiplier: float = 1.0,
    pitch_type_signal_multiplier: Optional[float] = None,
) -> BatterMatchupResult:
    """
    ROOT CAUSE OF THE CRISTOPHER SANCHEZ CRASH: the previous fallback chain
    for pitcher_rate/batter_rate was a two-tier ternary
    (`split.shrunk_rate if split and split.shrunk_rate is not None else
    (season.shrunk_rate if season else league_average)`). If `season`
    EXISTED as a SampleStat object but its OWN `.shrunk_rate` field was
    itself None (e.g. zero observed events that season), the expression
    evaluated to that None -- the `if season` truthiness check only
    verified the SampleStat object existed, not that its rate was usable
    -- and that None flowed straight into log5(), which had no input
    validation and crashed with a bare TypeError. The chain also skipped
    the career tier entirely, jumping straight from season to league
    average.

    Fixed by resolving BOTH pitcher_rate and batter_rate through the same
    shared, fully-validated 4-tier hierarchy (split -> season -> career ->
    league average), via app.features.probability_math.
    resolve_rate_with_fallback -- which only ever accepts a candidate that
    passes is_valid_rate() (not None, numeric, finite, in [0,1]), so a
    None can never reach log5() from here again. log5() itself was also
    hardened to raise a clear ValueError instead of crashing, as a second
    line of defense.
    """
    side = batter.expected_side_today or batter.bat_side or "R"
    throws = pitcher.throws or "R"

    if side == "L":
        pitcher_split = pitcher.k_rate_vs_lhb
        batter_split = batter.k_rate_vs_lhp
        league_rate = get_league_average("league_k_rate_vs_lhp")
    else:
        pitcher_split = pitcher.k_rate_vs_rhb
        batter_split = batter.k_rate_vs_rhp
        league_rate = get_league_average("league_k_rate_vs_rhp")

    league_k_rate = get_league_average("league_k_rate")

    pitcher_resolution = resolve_rate_with_fallback(
        [
            ("split", pitcher_split.shrunk_rate if pitcher_split else None),
            ("season", pitcher.k_rate_season.shrunk_rate if pitcher.k_rate_season else None),
            ("career", pitcher.k_rate_career.shrunk_rate if pitcher.k_rate_career else None),
            ("league_average", league_k_rate),
        ],
        "pitcher",
    )
    batter_resolution = resolve_rate_with_fallback(
        [
            ("split", batter_split.shrunk_rate if batter_split else None),
            ("season", batter.k_rate_overall.shrunk_rate if batter.k_rate_overall else None),
            ("career", batter.k_rate_career.shrunk_rate if batter.k_rate_career else None),
            ("league_average", league_k_rate),
        ],
        "batter",
    )

    pitcher_rate = pitcher_resolution.value
    batter_rate = batter_resolution.value

    base_prob = _log5(pitcher_rate, batter_rate, league_rate)

    adjustments = {}
    notes = []
    sample_warning = False

    fallback_notes = list(pitcher_resolution.notes) + list(batter_resolution.notes)
    notes.extend(fallback_notes)

    if pitcher_split and pitcher_split.is_small_sample:
        sample_warning = True
        notes.append("Pitcher split sample is small; heavily shrunk toward overall/league rate.")
    if batter_split and batter_split.is_small_sample:
        sample_warning = True
        notes.append("Batter split sample is small; heavily shrunk toward overall/league rate.")

    # Recent form: blended in at 15% weight, only when available, and only
    # nudges the probability (never overrides the base rate).
    recent_rates = [r for r in (batter.k_rate_last_7d, batter.k_rate_last_14d, batter.k_rate_last_30d) if r is not None]
    if recent_rates:
        recent_avg = sum(recent_rates) / len(recent_rates)
        recent_component = 0.15 * recent_avg + 0.85 * base_prob
        adjustments["recent_form_blend"] = round(recent_component - base_prob, 4)
        base_prob = recent_component

    running_mult = 1.0
    if pitch_type_signal_multiplier is not None:
        capped = min(max(pitch_type_signal_multiplier, 0.9), 1.1)  # cap pitch-type influence
        running_mult *= capped
        adjustments["pitch_type_signal"] = capped

    park_mult = min(max(ballpark_k_factor, 0.9), 1.1)
    running_mult *= park_mult
    adjustments["ballpark"] = park_mult

    weather_mult = min(max(weather_k_multiplier, 0.95), 1.05)
    running_mult *= weather_mult
    adjustments["weather"] = weather_mult

    ump_mult = min(max(umpire_k_multiplier, 0.95), 1.05)
    running_mult *= ump_mult
    adjustments["umpire"] = ump_mult

    running_mult = min(max(running_mult, 1 - MAX_TOTAL_ADJUSTMENT), 1 + MAX_TOTAL_ADJUSTMENT)

    final_prob = min(max(base_prob * running_mult, 0.01), 0.75)

    return BatterMatchupResult(
        player_id=batter.player_id,
        name=batter.name,
        batting_order=batter.batting_order,
        log5_base_probability=round(_log5(pitcher_rate, batter_rate, league_rate), 4),
        adjusted_probability=round(final_prob, 4),
        adjustments_applied=adjustments,
        sample_size_warning=sample_warning,
        notes=notes,
        pitcher_rate_source=pitcher_resolution.source,
        batter_rate_source=batter_resolution.source,
        pitcher_rate_fallback_used=pitcher_resolution.fallback_used,
        batter_rate_fallback_used=batter_resolution.fallback_used,
        fallback_notes=fallback_notes,
    )
