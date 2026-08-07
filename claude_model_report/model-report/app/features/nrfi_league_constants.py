"""
League-average NRFI/YRFI constants used as shrinkage priors.

Seeded from commonly-cited, publicly reported historical NRFI base rates
(MLB-wide first-inning scoring has run in roughly the high-20s to low-30s
percent per team-half over recent seasons, which combines to a whole-game
NRFI rate usually cited in the low-to-mid 50s percent). These are
documented starting points, refreshable once enough of your own backfilled
history accumulates (see app/training/nrfi_league_average_refresh.py) --
not a live-computed statistic.
"""
from __future__ import annotations

NRFI_LEAGUE_AVERAGES = {
    "season": 2025,
    # Probability a single team's half-inning (top or bottom of the 1st) goes scoreless.
    "league_scoreless_half_inning_rate": 0.715,
    # Derived if independent: 0.715 * 0.715 ~= 0.511. Kept as its own
    # documented constant (rather than only derived) since it's the figure
    # most directly comparable to backtesting output and sportsbook lines.
    "league_game_nrfi_rate": 0.51,
    "league_first_inning_era": 4.35,   # runs-per-9 equivalent, 1st inning only
    "league_first_inning_whip": 1.24,
    "league_first_inning_avg": 0.252,
    "league_first_inning_obp": 0.325,
    "league_first_inning_slg": 0.410,
    "league_first_inning_ops": 0.735,
    "league_first_inning_k_pct": 0.225,
    "league_first_inning_bb_pct": 0.085,
    "league_first_inning_hr_rate": 0.028,
    "league_first_inning_avg_pitches": 18.0,  # per starter, per 1st inning
}


def get_nrfi_league_average(key: str) -> float:
    if key not in NRFI_LEAGUE_AVERAGES:
        raise KeyError(f"No NRFI league average defined for '{key}'")
    return NRFI_LEAGUE_AVERAGES[key]
