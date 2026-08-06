"""
League-average constants used as shrinkage priors.

These are seeded from recent, publicly reported MLB-wide season averages and
are intentionally treated as *defaults* that should be refreshed each season
(e.g. via `python main.py update-results` accumulating enough games to
recompute them empirically -- see app/training/league_average_refresh.py).
Do not treat these as permanently fixed; they are a documented starting
point, not a scraped live value.
"""
from __future__ import annotations

LEAGUE_AVERAGES = {
    "season": 2025,
    "league_k_rate": 0.224,          # strikeouts per plate appearance, league-wide
    "league_k_rate_vs_rhp": 0.222,
    "league_k_rate_vs_lhp": 0.228,
    "league_bb_rate": 0.083,
    "league_contact_rate": 0.755,
    "league_zone_contact_rate": 0.835,
    "league_chase_rate": 0.283,
    "league_swstr_rate": 0.112,
    "league_first_pitch_strike_rate": 0.615,
    "league_runs_per_game": 4.4,
    "league_pa_per_game_per_team": 38.5,
    "league_pitches_per_start_avg": 88.0,
    "league_ip_per_start_avg": 5.1,
    "league_bf_per_start_avg": 22.5,
    "league_avg": 0.248,   # batting average allowed, league-wide (used as BvP shrinkage prior)
    "league_obp": 0.318,
    "league_slg": 0.405,
}


def get_league_average(key: str) -> float:
    if key not in LEAGUE_AVERAGES:
        raise KeyError(f"No league average defined for '{key}'")
    return LEAGUE_AVERAGES[key]
