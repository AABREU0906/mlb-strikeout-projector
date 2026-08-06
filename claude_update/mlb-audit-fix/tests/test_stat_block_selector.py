"""
Tests for the root-cause fix behind the workload bug (impossible values
like 27.9 expected innings / 126 expected batters faced / 531 expected
pitches). See app/data_sources/mlb_stat_block_selector.py for the full
root-cause writeup.
"""
import pytest

from app.data_sources.mlb_stat_block_selector import select_stat_block


def _block(splits):
    return [{"type": {"displayName": "season"}, "group": {"displayName": "pitching"}, "splits": splits}]


def test_reproduces_and_fixes_the_reported_bug():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "S", "team": {"name": "Mets"},
         "stat": {"gamesStarted": 1, "inningsPitched": "2.0", "battersFaced": 10,
                   "numberOfPitches": 35, "strikeOuts": 3, "baseOnBalls": 1}},
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Mets"},
         "stat": {"gamesStarted": 20, "inningsPitched": "112.1", "battersFaced": 470,
                   "numberOfPitches": 1850, "strikeOuts": 130, "baseOnBalls": 35}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert result.stat["gamesStarted"] == 20
    assert result.stat["inningsPitched"] == "112.1"
    ip_per_start = (112 + 1 / 3) / 20
    assert 0.5 <= ip_per_start <= 9.0


def test_excludes_minor_league_split_mixed_with_mlb():
    stats = _block([
        {"season": "2026", "sport": {"id": 11}, "league": {"name": "International League"}, "gameType": "R",
         "team": {"name": "Scranton (AAA)"},
         "stat": {"gamesStarted": 4, "inningsPitched": "22.0", "battersFaced": 95,
                   "numberOfPitches": 380, "strikeOuts": 20, "baseOnBalls": 8}},
        {"season": "2026", "sport": {"id": 1}, "league": {"name": "American League"}, "gameType": "R",
         "team": {"name": "New York Yankees"},
         "stat": {"gamesStarted": 6, "inningsPitched": "33.2", "battersFaced": 140,
                   "numberOfPitches": 531, "strikeOuts": 30, "baseOnBalls": 10}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert result.stat["gamesStarted"] == 6
    assert result.stat["battersFaced"] == 140


def test_aggregates_genuine_mid_season_trade():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Miami Marlins"},
         "stat": {"gamesStarted": 12, "inningsPitched": "68.1", "battersFaced": 290,
                   "numberOfPitches": 1100, "strikeOuts": 65, "baseOnBalls": 22}},
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Los Angeles Dodgers"},
         "stat": {"gamesStarted": 15, "inningsPitched": "89.2", "battersFaced": 370,
                   "numberOfPitches": 1450, "strikeOuts": 90, "baseOnBalls": 25}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert result.is_aggregated_from_multiple_splits
    assert result.stat["gamesStarted"] == 27
    assert result.stat["inningsPitched"] == "158.0"
    assert result.stat["battersFaced"] == 660


def test_prefers_api_provided_combined_total_over_manual_aggregation():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Marlins"},
         "stat": {"gamesStarted": 12, "inningsPitched": "68.1", "battersFaced": 290}},
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Dodgers"},
         "stat": {"gamesStarted": 15, "inningsPitched": "89.2", "battersFaced": 370}},
        {"season": "2026", "sport": {"id": 1}, "gameType": "R",
         "stat": {"gamesStarted": 27, "inningsPitched": "158.0", "battersFaced": 660}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert not result.is_aggregated_from_multiple_splits
    assert result.stat["gamesStarted"] == 27


def test_postseason_only_data_is_refused_not_mislabeled():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "P", "team": {"name": "Astros"},
         "stat": {"gamesStarted": 3, "inningsPitched": "18.0", "battersFaced": 70}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert not result.ok
    assert any("gameType" in r for r in result.rejected_reasons)


def test_wrong_season_split_is_rejected():
    stats = _block([
        {"season": "2025", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Reds"},
         "stat": {"gamesStarted": 30, "inningsPitched": "180.0", "battersFaced": 750}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert not result.ok


def test_exact_type_match_rejects_near_miss_names():
    stats = [{"type": {"displayName": "seasonAdvanced"}, "group": {"displayName": "pitching"},
              "splits": [{"stat": {"gamesStarted": 30}}]}]
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert not result.ok


def test_exact_group_match_rejects_hitting_when_pitching_requested():
    stats = [{"type": {"displayName": "season"}, "group": {"displayName": "hitting"},
              "splits": [{"stat": {"plateAppearances": 500}}]}]
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert not result.ok


def test_empty_stats_list_returns_unresolved_not_crash():
    result = select_stat_block([], "season", "pitching", season=2026)
    assert not result.ok
    assert result.n_candidate_blocks == 0


def test_split_with_no_level_info_is_accepted_not_rejected_by_default():
    stats = _block([{"season": "2026", "stat": {"gamesStarted": 30, "inningsPitched": "180.0"}}])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok


def test_relief_pitcher_zero_starts_produces_zero_not_crash():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Rays"},
         "stat": {"gamesStarted": 0, "gamesPitched": 45, "inningsPitched": "48.0", "battersFaced": 210}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert result.stat["gamesStarted"] == 0


def test_invalid_innings_notation_excluded_from_aggregation():
    stats = _block([
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "A"},
         "stat": {"gamesStarted": 10, "inningsPitched": "55.5", "battersFaced": 230}},
        {"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "B"},
         "stat": {"gamesStarted": 10, "inningsPitched": "50.1", "battersFaced": 210}},
    ])
    result = select_stat_block(stats, "season", "pitching", season=2026)
    assert result.ok
    assert result.stat["gamesStarted"] == 20
    assert result.stat["inningsPitched"] == "50.1"
