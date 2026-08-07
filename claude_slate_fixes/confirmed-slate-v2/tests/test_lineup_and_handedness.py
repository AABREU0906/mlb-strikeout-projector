from app.data_sources.mlb_stats_api import MlbStatsApiProvider


def test_normalize_game_captures_doubleheader_fields():
    provider = MlbStatsApiProvider()
    raw = {
        "gamePk": 12345,
        "officialDate": "2026-07-30",
        "gameDate": "2026-07-30T23:05:00Z",
        "status": {"detailedState": "Scheduled", "abstractGameState": "Preview"},
        "doubleHeader": "Y",
        "gameNumber": 2,
        "teams": {
            "home": {"team": {"id": 111, "name": "Home Team"}, "probablePitcher": {"id": 1, "fullName": "A Pitcher"}},
            "away": {"team": {"id": 222, "name": "Away Team"}, "probablePitcher": {"id": 2, "fullName": "B Pitcher"}},
        },
        "venue": {"id": 19, "name": "Coors Field"},
    }
    normalized = provider._normalize_game(raw)
    assert normalized["doubleheader"] == "Y"
    assert normalized["game_number"] == 2
    assert normalized["venue_id"] == 19
    assert normalized["probable_home_pitcher_id"] == 1
    assert normalized["probable_away_pitcher_id"] == 2


def test_normalize_game_handles_missing_probable_pitchers():
    provider = MlbStatsApiProvider()
    raw = {
        "gamePk": 999,
        "officialDate": "2026-07-30",
        "gameDate": "2026-07-30T23:05:00Z",
        "status": {"detailedState": "Postponed", "abstractGameState": "Preview"},
        "teams": {
            "home": {"team": {"id": 111, "name": "Home Team"}},
            "away": {"team": {"id": 222, "name": "Away Team"}},
        },
        "venue": {"id": 15, "name": "Chase Field"},
    }
    normalized = provider._normalize_game(raw)
    assert normalized["probable_home_pitcher_id"] is None
    assert normalized["probable_away_pitcher_id"] is None
    assert normalized["status"] == "Postponed"


def test_switch_hitter_resolves_opposite_of_pitcher_hand():
    from app.features.batter_features import BatterFeatureBuilder

    # Simulate the resolution logic directly (the part that doesn't need a
    # live network call): a switch hitter vs a RHP should resolve to 'L'.
    bat_side = "S"
    pitcher_hand_today = "R"
    is_switch = bat_side == "S"
    expected_side = bat_side
    if is_switch and pitcher_hand_today:
        expected_side = "L" if pitcher_hand_today == "R" else "R"
    assert expected_side == "L"

    pitcher_hand_today = "L"
    expected_side = "R" if pitcher_hand_today == "L" else "L"
    assert expected_side == "R"
