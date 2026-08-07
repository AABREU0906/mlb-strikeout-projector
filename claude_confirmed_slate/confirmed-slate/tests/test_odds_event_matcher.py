from app.services.odds_event_matcher import canonical_team_name, match_event_to_game


def test_team_name_variants_normalize_to_same_canonical_form():
    assert canonical_team_name("Yankees") == canonical_team_name("New York Yankees")


def test_unrecognized_team_returns_none():
    assert canonical_team_name("Not A Real Team") is None


def test_none_input_returns_none():
    assert canonical_team_name(None) is None


def test_case_insensitive_matching():
    assert canonical_team_name("yankees") == "New York Yankees"


def test_basic_event_match_found():
    events = [
        {"id": "evt-1", "home_team": "New York Yankees", "away_team": "Boston Red Sox", "commence_time": "2026-07-15T19:05:00Z"},
        {"id": "evt-2", "home_team": "Los Angeles Dodgers", "away_team": "San Francisco Giants", "commence_time": "2026-07-15T22:10:00Z"},
    ]
    matched = match_event_to_game(events, "Yankees", "Red Sox", "2026-07-15T19:05:00Z")
    assert matched is not None
    assert matched.event_id == "evt-1"


def test_no_matching_event_returns_none():
    events = [{"id": "evt-1", "home_team": "New York Yankees", "away_team": "Boston Red Sox", "commence_time": "2026-07-15T19:05:00Z"}]
    matched = match_event_to_game(events, "Chicago Cubs", "Milwaukee Brewers", "2026-07-15T19:00:00Z")
    assert matched is None


def test_doubleheader_game_2_matched_via_time_proximity():
    dh_events = [
        {"id": "dh-game1", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T17:00:00Z"},
        {"id": "dh-game2", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T21:00:00Z"},
    ]
    matched = match_event_to_game(dh_events, "Cubs", "Brewers", "2026-07-15T20:45:00Z", max_start_time_diff_minutes=60)
    assert matched is not None
    assert matched.event_id == "dh-game2"


def test_doubleheader_game_1_matched_via_time_proximity():
    dh_events = [
        {"id": "dh-game1", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T17:00:00Z"},
        {"id": "dh-game2", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T21:00:00Z"},
    ]
    matched = match_event_to_game(dh_events, "Cubs", "Brewers", "2026-07-15T17:10:00Z", max_start_time_diff_minutes=60)
    assert matched is not None
    assert matched.event_id == "dh-game1"


def test_ambiguous_match_never_guesses():
    ambiguous_events = [
        {"id": "amb-1", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T19:00:00Z"},
        {"id": "amb-2", "home_team": "Chicago Cubs", "away_team": "Milwaukee Brewers", "commence_time": "2026-07-15T19:30:00Z"},
    ]
    matched = match_event_to_game(ambiguous_events, "Cubs", "Brewers", "2026-07-15T19:15:00Z", max_start_time_diff_minutes=180)
    assert matched is None


def test_unrecognized_team_name_refuses_match():
    events = [{"id": "evt-1", "home_team": "New York Yankees", "away_team": "Boston Red Sox", "commence_time": "2026-07-15T19:05:00Z"}]
    matched = match_event_to_game(events, "Some Minor League Team", "Boston Red Sox", "2026-07-15T19:05:00Z")
    assert matched is None


def test_wrong_side_home_away_swap_does_not_match():
    events = [{"id": "evt-1", "home_team": "New York Yankees", "away_team": "Boston Red Sox", "commence_time": "2026-07-15T19:05:00Z"}]
    matched = match_event_to_game(events, "Boston Red Sox", "New York Yankees", "2026-07-15T19:05:00Z")
    assert matched is None
