"""
Tests for app/services/confirmed_slate_odds.py. All Odds API calls are
mocked -- this file makes zero live network requests.
"""
import app.services.confirmed_slate_odds as odds_mod
from app.services.confirmed_slate_odds import OddsRunSession, extract_fanduel_pitcher_outcomes


class _FakeResp:
    def __init__(self, json_body, headers):
        self.json_body = json_body
        self.headers = headers
        self.from_cache = False


def _make_fake_get_json(call_log):
    def fake_get_json(url, params=None, cache_category="misc", cache_ttl_seconds=0, **kw):
        call_log.append(url)
        if "events/" in url and url.endswith("/odds"):
            return _FakeResp(
                {"bookmakers": [{"key": "fanduel", "markets": [{
                    "key": "pitcher_strikeouts", "last_update": "2026-07-15T18:00:00Z",
                    "outcomes": [
                        {"description": "Cristopher Sanchez", "name": "Over", "point": 5.5, "price": -110},
                        {"description": "Cristopher Sanchez", "name": "Under", "point": 5.5, "price": -120},
                    ],
                }]}]},
                {"x-requests-remaining": "450", "x-requests-used": "50"},
            )
        if url.endswith("/events"):
            return _FakeResp(
                [{"id": "evt-1", "home_team": "Philadelphia Phillies", "away_team": "New York Mets", "commence_time": "2026-07-15T19:05:00Z"}],
                {"x-requests-remaining": "460", "x-requests-used": "40"},
            )
        return None
    return fake_get_json


def test_events_list_fetched_at_most_once(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    events1 = session.get_events()
    events2 = session.get_events()
    events3 = session.get_events()

    assert events1 == events2 == events3
    assert session.events_list_calls_made == 1
    assert sum(1 for u in call_log if u.endswith("/events")) == 1


def test_event_odds_fetched_at_most_once_per_event(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    odds1 = session.get_event_odds("evt-1")
    odds2 = session.get_event_odds("evt-1")

    assert odds1 == odds2
    assert session.event_odds_calls_made == 1


def test_same_event_never_queried_twice_in_one_run(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    session.get_event_odds("evt-1")
    session.get_event_odds("evt-1")
    session.get_event_odds("evt-1")

    assert sum(1 for u in call_log if "/odds" in u) == 1


def test_different_events_each_fetched_once(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    session.get_event_odds("evt-1")
    session.get_event_odds("evt-2")
    session.get_event_odds("evt-1")

    assert session.event_odds_calls_made == 2


def test_credit_headers_captured(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    session.get_events()
    session.get_event_odds("evt-1")

    assert session.credits_remaining == 450
    assert session.credits_used_this_run == 50


def test_missing_api_key_makes_zero_calls(monkeypatch):
    call_log = []
    monkeypatch.setattr(odds_mod.http_client, "get_json", _make_fake_get_json(call_log))
    session = OddsRunSession(api_key=None, base_url="https://api.the-odds-api.com/v4")

    events = session.get_events()
    odds = session.get_event_odds("evt-1")

    assert events == []
    assert odds is None
    assert len(call_log) == 0
    assert not session.is_configured()


def test_api_timeout_does_not_crash(monkeypatch):
    def failing_get_json(*a, **kw):
        return None

    monkeypatch.setattr(odds_mod.http_client, "get_json", failing_get_json)
    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")

    events = session.get_events()
    odds = session.get_event_odds("evt-1")

    assert events == []
    assert odds is None


def test_extract_fanduel_outcomes_filters_bookmaker():
    event_odds = {
        "bookmakers": [
            {"key": "draftkings", "markets": [{"key": "pitcher_strikeouts", "outcomes": [
                {"description": "Some Pitcher", "name": "Over", "point": 6.5, "price": 100},
            ]}]},
            {"key": "fanduel", "markets": [{"key": "pitcher_strikeouts", "last_update": "2026-07-15T18:00:00Z", "outcomes": [
                {"description": "Cristopher Sanchez", "name": "Over", "point": 5.5, "price": -110},
                {"description": "Cristopher Sanchez", "name": "Under", "point": 5.5, "price": -120},
            ]}]},
        ]
    }
    outcomes = extract_fanduel_pitcher_outcomes(event_odds)
    assert len(outcomes) == 2
    assert all(o["pitcher_name"] == "Cristopher Sanchez" for o in outcomes)


def test_extract_fanduel_outcomes_ignores_alternate_markets():
    event_odds = {
        "bookmakers": [
            {"key": "fanduel", "markets": [
                {"key": "pitcher_strikeouts_alternate", "outcomes": [{"description": "X", "name": "Over", "point": 7.5, "price": 200}]},
                {"key": "pitcher_strikeouts", "last_update": "2026-07-15T18:00:00Z", "outcomes": [
                    {"description": "Cristopher Sanchez", "name": "Over", "point": 5.5, "price": -110},
                    {"description": "Cristopher Sanchez", "name": "Under", "point": 5.5, "price": -120},
                ]},
            ]},
        ]
    }
    outcomes = extract_fanduel_pitcher_outcomes(event_odds)
    assert len(outcomes) == 2
    assert all(o["point"] == 5.5 for o in outcomes)


def test_extract_fanduel_outcomes_empty_response_no_crash():
    assert extract_fanduel_pitcher_outcomes({}) == []
    assert extract_fanduel_pitcher_outcomes(None) == []


# --- Bug 1 fix: credits_used_this_run must be an intra-run delta, not the
# raw cumulative x-requests-used header. ---

def test_first_run_uses_credits_delta_computed_correctly(monkeypatch):
    """First run: events call (500 remaining) + 4 event-odds calls,
    ending at 493 remaining -- credits_used_this_run must be 7 (the real
    delta), matching the actual reported scenario."""
    responses = [
        _FakeResp([{"id": f"evt-{i}"} for i in range(4)], {"x-requests-remaining": "500", "x-requests-used": "3"}),
        _FakeResp({"bookmakers": []}, {"x-requests-remaining": "498", "x-requests-used": "5"}),
        _FakeResp({"bookmakers": []}, {"x-requests-remaining": "496", "x-requests-used": "7"}),
        _FakeResp({"bookmakers": []}, {"x-requests-remaining": "495", "x-requests-used": "8"}),
        _FakeResp({"bookmakers": []}, {"x-requests-remaining": "493", "x-requests-used": "10"}),
    ]
    it = iter(responses)
    monkeypatch.setattr(odds_mod.http_client, "get_json", lambda *a, **kw: next(it))

    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")
    session.get_events()
    for i in range(4):
        session.get_event_odds(f"evt-{i}")

    assert session.credits_remaining == 493
    assert session.credits_used_this_run == 7
    assert session.credits_used_cumulative_account == 10


def test_second_incremental_run_makes_zero_prop_calls(monkeypatch):
    """Second run: only the events call fires (nothing newly confirmed
    needs odds) -- event_odds_calls_made must be 0."""
    responses = [_FakeResp([], {"x-requests-remaining": "493", "x-requests-used": "10"})]
    it = iter(responses)
    monkeypatch.setattr(odds_mod.http_client, "get_json", lambda *a, **kw: next(it))

    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")
    session.get_events()

    assert session.event_odds_calls_made == 0


def test_second_run_remaining_credits_unchanged(monkeypatch):
    responses = [_FakeResp([], {"x-requests-remaining": "493", "x-requests-used": "10"})]
    it = iter(responses)
    monkeypatch.setattr(odds_mod.http_client, "get_json", lambda *a, **kw: next(it))

    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")
    session.get_events()

    assert session.credits_remaining == 493


def test_second_run_displays_zero_credits_used_this_run(monkeypatch):
    """THE EXACT REPORTED BUG: a run making zero player-prop calls (only
    the events-list call) must show credits_used_this_run == 0, not the
    raw cumulative x-requests-used value (which was 7 in the report,
    despite remaining credits being provably unchanged)."""
    responses = [_FakeResp([], {"x-requests-remaining": "493", "x-requests-used": "10"})]
    it = iter(responses)
    monkeypatch.setattr(odds_mod.http_client, "get_json", lambda *a, **kw: next(it))

    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")
    session.get_events()

    assert session.credits_used_this_run == 0
    # The raw cumulative figure is still captured, but under its own,
    # clearly-distinct name -- never conflated with "this run" again.
    assert session.credits_used_cumulative_account == 10


def test_zero_calls_made_returns_zero_credits_used():
    """No calls at all this run (e.g. --no-odds or missing key) -> 0,
    known with certainty, not None/unavailable."""
    session = OddsRunSession(api_key=None, base_url="https://api.the-odds-api.com/v4")
    assert session.credits_used_this_run == 0


def test_credits_used_this_run_and_cumulative_are_distinct_values(monkeypatch):
    """Sanity check that the two numbers are never silently the same
    computation wearing two labels -- they diverge exactly when the
    bug's scenario occurs (0 delta but nonzero cumulative)."""
    responses = [_FakeResp([], {"x-requests-remaining": "493", "x-requests-used": "10"})]
    it = iter(responses)
    monkeypatch.setattr(odds_mod.http_client, "get_json", lambda *a, **kw: next(it))

    session = OddsRunSession(api_key="fake-key", base_url="https://api.the-odds-api.com/v4")
    session.get_events()

    assert session.credits_used_this_run != session.credits_used_cumulative_account
    assert session.credits_used_this_run == 0
    assert session.credits_used_cumulative_account == 10
