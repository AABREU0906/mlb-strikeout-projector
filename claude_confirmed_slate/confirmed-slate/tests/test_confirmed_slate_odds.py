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
