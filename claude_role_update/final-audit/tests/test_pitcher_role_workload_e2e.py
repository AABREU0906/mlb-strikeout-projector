"""
End-to-end test: PitcherFeatureBuilder (with a fake provider returning a
realistic Mason-Barnett-shaped payload, including a mixed relief/starting
game log) through role-aware workload resolution, through
stage1_workload.py, through betting-edge confidence -- confirming the
fix holds across the entire real code path, not just the pure functions.
"""
import pytest

from app.features.pitcher_features import PitcherFeatureBuilder
from app.markets.edge_analysis import analyze_betting_edge
from app.projections.stage1_workload import estimate_workload


class _FakeSourcedPayload:
    def __init__(self, data):
        self.data = data


class _FakeProvider:
    """Always returns the current-season payload regardless of which
    season is requested -- sufficient for scenarios where tier A (recent
    MLB starts) resolves the workload and the lazy previous-season fetch
    never fires. A dedicated test below exercises the lazy-fetch path
    with a provider that actually distinguishes seasons."""

    def __init__(self, payload):
        self._payload = payload
        self.base = "https://statsapi.mlb.com/api/v1"

    def get_pitcher_stats(self, pitcher_id, season):
        return _FakeSourcedPayload(self._payload)


class _SeasonAwareFakeProvider:
    """Distinguishes current vs. previous season by the actual season
    argument, for testing the lazy previous-season fetch path."""

    def __init__(self, current_season, current_payload, previous_payload):
        self._current_season = current_season
        self._current_payload = current_payload
        self._previous_payload = previous_payload
        self.base = "https://statsapi.mlb.com/api/v1"

    def get_pitcher_stats(self, pitcher_id, season):
        if season == self._current_season:
            return _FakeSourcedPayload(self._current_payload)
        return _FakeSourcedPayload(self._previous_payload)


def _mason_barnett_payload():
    """Realistic season block (gamesStarted=1, gamesPitched=15, season
    totals dominated by relief work) PLUS a game log containing both
    relief appearances and his one real start -- exactly the shape that
    exposed the original bug."""
    gamelog_splits = [
        {"date": "2026-04-05", "stat": {"gamesStarted": 0, "inningsPitched": "1.0", "battersFaced": 4, "numberOfPitches": 15, "strikeOuts": 1, "baseOnBalls": 0}},
        {"date": "2026-04-10", "stat": {"gamesStarted": 0, "inningsPitched": "1.1", "battersFaced": 5, "numberOfPitches": 18, "strikeOuts": 2, "baseOnBalls": 1}},
        {"date": "2026-07-20", "stat": {"gamesStarted": 1, "inningsPitched": "5.0", "battersFaced": 22, "numberOfPitches": 88, "strikeOuts": 6, "baseOnBalls": 2}},
        {"date": "2026-04-15", "stat": {"gamesStarted": 0, "inningsPitched": "2.0", "battersFaced": 8, "numberOfPitches": 32, "strikeOuts": 1, "baseOnBalls": 1}},
    ]
    return {
        "person": {"id": 686930, "fullName": "Mason Barnett", "pitchHand": {"code": "R"}},
        "stats": [
            {
                "type": {"displayName": "season"},
                "group": {"displayName": "pitching"},
                "splits": [{
                    "season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Athletics"},
                    "stat": {
                        "gamesStarted": 1, "gamesPitched": 15,
                        "inningsPitched": "48.0", "battersFaced": 210, "numberOfPitches": 780,
                        "strikeOuts": 45, "baseOnBalls": 18,
                    },
                }],
            },
            {
                "type": {"displayName": "career"},
                "group": {"displayName": "pitching"},
                "splits": [{"season": "2026", "sport": {"id": 1}, "gameType": "R",
                            "stat": {"gamesStarted": 1, "gamesPitched": 15, "inningsPitched": "48.0",
                                     "battersFaced": 210, "strikeOuts": 45}}],
            },
            {
                "type": {"displayName": "gameLog"},
                "group": {"displayName": "pitching"},
                "splits": gamelog_splits,
            },
        ],
    }


def _empty_pitching_payload():
    return {
        "person": {"id": 999999, "fullName": "No Data Pitcher", "pitchHand": {"code": "R"}},
        "stats": [
            {"type": {"displayName": "season"}, "group": {"displayName": "pitching"}, "splits": []},
            {"type": {"displayName": "gameLog"}, "group": {"displayName": "pitching"}, "splits": []},
        ],
    }


@pytest.fixture
def mason_barnett_profile(monkeypatch):
    import app.features.pitcher_features as pf_module

    monkeypatch.setattr(pf_module.http_client, "get_json", lambda *a, **k: None)

    builder = PitcherFeatureBuilder(provider=_FakeProvider(_mason_barnett_payload()))
    return builder.build(pitcher_id=686930, season=2026)


def test_role_correctly_classified_as_reliever_or_swingman(mason_barnett_profile):
    assert mason_barnett_profile.workload_role in ("reliever", "swingman")
    assert mason_barnett_profile.games_started == 1
    assert mason_barnett_profile.games_pitched == 15


def test_workload_uses_real_start_not_season_totals(mason_barnett_profile):
    """THE CORE FIX, verified through the real PitcherFeatureBuilder code
    path (game log fetch + parsing included), not just the pure function."""
    assert mason_barnett_profile.avg_innings_per_start is not None
    assert mason_barnett_profile.avg_innings_per_start != 48.0
    assert mason_barnett_profile.avg_innings_per_start == pytest.approx(5.0, abs=0.5)
    assert mason_barnett_profile.workload_source == "mlb_recent_starts"
    assert mason_barnett_profile.workload_source_level == "MLB"
    assert mason_barnett_profile.workload_fallback_used is True


def test_workload_estimate_reflects_thin_sample(mason_barnett_profile):
    workload = estimate_workload(pitcher=mason_barnett_profile, opponent_team=None)
    assert 0.5 <= workload.expected_innings <= 9.0
    assert workload.start_specific_sample_size == 1
    assert workload.workload_source == "mlb_recent_starts"


def test_full_chain_confidence_and_recommendation(mason_barnett_profile):
    workload = estimate_workload(pitcher=mason_barnett_profile, opponent_team=None)

    analysis = analyze_betting_edge(
        over_odds=+250,
        under_odds=-350,
        model_over_probability=0.55,
        lineup_confirmed=True,
        pitcher_confirmed=True,
        workload_warning=workload.workload_fallback_used,
        workload_all_metrics_fallback=workload.workload_all_metrics_fallback,
    )
    assert analysis.confidence != "HIGH"


def test_lazy_previous_season_fetch_when_no_current_data(monkeypatch):
    """A pitcher with zero usable current-season starts (e.g. recalled
    mid-season with no MLB starts yet this year) should trigger the lazy
    previous-season fetch and correctly resolve via tier C."""
    import app.features.pitcher_features as pf_module

    monkeypatch.setattr(pf_module.http_client, "get_json", lambda *a, **k: None)

    current_payload = {
        "person": {"id": 555555, "fullName": "Recalled Pitcher", "pitchHand": {"code": "L"}},
        "stats": [
            {"type": {"displayName": "season"}, "group": {"displayName": "pitching"},
             "splits": [{"season": "2026", "sport": {"id": 1}, "gameType": "R", "team": {"name": "Cubs"},
                         "stat": {"gamesStarted": 0, "gamesPitched": 3, "inningsPitched": "3.0",
                                  "battersFaced": 14, "numberOfPitches": 55, "strikeOuts": 3, "baseOnBalls": 1}}]},
            {"type": {"displayName": "gameLog"}, "group": {"displayName": "pitching"}, "splits": []},
        ],
    }
    previous_payload = {
        "person": {"id": 555555, "fullName": "Recalled Pitcher"},
        "stats": [
            {"type": {"displayName": "gameLog"}, "group": {"displayName": "pitching"}, "splits": [
                {"date": "2025-08-15", "stat": {"gamesStarted": 1, "inningsPitched": "5.1", "battersFaced": 23, "numberOfPitches": 89, "strikeOuts": 5, "baseOnBalls": 2}},
            ]},
        ],
    }
    provider = _SeasonAwareFakeProvider(2026, current_payload, previous_payload)
    profile = PitcherFeatureBuilder(provider=provider).build(pitcher_id=555555, season=2026)

    assert profile.workload_source == "mlb_previous_season_starts"
    assert profile.avg_innings_per_start == pytest.approx(5 + 1 / 3, abs=0.01)


def test_no_data_anywhere_falls_through_to_unresolved(monkeypatch):
    """A pitcher with no usable data at all (current season, previous
    season, or season totals) must resolve to 'unresolved', not crash and
    not fabricate a plausible-looking number."""
    import app.features.pitcher_features as pf_module

    monkeypatch.setattr(pf_module.http_client, "get_json", lambda *a, **k: None)

    provider = _SeasonAwareFakeProvider(2026, _empty_pitching_payload(), _empty_pitching_payload())
    profile = PitcherFeatureBuilder(provider=provider).build(pitcher_id=999999, season=2026)

    assert profile.avg_innings_per_start is None
    assert profile.workload_source == "unresolved"
    assert profile.workload_data_valid is False

    workload = estimate_workload(pitcher=profile, opponent_team=None)
    # Falls through to stage1_workload.py's existing league-average
    # fallback (tier F) -- still produces a sane, bounded estimate.
    assert 0.1 <= workload.expected_innings <= 9.0
    assert workload.workload_all_metrics_fallback is True
