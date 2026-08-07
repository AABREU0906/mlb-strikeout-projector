"""
End-to-end tests for the workload-fallback -> confidence -> recommendation
propagation bug and its fix.

Per the audit requirement, these tests begin at PitcherFeatureBuilder (not
just stage1_workload.py in isolation), using a fake provider that returns a
realistic MLB Stats API payload shaped so that PitcherFeatureBuilder's own
guardrail rejects the per-start averages and nulls them out -- exactly
reproducing the Mason Barnett scenario, where workload fallback originates
upstream of the workload model itself.
"""
import types

import pytest

from app.features.pitcher_features import PitcherFeatureBuilder
from app.markets.edge_analysis import analyze_betting_edge, determine_confidence, determine_edge_grade
from app.projections.stage1_workload import estimate_workload


class _FakeSourcedPayload:
    def __init__(self, data):
        self.data = data


class _FakeProvider:
    def __init__(self, stats_payload):
        self._stats_payload = stats_payload
        self.base = "https://statsapi.mlb.com/api/v1"

    def get_pitcher_stats(self, pitcher_id, season):
        return _FakeSourcedPayload(self._stats_payload)


def _mason_barnett_like_payload():
    """A realistic MLB Stats API 'person' + 'stats' payload where the
    'season'/'pitching' block's gamesStarted is present but tiny relative
    to a much larger innings/battersFaced/pitches total -- exactly the
    shape that trips PitcherFeatureBuilder's own plausibility guardrail,
    causing it to null out the avg_*_per_start fields BEFORE
    stage1_workload.py ever sees them."""
    return {
        "person": {
            "id": 686930,
            "fullName": "Mason Barnett",
            "pitchHand": {"code": "R"},
        },
        "stats": [
            {
                "type": {"displayName": "season"},
                "group": {"displayName": "pitching"},
                "splits": [
                    {
                        "season": "2026",
                        "sport": {"id": 1},
                        "gameType": "R",
                        "team": {"name": "Athletics"},
                        "stat": {
                            "gamesStarted": 1,
                            "inningsPitched": "48.0",
                            "battersFaced": 210,
                            "numberOfPitches": 780,
                            "strikeOuts": 45,
                            "baseOnBalls": 18,
                        },
                    }
                ],
            },
            {
                "type": {"displayName": "career"},
                "group": {"displayName": "pitching"},
                "splits": [
                    {
                        "season": "2026",
                        "sport": {"id": 1},
                        "gameType": "R",
                        "stat": {
                            "gamesStarted": 1,
                            "inningsPitched": "48.0",
                            "battersFaced": 210,
                            "strikeOuts": 45,
                        },
                    }
                ],
            },
        ],
    }


@pytest.fixture
def mason_barnett_profile(monkeypatch):
    """Builds a real PitcherProfile via PitcherFeatureBuilder.build(),
    using a fake provider for the primary stats call and short-circuiting
    the handedness-splits HTTP call (an already-handled 'unavailable'
    path, not a fabrication) so the test needs no real network access."""
    import app.features.pitcher_features as pf_module

    monkeypatch.setattr(pf_module.http_client, "get_json", lambda *a, **k: None)

    builder = PitcherFeatureBuilder(provider=_FakeProvider(_mason_barnett_like_payload()))
    return builder.build(pitcher_id=686930, season=2026)


def test_pitcher_feature_builder_rejects_implausible_workload(mason_barnett_profile):
    assert mason_barnett_profile.avg_innings_per_start is None
    assert mason_barnett_profile.avg_bf_per_start is None
    assert mason_barnett_profile.avg_pitches_per_start is None


def test_workload_estimate_detects_all_three_fallback(mason_barnett_profile):
    workload = estimate_workload(pitcher=mason_barnett_profile, opponent_team=None)

    assert workload.workload_fallback_used is True
    assert workload.workload_all_metrics_fallback is True
    assert workload.workload_fallback_count == 3
    assert workload.workload_data_valid is False
    assert workload.workload_confidence_penalty > 0.4

    assert 0.5 <= workload.expected_innings <= 9.0
    assert 3.0 <= workload.expected_batters_faced <= 45.0


def test_confidence_capped_at_medium_when_all_metrics_fallback():
    selected = types.SimpleNamespace(probability_edge_vs_price=0.15)
    confidence = determine_confidence(
        selected=selected,
        lineup_confirmed=True,
        pitcher_confirmed=True,
        workload_warning=True,
        workload_all_metrics_fallback=True,
        injury_warning=False,
        weather_warning=False,
        stale_data=False,
        model_sample_size=1000,
    )
    assert confidence != "HIGH"
    assert confidence in ("MEDIUM", "LOW", "AVOID")


def test_confidence_can_be_high_without_workload_fallback():
    selected = types.SimpleNamespace(probability_edge_vs_price=0.15)
    confidence = determine_confidence(
        selected=selected,
        lineup_confirmed=True,
        pitcher_confirmed=True,
        workload_warning=False,
        workload_all_metrics_fallback=False,
        model_sample_size=1000,
    )
    assert confidence == "HIGH"


def test_elite_grade_suppressed_when_all_metrics_fallback():
    analysis = types.SimpleNamespace(expected_value=0.15, probability_edge_vs_price=0.08)
    grade, stars = determine_edge_grade(analysis, workload_all_metrics_fallback=True)
    assert grade != "Elite estimated edge"
    assert stars < 5


def test_strong_grade_suppressed_when_all_metrics_fallback():
    analysis = types.SimpleNamespace(expected_value=0.07, probability_edge_vs_price=0.045)
    grade, stars = determine_edge_grade(analysis, workload_all_metrics_fallback=True)
    assert grade != "Strong estimated edge"


def test_elite_grade_still_possible_without_workload_fallback():
    analysis = types.SimpleNamespace(expected_value=0.15, probability_edge_vs_price=0.08)
    grade, stars = determine_edge_grade(analysis, workload_all_metrics_fallback=False)
    assert grade == "Elite estimated edge"
    assert stars == 5


def test_full_edge_analysis_end_to_end_mason_barnett_scenario(mason_barnett_profile):
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
    assert analysis.grade not in ("Elite estimated edge", "Strong estimated edge")


def test_pass_preferred_when_workload_fallback_drives_confidence_to_avoid():
    """Uses a large enough edge to initially QUALIFY (selected is not
    None) so this genuinely exercises the new override branch, rather
    than a pre-existing 'edge too small to qualify' PASS path."""
    analysis = analyze_betting_edge(
        over_odds=-110,
        under_odds=-110,
        model_over_probability=0.60,  # clearly qualifying edge on its own
        lineup_confirmed=False,
        pitcher_confirmed=False,
        workload_warning=True,
        workload_all_metrics_fallback=True,
        injury_warning=True,
        stale_data=True,
        model_sample_size=20,
    )
    assert analysis.confidence == "AVOID"
    assert analysis.recommended_side == "PASS"
    assert analysis.selected is None


def test_recommendation_not_forced_to_pass_when_fallback_absent():
    analysis = analyze_betting_edge(
        over_odds=+250,
        under_odds=-350,
        model_over_probability=0.55,
        lineup_confirmed=True,
        pitcher_confirmed=True,
        workload_warning=False,
        workload_all_metrics_fallback=False,
    )
    assert analysis.recommended_side in ("OVER", "UNDER")
    assert analysis.selected is not None
