from app.schemas.player import PitcherProfile, SampleStat
from app.projections.stage1_workload import estimate_workload


def _pitcher(**overrides):
    base = dict(
        player_id=1, name="Test Pitcher", throws="R",
        avg_innings_per_start=6.0, avg_bf_per_start=24.0, avg_pitches_per_start=95.0,
        bb_rate_season=SampleStat(shrunk_rate=0.08),
    )
    base.update(overrides)
    return PitcherProfile(**base)


def test_normal_workload_uses_season_averages():
    est = estimate_workload(_pitcher(), opponent_team=None)
    assert 5.0 <= est.expected_innings <= 6.5


def test_opener_caps_workload():
    est = estimate_workload(_pitcher(), opponent_team=None, is_opener=True)
    assert est.expected_innings <= 2.0
    assert est.workload_confidence_penalty > 0.2


def test_tandem_risk_caps_workload():
    est = estimate_workload(_pitcher(), opponent_team=None, is_tandem_risk=True)
    assert est.expected_innings <= 4.0


def test_announced_pitch_limit_caps_workload():
    est = estimate_workload(_pitcher(), opponent_team=None, announced_pitch_limit=50)
    assert est.expected_pitch_count <= 50


def test_short_rest_reduces_workload():
    normal = estimate_workload(_pitcher(), opponent_team=None)
    short = estimate_workload(_pitcher(), opponent_team=None, short_rest=True)
    assert short.expected_innings < normal.expected_innings


def test_rehab_assignment_caps_workload():
    est = estimate_workload(_pitcher(), opponent_team=None, recent_rehab_assignment=True)
    assert est.expected_innings <= 4.5


def test_probabilities_between_zero_and_one():
    est = estimate_workload(_pitcher(), opponent_team=None)
    for p in (est.prob_complete_5, est.prob_complete_6, est.prob_complete_7, est.prob_early_exit):
        assert 0.0 <= p <= 1.0
