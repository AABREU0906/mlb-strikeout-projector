"""
Comprehensive tests for the role-aware workload fix
(app/features/pitcher_role_workload.py), covering every scenario listed in
the fix request. Pure-function tests -- no network, no pydantic required
for this file specifically (see test_pitcher_role_workload_e2e.py for the
PitcherFeatureBuilder-based end-to-end fixture, which does need pydantic).
"""
import pytest

from app.features.pitcher_role_workload import (
    MIN_STARTS_FOR_RECENT_WINDOW,
    START_RATIO_MISMATCH_THRESHOLD,
    StartAppearance,
    assess_workload_role,
    compute_start_specific_averages,
    extract_start_appearances,
    resolve_workload,
)


def _start(date, ip, bf, pitches, k=0, bb=0):
    return StartAppearance(date=date, innings_pitched=ip, batters_faced=bf, pitches=pitches, strikeouts=k, walks=bb)


def test_one_start_fourteen_relief_appearances():
    role = assess_workload_role(games_started=1, games_pitched=15)
    assert role.role_mismatch is True
    assert role.workload_role in ("reliever", "swingman")
    assert role.start_ratio == pytest.approx(1 / 15)


def test_mason_barnett_end_to_end_resolution():
    role = assess_workload_role(games_started=1, games_pitched=15)
    starts = [_start("2026-07-20", 5.0, 22, 88, k=6, bb=2)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=starts, previous_season_starts=None,
        season_total_avg_ip=27.9, season_total_avg_bf=126.0, season_total_avg_pitches=531.0,
    )
    assert res.avg_innings_per_start == 5.0
    assert res.avg_innings_per_start != 27.9
    assert res.workload_source == "mlb_recent_starts"
    assert res.workload_source_level == "MLB"
    assert res.workload_fallback_used is True


def test_zero_starts_many_relief_appearances():
    role = assess_workload_role(games_started=0, games_pitched=40)
    assert role.role_mismatch is True
    assert role.workload_role == "reliever"

    res = resolve_workload(
        role_assessment=role, current_season_starts=[], previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "unresolved"
    assert res.workload_source_level == "unavailable"
    assert res.workload_data_valid is False


def test_mostly_starter_profile_uses_season_totals():
    role = assess_workload_role(games_started=28, games_pitched=29)
    assert role.role_mismatch is False
    assert role.workload_role == "starter"

    res = resolve_workload(
        role_assessment=role, current_season_starts=[], previous_season_starts=None,
        season_total_avg_ip=5.8, season_total_avg_bf=24.0, season_total_avg_pitches=95.0,
    )
    assert res.avg_innings_per_start == 5.8
    assert res.workload_source == "mlb_season_totals"
    assert res.workload_fallback_used is False


def test_reliever_converted_to_starter_uses_recent_starts():
    role = assess_workload_role(games_started=3, games_pitched=20)
    assert role.role_mismatch is True

    starts = [
        _start("2026-07-25", 6.0, 24, 92),
        _start("2026-07-19", 5.1, 23, 89),
        _start("2026-07-13", 5.2, 22, 90),
    ]
    res = resolve_workload(
        role_assessment=role, current_season_starts=starts, previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "mlb_recent_starts"
    assert res.start_specific_sample_size == 3
    assert res.avg_innings_per_start is not None


def test_starter_role_from_season_data_not_disrupted_by_fix():
    role = assess_workload_role(games_started=20, games_pitched=22)
    assert role.role_mismatch is False
    assert role.workload_role == "starter"


def test_first_mlb_start():
    role = assess_workload_role(games_started=1, games_pitched=1)
    starts = [_start("2026-08-01", 4.2, 19, 78)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=starts, previous_season_starts=None,
        season_total_avg_ip=4.67, season_total_avg_bf=19.0, season_total_avg_pitches=78.0,
    )
    assert res.avg_innings_per_start is not None


def test_one_mlb_start_no_milb_integration_does_not_crash():
    role = assess_workload_role(games_started=1, games_pitched=10)
    starts = [_start("2026-06-01", 4.0, 18, 75)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=starts, previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "mlb_recent_starts"
    assert res.workload_source_level == "MLB"


def test_bulk_reliever_role_mismatch():
    role = assess_workload_role(games_started=2, games_pitched=25)
    assert role.role_mismatch is True
    assert role.start_ratio < START_RATIO_MISMATCH_THRESHOLD


def test_mixed_gamelog_excludes_relief_from_start_averages():
    gamelog = [
        {"date": "2026-04-01", "stat": {"gamesStarted": 1, "inningsPitched": "6.0", "battersFaced": 24, "numberOfPitches": 95}},
        {"date": "2026-04-05", "stat": {"gamesStarted": 0, "inningsPitched": "1.0", "battersFaced": 4, "numberOfPitches": 15}},
        {"date": "2026-04-10", "stat": {"gamesStarted": 1, "inningsPitched": "5.2", "battersFaced": 23, "numberOfPitches": 91}},
        {"date": "2026-04-14", "stat": {"gamesStarted": 0, "inningsPitched": "2.0", "battersFaced": 8, "numberOfPitches": 30}},
        {"date": "2026-04-18", "stat": {"gamesStarted": 0, "inningsPitched": "0.1", "battersFaced": 1, "numberOfPitches": 5}},
    ]
    starts = extract_start_appearances(gamelog)
    assert len(starts) == 2
    assert {s.date for s in starts} == {"2026-04-01", "2026-04-10"}

    avg = compute_start_specific_averages(starts)
    assert avg.avg_innings_per_start == pytest.approx((6.0 + (5 + 2 / 3)) / 2, abs=0.01)
    assert avg.sample_size == 2


def test_fallback_hierarchy_tier_a_recent_starts_preferred():
    role = assess_workload_role(games_started=1, games_pitched=10)
    many_starts = [_start(f"2026-0{i}-01", 5.0, 22, 88) for i in range(1, 7)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=many_starts, previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "mlb_recent_starts"
    assert res.start_specific_sample_size == 5


def test_fallback_hierarchy_tier_c_previous_season():
    role = assess_workload_role(games_started=1, games_pitched=10)
    prev_starts = [_start("2025-09-01", 5.5, 23, 90)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=[], previous_season_starts=prev_starts,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "mlb_previous_season_starts"


def test_fallback_hierarchy_fully_unresolved_reports_milb_honestly():
    role = assess_workload_role(games_started=0, games_pitched=5)
    res = resolve_workload(
        role_assessment=role, current_season_starts=[], previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.workload_source == "unresolved"
    assert any("milb" in r.lower() for r in res.reasons)
    assert res.workload_data_valid is False


def test_thin_sample_flagged_for_confidence_reduction():
    role = assess_workload_role(games_started=1, games_pitched=15)
    one_start = [_start("2026-07-20", 5.0, 22, 88)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=one_start, previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.start_specific_sample_size < MIN_STARTS_FOR_RECENT_WINDOW
    assert any("fewer than" in r for r in res.reasons)


def test_healthy_sample_not_flagged():
    role = assess_workload_role(games_started=1, games_pitched=10)
    five_starts = [_start(f"2026-0{i}-01", 5.5, 23, 90) for i in range(1, 6)]
    res = resolve_workload(
        role_assessment=role, current_season_starts=five_starts, previous_season_starts=None,
        season_total_avg_ip=None, season_total_avg_bf=None, season_total_avg_pitches=None,
    )
    assert res.start_specific_sample_size >= MIN_STARTS_FOR_RECENT_WINDOW
    assert not any("fewer than" in r for r in res.reasons)


def test_missing_games_pitched_treated_as_unsafe():
    role = assess_workload_role(games_started=5, games_pitched=None)
    assert role.role_mismatch is True
    assert role.workload_role == "unknown"


def test_extract_start_appearances_handles_string_flag():
    gamelog = [{"date": "2026-05-01", "stat": {"gamesStarted": "1", "inningsPitched": "6.0", "battersFaced": 24, "numberOfPitches": 95}}]
    starts = extract_start_appearances(gamelog)
    assert len(starts) == 1


def test_extract_start_appearances_handles_missing_flag():
    gamelog = [{"date": "2026-05-01", "stat": {"inningsPitched": "1.0", "battersFaced": 5, "numberOfPitches": 20}}]
    starts = extract_start_appearances(gamelog)
    assert len(starts) == 0


def test_invalid_innings_notation_in_gamelog_excluded_gracefully():
    gamelog = [{"date": "2026-05-01", "stat": {"gamesStarted": 1, "inningsPitched": "6.5", "battersFaced": 24, "numberOfPitches": 95}}]
    starts = extract_start_appearances(gamelog)
    assert len(starts) == 1
    assert starts[0].innings_pitched is None
