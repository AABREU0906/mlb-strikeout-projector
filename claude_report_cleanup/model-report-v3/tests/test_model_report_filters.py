"""
Tests for app/evaluation/model_report_filters.py -- pure functions, no
database or pydantic required.
"""
from app.evaluation.model_report_filters import (
    UNRECORDED_RECOMMENDATION,
    classify_recommendation,
    deduplicate_projections,
    is_invalid_projection,
)


def _row(**overrides):
    base = dict(
        projection_id="p1", game_id="g1", pitcher_id=1,
        validation_status="valid", expected_innings=5.8, expected_batters_faced=24.0,
        expected_pitch_count=95.0, final_blended_projection=6.0,
        lineup_status="confirmed", created_at_utc="2026-07-01T10:00:00",
        game_start_utc="2026-07-01T19:00:00",
    )
    base.update(overrides)
    return base


def test_broken_barnett_style_projection_excluded():
    row = _row(validation_status=None, expected_innings=27.9, expected_batters_faced=126,
               expected_pitch_count=531, final_blended_projection=10.03)
    invalid, reason = is_invalid_projection(row)
    assert invalid is True
    assert "expected_innings" in reason


def test_valid_barnett_style_projection_retained():
    row = _row(validation_status="valid", expected_innings=5.0, expected_batters_faced=22,
               expected_pitch_count=88, final_blended_projection=6.5)
    invalid, _ = is_invalid_projection(row)
    assert invalid is False


def test_explicit_invalid_status_always_excluded():
    row = _row(validation_status="invalid", expected_innings=5.0, expected_batters_faced=22)
    invalid, reason = is_invalid_projection(row)
    assert invalid is True
    assert "validation_status" in reason


def test_explicit_valid_status_trusted_even_with_borderline_values():
    row = _row(validation_status="valid", expected_innings=8.9)
    invalid, _ = is_invalid_projection(row)
    assert invalid is False


def test_legacy_row_negative_innings_excluded():
    row = _row(validation_status=None, expected_innings=-1.0)
    invalid, reason = is_invalid_projection(row)
    assert invalid is True
    assert "expected_innings" in reason


def test_legacy_row_negative_projection_excluded():
    row = _row(validation_status=None, final_blended_projection=-2.0)
    invalid, reason = is_invalid_projection(row)
    assert invalid is True


def test_legacy_row_excessive_batters_faced_excluded():
    row = _row(validation_status=None, expected_batters_faced=126)
    invalid, _ = is_invalid_projection(row)
    assert invalid is True


def test_legacy_row_excessive_pitch_count_excluded():
    row = _row(validation_status=None, expected_pitch_count=531)
    invalid, _ = is_invalid_projection(row)
    assert invalid is True


def test_legacy_row_plausible_values_retained():
    row = _row(validation_status=None, expected_innings=5.8, expected_batters_faced=24,
               expected_pitch_count=95, final_blended_projection=6.0)
    invalid, _ = is_invalid_projection(row)
    assert invalid is False


def test_legacy_row_missing_fields_not_falsely_flagged():
    row = _row(validation_status=None, expected_innings=None, expected_batters_faced=None,
               expected_pitch_count=None, final_blended_projection=None)
    invalid, _ = is_invalid_projection(row)
    assert invalid is False


def test_duplicate_projections_collapse_correctly():
    rows = [
        _row(projection_id="m1", created_at_utc="2026-07-01T10:00:00", lineup_status="projected"),
        _row(projection_id="m2", created_at_utc="2026-07-01T15:00:00", lineup_status="confirmed"),
        _row(projection_id="m3", created_at_utc="2026-07-01T12:00:00", lineup_status="projected"),
    ]
    canonical, excluded = deduplicate_projections(rows)
    assert len(canonical) == 1
    assert len(excluded) == 2


def test_confirmed_lineup_preferred_in_dedup():
    rows = [
        _row(projection_id="a1", lineup_status="projected", created_at_utc="2026-07-01T16:00:00"),
        _row(projection_id="a2", lineup_status="confirmed", created_at_utc="2026-07-01T10:00:00"),
    ]
    canonical, _ = deduplicate_projections(rows)
    assert canonical[0]["projection_id"] == "a2"


def test_latest_pregame_projection_selected_when_lineup_status_ties():
    rows = [
        _row(projection_id="b1", lineup_status="confirmed", created_at_utc="2026-07-01T10:00:00"),
        _row(projection_id="b2", lineup_status="confirmed", created_at_utc="2026-07-01T17:00:00"),
    ]
    canonical, _ = deduplicate_projections(rows)
    assert canonical[0]["projection_id"] == "b2"


def test_postgame_rerun_excluded_in_favor_of_pregame_row():
    rows = [
        _row(projection_id="c1", created_at_utc="2026-07-01T15:00:00", game_start_utc="2026-07-01T19:00:00"),
        _row(projection_id="c2", created_at_utc="2026-07-01T22:00:00", game_start_utc="2026-07-01T19:00:00"),
    ]
    canonical, excluded = deduplicate_projections(rows)
    assert canonical[0]["projection_id"] == "c1"
    assert excluded[0]["projection_id"] == "c2"


def test_valid_preferred_over_invalid_regardless_of_other_factors():
    rows = [
        _row(projection_id="d1", validation_status="invalid", lineup_status="confirmed",
             created_at_utc="2026-07-01T18:00:00"),
        _row(projection_id="d2", validation_status="valid", lineup_status="projected",
             created_at_utc="2026-07-01T09:00:00"),
    ]
    canonical, _ = deduplicate_projections(rows)
    assert canonical[0]["projection_id"] == "d2"


def test_single_row_no_dedup_needed():
    rows = [_row(projection_id="e1")]
    canonical, excluded = deduplicate_projections(rows)
    assert len(canonical) == 1
    assert len(excluded) == 0


def test_different_pitchers_same_game_not_deduplicated():
    rows = [
        _row(projection_id="f1", pitcher_id=100),
        _row(projection_id="f2", pitcher_id=200),
    ]
    canonical, excluded = deduplicate_projections(rows)
    assert len(canonical) == 2
    assert len(excluded) == 0


def test_different_games_same_pitcher_not_deduplicated():
    rows = [
        _row(projection_id="g1a", game_id="game-1"),
        _row(projection_id="g1b", game_id="game-2"),
    ]
    canonical, excluded = deduplicate_projections(rows)
    assert len(canonical) == 2


def test_dedup_never_loses_a_row_between_canonical_and_excluded():
    rows = [_row(projection_id=f"p{i}", created_at_utc=f"2026-07-01T{10+i}:00:00") for i in range(5)]
    canonical, excluded = deduplicate_projections(rows)
    all_ids = {r["projection_id"] for r in canonical} | {r["projection_id"] for r in excluded}
    assert all_ids == {f"p{i}" for i in range(5)}
    assert len(canonical) == 1
    assert len(excluded) == 4


def test_null_recommendation_becomes_unknown():
    assert classify_recommendation(None) == UNRECORDED_RECOMMENDATION
    assert UNRECORDED_RECOMMENDATION == "UNKNOWN"


def test_actual_pass_remains_pass():
    assert classify_recommendation("PASS") == "PASS"


def test_over_and_under_classify_normally():
    assert classify_recommendation("OVER") == "OVER"
    assert classify_recommendation("UNDER") == "UNDER"


def test_unexpected_value_classifies_as_unknown_not_crash():
    assert classify_recommendation("garbage_value") == "UNKNOWN"
    assert classify_recommendation("") == "UNKNOWN"
