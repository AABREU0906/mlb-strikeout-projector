import pytest

from app.features.nrfi_rate_calculations import (
    cascading_shrinkage,
    compute_slash_line,
    last_n,
    rate_of,
    split_day_night,
    split_home_away,
    split_season,
    to_shrunk_rate,
)


@pytest.fixture
def sample_records():
    return [
        {"game_date": "2026-04-01", "season": 2026, "is_home": True, "day_night": "night",
         "scoreless": True, "game_is_nrfi": True, "runs_allowed": 0, "hits_allowed": 0,
         "walks_allowed": 0, "strikeouts": 2, "home_runs_allowed": 0, "at_bats_faced": 3,
         "total_bases_allowed": 0, "plate_appearances_faced": 3, "pitches_thrown": 12},
        {"game_date": "2026-04-08", "season": 2026, "is_home": False, "day_night": "day",
         "scoreless": False, "game_is_nrfi": False, "runs_allowed": 2, "hits_allowed": 2,
         "walks_allowed": 1, "strikeouts": 0, "home_runs_allowed": 1, "at_bats_faced": 4,
         "total_bases_allowed": 5, "plate_appearances_faced": 5, "pitches_thrown": 22},
        {"game_date": "2026-04-15", "season": 2026, "is_home": True, "day_night": "night",
         "scoreless": True, "game_is_nrfi": True, "runs_allowed": 0, "hits_allowed": 1,
         "walks_allowed": 0, "strikeouts": 1, "home_runs_allowed": 0, "at_bats_faced": 3,
         "total_bases_allowed": 1, "plate_appearances_faced": 3, "pitches_thrown": 14},
        {"game_date": "2025-09-01", "season": 2025, "is_home": True, "day_night": "night",
         "scoreless": True, "game_is_nrfi": True, "runs_allowed": 0, "hits_allowed": 0,
         "walks_allowed": 0, "strikeouts": 2, "home_runs_allowed": 0, "at_bats_faced": 3,
         "total_bases_allowed": 0, "plate_appearances_faced": 3, "pitches_thrown": 11},
    ]


def test_rate_of_basic(sample_records):
    r = rate_of(sample_records, "scoreless")
    assert r.made_rate == 0.75
    assert r.n == 4


def test_rate_of_excludes_none_from_denominator():
    records = [{"scoreless": True}, {"scoreless": None}, {"scoreless": False}]
    r = rate_of(records, "scoreless")
    assert r.n == 2


def test_rate_of_empty_returns_none():
    r = rate_of([], "scoreless")
    assert r.made_rate is None
    assert r.n == 0


def test_last_n_orders_by_date_descending(sample_records):
    result = last_n(sample_records, 2)
    assert [r["game_date"] for r in result] == ["2026-04-15", "2026-04-08"]


def test_split_home_away(sample_records):
    home, away = split_home_away(sample_records)
    assert len(home) == 3
    assert len(away) == 1


def test_split_day_night(sample_records):
    day, night = split_day_night(sample_records)
    assert len(day) == 1
    assert len(night) == 3


def test_split_season(sample_records):
    this_season, other = split_season(sample_records, 2026)
    assert len(this_season) == 3
    assert len(other) == 1


def test_slash_line_era_and_avg(sample_records):
    sl = compute_slash_line(
        sample_records, "runs_allowed", "hits_allowed", "walks_allowed", "strikeouts",
        "home_runs_allowed", "at_bats_faced", "total_bases_allowed", "plate_appearances_faced", "pitches_thrown",
    )
    assert sl.n_starts_with_data == 4
    assert sl.era == pytest.approx(4.5, abs=1e-6)
    assert sl.whip == pytest.approx(1.0, abs=1e-6)


def test_slash_line_missing_data_stays_none_not_zero():
    partial = [{"plate_appearances_faced": 3, "runs_allowed": None, "hits_allowed": 1, "walks_allowed": 0,
                "strikeouts": 1, "home_runs_allowed": 0, "at_bats_faced": 3, "total_bases_allowed": 1,
                "pitches_thrown": 10}]
    sl = compute_slash_line(
        partial, "runs_allowed", "hits_allowed", "walks_allowed", "strikeouts",
        "home_runs_allowed", "at_bats_faced", "total_bases_allowed", "plate_appearances_faced", "pitches_thrown",
    )
    assert sl.era is None
    assert sl.avg is not None


def test_slash_line_empty_input():
    sl = compute_slash_line([], "runs_allowed", "hits_allowed", "walks_allowed", "strikeouts",
                             "home_runs_allowed", "at_bats_faced", "total_bases_allowed",
                             "plate_appearances_faced", "pitches_thrown")
    assert sl.n_starts_with_data == 0
    assert sl.era is None


def test_to_shrunk_rate_small_sample_pulls_to_prior():
    r = rate_of([{"scoreless": True}] * 3 + [{"scoreless": False}], "scoreless")
    shrunk = to_shrunk_rate(r, prior=0.715, stabilization_n=20)
    assert shrunk.reliability < 0.3
    assert abs(shrunk.shrunk_rate - 0.715) < abs(0.75 - 0.715)


def test_to_shrunk_rate_zero_observations_returns_pure_prior():
    r = rate_of([], "scoreless")
    shrunk = to_shrunk_rate(r, prior=0.715, stabilization_n=20)
    assert shrunk.shrunk_rate == 0.715


def test_cascading_shrinkage_tiny_bvp_does_not_dominate():
    final_rate, detail = cascading_shrinkage(
        league_rate=0.320,
        season_rate=0.360, season_n=600, season_stabilization_n=200,
        vs_hand_rate=0.350, vs_hand_n=300, vs_hand_stabilization_n=200,
        bvp_rate=0.667, bvp_n=3, bvp_stabilization_n=60,
    )
    assert abs(final_rate - detail["hand_adjusted_prior_after_shrinkage"]) < 0.02


def test_cascading_shrinkage_large_bvp_sample_moves_estimate():
    final_rate, detail = cascading_shrinkage(
        league_rate=0.320,
        season_rate=0.360, season_n=600, season_stabilization_n=200,
        vs_hand_rate=0.350, vs_hand_n=300, vs_hand_stabilization_n=200,
        bvp_rate=0.150, bvp_n=50, bvp_stabilization_n=60,
    )
    assert final_rate < detail["hand_adjusted_prior_after_shrinkage"] - 0.03


def test_cascading_shrinkage_no_data_anywhere_falls_back_to_league():
    final_rate, _ = cascading_shrinkage(
        league_rate=0.320,
        season_rate=None, season_n=None, season_stabilization_n=200,
        vs_hand_rate=None, vs_hand_n=None, vs_hand_stabilization_n=200,
        bvp_rate=None, bvp_n=None, bvp_stabilization_n=60,
    )
    assert final_rate == 0.320


def test_cascading_shrinkage_missing_bvp_only_falls_back_to_hand_adjusted():
    final_rate, detail = cascading_shrinkage(
        league_rate=0.320,
        season_rate=0.360, season_n=600, season_stabilization_n=200,
        vs_hand_rate=0.350, vs_hand_n=300, vs_hand_stabilization_n=200,
        bvp_rate=None, bvp_n=None, bvp_stabilization_n=60,
    )
    assert final_rate == detail["hand_adjusted_prior_after_shrinkage"]
