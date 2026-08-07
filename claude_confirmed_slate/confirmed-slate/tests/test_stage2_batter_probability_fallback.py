"""
Regression tests for the Cristopher Sanchez crash fix.

Root cause: the previous pitcher_rate/batter_rate fallback logic was a
two-tier ternary that checked `if season_sample:` (object truthiness) but
not `season_sample.shrunk_rate is not None` for that specific tier. A
SampleStat object can exist (be truthy) while its own `.shrunk_rate` is
None (e.g. zero observed events). That None then flowed straight into
log5(), which had no input validation, crashing with:
    TypeError: '>' not supported between instances of 'float' and 'NoneType'

Fixed via a shared, fully-validated 4-tier fallback resolver (split ->
season -> career -> league average) plus defensive validation inside
log5() itself as a second line of defense.
"""
from app.projections.stage2_batter_probability import compute_batter_matchup_probability
from app.schemas.player import BatterProfile, PitcherProfile, SampleStat


def _sample(shrunk_rate=None, small=False):
    return SampleStat(observed_rate=None, observed_n=None, shrunk_rate=shrunk_rate, reliability=0.5, is_small_sample=small)


def _pitcher(**overrides):
    kwargs = dict(
        player_id=1, name="Test Pitcher", throws="R",
        k_rate_vs_rhb=_sample(0.24), k_rate_vs_lhb=_sample(0.26),
        k_rate_season=_sample(0.245), k_rate_career=_sample(0.24),
    )
    kwargs.update(overrides)
    return PitcherProfile(**kwargs)


def _batter(**overrides):
    kwargs = dict(
        player_id=2, name="Test Batter", batting_order=1, bat_side="R", expected_side_today="R",
        k_rate_vs_rhp=_sample(0.21), k_rate_vs_lhp=_sample(0.23),
        k_rate_overall=_sample(0.20), k_rate_career=_sample(0.195),
        k_rate_last_7d=None, k_rate_last_14d=None, k_rate_last_30d=None,
    )
    kwargs.update(overrides)
    return BatterProfile(**kwargs)


def test_missing_pitcher_vs_rhb_split_falls_to_season():
    result = compute_batter_matchup_probability(_pitcher(k_rate_vs_rhb=None), _batter(expected_side_today="R"))
    assert result.pitcher_rate_source == "season"
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_missing_pitcher_vs_lhb_split_falls_to_season():
    result = compute_batter_matchup_probability(_pitcher(k_rate_vs_lhb=None), _batter(expected_side_today="L"))
    assert result.pitcher_rate_source == "season"


def test_missing_batter_vs_rhp_split_falls_to_season():
    result = compute_batter_matchup_probability(_pitcher(), _batter(k_rate_vs_rhp=None, expected_side_today="R"))
    assert result.batter_rate_source == "season"


def test_missing_batter_vs_lhp_split_falls_to_season():
    result = compute_batter_matchup_probability(_pitcher(), _batter(k_rate_vs_lhp=None, expected_side_today="L"))
    assert result.batter_rate_source == "season"


def test_both_splits_missing_does_not_crash():
    result = compute_batter_matchup_probability(
        _pitcher(k_rate_vs_rhb=None), _batter(k_rate_vs_rhp=None, expected_side_today="R")
    )
    assert result.pitcher_rate_source == "season"
    assert result.batter_rate_source == "season"
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_season_fallback_when_split_object_exists_but_rate_is_none():
    result = compute_batter_matchup_probability(
        _pitcher(k_rate_vs_rhb=_sample(None)), _batter(expected_side_today="R")
    )
    assert result.pitcher_rate_source == "season"
    assert result.pitcher_rate_fallback_used


def test_career_fallback_when_split_and_season_both_invalid():
    result = compute_batter_matchup_probability(
        _pitcher(k_rate_vs_rhb=_sample(None), k_rate_season=_sample(None)),
        _batter(expected_side_today="R"),
    )
    assert result.pitcher_rate_source == "career"
    assert result.pitcher_rate_fallback_used


def test_league_average_fallback_when_everything_invalid():
    result = compute_batter_matchup_probability(
        _pitcher(k_rate_vs_rhb=None, k_rate_season=None, k_rate_career=None),
        _batter(k_rate_vs_rhp=None, k_rate_overall=None, k_rate_career=None, expected_side_today="R"),
    )
    assert result.pitcher_rate_source == "league_average"
    assert result.batter_rate_source == "league_average"
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_left_handed_pitcher_projects_normally():
    result = compute_batter_matchup_probability(_pitcher(throws="L"), _batter(expected_side_today="R"))
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_right_handed_pitcher_projects_normally():
    result = compute_batter_matchup_probability(_pitcher(throws="R"), _batter(expected_side_today="L"))
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_switch_hitter_uses_resolved_side():
    result = compute_batter_matchup_probability(
        _pitcher(throws="R"), _batter(bat_side="S", expected_side_today="L")
    )
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_boundary_zero_split_rate_is_used_not_treated_as_missing():
    result = compute_batter_matchup_probability(_pitcher(k_rate_vs_rhb=_sample(0.0)), _batter(expected_side_today="R"))
    assert result.pitcher_rate_source == "split"
    assert not result.pitcher_rate_fallback_used


def test_boundary_one_rate_does_not_crash():
    result = compute_batter_matchup_probability(_pitcher(k_rate_vs_rhb=_sample(1.0)), _batter(expected_side_today="R"))
    assert 0.0 <= result.adjusted_probability <= 1.0


def test_fallback_note_is_visible_when_fallback_used():
    result = compute_batter_matchup_probability(_pitcher(k_rate_vs_rhb=None), _batter(expected_side_today="R"))
    assert len(result.fallback_notes) >= 1
    assert any("season" in note for note in result.fallback_notes)
    assert all(note in result.notes for note in result.fallback_notes)


def test_no_fallback_note_when_all_splits_available():
    result = compute_batter_matchup_probability(_pitcher(), _batter())
    assert result.fallback_notes == []
    assert not result.pitcher_rate_fallback_used
    assert not result.batter_rate_fallback_used


def test_cristopher_sanchez_fixture_reproduces_and_fixes_the_crash():
    """Reproduces the exact reported failure mode: a left-handed pitcher
    whose vs-RHB split SampleStat exists but has shrunk_rate=None, AND
    whose season SampleStat ALSO exists but has shrunk_rate=None. The old
    code's `if pitcher.k_rate_season` truthiness check would have let
    that None through to log5() and crashed with a bare TypeError."""
    cristopher_sanchez = _pitcher(
        name="Cristopher Sanchez",
        throws="L",
        k_rate_vs_rhb=_sample(None),
        k_rate_vs_lhb=_sample(0.29),
        k_rate_season=_sample(None),
        k_rate_career=_sample(0.245),
    )
    opposing_batter = _batter(expected_side_today="R")

    result = compute_batter_matchup_probability(cristopher_sanchez, opposing_batter)

    assert result.pitcher_rate_source == "career"
    assert result.pitcher_rate_fallback_used is True
    assert len(result.fallback_notes) > 0

    assert 0.0 <= result.adjusted_probability <= 1.0
    assert 0.0 <= result.log5_base_probability <= 1.0


def test_log5_never_receives_none_across_full_matrix_of_missing_data():
    pitcher_variants = [
        _pitcher(),
        _pitcher(k_rate_vs_rhb=None),
        _pitcher(k_rate_vs_rhb=_sample(None)),
        _pitcher(k_rate_vs_rhb=None, k_rate_season=None),
        _pitcher(k_rate_vs_rhb=_sample(None), k_rate_season=_sample(None)),
        _pitcher(k_rate_vs_rhb=None, k_rate_season=None, k_rate_career=None),
    ]
    batter_variants = [
        _batter(),
        _batter(k_rate_vs_rhp=None),
        _batter(k_rate_vs_rhp=_sample(None)),
        _batter(k_rate_vs_rhp=None, k_rate_overall=None),
        _batter(k_rate_vs_rhp=None, k_rate_overall=None, k_rate_career=None),
    ]
    for p in pitcher_variants:
        for b in batter_variants:
            result = compute_batter_matchup_probability(p, b)
            assert 0.0 <= result.adjusted_probability <= 1.0
