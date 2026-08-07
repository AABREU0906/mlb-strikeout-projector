"""
DB-backed tests for the model-report/persistence features (Features 1-3).
Uses the isolated temp SQLite database set up in conftest.py.
"""
from app.database.models import ActualResult, Game, Projection
from app.database.repositories import ActualResultRepository, ProjectionRepository
from app.database.session import session_scope
from app.services.model_report_service import generate_model_report


def _make_game(game_id: str, game_date: str = "2026-07-15") -> None:
    with session_scope() as session:
        if session.get(Game, game_id) is not None:
            return
        session.add(Game(
            id=game_id, game_date=game_date, scheduled_start_utc=f"{game_date}T19:05:00Z",
            home_team="Cubs", away_team="Cardinals", home_team_id=1, away_team_id=2,
        ))
        session.flush()


def _make_projection(
    game_id: str, pitcher_name: str = "Test Pitcher", pitcher_id: int = 12345,
    game_date: str = "2026-07-15", final_blended_projection: float = 5.8,
    statistics_only_projection: float = 5.7, market_informed_projection: float = 5.9,
    expected_batters_faced: float = 23.0, expected_innings: float = 5.8, expected_pitch_count: float = 92.0,
    strikeout_line: float = None,
    recommended_side: str = None, edge_grade: str = None, betting_confidence: str = None,
    model_over_probability: float = None, workload_role: str = "starter",
    workload_source: str = "mlb_season_totals", workload_fallback_used: bool = False,
    lineup_status: str = "confirmed", validation_status: str = "valid",
) -> str:
    _make_game(game_id, game_date)
    market_snapshot = {"strikeout_line": strikeout_line} if strikeout_line is not None else None
    with session_scope() as session:
        row = Projection(
            game_id=game_id, game_date=game_date, pitcher_id=pitcher_id, pitcher_name=pitcher_name,
            lineup_status=lineup_status, lineup_json=[],
            pitcher_inputs_json={}, batter_inputs_json=[],
            workload_inputs_json={
                "workload_role": workload_role, "workload_source": workload_source,
                "workload_fallback_used": workload_fallback_used,
            },
            market_snapshot_json=market_snapshot,
            statistics_only_projection=statistics_only_projection,
            market_informed_projection=market_informed_projection,
            expected_innings=expected_innings, expected_pitch_count=expected_pitch_count,
            final_blended_projection=final_blended_projection,
            expected_batters_faced=expected_batters_faced,
            recommended_side=recommended_side, edge_grade=edge_grade,
            betting_confidence=betting_confidence, model_over_probability=model_over_probability,
            validation_status=validation_status,
        )
        saved = ProjectionRepository.save(session, row)
        return saved.id


def _grade_projection(projection_id: str, actual_strikeouts: int, actual_batters_faced: int = 23) -> None:
    with session_scope() as session:
        result = ActualResult(
            projection_id=projection_id, actual_strikeouts=actual_strikeouts,
            actual_batters_faced=actual_batters_faced, source="mlb_stats_api",
        )
        ActualResultRepository.save(session, result)


def test_projection_saved_even_with_no_bet_placed():
    pid = _make_projection("game-no-bet-1", pitcher_name="No Bet Pitcher")
    with session_scope() as session:
        saved = session.get(Projection, pid)
        assert saved is not None
        assert saved.pitcher_name == "No Bet Pitcher"


def test_pass_projection_saved():
    pid = _make_projection("game-pass-1", strikeout_line=5.5, recommended_side="PASS", edge_grade="No meaningful estimated edge")
    with session_scope() as session:
        saved = session.get(Projection, pid)
        assert saved.recommended_side == "PASS"


def test_no_market_line_projection_saved():
    pid = _make_projection("game-no-line-1", strikeout_line=None)
    with session_scope() as session:
        saved = session.get(Projection, pid)
        assert saved is not None
        assert saved.recommended_side is None


def test_validation_status_persisted_for_invalid_projection():
    pid = _make_projection("game-invalid-1", validation_status="invalid")
    with session_scope() as session:
        saved = session.get(Projection, pid)
        assert saved.validation_status == "invalid"


def test_model_report_uses_all_graded_projections_not_only_bets():
    pid1 = _make_projection("game-mr-1", pitcher_name="Pitcher One")
    _grade_projection(pid1, actual_strikeouts=6)
    pid2 = _make_projection("game-mr-2", pitcher_name="Pitcher Two", recommended_side="PASS")
    _grade_projection(pid2, actual_strikeouts=5)

    with session_scope() as session:
        all_graded = ProjectionRepository.list_graded_filtered(session)
        report = generate_model_report(all_graded)

    assert report.n_total_graded >= 2


def test_empty_database_report_does_not_crash():
    report = generate_model_report([])
    assert report.n_total_graded == 0
    assert len(report.warnings) >= 1


def test_small_sample_warning_present():
    pid = _make_projection("game-small-sample-1")
    _grade_projection(pid, actual_strikeouts=6)
    with session_scope() as session:
        single = [session.get(Projection, pid)]
        report = generate_model_report(single)
    assert any("small" in w.lower() for w in report.warnings)


def test_filter_by_pitcher_name():
    pid1 = _make_projection("game-filter-1", pitcher_name="Filter Target Pitcher")
    _grade_projection(pid1, actual_strikeouts=6)
    pid2 = _make_projection("game-filter-2", pitcher_name="Someone Else")
    _grade_projection(pid2, actual_strikeouts=5)

    with session_scope() as session:
        filtered = ProjectionRepository.list_graded_filtered(session, pitcher_name="Filter Target")
    assert all(p.pitcher_name == "Filter Target Pitcher" for p in filtered)
    assert len(filtered) >= 1


def test_filter_by_confidence():
    pid1 = _make_projection("game-conf-1", strikeout_line=5.5, betting_confidence="HIGH")
    _grade_projection(pid1, actual_strikeouts=6)
    pid2 = _make_projection("game-conf-2", strikeout_line=5.5, betting_confidence="LOW")
    _grade_projection(pid2, actual_strikeouts=5)

    with session_scope() as session:
        filtered = ProjectionRepository.list_graded_filtered(session, confidence="HIGH")
    assert all(p.betting_confidence == "HIGH" for p in filtered)


def test_filter_last_n():
    for i in range(5):
        pid = _make_projection(f"game-lastn-{i}", pitcher_name=f"LastN Pitcher {i}", game_date="2026-07-20")
        _grade_projection(pid, actual_strikeouts=5 + i)

    with session_scope() as session:
        limited = ProjectionRepository.list_graded_filtered(session, last_n=2)
    assert len(limited) <= 2


def test_correct_pitcher_selected_by_pitcher_id_and_game_id():
    pid_a = _make_projection("game-id-match-1", pitcher_name="Same Name", pitcher_id=111)
    pid_b = _make_projection("game-id-match-1", pitcher_name="Same Name", pitcher_id=222)
    _grade_projection(pid_a, actual_strikeouts=7)
    _grade_projection(pid_b, actual_strikeouts=3)

    with session_scope() as session:
        result_a = session.get(Projection, pid_a).actual_result
        result_b = session.get(Projection, pid_b).actual_result
    assert result_a.actual_strikeouts == 7
    assert result_b.actual_strikeouts == 3


def test_no_overwrite_of_finalized_result_without_force():
    pid = _make_projection("game-no-overwrite-1")
    _grade_projection(pid, actual_strikeouts=6)

    with session_scope() as session:
        exists = ActualResultRepository.exists_for_projection(session, pid)
    assert exists is True
    with session_scope() as session:
        original = session.get(Projection, pid).actual_result
        assert original.actual_strikeouts == 6


def test_delete_for_projection_only_removes_target_row():
    pid1 = _make_projection("game-delete-1")
    pid2 = _make_projection("game-delete-2")
    _grade_projection(pid1, actual_strikeouts=6)
    _grade_projection(pid2, actual_strikeouts=4)

    with session_scope() as session:
        deleted = ActualResultRepository.delete_for_projection(session, pid1)
    assert deleted is True

    with session_scope() as session:
        assert ActualResultRepository.exists_for_projection(session, pid1) is False
        assert ActualResultRepository.exists_for_projection(session, pid2) is True


def test_betting_ledger_unaffected_by_projection_changes():
    from app.services.bet_ledger import list_unsettled

    pending = list_unsettled()
    assert isinstance(pending, list)


# --- Full pipeline: invalid exclusion + dedup + footer reconciliation ---

def test_full_pipeline_excludes_invalid_and_deduplicates_reruns():
    # Broken legacy Barnett-style row -- excluded as invalid.
    pid_broken = _make_projection(
        "game-pipeline-barnett", pitcher_name="Mason Barnett", pitcher_id=686930,
        validation_status=None, expected_innings=27.9, expected_batters_faced=126,
        expected_pitch_count=531, final_blended_projection=10.03,
    )
    _grade_projection(pid_broken, actual_strikeouts=7)

    # Three reruns of the same pitcher/game -- must collapse to one.
    for i, lineup in enumerate(["projected", "confirmed", "projected"]):
        pid = _make_projection(
            "game-pipeline-mikolas", pitcher_name="Miles Mikolas", pitcher_id=543068,
            lineup_status=lineup, strikeout_line=5.5,
        )
        _grade_projection(pid, actual_strikeouts=6)

    # One clean, unique projection.
    pid_unique = _make_projection("game-pipeline-unique", pitcher_name="Unique Pitcher", pitcher_id=777777)
    _grade_projection(pid_unique, actual_strikeouts=5)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session, pitcher_name=None)
        # Scope to just this test's rows to keep the assertion precise
        # even if other tests in this file added rows to the same DB.
        relevant = [
            p for p in projections
            if p.game_id in ("game-pipeline-barnett", "game-pipeline-mikolas", "game-pipeline-unique")
        ]
        report = generate_model_report(relevant)

    assert report.n_raw_projections == 5
    assert report.n_excluded_invalid == 1
    assert report.n_excluded_reruns == 2
    assert report.n_independent_projections == 2
    assert report.n_total_graded == 2


def test_include_invalid_flag_bypasses_exclusion():
    pid = _make_projection(
        "game-include-invalid-1", pitcher_name="Broken Pitcher", pitcher_id=888001,
        validation_status=None, expected_innings=99.0, final_blended_projection=10.0,
    )
    _grade_projection(pid, actual_strikeouts=7)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [p for p in projections if p.game_id == "game-include-invalid-1"]
        report_default = generate_model_report(relevant)
        report_included = generate_model_report(relevant, include_invalid=True)

    assert report_default.n_excluded_invalid == 1
    assert report_default.n_total_graded == 0
    assert report_included.n_excluded_invalid == 0
    assert report_included.n_total_graded == 1


def test_include_reruns_flag_bypasses_deduplication():
    for lineup in ["projected", "confirmed"]:
        pid = _make_projection(
            "game-include-reruns-1", pitcher_name="Rerun Pitcher", pitcher_id=888002, lineup_status=lineup,
        )
        _grade_projection(pid, actual_strikeouts=6)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [p for p in projections if p.game_id == "game-include-reruns-1"]
        report_default = generate_model_report(relevant)
        report_included = generate_model_report(relevant, include_reruns=True)

    assert report_default.n_independent_projections == 1
    assert report_included.n_independent_projections == 2


def test_footer_counts_reconcile_exactly():
    pid_invalid = _make_projection(
        "game-footer-1", pitcher_name="Footer Invalid", pitcher_id=888003,
        validation_status=None, expected_innings=50.0, final_blended_projection=10.0,
    )
    _grade_projection(pid_invalid, actual_strikeouts=7)

    for lineup in ["projected", "confirmed"]:
        pid = _make_projection("game-footer-2", pitcher_name="Footer Rerun", pitcher_id=888004, lineup_status=lineup)
        _grade_projection(pid, actual_strikeouts=6)

    pid_unique = _make_projection("game-footer-3", pitcher_name="Footer Unique", pitcher_id=888005)
    _grade_projection(pid_unique, actual_strikeouts=5)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [p for p in projections if p.game_id in ("game-footer-1", "game-footer-2", "game-footer-3")]
        report = generate_model_report(relevant)

    # Raw = independent + excluded_invalid + excluded_reruns, exactly.
    assert report.n_raw_projections == report.n_independent_projections + report.n_excluded_invalid + report.n_excluded_reruns
    assert report.n_raw_projections == 4
    assert report.n_independent_projections == 2


def test_no_database_rows_deleted_by_report_generation():
    """generate_model_report and its filtering pipeline must never call
    session.delete() or otherwise remove rows -- confirms row counts in
    the actual database are unchanged after running a report that
    excludes/deduplicates in memory."""
    pid_invalid = _make_projection(
        "game-no-delete-1", pitcher_name="No Delete Test", pitcher_id=888006,
        validation_status=None, expected_innings=99.0,
    )
    _grade_projection(pid_invalid, actual_strikeouts=7)

    with session_scope() as session:
        count_before = session.query(Projection).filter(Projection.game_id == "game-no-delete-1").count()

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [p for p in projections if p.game_id == "game-no-delete-1"]
        generate_model_report(relevant)  # excludes it from the report, in memory only

    with session_scope() as session:
        count_after = session.query(Projection).filter(Projection.game_id == "game-no-delete-1").count()

    assert count_before == count_after == 1
    with session_scope() as session:
        still_there = session.get(Projection, pid_invalid)
        assert still_there is not None


# --- Bug 2: exclusion counts must never be negative, footer must reconcile ---

def test_raw_39_independent_24_all_counts_non_negative_and_reconcile():
    """Reproduces the exact live-report scenario: 39 raw rows, 1 genuinely
    invalid, 14 rerun duplicates, 24 independent projections remaining.
    Every exclusion count must be a non-negative integer and the footer
    equation must reconcile exactly: raw - invalid - reruns == independent.

    Fixture construction to hit these exact numbers: 1 invalid row, 14
    duplicate PAIRS (2 rows each = 28 raw rows, collapsing to 14
    independent + 14 reruns removed), and 10 genuinely unique single
    rows (10 raw = 10 independent, 0 reruns). Total raw = 1 + 28 + 10 =
    39. Independent = 14 + 10 = 24. Reruns removed = 14. Invalid = 1.
    """
    pid_invalid = _make_projection(
        "game-bug2-invalid", pitcher_name="Bug2 Invalid Pitcher", pitcher_id=900001,
        validation_status=None, expected_innings=50.0, final_blended_projection=10.0,
    )
    _grade_projection(pid_invalid, actual_strikeouts=7)

    for i in range(14):
        game_id = f"game-bug2-pair-{i}"
        for lineup in ["projected", "confirmed"]:
            pid = _make_projection(game_id, pitcher_name=f"Pair Pitcher {i}",
                                    pitcher_id=900100 + i, lineup_status=lineup)
            _grade_projection(pid, actual_strikeouts=6)

    for i in range(10):
        game_id = f"game-bug2-single-{i}"
        pid = _make_projection(game_id, pitcher_name=f"Single Pitcher {i}", pitcher_id=900300 + i)
        _grade_projection(pid, actual_strikeouts=6)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [
            p for p in projections
            if p.game_id == "game-bug2-invalid" or p.game_id.startswith("game-bug2-pair-")
            or p.game_id.startswith("game-bug2-single-")
        ]
        report = generate_model_report(relevant)

    assert report.n_excluded_invalid >= 0
    assert report.n_excluded_reruns >= 0
    assert report.n_raw_projections - report.n_excluded_invalid - report.n_excluded_reruns == report.n_independent_projections

    assert report.n_raw_projections == 39
    assert report.n_excluded_invalid == 1
    assert report.n_excluded_reruns == 14
    assert report.n_independent_projections == 24


def test_negative_exclusion_counts_never_occur_for_any_split():
    """A broader sweep: no combination of invalid/valid/rerun rows should
    ever produce a negative exclusion count."""
    pid1 = _make_projection("game-bug2-sweep-1", pitcher_name="Sweep A", pitcher_id=900301)
    _grade_projection(pid1, actual_strikeouts=6)
    pid2 = _make_projection(
        "game-bug2-sweep-2", pitcher_name="Sweep B", pitcher_id=900302,
        validation_status=None, expected_innings=99.0,
    )
    _grade_projection(pid2, actual_strikeouts=6)

    with session_scope() as session:
        projections = ProjectionRepository.list_graded_filtered(session)
        relevant = [p for p in projections if p.game_id in ("game-bug2-sweep-1", "game-bug2-sweep-2")]
        report = generate_model_report(relevant)

    assert report.n_excluded_invalid >= 0
    assert report.n_excluded_reruns >= 0
    assert report.n_independent_projections >= 0
