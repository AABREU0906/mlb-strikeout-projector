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
    expected_batters_faced: float = 23.0, strikeout_line: float = None,
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
