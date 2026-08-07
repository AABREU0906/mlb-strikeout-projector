"""
Tests for app/services/confirmed_slate_service.py -- the main orchestrator.
All Odds API and MLB Stats API calls are mocked via monkeypatch; no live
network requests are made. The real projection engine is also mocked
(via ProjectionPipeline.run) since these tests are about ORCHESTRATION
logic (eligibility, credit conservation, incremental tracking, failure
isolation), not projection math itself.

Note: the fake projection_ids returned by the mocked pipeline.run() do
not correspond to real database rows (since the real
ProjectionRepository.save() call inside pipeline.run() never executes
here) -- update_projection_edge_outcome() gracefully no-ops when it can't
find the row, so this does not cause any test to fail; it just means
these tests validate ORCHESTRATION behavior, not the persisted row
contents (see test_model_report_persistence.py for real DB-row tests
against the actual pipeline).
"""
import types

import pytest

import app.services.confirmed_slate_service as svc


def _fake_result(final_projection=5.8, workload_source="mlb_season_totals"):
    workload = types.SimpleNamespace(
        prob_complete_5=0.75, prob_complete_6=0.55, prob_complete_7=0.30, prob_early_exit=0.15,
        workload_fallback_used=False, workload_fallback_count=0, workload_all_metrics_fallback=False,
        workload_source=workload_source, workload_role="starter",
    )
    probs = {i: 1 / 16 for i in range(16)}
    return types.SimpleNamespace(
        statistics_only_projection=final_projection, market_informed_projection=final_projection,
        final_blended_projection=final_projection, median_strikeouts=final_projection,
        std_dev=2.0, percentiles={10: 3, 25: 4, 50: 5, 75: 7, 90: 8}, probability_by_k=probs,
        expected_innings=5.8, expected_batters_faced=24.0, expected_pitch_count=95.0,
        workload=workload, market_used={"snapshot": None},
    )


def _fake_game(game_id="1001", home_team="Philadelphia Phillies", away_team="New York Mets",
                home_pitcher_id=111, away_pitcher_id=222,
                home_pitcher_name="Cristopher Sanchez", away_pitcher_name="Zack Wheeler",
                home_team_id=1, away_team_id=2):
    return {
        "game_id": game_id, "game_date": "2026-07-15", "scheduled_start_utc": "2026-07-15T19:05:00Z",
        "home_team": home_team, "away_team": away_team, "home_team_id": home_team_id, "away_team_id": away_team_id,
        "probable_home_pitcher_id": home_pitcher_id, "probable_home_pitcher_name": home_pitcher_name,
        "probable_away_pitcher_id": away_pitcher_id, "probable_away_pitcher_name": away_pitcher_name,
    }


@pytest.fixture
def patched_pipeline_run(monkeypatch):
    call_log = []

    def fake_run(self, game, pitcher_id, pitcher_is_home, season, manual_market=None, **kwargs):
        call_log.append({"game_id": game["game_id"], "pitcher_id": pitcher_id, "manual_market": manual_market})
        result = _fake_result()
        pitcher_name = game["probable_home_pitcher_name"] if pitcher_is_home else game["probable_away_pitcher_name"]
        return result, f"proj-{pitcher_id}-{game['game_id']}", "confirmed", "mlb_stats_api", pitcher_name

    monkeypatch.setattr(svc.ProjectionPipeline, "run", fake_run)
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: [])
    return call_log


def test_only_confirmed_games_processed(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="confirmed-1"), _fake_game(game_id="unconfirmed-1")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: (
        types.SimpleNamespace(data={"lineup": []}) if gid == "confirmed-1" else None
    ))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert result.summary.games_confirmed == 1
    assert result.summary.games_skipped == 1
    processed_games = {c["game_id"] for c in patched_pipeline_run}
    assert processed_games == {"confirmed-1"}


def test_unconfirmed_games_cause_zero_player_prop_calls(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="unconfirmed-only")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: None)

    odds_calls = []
    fake_session = types.SimpleNamespace(
        is_configured=lambda: True,
        get_events=lambda: odds_calls.append("events") or [],
        get_event_odds=lambda eid: odds_calls.append(("odds", eid)) or None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    )
    monkeypatch.setattr(svc, "OddsRunSession", lambda: fake_session)

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert result.summary.games_confirmed == 0
    assert not any(isinstance(c, tuple) and c[0] == "odds" for c in odds_calls)


def test_both_starters_projected_for_confirmed_game(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="both-starters")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    projected_pitcher_ids = {c["pitcher_id"] for c in patched_pipeline_run}
    assert projected_pitcher_ids == {111, 222}


def test_one_event_odds_call_serves_both_pitchers(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="shared-event-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))

    event_odds_call_count = {"n": 0}
    fake_event_odds = {"bookmakers": [{"key": "fanduel", "markets": [{"key": "pitcher_strikeouts", "last_update": "x", "outcomes": [
        {"description": "Cristopher Sanchez", "name": "Over", "point": 5.5, "price": -110},
        {"description": "Cristopher Sanchez", "name": "Under", "point": 5.5, "price": -120},
        {"description": "Zack Wheeler", "name": "Over", "point": 6.5, "price": -115},
        {"description": "Zack Wheeler", "name": "Under", "point": 6.5, "price": -105},
    ]}]}]}

    def fake_get_event_odds(eid):
        event_odds_call_count["n"] += 1
        return fake_event_odds

    monkeypatch.setattr(svc, "match_event_to_game", lambda events, h, a, t: types.SimpleNamespace(event_id="evt-shared"))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: True, get_events=lambda: [{"id": "evt-shared"}],
        get_event_odds=fake_get_event_odds,
        events_list_calls_made=1, event_odds_calls_made=1, credits_used_this_run=50, credits_used_cumulative_account=100, credits_remaining=450,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert event_odds_call_count["n"] == 1
    assert result.summary.fanduel_markets_found == 2


def test_incremental_run_skips_already_projected(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="incremental-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))
    monkeypatch.setattr(
        svc.ProjectionRepository, "exists_for_game_pitcher_date",
        staticmethod(lambda session, gid, pid, date: pid == 111),
    )

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    processed_pitcher_ids = {c["pitcher_id"] for c in patched_pipeline_run}
    assert processed_pitcher_ids == {222}
    assert result.summary.pitchers_already_projected_today == 1


def test_refresh_reprocesses_already_projected_games(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="refresh-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))
    monkeypatch.setattr(
        svc.ProjectionRepository, "exists_for_game_pitcher_date",
        staticmethod(lambda session, gid, pid, date: True),
    )

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True, refresh=True)

    processed_pitcher_ids = {c["pitcher_id"] for c in patched_pipeline_run}
    assert processed_pitcher_ids == {111, 222}


def test_no_odds_flag_makes_zero_odds_calls(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="no-odds-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))

    events_calls = {"n": 0}
    odds_calls = {"n": 0}
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: True,
        get_events=lambda: (events_calls.__setitem__("n", events_calls["n"] + 1), [])[1],
        get_event_odds=lambda eid: (odds_calls.__setitem__("n", odds_calls["n"] + 1), None)[1],
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert events_calls["n"] == 0
    assert odds_calls["n"] == 0


def test_missing_api_key_does_not_crash(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="no-key-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert result.summary.odds_api_configured is False
    assert len(result.rows) == 2


def test_ambiguous_event_safely_skipped(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="ambiguous-event-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "match_event_to_game", lambda events, h, a, t: None)
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: True, get_events=lambda: [{"id": "evt-1"}], get_event_odds=lambda eid: None,
        events_list_calls_made=1, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert len(result.rows) == 2
    assert all(r.fanduel_market_status == "unavailable" for r in result.rows)


def test_one_bad_pitcher_does_not_kill_the_slate(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="game-a", home_pitcher_id=111, away_pitcher_id=222),
             _fake_game(game_id="game-b", home_pitcher_id=333, away_pitcher_id=444)]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None, credits_used_cumulative_account=None, credits_remaining=None,
    ))

    def flaky_run(self, game, pitcher_id, pitcher_is_home, season, manual_market=None, **kwargs):
        if pitcher_id == 111:
            raise RuntimeError("simulated API timeout")
        result = _fake_result()
        pitcher_name = game["probable_home_pitcher_name"] if pitcher_is_home else game["probable_away_pitcher_name"]
        return result, f"proj-{pitcher_id}", "confirmed", "mlb_stats_api", pitcher_name

    monkeypatch.setattr(svc.ProjectionPipeline, "run", flaky_run)

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert len(result.failed_items) == 1
    assert len(result.rows) == 3


def test_credit_header_values_propagate_to_summary(monkeypatch, patched_pipeline_run):
    games = [_fake_game(game_id="credits-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: True, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=1, event_odds_calls_made=1, credits_used_this_run=50, credits_used_cumulative_account=100, credits_remaining=450,
    ))

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert result.summary.credits_used_this_run == 50
    assert result.summary.credits_used_cumulative_account == 100
    assert result.summary.credits_remaining == 450


def test_summary_default_counts_are_zero_and_non_negative():
    summary = svc.ConfirmedSlateSummary()
    assert summary.games_today == 0
    assert summary.games_confirmed == 0
    assert summary.pitchers_projected == 0


# --- Bug 2: "already projected" must be tracked in explicit units
# (games vs. pitchers), not conflated into one pitcher-count field. ---

def test_games_already_processed_today_counts_games_not_pitchers(monkeypatch, patched_pipeline_run):
    """The exact reported scenario: 1 game with both pitchers already
    projected today. games_already_processed_today must be 1 (one GAME),
    while pitchers_already_projected_today must be 2 (two PITCHERS) --
    these are different units and must not be conflated."""
    games = [_fake_game(game_id="fully-done-game")]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None,
        credits_used_cumulative_account=None, credits_remaining=None,
    ))
    monkeypatch.setattr(
        svc.ProjectionRepository, "exists_for_game_pitcher_date",
        staticmethod(lambda session, gid, pid, date: True),  # both pitchers already done
    )

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert result.summary.games_already_processed_today == 1
    assert result.summary.pitchers_already_projected_today == 2
    assert result.summary.new_games_projected == 0
    assert result.summary.pitchers_projected == 0
    assert len(patched_pipeline_run) == 0  # zero pipeline.run() calls -- no duplicate saves


def test_partially_done_game_not_counted_as_games_already_processed(monkeypatch, patched_pipeline_run):
    """A game with ONE pitcher already done and one new pitcher must NOT
    count toward games_already_processed_today (the GAME as a whole
    still needs processing), but the one already-done pitcher must still
    count in the pitcher-level tally."""
    games = [_fake_game(game_id="partial-game", home_pitcher_id=111, away_pitcher_id=222)]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None,
        credits_used_cumulative_account=None, credits_remaining=None,
    ))
    monkeypatch.setattr(
        svc.ProjectionRepository, "exists_for_game_pitcher_date",
        staticmethod(lambda session, gid, pid, date: pid == 111),  # only one of the two done
    )

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert result.summary.games_already_processed_today == 0  # game still had work to do
    assert result.summary.pitchers_already_projected_today == 1
    assert result.summary.new_games_projected == 1
    assert result.summary.pitchers_projected == 1


def test_summary_counts_reconcile_across_mixed_scenario(monkeypatch, patched_pipeline_run):
    """One fully-done game, one brand-new game -- every summary count
    must reconcile exactly, in the correct units."""
    games = [
        _fake_game(game_id="done-game", home_pitcher_id=111, away_pitcher_id=222),
        _fake_game(game_id="new-game", home_pitcher_id=333, away_pitcher_id=444),
    ]
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: games)
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None,
        credits_used_cumulative_account=None, credits_remaining=None,
    ))
    monkeypatch.setattr(
        svc.ProjectionRepository, "exists_for_game_pitcher_date",
        staticmethod(lambda session, gid, pid, date: gid == "done-game"),
    )

    result = svc.run_confirmed_slate(game_date="2026-07-15", no_odds=True)

    assert result.summary.games_confirmed == 2
    assert result.summary.games_already_processed_today == 1
    assert result.summary.new_games_projected == 1
    assert result.summary.pitchers_already_projected_today == 2
    assert result.summary.pitchers_projected == 2
    # Reconciliation: every pitcher in a confirmed game is accounted for
    # exactly once, either as already-projected or newly-projected.
    total_pitchers_in_confirmed_games = 4
    assert result.summary.pitchers_already_projected_today + result.summary.pitchers_projected == total_pitchers_in_confirmed_games


# --- Diff/Projection data-consistency fix ---

def _fake_result_with_market(statistics_only, final_blended, line):
    """A fake ProjectionResult whose statistics_only_projection and
    final_blended_projection deliberately DIFFER (as they legitimately do
    once market data is blended in) -- this is the exact condition that
    exposed the bug, so the fixture must exercise it, not use equal
    values that would mask the issue.

    probability_by_k is built as a tight distribution centered on
    round(statistics_only) (not a uniform 0-15 spread) so its implied
    mean is close to statistics_only_projection -- otherwise
    validate_projection's distribution-consistency check would correctly
    reject the fixture as invalid, since probability_by_k is compared
    against statistics_only_projection specifically (see
    check_projection_vs_distribution_mean)."""
    workload = types.SimpleNamespace(
        prob_complete_5=0.75, prob_complete_6=0.55, prob_complete_7=0.30, prob_early_exit=0.15,
        workload_fallback_used=False, workload_fallback_count=0, workload_all_metrics_fallback=False,
        workload_source="mlb_season_totals", workload_role="starter",
    )
    center = round(statistics_only)
    probs = {k: 0.0 for k in range(16)}
    probs[max(0, min(15, center))] = 1.0
    market_snapshot_dict = {
        "source": "the_odds_api", "retrieved_at": "2026-07-15T18:00:00Z", "manual_override": True,
        "strikeout_line": line, "over_odds": -110, "under_odds": -110,
    }
    return types.SimpleNamespace(
        statistics_only_projection=statistics_only, market_informed_projection=final_blended,
        final_blended_projection=final_blended, median_strikeouts=final_blended,
        std_dev=2.0, percentiles={10: 3, 25: 4, 50: 5, 75: 7, 90: 8}, probability_by_k=probs,
        expected_innings=5.8, expected_batters_faced=24.0, expected_pitch_count=95.0,
        workload=workload, market_used={"snapshot": market_snapshot_dict},
    )


def test_displayed_diff_matches_displayed_projection_minus_line(monkeypatch):
    """Reproduces the exact reported bug: statistics_only_projection
    (4.36) and final_blended_projection (4.30) legitimately diverge.
    projection_minus_line on the resulting row MUST be computed from
    final_blended_projection (the value actually shown as 'Projection'),
    so Diff == Projection - Line reconciles exactly, matching the Noah
    Schultz example from the report (4.30 vs line 3.5 -> +0.80)."""
    game = _fake_game(game_id="diff-bug-game")
    monkeypatch.setattr(svc.ProjectionPipeline, "get_schedule", lambda self, date: [game])
    monkeypatch.setattr(svc.MlbStatsApiProvider, "get_confirmed_lineup", lambda self, gid, tid: types.SimpleNamespace(data={"lineup": []}))
    monkeypatch.setattr(svc, "OddsRunSession", lambda: types.SimpleNamespace(
        is_configured=lambda: False, get_events=lambda: [], get_event_odds=lambda eid: None,
        events_list_calls_made=0, event_odds_calls_made=0, credits_used_this_run=None,
        credits_used_cumulative_account=None, credits_remaining=None,
    ))

    fake_edge_analysis = types.SimpleNamespace(
        recommended_side="OVER", grade="Moderate estimated edge", confidence="MEDIUM",
        selected=types.SimpleNamespace(expected_value=0.05),
        over=types.SimpleNamespace(model_probability=0.55),
        under=types.SimpleNamespace(model_probability=0.45),
    )
    monkeypatch.setattr("app.reporting.display.print_market_comparison", lambda *a, **kw: fake_edge_analysis)

    def fake_run(self, game, pitcher_id, pitcher_is_home, season, manual_market=None, **kwargs):
        # 4.36 stats-only vs 4.30 blended -- the exact Noah Schultz divergence.
        result = _fake_result_with_market(statistics_only=4.36, final_blended=4.30, line=3.5)
        pitcher_name = game["probable_home_pitcher_name"] if pitcher_is_home else game["probable_away_pitcher_name"]
        return result, f"proj-{pitcher_id}", "confirmed", "mlb_stats_api", pitcher_name

    monkeypatch.setattr(svc.ProjectionPipeline, "run", fake_run)

    result = svc.run_confirmed_slate(game_date="2026-07-15")

    assert len(result.rows) == 2
    for row in result.rows:
        assert row.final_blended_projection == 4.30
        assert row.projection_minus_line is not None
        # THE CORE ASSERTION: Diff must reconcile exactly against the
        # displayed Projection value, not a different internal number.
        assert abs(row.projection_minus_line - (row.final_blended_projection - row.strikeout_line)) < 1e-9
        assert abs(row.projection_minus_line - 0.80) < 0.005  # matches the reported example exactly


def test_all_four_reported_examples_reconcile_after_rounding(monkeypatch):
    """Runs all four pitchers from the bug report through the real
    per-pitcher projection function and confirms every one now reconciles
    Diff == round(Projection - Line, 2), not the previously-buggy value."""
    from app.services.confirmed_slate_service import _project_one_pitcher

    reported_examples = [
        ("Noah Schultz", 4.36, 4.30, 3.5, 0.80),
        ("Payton Tolle", 5.93, 5.88, 6.5, -0.62),
        ("Max Fried", 5.21, 5.18, 5.5, -0.32),
        ("Parker Messick", 6.02, 5.96, 5.5, 0.46),
    ]

    fake_edge_analysis = types.SimpleNamespace(
        recommended_side="OVER", grade="Moderate estimated edge", confidence="MEDIUM",
        selected=types.SimpleNamespace(expected_value=0.05),
        over=types.SimpleNamespace(model_probability=0.55),
        under=types.SimpleNamespace(model_probability=0.45),
    )
    monkeypatch.setattr("app.reporting.display.print_market_comparison", lambda *a, **kw: fake_edge_analysis)
    monkeypatch.setattr(svc, "update_projection_edge_outcome", lambda *a, **kw: None)

    for name, stats_only, blended, line, expected_diff in reported_examples:
        pipeline = types.SimpleNamespace()

        def fake_run(game, pitcher_id, pitcher_is_home, season, manual_market=None, **kwargs):
            result = _fake_result_with_market(statistics_only=stats_only, final_blended=blended, line=line)
            return result, f"proj-{name}", "confirmed", "mlb_stats_api", name

        pipeline.run = fake_run
        game = _fake_game(game_id=f"game-{name}")

        row, error = _project_one_pitcher(
            pipeline, game, pitcher_id=111, is_home=True, season=2026,
            manual_market=None, lineup_status_for_display="confirmed",
        )

        assert error is None, f"{name}: unexpected error {error}"
        assert row.final_blended_projection == blended
        assert abs(row.projection_minus_line - round(blended - line, 2)) < 0.005, (
            f"{name}: Diff {row.projection_minus_line} does not reconcile with "
            f"Projection {blended} - Line {line} = {round(blended - line, 2)}"
        )
        assert abs(row.projection_minus_line - expected_diff) < 0.005, (
            f"{name}: expected corrected Diff {expected_diff}, got {row.projection_minus_line}"
        )
