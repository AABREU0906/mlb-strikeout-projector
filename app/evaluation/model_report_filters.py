"""
Historical-data quality filtering for `python main.py model-report`.

Everything here operates on ALREADY-SAVED Projection rows -- it never
touches the projection engine, workload model, Monte Carlo simulation,
NRFI/YRFI model, or edge-grading logic in any way. It answers three
questions about historical data quality:

  1. Was this a genuinely invalid projection (the pre-fix workload bug,
     or any other historically-recorded/derivable invalidity)?
  2. Is this a rerun of the same (game, pitcher) rather than an
     independent observation?
  3. Was a recommendation ever actually computed for this row, or is the
     NULL simply "this predates recommended_side being stored" (which
     must NEVER be silently treated as a genuine PASS)?

No database rows are ever deleted by anything in this module -- these are
purely IN-MEMORY exclusions applied before metrics are computed.
"""
from __future__ import annotations

from app.validation.bounds import (
    MAX_BATTERS_FACED_PER_START,
    MAX_INNINGS_PER_START,
    MAX_PITCHES_PER_START,
)

UNRECORDED_RECOMMENDATION = "UNKNOWN"


def classify_recommendation(recommended_side) -> str:
    """NULL must never become PASS -- this is the single choke point that
    enforces that."""
    if recommended_side in ("OVER", "UNDER", "PASS"):
        return recommended_side
    return UNRECORDED_RECOMMENDATION


def is_invalid_projection(row: dict) -> tuple[bool, str]:
    """Returns (is_invalid, reason)."""
    if row.get("validation_status") == "invalid":
        return True, "validation_status=invalid"

    if row.get("validation_status") is not None:
        return False, ""

    reasons = []
    expected_innings = row.get("expected_innings")
    expected_bf = row.get("expected_batters_faced")
    expected_pitches = row.get("expected_pitch_count")
    final_projection = row.get("final_blended_projection")

    if expected_innings is not None and (expected_innings > MAX_INNINGS_PER_START or expected_innings < 0):
        reasons.append(f"expected_innings={expected_innings} outside plausible range")
    if expected_bf is not None and (expected_bf > MAX_BATTERS_FACED_PER_START or expected_bf < 0):
        reasons.append(f"expected_batters_faced={expected_bf} outside plausible range")
    if expected_pitches is not None and (expected_pitches > MAX_PITCHES_PER_START or expected_pitches < 0):
        reasons.append(f"expected_pitch_count={expected_pitches} outside plausible range")
    if final_projection is not None and final_projection < 0:
        reasons.append(f"final_blended_projection={final_projection} is negative")

    if reasons:
        return True, "legacy sanity check: " + "; ".join(reasons)
    return False, ""


def _canonical_sort_key(row: dict):
    is_valid, _ = is_invalid_projection(row)
    is_valid = not is_valid

    created_before_first_pitch = True
    if row.get("created_at_utc") and row.get("game_start_utc"):
        created_before_first_pitch = row["created_at_utc"] <= row["game_start_utc"]

    lineup_confirmed = row.get("lineup_status") == "confirmed"
    recency_key = row.get("created_at_utc") or ""

    return (
        is_valid,
        created_before_first_pitch,
        lineup_confirmed,
        recency_key,
        row.get("projection_id") or "",
    )


def deduplicate_projections(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Groups by (game_id, pitcher_id) and keeps exactly one canonical row
    per group. Returns (canonical_rows, excluded_rerun_rows). Never
    mutates or deletes anything."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("game_id"), row.get("pitcher_id"))
        groups.setdefault(key, []).append(row)

    canonical: list[dict] = []
    excluded: list[dict] = []
    for group_rows in groups.values():
        if len(group_rows) == 1:
            canonical.append(group_rows[0])
            continue
        winner = max(group_rows, key=_canonical_sort_key)
        canonical.append(winner)
        excluded.extend(r for r in group_rows if r is not winner)

    return canonical, excluded
