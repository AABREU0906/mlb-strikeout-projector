"""
Pure metric-computation functions for `python main.py model-report`.

Deliberately separated from the ORM/CLI layer: every function here takes
plain dicts (one per graded projection) and returns plain dicts/lists, so
the actual math is testable without a database or pydantic. See
app/services/model_report_service.py for the extraction layer that turns
Projection/ActualResult ORM rows into these plain dicts.
"""
from __future__ import annotations

import math
from typing import Optional

CALIBRATION_BUCKETS = [
    (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 0.60),
    (0.60, 0.65), (0.65, 0.70), (0.70, 1.01),
]
MIN_BUCKET_SAMPLE_FOR_CALIBRATION_CLAIM = 10
MIN_SAMPLE_FOR_STRONG_CONCLUSIONS = 20


def compute_error_metrics(rows: list[dict], projection_field: str = "final_blended_projection") -> dict:
    pairs = [
        (row[projection_field], row["actual_strikeouts"])
        for row in rows
        if row.get(projection_field) is not None and row.get("actual_strikeouts") is not None
    ]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None, "median_ae": None,
                "pct_within_0_5": None, "pct_within_1_0": None, "pct_within_2_0": None}

    signed_errors = [proj - actual for proj, actual in pairs]  # final_projection - actual, per spec
    abs_errors = sorted(abs(e) for e in signed_errors)

    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e * e for e in signed_errors) / n)
    bias = sum(signed_errors) / n
    median_ae = abs_errors[n // 2] if n % 2 == 1 else (abs_errors[n // 2 - 1] + abs_errors[n // 2]) / 2

    within = lambda t: sum(1 for e in abs_errors if e <= t) / n

    return {
        "n": n,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "bias": round(bias, 3),
        "median_ae": round(median_ae, 3),
        "pct_within_0_5": round(within(0.5) * 100, 1),
        "pct_within_1_0": round(within(1.0) * 100, 1),
        "pct_within_2_0": round(within(2.0) * 100, 1),
    }


def compute_directional_metrics(rows: list[dict]) -> dict:
    """
    BUG FIX: previously, `r.get("recommended_side") in (None, "PASS")`
    silently folded "no recommendation was ever recorded" (rows that
    predate recommended_side being persisted) into "genuine PASS
    recommendation" -- producing misleading output like "0 Over, 0
    Under, 27 PASS" when in reality 27 rows simply never had a
    recommendation computed/stored at all. NULL now classifies as
    UNKNOWN (see app.evaluation.model_report_filters.classify_recommendation)
    and is excluded from every recommendation-statistic percentage below
    -- the denominator for projected_over/under/pass percentages is the
    count of rows with a KNOWN recommendation, not the full `with_line`
    count.
    """
    from app.evaluation.model_report_filters import UNRECORDED_RECOMMENDATION, classify_recommendation

    with_line = [r for r in rows if r.get("strikeout_line") is not None and r.get("actual_strikeouts") is not None]
    n = len(with_line)
    if n == 0:
        return {"n": 0}

    classified = [classify_recommendation(r.get("recommended_side")) for r in with_line]
    known_rows = [r for r, c in zip(with_line, classified) if c != UNRECORDED_RECOMMENDATION]
    n_unknown = n - len(known_rows)
    n_known = len(known_rows)

    proj_over = sum(1 for r in known_rows if r.get("recommended_side") == "OVER")
    proj_under = sum(1 for r in known_rows if r.get("recommended_side") == "UNDER")
    proj_pass = sum(1 for r in known_rows if r.get("recommended_side") == "PASS")

    actual_over = sum(1 for r in with_line if r["actual_strikeouts"] > r["strikeout_line"])
    actual_under = sum(1 for r in with_line if r["actual_strikeouts"] < r["strikeout_line"])
    actual_push = n - actual_over - actual_under

    def _result_for(row):
        side = row.get("recommended_side")
        if side not in ("OVER", "UNDER"):
            return None
        actual, line = row["actual_strikeouts"], row["strikeout_line"]
        if actual == line:
            return "PUSH"
        went_over = actual > line
        won = (side == "OVER" and went_over) or (side == "UNDER" and not went_over)
        return "WIN" if won else "LOSS"

    over_results = [_result_for(r) for r in known_rows if r.get("recommended_side") == "OVER"]
    under_results = [_result_for(r) for r in known_rows if r.get("recommended_side") == "UNDER"]

    def _win_rate(results):
        decided = [r for r in results if r in ("WIN", "LOSS")]
        if not decided:
            return {"n": len(results), "wins": 0, "losses": 0, "pushes": sum(1 for r in results if r == "PUSH"), "win_rate": None}
        wins = sum(1 for r in decided if r == "WIN")
        return {
            "n": len(results), "wins": wins, "losses": len(decided) - wins,
            "pushes": sum(1 for r in results if r == "PUSH"),
            "win_rate": round(wins / len(decided) * 100, 1),
        }

    all_bet_results = [r for r in (over_results + under_results) if r in ("WIN", "LOSS")]
    overall_win_rate = round(sum(1 for r in all_bet_results if r == "WIN") / len(all_bet_results) * 100, 1) if all_bet_results else None

    return {
        "n": n,
        "n_known_recommendation": n_known,
        "n_unknown_recommendation": n_unknown,
        "projected_over": proj_over, "projected_over_pct": round(proj_over / n_known * 100, 1) if n_known else None,
        "projected_under": proj_under, "projected_under_pct": round(proj_under / n_known * 100, 1) if n_known else None,
        "projected_pass": proj_pass, "projected_pass_pct": round(proj_pass / n_known * 100, 1) if n_known else None,
        "actual_over": actual_over, "actual_over_pct": round(actual_over / n * 100, 1),
        "actual_under": actual_under, "actual_under_pct": round(actual_under / n * 100, 1),
        "actual_push": actual_push,
        "recommendation_win_rate": overall_win_rate,
        "over_results": _win_rate(over_results),
        "under_results": _win_rate(under_results),
        "pass_count": proj_pass,
    }


def compute_calibration(rows: list[dict]) -> list[dict]:
    usable = [
        r for r in rows
        if r.get("model_over_probability") is not None
        and r.get("strikeout_line") is not None
        and r.get("actual_strikeouts") is not None
    ]

    buckets = []
    for lo, hi in CALIBRATION_BUCKETS:
        in_bucket = [r for r in usable if lo <= r["model_over_probability"] < hi]
        label = f"{int(lo*100)}-{int(hi*100)}%" if hi <= 1.0 else f"{int(lo*100)}%+"
        n = len(in_bucket)
        if n == 0:
            buckets.append({"bucket": label, "n": 0, "avg_predicted": None, "actual_over_rate": None,
                             "calibration_gap": None, "brier": None, "reliable": False})
            continue

        avg_predicted = sum(r["model_over_probability"] for r in in_bucket) / n
        actual_overs = sum(1 for r in in_bucket if r["actual_strikeouts"] > r["strikeout_line"])
        actual_rate = actual_overs / n
        brier = sum((r["model_over_probability"] - (1.0 if r["actual_strikeouts"] > r["strikeout_line"] else 0.0)) ** 2 for r in in_bucket) / n

        buckets.append({
            "bucket": label, "n": n,
            "avg_predicted": round(avg_predicted * 100, 1),
            "actual_over_rate": round(actual_rate * 100, 1),
            "calibration_gap": round((actual_rate - avg_predicted) * 100, 1),
            "brier": round(brier, 4),
            "reliable": n >= MIN_BUCKET_SAMPLE_FOR_CALIBRATION_CLAIM,
        })

    return buckets


def compute_bias_by_group(rows: list[dict], group_field: str, projection_field: str = "final_blended_projection") -> list[dict]:
    """
    BUG FIX: this previously appended `actual - proj` to each group's
    error list, while compute_error_metrics' "bias" field (the overall
    Core Projection Accuracy number) uses `proj - actual`. Both were
    labeled "Bias" in the report, but with opposite sign conventions --
    e.g. Dylan Cease (projected 7.30, actual 10) showed as "+2.70" here
    while the documented convention ("Bias = final projection minus
    actual strikeouts") says it should be -2.70. Standardized to
    proj - actual everywhere something is labeled "Bias", matching the
    documented convention: positive = model overprojects, negative =
    model underprojects. (The separate PROJECTION ERROR REVIEW section
    intentionally uses actual - projected under the label "miss", which
    is a different, clearly-labeled metric and is NOT changed by this fix.)
    """
    groups: dict[str, list[float]] = {}
    for row in rows:
        key = row.get(group_field)
        proj = row.get(projection_field)
        actual = row.get("actual_strikeouts")
        if key is None or proj is None or actual is None:
            continue
        groups.setdefault(str(key), []).append(proj - actual)

    result = []
    for key, errors in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(errors)
        result.append({
            "group": key, "n": n,
            "avg_bias": round(sum(errors) / n, 3),
            "reliable": n >= MIN_BUCKET_SAMPLE_FOR_CALIBRATION_CLAIM,
        })
    return result


def line_bucket_for(line: Optional[float]) -> Optional[str]:
    if line is None:
        return None
    if line < 4.0:
        return "<4.0"
    if line < 5.0:
        return "4.0-4.5"
    if line < 6.0:
        return "5.0-5.5"
    if line < 7.0:
        return "6.0-6.5"
    if line < 8.0:
        return "7.0-7.5"
    return "8.0+"


def decompose_error(row: dict) -> Optional[dict]:
    """
        projected_k_rate_per_batter = final_blended_projection / expected_batters_faced
        actual_k_rate_per_batter = actual_strikeouts / actual_batters_faced
        workload_contribution = (actual_bf - expected_bf) * projected_k_rate_per_batter
        rate_contribution = actual_bf * (actual_k_rate_per_batter - projected_k_rate_per_batter)
    """
    projection = row.get("final_blended_projection")
    actual_k = row.get("actual_strikeouts")
    expected_bf = row.get("expected_batters_faced")
    actual_bf = row.get("actual_batters_faced")

    if None in (projection, actual_k, expected_bf, actual_bf) or expected_bf <= 0 or actual_bf <= 0:
        return None

    projected_k_rate = projection / expected_bf
    actual_k_rate = actual_k / actual_bf

    workload_contribution = (actual_bf - expected_bf) * projected_k_rate
    rate_contribution = actual_bf * (actual_k_rate - projected_k_rate)

    total_miss = actual_k - projection
    reconciled = workload_contribution + rate_contribution
    reconciliation_error = total_miss - reconciled

    return {
        "pitcher_name": row.get("pitcher_name"),
        "projected_ks": round(projection, 2),
        "actual_ks": actual_k,
        "total_miss": round(total_miss, 2),
        "expected_bf": round(expected_bf, 1),
        "actual_bf": actual_bf,
        "workload_contribution": round(workload_contribution, 2),
        "rate_contribution": round(rate_contribution, 2),
        "reconciliation_error": round(reconciliation_error, 3),
    }


def summarize_error_decomposition(rows: list[dict]) -> dict:
    decomposed = [d for d in (decompose_error(r) for r in rows) if d is not None]
    n = len(decomposed)
    if n == 0:
        return {"n": 0, "avg_workload_contribution": None, "avg_rate_contribution": None,
                "biggest_underprojections": [], "biggest_overprojections": [],
                "largest_workload_misses": [], "largest_rate_misses": []}

    avg_workload = sum(d["workload_contribution"] for d in decomposed) / n
    avg_rate = sum(d["rate_contribution"] for d in decomposed) / n

    by_miss = sorted(decomposed, key=lambda d: d["total_miss"])
    by_workload = sorted(decomposed, key=lambda d: abs(d["workload_contribution"]), reverse=True)
    by_rate = sorted(decomposed, key=lambda d: abs(d["rate_contribution"]), reverse=True)

    return {
        "n": n,
        "avg_workload_contribution": round(avg_workload, 3),
        "avg_rate_contribution": round(avg_rate, 3),
        "biggest_underprojections": by_miss[-5:][::-1],
        "biggest_overprojections": by_miss[:5],
        "largest_workload_misses": by_workload[:5],
        "largest_rate_misses": by_rate[:5],
    }
