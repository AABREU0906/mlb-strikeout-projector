"""
Converts stored Projection rows (which already snapshot every pregame input
used at the time) into flat feature vectors/targets for ML training. Only
fields that were captured pregame are used -- since Projection rows store
inputs as they existed before the game, this is leakage-safe by
construction as long as callers only pass rows with `created_at_utc`
strictly before the game's actual start (enforced by the caller).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from app.database.models import Projection

FEATURE_NAMES = [
    "pitcher_k_rate_season",
    "pitcher_k_rate_vs_hand_avg",
    "pitcher_bb_rate_season",
    "avg_innings_per_start",
    "avg_bf_per_start",
    "lineup_avg_k_rate",
    "lineup_is_confirmed",
    "expected_innings",
    "expected_batters_faced",
    "ballpark_k_factor_proxy",
    "workload_confidence_penalty",
    "news_warning_penalty",
]

MARKET_FEATURE_NAMES = FEATURE_NAMES + [
    "market_strikeout_line",
    "market_line_vs_stats_only_diff",
    "market_opponent_implied_runs",
    "market_game_total",
]


def extract_features(p: Projection, include_market: bool) -> Optional[np.ndarray]:
    pitcher = p.pitcher_inputs_json or {}
    batters = p.batter_inputs_json or []
    workload = p.workload_inputs_json or {}
    confidence_factors = p.confidence_factors_json or {}

    k_rate_season = ((pitcher.get("k_rate_season") or {}).get("shrunk_rate"))
    k_vs_r = ((pitcher.get("k_rate_vs_rhb") or {}).get("shrunk_rate"))
    k_vs_l = ((pitcher.get("k_rate_vs_lhb") or {}).get("shrunk_rate"))
    k_vs_hand_vals = [v for v in (k_vs_r, k_vs_l) if v is not None]
    k_vs_hand_avg = sum(k_vs_hand_vals) / len(k_vs_hand_vals) if k_vs_hand_vals else k_rate_season

    bb_rate = ((pitcher.get("bb_rate_season") or {}).get("shrunk_rate"))
    avg_ip = pitcher.get("avg_innings_per_start")
    avg_bf = pitcher.get("avg_bf_per_start")

    lineup_k_rates = []
    for b in batters:
        kr = (b.get("k_rate_overall") or {}).get("shrunk_rate")
        if kr is not None:
            lineup_k_rates.append(kr)
    lineup_avg_k = sum(lineup_k_rates) / len(lineup_k_rates) if lineup_k_rates else None

    required = [k_rate_season, k_vs_hand_avg, bb_rate, avg_ip, avg_bf, lineup_avg_k, p.expected_innings, p.expected_batters_faced]
    if any(v is None for v in required):
        return None

    features = [
        k_rate_season,
        k_vs_hand_avg,
        bb_rate,
        avg_ip,
        avg_bf,
        lineup_avg_k,
        1.0 if p.lineup_status == "confirmed" else 0.0,
        p.expected_innings,
        p.expected_batters_faced,
        1.0,  # ballpark_k_factor_proxy placeholder (already baked into expected_innings/probabilities upstream)
        workload.get("workload_confidence_penalty", 0.0) or 0.0,
        confidence_factors.get("injury_news_uncertainty", 0.0) or 0.0,
    ]

    if include_market:
        snap = p.market_snapshot_json or {}
        line = snap.get("strikeout_line")
        if line is None:
            return None
        stats_only = p.statistics_only_projection or 0.0
        features += [
            line,
            line - stats_only,
            snap.get("opponent_implied_runs") or 0.0,
            snap.get("game_total") or 0.0,
        ]

    return np.array(features, dtype=float)


def build_training_matrix(projections: list[Projection], include_market: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, y = [], []
    for p in projections:
        if not p.actual_result or p.actual_result.actual_strikeouts is None:
            continue
        feats = extract_features(p, include_market=include_market)
        if feats is None:
            continue
        X.append(feats)
        y.append(p.actual_result.actual_strikeouts)
    names = MARKET_FEATURE_NAMES if include_market else FEATURE_NAMES
    return (np.array(X), np.array(y), names) if X else (np.empty((0, len(names))), np.empty((0,)), names)
