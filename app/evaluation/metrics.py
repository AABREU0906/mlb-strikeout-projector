"""Pure metric functions used by evaluator.py and backtester.py."""
from __future__ import annotations

import math
from statistics import median
from typing import Optional


def mae(errors: list[float]) -> Optional[float]:
    if not errors:
        return None
    return sum(abs(e) for e in errors) / len(errors)


def rmse(errors: list[float]) -> Optional[float]:
    if not errors:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def medae(errors: list[float]) -> Optional[float]:
    if not errors:
        return None
    return median(abs(e) for e in errors)


def bias(errors: list[float]) -> Optional[float]:
    """Mean signed error: predicted - actual. Positive = overprojecting."""
    if not errors:
        return None
    return sum(errors) / len(errors)


def brier_score(prob_over_predictions: list[float], actual_over_outcomes: list[int]) -> Optional[float]:
    if not prob_over_predictions or len(prob_over_predictions) != len(actual_over_outcomes):
        return None
    n = len(prob_over_predictions)
    return sum((p - o) ** 2 for p, o in zip(prob_over_predictions, actual_over_outcomes)) / n


def log_loss(prob_over_predictions: list[float], actual_over_outcomes: list[int], eps: float = 1e-6) -> Optional[float]:
    if not prob_over_predictions or len(prob_over_predictions) != len(actual_over_outcomes):
        return None
    n = len(prob_over_predictions)
    total = 0.0
    for p, o in zip(prob_over_predictions, actual_over_outcomes):
        p = min(max(p, eps), 1 - eps)
        total += -(o * math.log(p) + (1 - o) * math.log(1 - p))
    return total / n


def calibration_buckets(prob_predictions: list[float], actual_outcomes: list[int], n_buckets: int = 5) -> list[dict]:
    """Groups predictions into probability buckets and reports predicted vs
    observed frequency in each -- the standard reliability-diagram data."""
    if not prob_predictions:
        return []
    buckets = [[] for _ in range(n_buckets)]
    for p, o in zip(prob_predictions, actual_outcomes):
        idx = min(int(p * n_buckets), n_buckets - 1)
        buckets[idx].append((p, o))

    out = []
    for i, bucket in enumerate(buckets):
        lo, hi = i / n_buckets, (i + 1) / n_buckets
        if not bucket:
            out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": 0, "avg_predicted": None, "observed_freq": None})
            continue
        avg_pred = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(o for _, o in bucket) / len(bucket)
        out.append({"range": f"{lo:.1f}-{hi:.1f}", "n": len(bucket), "avg_predicted": round(avg_pred, 3), "observed_freq": round(observed, 3)})
    return out


def over_under_accuracy(predicted_over: list[bool], actual_over: list[bool]) -> Optional[float]:
    if not predicted_over or len(predicted_over) != len(actual_over):
        return None
    correct = sum(1 for p, a in zip(predicted_over, actual_over) if p == a)
    return correct / len(predicted_over)


def roi_flat_stake(bets: list[dict]) -> Optional[dict]:
    """bets: list of {"odds": int American odds, "won": bool}. Returns ROI
    on flat $1 stakes. Only meaningful when real historical odds exist."""
    if not bets:
        return None
    total_staked = len(bets)
    total_return = 0.0
    for b in bets:
        odds = b["odds"]
        if b["won"]:
            total_return += (odds / 100.0) if odds > 0 else (100.0 / (-odds))
        else:
            total_return -= 1.0
    return {"n_bets": len(bets), "total_staked": total_staked, "net_return": round(total_return, 2), "roi_pct": round((total_return / total_staked) * 100, 2)}
