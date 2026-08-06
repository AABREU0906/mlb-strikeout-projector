"""
Stage 5: Monte Carlo Simulation.

Each iteration:
  1. Samples innings pitched from a distribution centered on the Stage 1
     workload estimate (truncated normal, spread widened by the workload
     confidence penalty) -- this captures early-exit / extended-outing
     uncertainty rather than assuming a fixed workload.
  2. Converts sampled innings to an integer plate-appearance count using
     the pitcher's own batters-faced-per-inning ratio (from Stage 1),
     plus Poisson noise on top to capture game-to-game BF variance at a
     fixed IP level (walks/hits extending innings).
  3. Cycles through the batting order for exactly that many plate
     appearances (Stage 3's pa_cycle_order), and for each plate
     appearance draws a Bernoulli strikeout outcome using that lineup
     spot's Stage 2 matchup probability.
  4. Sums strikeouts for the simulated game.

Reproducibility: pass an explicit `seed` to get identical results across
runs (used by backtesting so historical projections can be exactly
reproduced).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.projections.stage3_lineup_simulation import pa_cycle_order

MAX_STRIKEOUT_BUCKET = 15  # "15 or more" is the final displayed bucket


@dataclass
class SimulationResult:
    n_simulations: int
    raw_strikeouts: np.ndarray  # shape (n_simulations,)
    mean: float
    median: float
    std_dev: float
    percentiles: dict[int, float]
    probability_by_k: dict[int, float]  # 0..14 exact, 15 = "15 or more"
    most_likely_k: int


def run_monte_carlo(
    expected_innings: float,
    workload_spread: float,  # innings stdev, derived from workload confidence penalty
    bf_per_inning: float,
    batter_probabilities: list[float],  # length = lineup_size, indexed by spot-1
    n_simulations: int = 25000,
    seed: int | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    lineup_size = len(batter_probabilities)
    probs_array = np.array(batter_probabilities, dtype=float)

    # 1. Sample innings pitched (truncated normal, floor at 0.1, cap at 9.0)
    innings_samples = rng.normal(loc=expected_innings, scale=max(workload_spread, 0.15), size=n_simulations)
    innings_samples = np.clip(innings_samples, 0.1, 9.0)

    # 2. Convert to batters faced with added Poisson-ish noise.
    base_bf = innings_samples * bf_per_inning
    bf_noise = rng.normal(loc=0.0, scale=np.sqrt(np.maximum(base_bf, 1.0)) * 0.15, size=n_simulations)
    bf_samples = np.clip(np.round(base_bf + bf_noise), 1, 50).astype(int)

    strikeout_totals = np.zeros(n_simulations, dtype=int)

    # Vectorized-ish loop: batch by unique BF values to reduce Python overhead.
    unique_bfs, inverse_idx = np.unique(bf_samples, return_inverse=True)
    for u_i, bf in enumerate(unique_bfs):
        mask = inverse_idx == u_i
        count = int(mask.sum())
        if count == 0 or bf <= 0:
            continue
        spot_sequence = pa_cycle_order(lineup_size, int(bf))
        spot_probs = probs_array[[s - 1 for s in spot_sequence]]  # shape (bf,)
        # Draw (count, bf) Bernoulli outcomes at once.
        draws = rng.random((count, len(spot_probs))) < spot_probs[np.newaxis, :]
        game_strikeouts = draws.sum(axis=1)
        strikeout_totals[mask] = game_strikeouts

    mean = float(np.mean(strikeout_totals))
    median = float(np.median(strikeout_totals))
    std_dev = float(np.std(strikeout_totals))

    percentiles = {
        p: float(np.percentile(strikeout_totals, p)) for p in (10, 25, 50, 75, 90)
    }

    capped = np.minimum(strikeout_totals, MAX_STRIKEOUT_BUCKET)
    counts = np.bincount(capped, minlength=MAX_STRIKEOUT_BUCKET + 1)
    probability_by_k = {k: float(counts[k] / n_simulations) for k in range(MAX_STRIKEOUT_BUCKET + 1)}
    most_likely_k = int(np.argmax(counts))

    return SimulationResult(
        n_simulations=n_simulations,
        raw_strikeouts=strikeout_totals,
        mean=round(mean, 3),
        median=median,
        std_dev=round(std_dev, 3),
        percentiles={k: round(v, 2) for k, v in percentiles.items()},
        probability_by_k=probability_by_k,
        most_likely_k=most_likely_k,
    )
