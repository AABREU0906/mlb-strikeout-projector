"""
Stage 3: Lineup Simulation (expected-value layer; the full stochastic
simulation happens per-iteration in Stage 5's Monte Carlo engine, which
consumes this stage's per-batter PA-cycling logic each iteration).

Given an expected number of batters faced (a float from Stage 1) and a
9-spot batting order, batters earlier in the order are expected to bat one
extra time before those later in the order. This function returns the
expected plate-appearance count per lineup spot along with a simple
integer "PA cycling" generator that Stage 5 uses to walk through the
lineup slot-by-slot for exactly N plate appearances in one simulated game.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LineupSpotExposure:
    batting_order: int
    expected_pa: float
    availability_multiplier: float  # <1.0 reduces exposure for pinch-hit/platoon risk


def expected_pa_by_spot(
    expected_batters_faced: float,
    lineup_size: int = 9,
    availability_multipliers: dict[int, float] | None = None,
) -> list[LineupSpotExposure]:
    availability_multipliers = availability_multipliers or {}
    times_through = expected_batters_faced / lineup_size
    base_pa = int(times_through)
    fractional = times_through - base_pa
    extra_spots = round(fractional * lineup_size)

    results = []
    for spot in range(1, lineup_size + 1):
        pa = base_pa + (1 if spot <= extra_spots else 0)
        mult = availability_multipliers.get(spot, 1.0)
        results.append(LineupSpotExposure(batting_order=spot, expected_pa=pa * mult, availability_multiplier=mult))
    return results


def pa_cycle_order(lineup_size: int, n_plate_appearances: int, start_spot: int = 1) -> list[int]:
    """Returns the sequence of batting-order spots (1-indexed) that bat,
    in order, for n_plate_appearances starting at start_spot and cycling."""
    order = []
    spot = start_spot
    for _ in range(n_plate_appearances):
        order.append(spot)
        spot = spot + 1 if spot < lineup_size else 1
    return order
