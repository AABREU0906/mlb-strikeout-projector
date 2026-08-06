"""
Shared plausibility bounds for pitcher workload values.

These constants are the SINGLE source of truth for "what does a realistic
single MLB start look like" -- both `app/projections/stage1_workload.py`
(which builds the workload estimate) and
`app/validation/projection_validator.py` (which re-checks the final
projection before anything is displayed) import from here, so the two
layers of defense can never silently drift apart from each other.
"""
from __future__ import annotations

MIN_INNINGS_PER_START = 0.1
MAX_INNINGS_PER_START = 9.0

MIN_BATTERS_FACED_PER_START = 3.0
MAX_BATTERS_FACED_PER_START = 45.0

MIN_PITCHES_PER_START = 10.0
MAX_PITCHES_PER_START = 130.0

# Batters faced must scale with innings pitched within this band (a 9-out
# inning realistically sees between ~3 and ~6.5 batters).
MIN_BATTERS_FACED_PER_INNING = 3.0
MAX_BATTERS_FACED_PER_INNING = 6.5

# Pitches thrown must scale with batters faced within this band.
MIN_PITCHES_PER_BATTER_FACED = 2.5
MAX_PITCHES_PER_BATTER_FACED = 6.0

# A per-start average computed from season totals that falls outside these
# (looser) bounds indicates the underlying data selection was wrong, not
# just an unusual outing -- see app/data_sources/mlb_stat_block_selector.py
# for the root-cause fix this backs up.
MIN_PLAUSIBLE_AVG_INNINGS_PER_START = 0.5
MAX_PLAUSIBLE_AVG_INNINGS_PER_START = 9.0
MIN_PLAUSIBLE_AVG_BF_PER_START = 3.0
MAX_PLAUSIBLE_AVG_BF_PER_START = 45.0
MIN_PLAUSIBLE_AVG_PITCHES_PER_START = 10.0
MAX_PLAUSIBLE_AVG_PITCHES_PER_START = 130.0
