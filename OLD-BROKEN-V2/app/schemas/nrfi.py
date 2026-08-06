from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ShrunkRate(BaseModel):
    observed_rate: Optional[float] = None
    observed_n: int = 0
    shrunk_rate: Optional[float] = None
    reliability: float = 0.0
    is_small_sample: bool = True

    @classmethod
    def from_rate_result(cls, rate_result, prior: float, stabilization_n: float) -> "ShrunkRate":
        """Shared conversion from a pure app.features.nrfi_rate_calculations.RateResult
        into this schema, used identically by the pitcher and team feature
        builders so neither duplicates the wrapping logic."""
        from app.features.nrfi_rate_calculations import to_shrunk_rate

        result = to_shrunk_rate(rate_result, prior, stabilization_n)
        return cls(
            observed_rate=result.observed_rate if rate_result.made_rate is not None else None,
            observed_n=int(result.observed_n),
            shrunk_rate=result.shrunk_rate,
            reliability=result.reliability,
            is_small_sample=result.is_small_sample,
        )


class FirstInningSlashLineSchema(BaseModel):
    n_starts_with_data: int = 0
    era: Optional[float] = None
    whip: Optional[float] = None
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None
    k_pct: Optional[float] = None
    bb_pct: Optional[float] = None
    hr_rate: Optional[float] = None
    avg_pitches: Optional[float] = None


class PitcherFirstInningProfile(BaseModel):
    pitcher_id: int
    name: str
    throws: Optional[str] = None

    career_scoreless_rate: ShrunkRate
    season_scoreless_rate: ShrunkRate
    previous_season_scoreless_rate: ShrunkRate
    last_5_scoreless_rate: ShrunkRate
    last_10_scoreless_rate: ShrunkRate
    last_20_scoreless_rate: ShrunkRate
    home_scoreless_rate: ShrunkRate
    away_scoreless_rate: ShrunkRate
    day_scoreless_rate: ShrunkRate
    night_scoreless_rate: ShrunkRate

    # Distinct from the above: whether the WHOLE GAME was NRFI during this
    # pitcher's starts (differs from personal-scoreless when relief/openers
    # affected the 1st inning after the starter left).
    game_nrfi_rate_in_starts: ShrunkRate

    season_slash_line: FirstInningSlashLineSchema
    career_slash_line: FirstInningSlashLineSchema

    days_of_rest: Optional[int] = None
    previous_start_pitch_count: Optional[int] = None
    recent_velocity_change: Optional[float] = None  # None unless a velocity data source is wired

    injury_warning: bool = False
    opener_risk: bool = False
    pitch_limit_warning: bool = False

    data_completeness: float = 1.0
    missing_fields: list[str] = []


class BvPFactor(BaseModel):
    """One BvP metric (AVG, OBP, or SLG) with the full hierarchical
    shrinkage chain shown, per spec requirement to 'display the raw sample
    size and adjusted value' rather than hiding the regression."""

    raw_bvp_value: Optional[float] = None
    raw_bvp_n: int = 0
    season_prior: Optional[float] = None
    vs_hand_prior: Optional[float] = None
    final_adjusted_value: float
    reliability: float = 0.0


class BvPProfile(BaseModel):
    batter_id: int
    pitcher_id: int
    batter_name: str

    plate_appearances: int = 0
    at_bats: int = 0
    hits: int = 0
    singles: Optional[int] = None
    doubles: Optional[int] = None
    triples: Optional[int] = None
    home_runs: Optional[int] = None
    walks: int = 0
    strikeouts: Optional[int] = None

    avg: BvPFactor
    obp: BvPFactor
    slg: BvPFactor
    ops_adjusted: float

    exit_velocity: Optional[float] = None  # not wired -- requires Statcast/Baseball Savant, never fabricated
    hard_hit_pct: Optional[float] = None    # not wired -- same reason

    data_completeness: float = 1.0
    missing_fields: list[str] = []


class TeamFirstInningProfile(BaseModel):
    team_id: int
    team_name: str

    season_scoring_rate: ShrunkRate
    previous_season_scoring_rate: ShrunkRate
    last_5_scoring_rate: ShrunkRate
    last_10_scoring_rate: ShrunkRate
    last_20_scoring_rate: ShrunkRate
    last_30_scoring_rate: ShrunkRate
    home_scoring_rate: ShrunkRate
    away_scoring_rate: ShrunkRate
    day_scoring_rate: ShrunkRate
    night_scoring_rate: ShrunkRate

    season_slash_line: FirstInningSlashLineSchema

    avg_first_inning_runs: Optional[float] = None
    leadoff_reach_rate: Optional[float] = None  # None unless wired to PA-level leadoff tracking
    team_nrfi_record: Optional[str] = None  # e.g. "42-38" (games/appearances, not W-L)
    team_yrfi_record: Optional[str] = None

    data_completeness: float = 1.0
    missing_fields: list[str] = []
