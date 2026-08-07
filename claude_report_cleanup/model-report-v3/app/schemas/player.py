from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SampleStat(BaseModel):
    """A single rate stat with its sample size and shrinkage applied."""

    observed_rate: Optional[float] = None
    observed_n: Optional[float] = None
    shrunk_rate: Optional[float] = None
    reliability: Optional[float] = None
    is_small_sample: bool = False


class PitcherProfile(BaseModel):
    player_id: int
    name: str
    throws: Optional[str] = None  # 'R' | 'L'

    season_bf: Optional[int] = None
    career_bf: Optional[int] = None

    k_rate_season: Optional[SampleStat] = None
    k_rate_career: Optional[SampleStat] = None
    k_per_9: Optional[float] = None
    k_per_bf: Optional[float] = None
    k_per_start: Optional[float] = None

    k_rate_vs_rhb: Optional[SampleStat] = None
    k_rate_vs_lhb: Optional[SampleStat] = None
    bb_rate_vs_rhb: Optional[SampleStat] = None
    bb_rate_vs_lhb: Optional[SampleStat] = None
    bf_vs_rhb: Optional[int] = None
    bf_vs_lhb: Optional[int] = None

    swstr_rate: Optional[SampleStat] = None
    chase_rate_induced: Optional[SampleStat] = None
    contact_rate_allowed: Optional[SampleStat] = None
    zone_contact_rate_allowed: Optional[SampleStat] = None
    first_pitch_strike_rate: Optional[SampleStat] = None
    bb_rate_season: Optional[SampleStat] = None

    avg_innings_per_start: Optional[float] = None
    avg_bf_per_start: Optional[float] = None
    avg_pitches_per_start: Optional[float] = None
    recent_pitch_counts: list[int] = []
    recent_innings: list[float] = []
    recent_bf: list[int] = []
    recent_strikeouts: list[int] = []

    # Role-aware workload metadata (see app/features/pitcher_role_workload.py).
    # These describe HOW avg_innings_per_start/avg_bf_per_start/
    # avg_pitches_per_start above were derived, so downstream consumers
    # (WorkloadEstimate, confidence scoring, display) never have to guess.
    games_pitched: Optional[int] = None
    games_started: Optional[int] = None
    start_ratio: Optional[float] = None
    workload_role: Optional[str] = None  # "starter" | "reliever" | "swingman" | "unknown"
    workload_source: Optional[str] = None  # "mlb_season_totals" | "mlb_recent_starts" | "mlb_season_starts_only" | "mlb_previous_season_starts" | "unresolved"
    workload_source_level: Optional[str] = None  # "MLB" | "unavailable"
    start_specific_sample_size: int = 0
    workload_data_valid: bool = True
    workload_fallback_used: bool = False
    workload_reasons: list[str] = []

    rest_days: Optional[int] = None
    short_rest: bool = False
    extra_rest: bool = False
    recent_skipped_start: bool = False
    recent_rehab_assignment: bool = False
    opener_status: bool = False
    tandem_risk: bool = False

    data_completeness: float = 1.0  # fraction of expected fields populated
    missing_fields: list[str] = []


class BatterProfile(BaseModel):
    player_id: int
    name: str
    batting_order: Optional[int] = None
    bat_side: Optional[str] = None  # 'R' | 'L' | 'S' (switch)
    is_switch_hitter: bool = False
    expected_side_today: Optional[str] = None  # resolved side vs today's pitcher hand

    season_pa: Optional[int] = None
    career_pa: Optional[int] = None

    k_rate_overall: Optional[SampleStat] = None
    k_rate_career: Optional[SampleStat] = None
    k_rate_vs_rhp: Optional[SampleStat] = None
    k_rate_vs_lhp: Optional[SampleStat] = None
    pa_vs_rhp: Optional[int] = None
    pa_vs_lhp: Optional[int] = None

    k_rate_last_7d: Optional[float] = None
    k_rate_last_14d: Optional[float] = None
    k_rate_last_30d: Optional[float] = None

    contact_rate: Optional[SampleStat] = None
    zone_contact_rate: Optional[SampleStat] = None
    chase_rate: Optional[SampleStat] = None
    swstr_rate: Optional[SampleStat] = None
    bb_rate: Optional[SampleStat] = None

    expected_pa: Optional[float] = None

    injury_status: Optional[str] = None
    recent_il_activation: bool = False
    recent_missed_games: int = 0
    playing_time_limitation: Optional[str] = None
    pinch_hit_risk: bool = False
    platoon_substitution_risk: bool = False

    data_completeness: float = 1.0
    missing_fields: list[str] = []


class TeamProfile(BaseModel):
    team_id: int
    team_name: str

    k_rate_overall: Optional[float] = None
    k_rate_vs_rhp: Optional[float] = None
    k_rate_vs_lhp: Optional[float] = None
    k_rate_last_7d: Optional[float] = None
    k_rate_last_14d: Optional[float] = None
    k_rate_last_30d: Optional[float] = None
    bb_rate: Optional[float] = None
    contact_rate: Optional[float] = None
    chase_rate: Optional[float] = None
    swstr_rate: Optional[float] = None
    pa_per_game: Optional[float] = None
    runs_per_game: Optional[float] = None
    wrc_plus_or_similar: Optional[float] = None
    home_road_split_note: Optional[str] = None
