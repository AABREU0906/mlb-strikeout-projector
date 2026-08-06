"""
Pure calculation functions over first-inning "start records" -- plain dicts,
not ORM objects, so this module has zero database/pydantic dependency and
can be fully unit tested in isolation. The feature builders
(nrfi_pitcher_features.py, nrfi_team_features.py) are responsible for
converting FirstInningGameResult rows into these plain records; everything
below is pure aggregation math.

A pitcher "start record" dict has keys:
  game_date, season, is_home, day_night, scoreless (bool|None),
  game_is_nrfi (bool|None), runs_allowed, hits_allowed, walks_allowed,
  strikeouts, home_runs_allowed, at_bats_faced, total_bases_allowed,
  plate_appearances_faced, pitches_thrown (any may be None if unknown --
  never fabricated).

A team "game record" dict has keys:
  game_date, season, is_home, day_night, runs_scored (int|None),
  scored (bool|None), hits, walks, strikeouts, home_runs, at_bats,
  total_bases, plate_appearances.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateResult:
    made_rate: Optional[float]
    n: int


def rate_of(records: list[dict], flag_key: str) -> RateResult:
    """Fraction of records where records[i][flag_key] is True, counting
    only records where the flag is not None (unknown records are excluded
    from both numerator and denominator, never treated as 0 or False)."""
    known = [r for r in records if r.get(flag_key) is not None]
    if not known:
        return RateResult(made_rate=None, n=0)
    made = sum(1 for r in known if r[flag_key])
    return RateResult(made_rate=made / len(known), n=len(known))


def cascading_shrinkage(
    league_rate: float,
    season_rate: Optional[float], season_n: Optional[float], season_stabilization_n: float,
    vs_hand_rate: Optional[float], vs_hand_n: Optional[float], vs_hand_stabilization_n: float,
    bvp_rate: Optional[float], bvp_n: Optional[float], bvp_stabilization_n: float,
):
    """Three-level hierarchical (cascading) shrinkage, used for
    batter-vs-pitcher regression per project spec: 'Blend small-sample
    batter-versus-pitcher results toward (1) batter performance versus the
    pitcher's throwing arm, (2) batter season baseline, (3) league average.'

    Each level's OWN prior is the *already-shrunk* result of the level
    below it, not the raw league constant directly -- this is standard
    multilevel/hierarchical empirical Bayes and ensures a batter with a
    small vs-hand sample still gets pulled toward season, which itself is
    pulled toward league, before that combined prior is used to regress the
    (usually tiny) BvP sample. Returns (final_rate, level_detail_dict).
    """
    from app.features.shrinkage import shrink_rate

    # Level 3 (deepest fallback): season rate shrunk toward league average.
    if season_rate is not None and season_n:
        season_result = shrink_rate(season_rate, season_n, league_rate, season_stabilization_n)
        season_prior = season_result.shrunk_rate
    else:
        season_prior = league_rate

    # Level 2: vs-pitcher-hand rate shrunk toward the (already-shrunk) season baseline.
    if vs_hand_rate is not None and vs_hand_n:
        vs_hand_result = shrink_rate(vs_hand_rate, vs_hand_n, season_prior, vs_hand_stabilization_n)
        hand_adjusted_prior = vs_hand_result.shrunk_rate
    else:
        hand_adjusted_prior = season_prior

    # Level 1 (shallowest, usually tiny sample): raw BvP shrunk toward the
    # hand-adjusted prior. BvP samples like "2-for-3" get pulled hard here.
    if bvp_rate is not None and bvp_n:
        bvp_result = shrink_rate(bvp_rate, bvp_n, hand_adjusted_prior, bvp_stabilization_n)
        final_rate = bvp_result.shrunk_rate
        final_reliability = bvp_result.reliability
    else:
        final_rate = hand_adjusted_prior
        final_reliability = 0.0

    return final_rate, {
        "league_rate": league_rate,
        "season_prior_after_shrinkage": round(season_prior, 4),
        "hand_adjusted_prior_after_shrinkage": round(hand_adjusted_prior, 4),
        "final_rate": round(final_rate, 4),
        "bvp_reliability": round(final_reliability, 4),
        "bvp_n": bvp_n or 0,
    }


def to_shrunk_rate(rate_result: "RateResult", prior: float, stabilization_n: float):
    """Applies shrink_rate() to a RateResult, handling the zero-sample case
    explicitly (falls back to a pure-prior ShrinkageResult rather than
    dividing by zero). Returns an app.features.shrinkage.ShrinkageResult;
    schema conversion (to the pydantic ShrunkRate) is the caller's job so
    this module stays free of pydantic/schema dependencies."""
    from app.features.shrinkage import shrink_rate

    if rate_result.made_rate is None:
        return shrink_rate(0.0, 0, prior, stabilization_n)
    return shrink_rate(rate_result.made_rate, rate_result.n, prior, stabilization_n)


def last_n(records: list[dict], n: int) -> list[dict]:
    ordered = sorted(records, key=lambda r: r["game_date"], reverse=True)
    return ordered[:n]


def split_home_away(records: list[dict]) -> tuple[list[dict], list[dict]]:
    home = [r for r in records if r.get("is_home") is True]
    away = [r for r in records if r.get("is_home") is False]
    return home, away


def split_day_night(records: list[dict]) -> tuple[list[dict], list[dict]]:
    day = [r for r in records if r.get("day_night") == "day"]
    night = [r for r in records if r.get("day_night") == "night"]
    return day, night


def split_season(records: list[dict], season: int) -> tuple[list[dict], list[dict]]:
    this_season = [r for r in records if r.get("season") == season]
    other_seasons = [r for r in records if r.get("season") != season]
    return this_season, other_seasons


@dataclass
class FirstInningSlashLine:
    n_starts_with_data: int
    era: Optional[float] = None   # runs allowed per 9 "innings" (1 first-inning = 1/9)
    whip: Optional[float] = None  # (H+BB) per first inning, averaged
    avg: Optional[float] = None   # batting average allowed
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None
    k_pct: Optional[float] = None
    bb_pct: Optional[float] = None
    hr_rate: Optional[float] = None  # HR per PA faced
    avg_pitches: Optional[float] = None


def compute_slash_line(records: list[dict], runs_key: str, hits_key: str, walks_key: str,
                        k_key: str, hr_key: str, ab_key: str, tb_key: str, pa_key: str,
                        pitches_key: str) -> FirstInningSlashLine:
    """Generic slash-line aggregator; the caller supplies which dict keys
    hold runs/hits/walks/etc. so this same function serves both the
    pitcher-allowed view and the team-hitting view without duplication."""
    usable = [r for r in records if r.get(pa_key) is not None]
    if not usable:
        return FirstInningSlashLine(n_starts_with_data=0)

    total_runs = sum(r.get(runs_key) or 0 for r in usable if r.get(runs_key) is not None)
    n_runs_known = sum(1 for r in usable if r.get(runs_key) is not None)
    total_hits = sum(r.get(hits_key) or 0 for r in usable)
    total_walks = sum(r.get(walks_key) or 0 for r in usable)
    total_k = sum(r.get(k_key) or 0 for r in usable)
    total_hr = sum(r.get(hr_key) or 0 for r in usable)
    total_ab = sum(r.get(ab_key) or 0 for r in usable)
    total_tb = sum(r.get(tb_key) or 0 for r in usable)
    total_pa = sum(r.get(pa_key) or 0 for r in usable)
    pitches = [r.get(pitches_key) for r in usable if r.get(pitches_key) is not None]

    era = (total_runs / n_runs_known) * 9 if n_runs_known else None
    whip = (total_hits + total_walks) / n_runs_known if n_runs_known else None
    avg = (total_hits / total_ab) if total_ab else None
    obp = ((total_hits + total_walks) / (total_ab + total_walks)) if (total_ab + total_walks) else None
    slg = (total_tb / total_ab) if total_ab else None
    ops = (obp + slg) if (obp is not None and slg is not None) else None
    k_pct = (total_k / total_pa) if total_pa else None
    bb_pct = (total_walks / total_pa) if total_pa else None
    hr_rate = (total_hr / total_pa) if total_pa else None
    avg_pitches = (sum(pitches) / len(pitches)) if pitches else None

    return FirstInningSlashLine(
        n_starts_with_data=len(usable),
        era=round(era, 3) if era is not None else None,
        whip=round(whip, 3) if whip is not None else None,
        avg=round(avg, 3) if avg is not None else None,
        obp=round(obp, 3) if obp is not None else None,
        slg=round(slg, 3) if slg is not None else None,
        ops=round(ops, 3) if ops is not None else None,
        k_pct=round(k_pct, 3) if k_pct is not None else None,
        bb_pct=round(bb_pct, 3) if bb_pct is not None else None,
        hr_rate=round(hr_rate, 4) if hr_rate is not None else None,
        avg_pitches=round(avg_pitches, 1) if avg_pitches is not None else None,
    )
