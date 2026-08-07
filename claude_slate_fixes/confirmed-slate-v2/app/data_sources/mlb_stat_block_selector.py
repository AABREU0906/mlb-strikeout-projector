"""
Central stat-block selection for MLB Stats API responses.

ROOT CAUSE OF THE WORKLOAD BUG this module fixes:

The previous per-file `_find_stat_block()` helpers (formerly duplicated in
both pitcher_features.py and batter_features.py) did two unsafe things:

  1. Matched `type.displayName` and `group.displayName` with a loose
     substring check (`stat_type.lower() in block_type`), which is fragile
     against any MLB API naming variance.
  2. When a stat block's `splits` array contained MORE THAN ONE split --
     which genuinely happens for a pitcher who changed teams mid-season,
     or when the API returns split-level entries for different game types
     (regular season vs. postseason vs. spring training) or sport levels
     (MLB vs. a minor-league affiliate) under the same nominal "season"
     type -- the code always took `splits[0]` with no verification of
     which split that actually was.

Because innings/battersFaced/numberOfPitches and gamesStarted were always
read from the SAME split, per-start averages computed from a correctly
matched single split were never literally "impossible" on their own terms.
The impossible per-start values (e.g. 27.9 IP, 126 BF, 531 pitches --
numbers that are internally consistent with each other, just clearly not
"one start's worth") are the signature of `splits[0]` landing on a split
whose `gamesStarted` belonged to a much smaller sub-population (e.g. a
post-trade stint, or a level/game-type other than MLB regular season)
than the counting stats actually represented, OR of a genuinely different
level/game-type split being selected outright.

This module eliminates that failure mode at the source by:
  - Requesting `sportId=1` (MLB) and `gameType=R` (regular season) as
    EXPLICIT parameters on the underlying API call (see mlb_stats_api.py),
    removing ambiguity about level/game-type rather than hoping the API's
    default behavior is what's expected.
  - Using EXACT (not substring) type/group matching.
  - When a block still contains multiple splits after level/game-type
    filtering (the legitimate traded-mid-season case), safely SUMMING the
    counting stats across all remaining splits rather than blindly using
    the first one -- so a traded pitcher's full-season workload is
    correctly represented instead of silently truncated to one stint.
  - Returning a `StatBlockSelection` that records exactly how the block
    was chosen (or why none could be safely chosen), so callers can turn a
    failed/uncertain selection into a visible warning and a confidence
    reduction instead of silently substituting a wrong value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

_SUMMABLE_PITCHING_FIELDS = (
    "gamesStarted", "gamesPitched", "battersFaced", "numberOfPitches",
    "strikeOuts", "baseOnBalls", "hits", "homeRuns", "earnedRuns", "runs",
)
_SUMMABLE_HITTING_FIELDS = (
    "plateAppearances", "atBats", "hits", "doubles", "triples", "homeRuns",
    "strikeOuts", "baseOnBalls", "totalBases", "gamesPlayed",
)


@dataclass
class StatBlockSelection:
    stat: Optional[dict]
    is_aggregated_from_multiple_splits: bool = False
    n_candidate_blocks: int = 0
    n_splits_before_filtering: int = 0
    n_splits_used: int = 0
    rejected_reasons: list[str] = field(default_factory=list)
    selection_notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.stat is not None


def _outs_from_innings_pitched(value: object) -> Optional[int]:
    """Converts MLB's innings-pitched notation ("123.1" = 123 whole
    innings + 1 out, "123.2" = 123 whole innings + 2 outs) into a total
    out count, which is the only safe unit to SUM across multiple splits.
    Rejects any fractional part other than 0/1/2 (an invalid/unexpected
    notation) rather than guessing."""
    if value is None:
        return None
    try:
        whole, _, frac = str(value).partition(".")
        whole_innings = int(whole)
        outs_fraction = int(frac) if frac else 0
        if outs_fraction not in (0, 1, 2):
            return None
        return whole_innings * 3 + outs_fraction
    except (ValueError, TypeError):
        return None


def _outs_to_innings_pitched_str(outs: int) -> str:
    whole = outs // 3
    remainder = outs % 3
    return f"{whole}.{remainder}"


def _split_matches_mlb_regular_season(split: dict, season: int) -> tuple[bool, Optional[str]]:
    """Returns (matches, rejection_reason). A split is accepted only when
    every level/game-type field IT ACTUALLY PROVIDES is consistent with
    MLB regular season for the requested year; a field that is simply
    absent from the split is not, by itself, grounds for rejection (the
    API does not always echo every filter back on every split), but every
    field that IS present must check out."""
    sport = split.get("sport") or {}
    sport_id = sport.get("id")
    if sport_id is not None and sport_id != 1:
        return False, f"split.sport.id={sport_id} (not MLB)"

    league = split.get("league") or {}
    league_name = (league.get("name") or "").lower()
    if league_name and "major league" not in league_name and league_name not in ("al", "nl", "american league", "national league"):
        return False, f"split.league.name={league.get('name')!r} (not a recognized MLB league)"

    game_type = split.get("gameType")
    if game_type is not None and game_type != "R":
        return False, f"split.gameType={game_type!r} (not regular season)"

    split_season = split.get("season")
    if split_season is not None and str(split_season) != str(season):
        return False, f"split.season={split_season!r} (requested {season})"

    return True, None


def select_stat_block(
    stats: list[dict],
    stat_type: str,
    group: str,
    season: int,
    summable_fields: Optional[tuple[str, ...]] = None,
) -> StatBlockSelection:
    """The single, centrally-tested replacement for the old
    `_find_stat_block()` helpers. `summable_fields` defaults based on
    `group` ("pitching" -> pitching fields, "hitting" -> hitting fields)."""
    if summable_fields is None:
        summable_fields = _SUMMABLE_PITCHING_FIELDS if group == "pitching" else _SUMMABLE_HITTING_FIELDS

    selection = StatBlockSelection(stat=None)

    matching_blocks = [
        block
        for block in stats
        if (block.get("type", {}) or {}).get("displayName", "").strip().lower() == stat_type.strip().lower()
        and (block.get("group", {}) or {}).get("displayName", "").strip().lower() == group.strip().lower()
    ]
    selection.n_candidate_blocks = len(matching_blocks)

    if not matching_blocks:
        selection.rejected_reasons.append(
            f"No stat block found with exact type='{stat_type}', group='{group}'."
        )
        return selection

    all_splits: list[dict] = []
    for block in matching_blocks:
        all_splits.extend(block.get("splits", []) or [])
    selection.n_splits_before_filtering = len(all_splits)

    if not all_splits:
        selection.rejected_reasons.append(f"Matched block(s) for '{stat_type}'/'{group}' had no splits.")
        return selection

    accepted_splits = []
    for split in all_splits:
        ok, reason = _split_matches_mlb_regular_season(split, season)
        if ok:
            accepted_splits.append(split)
        else:
            selection.rejected_reasons.append(f"Rejected a split: {reason}")

    if not accepted_splits:
        selection.rejected_reasons.append(
            "Every split was rejected as non-MLB-regular-season; refusing to guess."
        )
        return selection

    selection.n_splits_used = len(accepted_splits)

    if len(accepted_splits) == 1:
        selection.stat = dict(accepted_splits[0].get("stat", {}) or {})
        selection.selection_notes.append("Single unambiguous MLB regular-season split used.")
        return selection

    combined = next((s for s in accepted_splits if not s.get("team")), None)
    if combined is not None:
        selection.stat = dict(combined.get("stat", {}) or {})
        selection.selection_notes.append(
            f"Multiple splits found ({len(accepted_splits)}); used the API-provided combined total."
        )
        return selection

    aggregated: dict = {}
    for field_name in summable_fields:
        total = None
        for split in accepted_splits:
            raw = (split.get("stat", {}) or {}).get(field_name)
            if raw is None:
                continue
            try:
                numeric = float(raw)
            except (ValueError, TypeError):
                continue
            total = (total or 0.0) + numeric
        if total is not None:
            aggregated[field_name] = int(total) if float(total).is_integer() else total

    total_outs = 0
    any_innings = False
    for split in accepted_splits:
        outs = _outs_from_innings_pitched((split.get("stat", {}) or {}).get("inningsPitched"))
        if outs is not None:
            total_outs += outs
            any_innings = True
    if any_innings:
        aggregated["inningsPitched"] = _outs_to_innings_pitched_str(total_outs)

    selection.stat = aggregated
    selection.is_aggregated_from_multiple_splits = True
    team_names = [s.get("team", {}).get("name", "?") for s in accepted_splits]
    selection.selection_notes.append(
        f"Aggregated {len(accepted_splits)} MLB regular-season splits (likely mid-season trade): "
        f"{', '.join(team_names)}."
    )
    return selection
