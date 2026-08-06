# NRFI/YRFI Data Dictionary

## Database tables (new)

### `first_inning_game_results`
Historical ground truth for one completed game's 1st inning. One row per
game, upserted (never duplicated) keyed by `game_id`.

| Column | Type | Meaning |
|---|---|---|
| `game_id` | str (unique) | MLB gamePk |
| `season` | int | |
| `home_team_id` / `away_team_id` | int | |
| `home_starting_pitcher_id` / `away_starting_pitcher_id` | int, nullable | From boxscore |
| `away_first_inning_runs` / `home_first_inning_runs` | int, nullable | From linescore |
| `away_pitcher_scoreless_first` / `home_pitcher_scoreless_first` | bool, nullable | **Personal** scoreless flag; `None` if a mid-1st-inning pitching change makes attribution uncertain |
| `is_nrfi` | bool, nullable | Whole-game 1st-inning outcome (0-0 through 1st) |
| `away_plate_appearances`...`home_total_bases` | int, nullable | Real 1st-inning offensive aggregates parsed from play-by-play |
| `away_pitcher_first_inning_pitches` / `home_pitcher_first_inning_pitches` | int, nullable | |
| `day_night` | str, nullable | `"day"` \| `"night"` |
| `venue_id` | int, nullable | |
| `source` / `retrieved_at_utc` | | Provenance |

### `nrfi_projections`
Pregame snapshot -- every input used, stored for full reproducibility
(mirrors the strikeout model's `projections` table design). Includes
`away_lineup_json`/`home_lineup_json`, per-side pitcher/team input JSON,
BvP input JSON, weather/umpire/context JSON, warnings, market snapshot,
resulting probabilities/Threat Scores/confidence/explanation, and model
version metadata.

### `nrfi_actual_results`
Postgame ground truth, one-to-one with `nrfi_projections`. **Pregame data
is never overwritten** -- this is a separate table, same design principle
as the strikeout model's `actual_results` table.

### `bets` (extended, not replaced)
Added columns: `market_type` (`'strikeouts'` | `'nrfi_yrfi'`, defaults to
`'strikeouts'` for all pre-existing rows), `nrfi_projection_id`,
`actual_nrfi_result`. `strikeout_line` and `pitcher_name` were loosened to
nullable. This was a live-data-preserving migration (see
`app/database/migrations.py`) -- verified against a simulated pre-existing
table to confirm zero data loss before being applied to the real schema.

---

## Feature vocabulary

| Term | Definition |
|---|---|
| **Scoreless-first rate** | Fraction of a pitcher's starts where they personally didn't allow a 1st-inning run |
| **Game NRFI rate in starts** | Fraction of a pitcher's starts where the *whole game* was NRFI (distinct from the above when relief pitching affected the 1st) |
| **Team scoring rate** | Fraction of a team's games where they scored in the 1st (either half, depending on home/away) |
| **Shrunk rate** | A rate after empirical-Bayes regression toward a documented prior; see `reliability` (0-1) for how much weight the observed data carries |
| **BvP hierarchical shrinkage** | 3-level cascade: raw BvP sample -> shrunk toward (already-shrunk) vs-pitcher-hand rate -> shrunk toward (already-shrunk) season rate -> shrunk toward league average |
| **Threat Score** | 0-100 supporting metric; NOT the calibrated probability |
| **Confidence score** | 0-100 data-quality/certainty metric; NOT the prediction strength |

---

## League constants (documented priors, `app/features/nrfi_league_constants.py`)

| Key | Value | Meaning |
|---|---|---|
| `league_scoreless_half_inning_rate` | 0.715 | P(one team's half-inning is scoreless) |
| `league_game_nrfi_rate` | 0.51 | P(whole game is NRFI), documented historical rate |
| `league_first_inning_era` | 4.35 | |
| `league_first_inning_whip` | 1.24 | |
| `league_first_inning_avg/obp/slg/ops` | 0.252/0.325/0.410/0.735 | |
| `league_first_inning_k_pct/bb_pct/hr_rate` | 0.225/0.085/0.028 | |
| `league_first_inning_avg_pitches` | 18.0 | Per starter, per 1st inning |

These are seed values, refreshable as your own backfilled history grows.
They are internally consistent with the model: a league-average-vs-average
matchup in `compute_nrfi_probability()` produces a modeled NRFI rate of
0.5112, matching the 0.51 documented constant almost exactly (verified).

---

## Play-by-play field mapping (defensive, unverified against live data)

`get_first_inning_result()` and its helpers parse:
- `liveData.linescore.innings[0].{away,home}.runs` -- high confidence, stable.
- `liveData.boxscore.teams.{home,away}.pitchers[0]` -- the starting
  pitcher ID convention; moderate-high confidence, standard MLB gumbo
  pattern.
- `liveData.plays.allPlays[].about.{inning,isTopInning}` and
  `.result.eventType` (`single`/`double`/`triple`/`home_run`/`walk`/
  `strikeout`/etc.) and `.playEvents[].isPitch` -- moderate confidence;
  standard MLB gumbo event vocabulary, but not independently verified
  against a live payload in this development environment.
- `playEvents[].details.event` containing `"substitution"` + `"pitch"`
  (case-insensitive) -- used to detect mid-1st-inning pitching changes.
  **Lowest-confidence mapping in the system** -- if the actual event
  string differs, this silently under-detects substitutions (fails safe:
  worst case is treating a personal-scoreless flag as known when it should
  be `None`, not a crash).

If any of these turn out to be wrong against live data, symptoms will be
`missing_fields` warnings or unexpectedly-empty aggregates -- not silent
wrong numbers, because every field defaults to `None`/empty on a parse
miss rather than a fabricated value.
