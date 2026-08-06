# NRFI/YRFI Model Methodology

## What NRFI and YRFI mean

**NRFI** = No Run First Inning: neither team scores in the top or bottom of
the 1st inning (the game is 0-0 after one full inning).

**YRFI** = Yes Run First Inning: at least one team scores in the 1st
inning. YRFI is simply `1 - P(NRFI)`.

These are among the most popular MLB betting markets because the 1st
inning is a relatively self-contained, well-defined event with two known
starting pitchers and (usually) known lineups.

---

## Data sources

| Data | Source | Notes |
|---|---|---|
| First-inning runs, starting pitchers, play-by-play | MLB Stats API (`liveData.linescore`, `liveData.boxscore`, `liveData.plays`) | Free, public, no key. Same API the strikeout model already uses. |
| Batter-vs-pitcher history | MLB Stats API `vsPlayer` stat type | Thinly documented; field mapping is defensive (see `nrfi_bvp_features.py`). |
| Weather | Open-Meteo | Reused from the strikeout model, same ballpark reference table. |
| Sportsbook NRFI/YRFI odds | Manual entry, or automated if `ODDS_API_KEY` is configured and the NRFI market is available from your provider | Same manual-entry-always-available design as the strikeout model. |
| Umpire | Manual entry via the existing umpire module, reused as a proxy for run environment (walk-rate effect) | See "Limitations" below. |

**Not wired (explicitly `None`, never fabricated):** exit velocity,
hard-hit%, barrel%, xwOBA -- these are Statcast metrics from Baseball
Savant, a separate data source this add-on does not integrate.
`recent_velocity_change` and `leadoff_reach_rate` are similarly left
`None` pending a verified data source.

---

## Feature engineering

### Pitcher first-inning profile
Career / season / previous-season / last-5/10/20-start / home-away /
day-night scoreless-first-inning rates, each run through the same
empirical-Bayes shrinkage formula as the strikeout model
(`app/features/shrinkage.py`), with a stabilization point of ~20 starts
(a start is the unit of observation here, not a plate appearance).

**Two distinct rates are tracked separately, per design requirement:**
- `season_scoreless_rate` -- did the pitcher *personally* avoid allowing a
  run before being pulled?
- `game_nrfi_rate_in_starts` -- was the *whole game* (including any relief
  pitching in the 1st) NRFI during this pitcher's starts?

These differ only when a starter is pulled mid-1st-inning (openers,
injuries); the backfill pipeline detects a pitching substitution during
the 1st via play-by-play and marks the personal-scoreless figure as
unknown (`None`) rather than guessing, in that case.

### Team first-inning profile
Same shrinkage treatment applied to team scoring rate (season, previous
season, last 5/10/20/30 games, home/away, day/night splits), plus a real
first-inning slash line (ERA/WHIP/AVG/OBP/SLG/OPS/K%/BB%/HR-rate/pitch
count) computed from parsed play-by-play, not just runs.

### Batter-vs-pitcher (BvP)
**Hierarchical (cascading) shrinkage**, per spec: a tiny BvP sample (e.g.
"2-for-3" against one specific pitcher) is blended toward, in priority
order: (1) the batter's rate vs. that pitcher's throwing arm, (2) the
batter's season baseline, (3) league average -- with each level itself
already shrunk toward the level below it before being used as the next
level's prior. See `app/features/nrfi_rate_calculations.cascading_shrinkage`.

This was verified against the exact scenario the spec calls out: a 3-PA
".667" BvP sample moves the final estimate by less than 0.02 from the
prior, while a 50-PA sample with a real signal moves it by 0.09+.

---

## Half-inning probability model

Two half-innings are modeled **separately**, never averaged:

```
p_away_scores = log5(away_team_scoring_rate, home_pitcher_run_allowed_rate, league_scoring_rate)
p_home_scores = log5(home_team_scoring_rate, away_pitcher_run_allowed_rate, league_scoring_rate)

P(NRFI) = (1 - p_away_scores) * (1 - p_home_scores)
P(YRFI) = 1 - P(NRFI)
```

log5 is the same function already used (and tested) by the strikeout
model's batter-matchup calculation, imported from a shared module
(`app.features.probability_math`) rather than duplicated.

On top of the log5 core, small **capped** adjustments apply for:
top-of-order (batting spots 1-5) BvP-informed OBP/SLG quality (+/-15% max),
ballpark run factor (+/-8% max), weather (+/-5% max), umpire (+/-5% max,
using the umpire's walk-rate effect as a documented proxy for run
environment). No single adjustment can overpower the core pitcher/team
matchup.

---

## First-Inning Threat Score (0-100)

A **supporting**, structurally separate metric -- never fed back into the
probability calculation. Built from a weighted combination of z-score-like
deviations from league average (OBP, SLG, K% [inverted -- lower K% raises
threat], BB%, HR rate, top-order BvP quality, recent-vs-season form),
mapped onto 0-100 via a bounded logistic transform so extreme inputs can
never produce a score below 0 or above 100.

League-average inputs produce a score of exactly 50.0 (verified).

---

## Confidence score (0-100)

Deducts points from a starting 100 for: unconfirmed pitcher (-22),
projected lineup (-15), small pitcher/team samples, missing BvP data,
missing weather, injury/opener uncertainty, stale data, and known poor
recent calibration. **The confidence function has no access to the
predicted probability at all** -- this is enforced by the function
signature itself, not just convention, so a 70% NRFI prediction built on
thin data still surfaces low confidence.

---

## Recommendation thresholds

NRFI/YRFI edge analysis reuses the existing `app/markets/edge_analysis.py`
math directly (American-odds conversion, vig removal, expected value,
edge grading) rather than reimplementing it. Grades: Elite / Strong /
Moderate / Small / No positive estimated edge, mapped to
Strong-NRFI/Lean-NRFI/PASS/Lean-YRFI/Strong-YRFI style recommendations.
Thresholds (`minimum_ev_to_recommend`, `minimum_price_edge_to_recommend`)
are configurable per call.

---

## Model versions

- **Baseline** (`nrfi-baseline-v0.1.0`): the transparent log5 half-inning
  model described above. Always available, no minimum data requirement.
- **Logistic regression** (`nrfi_logistic`): trained once
  `MIN_PROJECTIONS_FOR_ML_RETRAIN` graded historical games have
  accumulated (same threshold/reasoning as the strikeout model -- no ML on
  a tiny personal dataset). Walk-forward validated (never randomly
  shuffled), promoted only if it clears the same `evaluate_promotion()`
  gate the strikeout model uses (minimum validation observations, bounded
  fold variance, strict improvement over the active model).

---

## Limitations

- **Umpire run-environment effect** is a proxy (the umpire module's
  walk-rate multiplier), not a purpose-built run-environment model -- no
  free, ToS-verified structured umpire data source exists (same
  limitation as the strikeout model).
- **Statcast metrics** (xwOBA, barrel%, hard-hit%, exit velocity) are not
  wired; fields are explicitly `None`.
- **BvP endpoint field mapping** (`vsPlayer` stat type) is defensive but
  unverified against a live payload in this development environment --
  test against real data early and watch for `missing_fields` flags.
- **Expected first-inning runs** is a documented approximation (average
  runs-given-at-least-one-run ~= 0.62), not a separately fitted count
  model.
