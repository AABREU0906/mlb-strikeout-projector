# NRFI/YRFI Backtesting

## Running a backtest

```bash
python main.py nrfi-backtest --start-date 2026-04-01 --end-date 2026-06-30
```

Requires graded `NrfiProjection` rows in that date range (i.e. you ran
`nrfi-project` for those games and later `nrfi-update-results`). This
replays projections that were **actually generated and stored** by this
system -- there is no separate "backtest engine" vs. "live engine" split,
by design, so historical and live projections are evaluated identically.

## What's reported

- **Accuracy** -- fraction of games where the model's >=50% side matched
  the actual outcome.
- **Brier score** and **log loss** -- probabilistic calibration, computed
  by the same `app/evaluation/metrics.py` functions the strikeout model
  uses (reused, not duplicated).
- **Calibration buckets** -- predicted-probability vs. observed-frequency
  in 5 bins, the standard reliability-diagram data.
- **Confusion matrix** and **NRFI/YRFI precision & recall** -- separately,
  since a model can be good at calling NRFI but weak at calling YRFI (or
  vice versa), and a single accuracy number would hide that.
- **Accuracy by month** and **by lineup status** (confirmed vs. projected)
  -- so you can see whether the model's accuracy holds up before official
  lineups post.
- **Model accuracy vs. an always-predict-NRFI baseline** -- because NRFI is
  the more common outcome historically (~51%), a model needs to clearly
  beat "always guess NRFI" to be adding value, not just look good in
  isolation.

## Data-leakage prevention

The feature builders (`PitcherFirstInningFeatureBuilder`,
`TeamFirstInningFeatureBuilder`) query
`FirstInningGameResultRepository.list_pitcher_history` /
`list_team_history`, both of which filter `game_date < before_date` **at
the repository/SQL level** -- not as an afterthought filter applied to
already-fetched data. The same discipline applies to the training feature
matrix in `nrfi_retrain.py`, which walks forward in date order and only
ever uses each pitcher/team's history strictly before the row being
featurized.

If you want an explicit leakage audit similar to the strikeout model's
`backtest` command (which flags rows where a stored timestamp is after the
game's start), that same style of check applies here too: every
`NrfiProjection` row stores its own `created_at_utc` and the game's
`game_start_utc`, so a manual check (`created_at_utc < game_start_utc`)
against stored rows will catch any projection accidentally run using
post-game data.

## Deep historical backtesting (seasons before you started using this tool)

Requires importing historical games via the existing CSV import templates
(`python main.py import-data --write-templates`), which produce rows in
the same schema this backtester consumes. There is currently no NRFI-specific
CSV importer wired up (only `games` and `actual_results` have loaders in
`historical_import.py`); adding `first_inning_results` as a new importable
entity there is a natural next step if you want to seed several seasons of
history at once rather than backfilling forward from today.

## Model comparison

`nrfi-train` reports walk-forward log loss for the logistic-regression
candidate and only promotes it if it beats the currently active model.
Use `python main.py models` to see every trained NRFI/YRFI model version
(the same `ModelVersion` table the strikeout model uses, filtered by
`model_type='nrfi_logistic'`), and `rollback --version <label>` to revert
if a newly promoted model underperforms in practice.
