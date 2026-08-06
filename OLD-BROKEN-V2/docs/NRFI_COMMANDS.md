# NRFI/YRFI Command Reference

All commands below are additive -- every existing strikeout-projector
command (`project`, `update-results`, `evaluate`, `retrain`, `history`,
`backtest`, `settle-bets`, `bet-history`, `export-bets`, `import-data`,
`models`, `rollback`) is unchanged and continues to work exactly as
before.

## Daily workflow

```bash
python main.py nrfi-project                 # interactive: pick a game, get NRFI/YRFI projection
python main.py nrfi-project --date 2026-07-30

python main.py both                         # run strikeout AND NRFI/YRFI projections for one game, in sequence
python main.py both --date 2026-07-30

python main.py menu                         # show the full command menu
```

`nrfi-project` requires both starting pitchers to be announced (probable
or confirmed) -- if the schedule feed hasn't posted them yet, it will tell
you to try again closer to game time rather than guessing.

## Historical database

```bash
# Backfill a date range
python main.py nrfi-backfill --start-date 2026-04-01 --end-date 2026-04-30

# Backfill an entire season (uses a generous Mar 15 - Nov 5 window)
python main.py nrfi-backfill --season 2025

# Just catch up the last few days
python main.py nrfi-backfill --recent-days 3
```

All three are **resumable**: if interrupted, re-running the same command
skips already-complete games (a fast DB check, no re-fetch) and only
retries games that weren't final yet last time.

## Grading and evaluation

```bash
python main.py nrfi-update-results          # grade completed games against pending NRFI/YRFI projections
python main.py nrfi-history                 # browse past NRFI/YRFI projections and outcomes
python main.py nrfi-history --date 2026-07-30 --team Yankees
python main.py nrfi-backtest --start-date 2026-04-01 --end-date 2026-06-30
```

## Model training

```bash
python main.py nrfi-train
```

Declines to train below `MIN_PROJECTIONS_FOR_ML_RETRAIN` graded games
(same threshold/env var as the strikeout model); the transparent log5
baseline stays active until then.

## Betting

```bash
python main.py nrfi-bet-history             # NRFI/YRFI-specific betting ledger and totals
python main.py bet-history                  # strikeout betting ledger (unchanged)
python main.py export-bets                  # exports BOTH market types to one CSV (market_type column distinguishes them)
```

Bet recording happens inline during `nrfi-project` (after you view the
edge analysis, you're asked whether you placed a bet) -- there's no
separate "record bet" command, matching the existing strikeout workflow's
pattern.

## Suggested first-time setup

```bash
pip install -r requirements.txt
cp .env.example .env
python main.py nrfi-backfill --recent-days 30    # get some history to work with
python main.py nrfi-project                       # try a live projection
```
