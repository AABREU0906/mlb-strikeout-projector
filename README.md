# MLB Pitcher Strikeout Projection System

A local, terminal-based system for projecting MLB starting-pitcher strikeouts,
combining real MLB statistics, weather/ballpark context, sportsbook market
data, and Monte Carlo simulation — with every projection stored, gradeable,
and evaluable over time.

This is a personal research/analysis tool. It is **not** a guaranteed
prediction system, and nothing it outputs should be treated as a
guaranteed or certain betting outcome.

---

## 1. What it does

Each day, you run `python main.py`, pick a game and a starting pitcher, and
the system:

1. Pulls the day's schedule and probable pitchers (MLB Stats API — free,
   no key required).
2. Checks whether the **official confirmed lineup** has posted for the
   pitcher's opponent; if not, falls back to a clearly-labeled
   **projected lineup** (the opponent's most recent actual lineup) and
   reduces model confidence accordingly.
3. Builds shrinkage-adjusted statistical profiles for the pitcher and
   every batter (season, career, and handedness-split strikeout/walk
   rates, recent 7/14/30-day form).
4. Pulls weather for the ballpark (Open-Meteo, free/keyless) and applies a
   documented, capped ballpark strikeout factor.
5. Resolves sportsbook market data — automatically if you've configured
   `ODDS_API_KEY`, or via manual entry, which always overrides automated
   data when both are present.
6. Runs a 5-stage projection: expected workload → batter-level matchup
   probability (log5) → lineup simulation → market-informed adjustment →
   25,000+ iteration Monte Carlo simulation.
7. Displays a full terminal report: point projections (statistics-only,
   market-informed, blended), percentile distribution, batter matchup
   table, sportsbook comparison with vig removal, ranked positive/negative
   factors, and a High/Medium/Low/Avoid confidence rating.
8. Saves everything — every input, warning, market snapshot, and the full
   simulated distribution — to a local SQLite database, so the projection
   can be graded and audited later.

Afterward:

- `python main.py update-results` fills in actual outcomes for completed
  games (stored separately from the pregame projection — pregame data is
  never overwritten).
- `python main.py evaluate` reports MAE/RMSE/Brier/log-loss/calibration,
  broken out by confidence rating, lineup status, and pitcher handedness,
  and directly compares the statistics-only vs. market-informed vs.
  blended models so you can see whether market data is actually helping.
- `python main.py retrain` trains a Poisson-regression ML model on your
  accumulated graded history, walk-forward validates it, and only
  activates it if it clears documented promotion criteria — below a
  minimum data threshold (default 150 graded projections) it declines to
  train and the transparent baseline formulas stay active.
- `python main.py backtest --start-date ... --end-date ...` replays stored
  graded projections in a date window with an explicit data-leakage audit.

---

## 2. Architecture

```
main.py                    Entry point
app/
  config/                  Settings (env-driven) + logging
  cli/                     Typer commands + interactive terminal prompts
  database/                SQLAlchemy models, session, repositories
  data_sources/             Provider interfaces + real implementations
                            (MLB Stats API, Open-Meteo, The Odds API, news)
  features/                 Shrinkage math, pitcher/batter/team feature
                            builders, umpire regression, league constants
  markets/                  Odds math (vig removal), market snapshot service
  projections/              Stage 1-4 (workload, batter probability,
                            lineup simulation, market adjustment),
                            confidence rating, explanation generator,
                            the orchestrating engine
  simulation/                Monte Carlo engine
  evaluation/                Metrics, evaluator, backtester
  training/                  Feature extraction, walk-forward validation,
                            model promotion rules, retrain orchestrator,
                            rollback
  services/                  Pipeline orchestration, persistence, results
                            updater, history browser, CSV import
  reporting/                  Rich terminal display
tests/                       pytest suite
data/                        cache/ imports/ exports/ database/
saved_models/                 Versioned trained model artifacts (joblib)
logs/                         Rotating log files
```

**Provider interfaces** (`app/data_sources/base.py`) mean every external
data source is swappable — e.g. replacing The Odds API with a different
odds vendor only touches `app/data_sources/odds_api.py` and
`app/markets/market_service.py`; nothing in the projection engine changes.

---

## 3. Installation (macOS Terminal)

```bash
cd path/to/mlb-strikeout-projector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

You do **not** need to fill in any API keys to get started — MLB Stats API
and Open-Meteo weather are free and keyless. Sportsbook odds automation
needs `ODDS_API_KEY` from https://the-odds-api.com; without it, use manual
line entry when prompted (fully supported, not a degraded mode).

### Daily use after install

```bash
cd path/to/mlb-strikeout-projector
source .venv/bin/activate
python main.py                    # today's interactive projection workflow
python main.py update-results     # after games finish
python main.py evaluate           # periodically, to check model accuracy
python main.py history            # browse past projections
```

---

## 4. Environment variables

See `.env.example` for the full list with defaults. Key ones:

| Variable | Purpose | Required? |
|---|---|---|
| `ODDS_API_KEY` | The Odds API key for automated sportsbook lines | No — manual entry works without it |
| `NEWS_API_KEY` | Optional licensed headline search | No — manual warning entry works without it |
| `DATABASE_PATH` | SQLite file location | No, has a default |
| `DEFAULT_MONTE_CARLO_ITERATIONS` | Simulation count (min 25,000 per spec) | No, defaults to 25000 |
| `MIN_PROJECTIONS_FOR_ML_RETRAIN` | Graded-projection threshold before ML retraining activates | No, defaults to 150 |

---

## 5. Manual data entry

- **Sportsbook lines**: at the `project` prompt, choose to enter/override
  the strikeout line, over/under odds, game total, and opponent implied
  runs. Manual values always take precedence over automated feed data.
- **News/injury warnings**: enter player, issue, source, and — critically —
  a required **confidence level** (`confirmed` / `reported` / `inferred` /
  `speculative`). The system never presents an unconfirmed report as fact;
  confidence is a mandatory field, not inferred.
- **Lineup correction**: the pipeline's `resolve_lineup()` accepts a
  manual lineup list (player IDs + batting order) when automated sources
  are wrong or unavailable; it still pulls real season statistics for
  those player IDs rather than fabricating anything.

---

## 6. Modeling methodology

### Shrinkage
Every rate stat (batter/pitcher strikeout rate, walk rate, contact rate,
etc.) is shrunk toward a league-average prior using a documented
stabilization-point formula (`app/features/shrinkage.py`):

```
shrunk_rate = (n·observed_rate + k·prior_rate) / (n + k)
```

where `k` is a sabermetrically-documented stabilization point (e.g. ~60 PA
for overall batter K rate, ~110–120 PA/BF for handedness splits). Small
samples get pulled hard toward league average; large samples stay close to
observed.

### Batter-matchup probability: log5
Pitcher-vs-hand and batter-vs-hand strikeout rates are combined via the
standard sabermetric **log5** method (not a simple average — explicitly
disallowed by design):

```
log5(p, b, league) = (p·b/league) / (p·b/league + (1-p)(1-b)/(1-league))
```

Small, capped multiplicative adjustments (each capped in magnitude) are
then layered on for recent form, ballpark factor, weather, and umpire
tendency, so no single modest-sample signal can dominate.

### Workload model
A documented formula-based estimate of innings/batters-faced/pitch count
from season and recent-start averages, with hard caps (not just
percentage nudges) for openers, tandem risk, announced pitch limits, short
rest, and rehab-assignment starts — the system never assumes a normal
workload when there's credible evidence against it.

### Monte Carlo simulation
25,000+ iterations per projection (configurable). Each iteration samples
innings pitched from a distribution centered on the workload estimate
(spread widens with workload uncertainty), converts to batters faced,
cycles through the batting order for that many plate appearances, and
draws a Bernoulli strikeout outcome per PA using that lineup spot's
matchup probability. Supports a reproducible random seed.

### Statistics-only vs. market-informed
Two full parallel projections are always computed and displayed — the
market-informed path never silently replaces the statistics-only one.
Market features (consensus line, opponent implied runs, game total, line
movement) apply a separate, capped adjustment multiplier
(`app/projections/stage4_market_adjustment.py`) with conservative
documented default weights, since — per the project's own rules — you
don't get to invent blending weights; they should come from validated
history. `retrain`/`evaluate` are what let those weights (eventually) be
replaced with walk-forward-validated ones.

### Vig removal
Standard proportional two-way de-vig: convert American odds to implied
probabilities, then normalize by their sum so they total exactly 1.0. See
`app/markets/odds_math.py` for the documented method and its
simplification (proportional vig distribution, not Shin's method).

### Confidence rating
A documented, itemized penalty score (unconfirmed lineup/pitcher, data
completeness gaps, workload/news/weather uncertainty, market
disagreement, simulation variance) maps to High/Medium/Low/Avoid — always
traceable to specific listed factors, never a black-box judgment.

### Retraining and model promotion
Below `MIN_PROJECTIONS_FOR_ML_RETRAIN` graded projections, the system
stays on the transparent baseline formulas above — it will not train an
ML model on a tiny personal dataset. Above that threshold, `retrain`
builds leakage-safe feature vectors from stored pregame snapshots, runs
strict **walk-forward** (never randomly-shuffled) validation with a
Poisson regression (appropriate for count data), and only promotes a new
model if it clears documented bars: minimum validation observations,
bounded fold-to-fold variance, and a strict MAE improvement over the
currently active model. Every version — promoted or not — is persisted
with full metadata and can be restored with `python main.py rollback
--version <label>`.

---

## 7. Database & storage

SQLite at `data/database/strikeout_projector.db` (path configurable).
Every projection stores its full pregame snapshot (pitcher/batter/team
inputs, weather, warnings, market data, simulation distribution,
confidence factors) so any historical projection can be fully
reconstructed and audited. Postgame results are stored in a **separate**
table (`actual_results`) and never overwrite the pregame `projections`
row.

---

## 8. Evaluation and backtesting

`evaluate` computes MAE/RMSE/MedAE/bias for each of the three projection
paths (statistics-only / market-informed / blended) plus Brier
score/log-loss/calibration/O-U accuracy for the probabilistic side, broken
out by confidence rating, lineup status, and pitcher handedness — so you
can see directly whether market data is earning its place in the blend,
rather than assuming it does.

`backtest` replays stored graded projections within a date window and
explicitly flags any row where the recorded market or lineup timestamp is
*after* the game's start (a leakage check), plus breaks accuracy down by
month, confidence, lineup status, and market-data availability.

Deep historical backtesting (seasons before you started running this
tool) requires importing historical data via
`python main.py import-data --write-templates`, which generates documented
CSV schemas in `data/imports/templates/` for games, pitchers, batters,
lineups, weather, sportsbook markets, and results.

---

## 9. Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest
```

The suite covers shrinkage math, log5, vig removal/odds conversion, Monte
Carlo reproducibility and bucket correctness, workload-model hard caps
(openers/tandem/pitch limits/short rest), confidence-rating thresholds,
walk-forward validation and model-promotion rules, and evaluation metrics.
Every test in this suite was verified passing during development wherever
the sandbox's tooling allowed direct execution (pure-Python/NumPy modules);
the remaining Pydantic/SQLAlchemy-dependent tests are ready to run the
moment you `pip install -r requirements.txt` locally.

---

## 10. Troubleshooting

- **"No games scheduled"**: double check the date format (`YYYY-MM-DD`)
  and that MLB actually has games that day.
- **No confirmed lineup found close to game time**: MLB typically posts
  official lineups ~1–2.5 hours before first pitch; before that, you'll
  see the projected-lineup warning, which is expected behavior, not a bug.
- **Odds are empty**: either set `ODDS_API_KEY` in `.env`, or use manual
  entry when prompted — this is a supported path, not a fallback of last
  resort.
- **`retrain` says "below minimum"**: keep running `update-results` after
  games complete; the ML path activates automatically once you cross
  `MIN_PROJECTIONS_FOR_ML_RETRAIN`.
- **Field-mapping errors from MLB Stats API**: this is an undocumented
  public API and its JSON shape can drift. Parsing in
  `app/features/pitcher_features.py` / `batter_features.py` is written
  defensively (missing fields become `None` and get flagged in
  `missing_fields`, never fabricated) — if you see a `missing_fields`
  warning spike, check the live payload shape against the parsing code.

---

## 11. Data limitations (read this)

- **Umpire tendency data**: no free, structured, ToS-clear API exists for
  this. The system supports manual entry (heavily regressed toward
  league-average) and defaults to fully neutral otherwise — it never
  fabricates a tendency.
- **News/injury classification**: there's no free structured feed for
  MLB injury news. Automated headline search only works with a licensed
  `NEWS_API_KEY` and returns raw headlines (no auto-classification of
  confidence). The reliable path is manual entry, which is why confidence
  is a required field there.
- **Bullpen strength**: not yet wired to a real bullpen-quality feed;
  currently a documented neutral placeholder in the workload model.
- **Ballpark factors**: seeded from a small, documented reference table
  covering a handful of well-known parks; unmapped parks fall back to
  neutral (1.00) factors rather than a guess.
- **The Odds API event matching**: matching a specific pitcher's props to
  the correct event conservatively requires team/date resolution the
  current implementation doesn't fully automate yet (see comments in
  `app/data_sources/odds_api.py`); manual entry is the reliable path today.

## 12. Legal and responsible-use considerations

This tool does not access any paywalled, authenticated, or bot-protected
content, and respects robots.txt for any HTML fetch path. It uses only
public/free JSON APIs (MLB Stats API, Open-Meteo) and a licensed odds
aggregator (The Odds API, requires your own key/ToS acceptance). No
projection or edge label in this system is ever presented as guaranteed,
safe, or certain — see `app/markets/odds_math.classify_edge()` for the
neutral labeling used throughout. If you use this for wagering decisions,
you are solely responsible for complying with the gambling laws in your
jurisdiction.

## Betting ledger

The app stores bets you actually placed in the same SQLite database as the projection history, but in a separate `bets` table. Before showing today's schedule, it checks for unresolved bets from prior dates and offers to settle them.

Commands:

```bash
python main.py settle-bets
python main.py bet-history
python main.py export-bets
```

A CSV backup is automatically written to `data/exports/bets.csv` whenever a bet is added or settled.
