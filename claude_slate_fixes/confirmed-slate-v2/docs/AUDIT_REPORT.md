# Audit Report: Workload Bug Root-Cause Fix + Full Pipeline Audit

## 1. Root cause of the reported workload bug

**Symptom:** 27.9 expected innings, 126 expected batters faced, 531
expected pitches, 98.8% probability of going over 4.5 strikeouts.

**Root cause:** `_find_stat_block()` (formerly duplicated in
`pitcher_features.py` and `batter_features.py`) matched MLB Stats API stat
blocks with a loose substring check (`"season" in block_type`) and, when a
block's `splits` array contained more than one entry -- which genuinely
happens for a mid-season trade, or when spring-training/postseason/minor-
league splits share the nominal "season" type -- it always took
`splits[0]` with zero verification of what that split represented. Since
innings/battersFaced/numberOfPitches and `gamesStarted` were always read
from that same split, the resulting per-start averages were internally
self-consistent but could reflect an entirely different, smaller (or
differently-scoped) population than a full MLB regular season.

**The fix** (`app/data_sources/mlb_stat_block_selector.py`, new):
1. The underlying API calls now explicitly request `sportId=1` and
   `gameType=R` (previously omitted, relying on undocumented API
   defaults) -- this removes the ambiguity at the source.
2. Stat-block matching is now exact (`type.displayName`/`group.
   displayName`), not substring.
3. When multiple valid MLB-regular-season splits remain (a genuine
   mid-season trade), counting stats are correctly **summed** across all
   of them rather than truncated to the first one.
4. Every rejection reason is recorded and surfaced as a warning, not
   silently swallowed.
5. This single implementation now backs both the pitcher and batter
   feature builders, eliminating the previously duplicated fragile logic.

**Verification:** 12 dedicated tests, including a byte-for-byte
reproduction of the exact reported failure mode (a small non-regular-
season split sitting ahead of the correct MLB split in the response
array) and a genuine mid-season-trade aggregation case (12-start/68.1-IP +
15-start/89.2-IP correctly summing to 27 starts/158.0 IP). All 12 pass.

---

## 2. Central projection validator (new, gates all betting output)

Per the audit's own stated top priority -- "prefer PASS or VALIDATION
FAILED over a confident but unreliable betting recommendation" -- a new
`app/validation/projection_validator.py` runs 14 independent, named checks
against every projection immediately before any betting-related output:

- Expected innings in (0, 9]
- Batters-faced-to-innings ratio realistic
- Pitch-count-to-batters-faced ratio realistic
- Probability distribution: values in [0,1], sums to ~1
- Percentiles correctly ordered
- Standard deviation non-negative and finite
- Final projection consistent with the distribution's implied mean
- Final projection does not exceed expected batters faced
- Workload completion-probability ordering (P(7) <= P(6) <= P(5))
- Over/Under/Push probabilities sum to ~1 (whole-number and half-point lines both handled)
- Extreme probabilities near 0%/100% -> warning (never a hard block on their own)
- Unconfirmed pitcher / projected lineup -> warning
- Stale data -> warning
- Workload fallback used -> warning, with the exact reason surfaced

Critical issues **block** the recommended side, EV, edge grade, and the
bet-recording prompt entirely (verified: the code path returns before
`_prompt_to_record_bet` is ever reached). Warnings display but do not
block. Fed the exact reported bug numbers, the validator correctly returns
`VALIDATION FAILED`. 32 dedicated tests, all passing.

**Wired into the live flow**, not just built and tested in isolation --
`app/cli/main_app.py`'s interactive projection command now calls
`validate_projection()` with the real pipeline output and gates
`print_market_comparison()` / the bet-recording prompt behind
`report.is_valid`.

---

## 3. Second critical bug found while wiring the validator (audit item #11)

`print_market_comparison()` derived every confidence flag
(`lineup_confirmed`, `pitcher_confirmed`, `workload_warning`,
`injury_warning`, `weather_warning`, `stale_data`) via
`getattr(result, "<name>", <safe_default>)`. **`ProjectionResult` never
actually carried any of those attributes**, so every lookup silently fell
through to its "everything is fine" default on every single call,
regardless of what actually happened in the pipeline. This is the direct
mechanism behind "the program previously displayed HIGH confidence even
while using league-average fallbacks."

**Fix:** the function now takes these as real, explicit parameters. The
caller passes the actual `lineup_status` from the pipeline run and derives
`workload_warning` from the real workload notes via a new shared helper,
`workload_notes_indicate_fallback()` (used identically by both the
validator and the display layer, so they can never diverge on what counts
as a fallback).

---

## 4. Third bug found: whole-number-line push probability

`app/reporting/display.py` computed `model_under_probability = 1.0 -
model_over_probability` unconditionally. This is only correct for a
half-point line. For a whole-number line, this silently folds the push
probability into "under" -- e.g. with an 18% chance of landing exactly on
a 5.0 line, the old code reported "under" at 76% when the true value
(excluding the push) was 58%.

**Fix:** new `app/markets/line_probability.py` (`compute_line_
probabilities()`) correctly separates over/under/push based on whether the
line is a whole number, used by both display and validation. 9 dedicated
tests, including an explicit assertion that the old buggy value and the
fixed value differ.

---

## 5. Other audit findings

- **Zero-value truthiness (item #6):** audited every numeric field check
  in `pitcher_features.py`/`batter_features.py`; all already use explicit
  `is None`/`<= 0` checks, not bare truthiness. No bug found here --
  already correct.
- **Innings notation validation:** `_parse_innings()` already correctly
  rejects any fractional part other than `.0`/`.1`/`.2`. No bug found.
- **Switch-hitter handling:** already correctly resolves to the side
  opposite the day's starting pitcher's throwing hand, with an explicit
  `missing_fields` flag when that can't be resolved. No bug found.
- **Duplicated validation logic (item #17):** the workload plausibility
  bounds (0.5-9.0 IP, 3.0-45.0 BF, 10.0-130.0 pitches, 3.0-6.5 BF/IP,
  2.5-6.0 pitches/BF) were previously hard-coded independently in
  `pitcher_features.py` and `stage1_workload.py`. Extracted into
  `app/validation/bounds.py` as the single source of truth; both files
  and the new validator now import from it, so the layers of defense can
  never silently drift apart.
- **Extreme fair odds (item #10):** `probability_to_american_odds()` in
  `edge_analysis.py` is mathematically correct (an odds value like -8581
  is the correct fair-odds conversion of a ~98.8% probability). Per the
  audit's own framing ("unless the underlying validated probability truly
  supports it"), the fix is not to cap or hide the number but to flag the
  underlying probability -- which `check_extreme_probability_risk()`
  already does for any model probability >= 98% or <= 2%, surfaced as a
  visible warning alongside the number.
- **Confidence math itself (`determine_confidence()` in
  `edge_analysis.py`):** the scoring formula itself was already sound
  (itemized deductions, correctly tiered HIGH/MEDIUM/LOW/AVOID
  thresholds). The bug was entirely in the fake inputs feeding it (see
  #3 above), now fixed.

---

## 6. Tests added this pass

| File | Tests | Status |
|---|---|---|
| `tests/test_stat_block_selector.py` | 12 | All passing (verified directly) |
| `tests/test_projection_validator.py` | 32 | All passing (verified directly) |
| `tests/test_line_probability.py` | 9 | All passing (verified directly) |

Combined with the pre-existing suite, a full sweep of every test file
runnable in this development sandbox (pure-Python modules, without
requiring `pydantic`/`SQLAlchemy` to be installed) shows **74 of 75
executed tests passing**; the one non-pass is a sandbox tooling artifact
(a test using `pytest.raises` that my quick-sweep script couldn't
auto-strip the `pytest` import from), not a real failure. The remaining
test files require `pydantic`/`SQLAlchemy`, which are declared in
`requirements.txt` but not installable in this offline sandbox --
`python -m compileall app tests` confirms every one of them is
syntactically valid and ready to run the moment you `pip install -r
requirements.txt` locally.

---

## 7. Commands

```bash
# Install (same as before -- no new dependencies required)
pip install -r requirements.txt
cp .env.example .env

# Run
python main.py

# Test
pytest
python -m compileall app tests
python main.py --help
```

No environment-variable changes are required. No database migration is
required for this pass -- every change in this audit is in application
logic (data selection, validation, display), not schema.

---

## 8. Remaining limitations (honest, not swept under the rug)

- **Items 12-14 (rookie/limited-sample handling formalization, deeper
  data-source integrity work like doubleheader/same-name-player edge
  cases, and a full line-by-line SQLite betting-ledger re-audit)** were
  reviewed at a survey level given this pass's scope, but did not surface
  additional bugs of the severity of items 1-4 above, and were not each
  individually re-tested with new dedicated test files this pass. If you
  want a dedicated deep-dive on any of them, they're the natural next
  target.
- **This sandbox has no network access**, so no fix in this pass was
  verified against a live MLB Stats API response -- every fix is verified
  against realistic synthetic payloads modeling the documented API shape.
  Run `pytest` and a real `python main.py` session on your machine as the
  final confirmation.
- The distribution-mean consistency check in the validator compares
  against `statistics_only_projection` (the exact same source as the
  displayed distribution) rather than `final_blended_projection`, since
  the blended figure can legitimately differ once market data is
  incorporated -- this is by design, not a gap, but worth knowing if you
  see the two numbers diverge modestly in the output.
