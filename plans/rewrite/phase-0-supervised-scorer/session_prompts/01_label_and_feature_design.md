# Session prompt — Phase 0, Session 01: label + feature dataset

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Produce a labelled, featurised parquet dataset of historical opening
opportunities for the Phase 0 supervised scorer. **No model training
in this session.** The dataset is the contract that Session 02
trains on.

Output: `data/scorer_v1/dataset.parquet` (or split files —
implementer's call) with one row per (date, market_id, runner_idx,
tick_idx, side) opportunity, columns:

- `label` — 0.0 / 1.0 / NaN per the strict mature definition.
- All ~25–35 features per `phase-0-supervised-scorer/purpose.md`.
- `date`, `market_id`, `runner_idx`, `tick_idx`, `side` — for
  joining and chronological splitting.

## What you need to read first

1. `plans/rewrite/README.md` — rewrite plan overview.
2. `plans/rewrite/phase-0-supervised-scorer/purpose.md` — locked
   label definition, feature set, success bars. **The label
   definition and feature list are not up for debate in this
   session; if you find a reason to change them, file it as a
   finding and stop.**
3. `plans/per-runner-credit/findings.md` — the H1 finding that
   motivates this whole phase. Specifically the section on why
   `fill_prob_head`'s label was wrong.
4. `CLAUDE.md` — sections "Bet accounting", "Order matching",
   "Equal-profit pair sizing", "Force-close at T−N", "
   `info[realised_pnl]` is last-race-only", "mature_prob_head
   feeds actor_head". These are the env behaviours your label
   simulator must match.
5. `env/exchange_matcher.py` — the matcher. **You will reuse this
   to simulate opens. Do NOT re-implement matching logic; import
   and call.**
6. `env/scalping_math.py` — `equal_profit_lay_stake`,
   `equal_profit_back_stake`. Reuse for sizing the simulated
   opens.
7. `env/bet_manager.py:103–133` — the `Bet` dataclass, the
   `force_close` and `close_leg` flags. **The label classifier
   reads these flags from simulated bets.**
8. `data/episode_builder.py` — `load_days`, the parquet schema for
   processed data. This is what you read FROM.
9. `agents/ppo_trainer.py:1604–1704` — the existing episode-end
   pair-outcome backfill loop. The label classifier in this
   session is the same logic, applied to the universe of
   opportunities (not just policy-chosen).

## What to do

### 1. Lock the file layout

Create `data/scorer_v1/` with subdirectories:

- `data/scorer_v1/dataset.parquet` (or `dataset/{date}.parquet`
  per-day shards if memory is tight).
- `data/scorer_v1/feature_spec.json` — the feature contract
  (names, dtypes, the function name that computes each).

### 2. Build the feature extractor (~30 min)

A pure function `extract_features(market_state, runner_idx, tick_idx,
side) -> dict` that returns the ~25–35 features in
`purpose.md`'s feature set.

Implementation notes:

- Reuse env helpers wherever they exist (`bm.get_*`, market state
  parsers in `env/betfair_env.py`).
- Velocity features (`traded_volume_last_30s`, etc.) need a
  rolling window — implement once, in a helper class.
- `time_to_off_seconds` comes from race metadata — compute
  consistently with the env's own `time_to_off` calculation.
- One-hot `side` and `market_type` at extract time, not later.
- NaN propagation: if a feature is unavailable (e.g. no LTP),
  emit NaN — the model handles missingness natively.

Feature names go into `feature_spec.json` in declaration order
so Session 02 (and Phase 1 actor wiring) reads them
deterministically.

### 3. Build the label generator (~60 min)

For each (date, market_id, runner_idx, tick_idx, side):

a. Check feasibility: LTP exists, opposite-side book has size,
   priceable side meets hard cap. If not feasible, label = NaN.

b. Simulate the open: call `ExchangeMatcher.match_back` (or
   `match_lay`) with the same junk-filter / hard-cap rules the
   env uses at training time. If the matcher refuses the open,
   label = NaN (not feasible).

c. Compute the equal-profit passive stake via
   `equal_profit_{lay,back}_stake` and simulate the passive
   resting on the opposite side from this tick onward.

d. Walk forward in the historical price book until ONE of:
   - The passive matches naturally (label = 1.0, outcome = matured).
   - `time_to_off_seconds <= force_close_threshold` is hit. Then
     simulate force-close via the relaxed matcher path. If force-
     close lands → label = 0.0, outcome = force_closed. If force-
     close refuses (no priceable opposite book) → label = 0.0,
     outcome = naked.
   - Race goes off without the passive matching → label = 0.0,
     outcome = naked.

e. **`agent_closed` outcome handling.** The historical data doesn't
   contain "the agent's `close_signal` decisions" — those depend on
   policy. For the supervised label, treat agent-closed pairs as
   matured (label = 1.0) — the policy COULD have closed them
   profitably and the supervised scorer should not penalise that.
   This matches the `mature_prob_head` strict label
   (CLAUDE.md "mature_prob_head feeds actor_head"):
   `force_close=True → 0.0`; everything else with both legs → 1.0.

   But since we're simulating WITHOUT a policy in this session,
   we never observe `agent_closed` directly. The label collapses
   to: matured naturally → 1.0; force-closed → 0.0; naked → 0.0.
   Same binary, fewer code paths.

### 4. Generate the dataset (~30 min wall time, plus runtime)

Iterate over all available days in `data/processed/`. For each day,
walk every market, runner, tick, side; emit one row per opportunity.

**Sub-sampling.** Per-tick state changes slowly relative to the
close window. Sub-sample by tick — emit every 5th tick. This cuts
the dataset by 5× without losing meaningful signal. Make this a
config knob (`tick_stride: int = 5`) so Session 02 can experiment.

**Memory.** ~12 days × ~300 markets × ~14 runners × ~200 ticks ×
2 sides = ~20M rows after sub-sampling. At ~30 columns × 4 bytes,
that's ~2.4 GB — tight for in-memory but workable. Per-day shards
keep memory bounded.

Estimated wall time: 30–90 minutes depending on env-call overhead.
If it's slower than 2 hours, profile before optimising — likely
candidates are repeated parquet reads or repeated history-walk
allocations.

### 5. Sanity-check the dataset (~15 min)

Before declaring done:

- Row count makes sense (in the 10–30M range after sub-sampling).
- Label distribution: matured-rate (label = 1.0) should land in
  the 15–35 % range based on cohort-M observations
  (`arbs_completed + arbs_closed` / `arbs_completed + arbs_naked
   + arbs_closed + arbs_force_closed`). Wildly different label
  rate is a red flag — investigate before shipping.
- NaN-feature rates are reasonable (< 5 % per column, except
  velocity features in the first few ticks of a market which can
  be higher).
- Spot-check 10 random rows by hand: the features make sense
  given the (date, market, tick) state.
- Per-date row counts are roughly proportional to market count
  (a date with 2× markets has 2× rows).

If any check fails, stop and investigate before writing the
parquet — easier to fix data generation than to debug a bad
dataset in Session 02.

### 6. Write findings (`session_01_findings.md`)

A short writeup in `plans/rewrite/phase-0-supervised-scorer/
session_01_findings.md` with:

- Final dataset row count and per-class label split.
- Train/val/test date splits decided (chronological per
  `purpose.md`).
- Per-feature NaN rate.
- Per-day row count summary.
- Wall time for generation.
- Any feature definitions where you had to make a judgement call
  (e.g. "I defined `ltp_change_last_30s` as the LTP delta over
  the last 30 ticks, not seconds — ticks are ~5s apart in this
  data"). Document so Session 02 (and Phase 1) read the same
  feature.
- Anything that surprised you during generation.

## Stop conditions

- Dataset generated, sanity checks pass, findings written →
  message operator "Phase 0 Session 01 complete, ready for
  Session 02", **stop**.
- Sanity check fails → write `session_01_findings.md` describing
  what failed and what you tried, **stop**. Do not generate
  Session 02's dataset on a known-bad input.
- You discover the locked label or feature list is wrong → write
  the finding, **stop**. Do not silently override the design.

## Hard constraints

- **Reuse env code, do NOT re-implement.** The matcher, sizer,
  force-close logic must come from `env/exchange_matcher.py`,
  `env/scalping_math.py`, `env/bet_manager.py`. If you
  re-implement, you'll diverge from how the env will behave at
  Phase 1 runtime and the scorer will be miscalibrated.
- **No env modifications.** Same rule as Phase −1. If you find a
  bug, file it as a finding; don't fix in this session.
- **No new tests for env / matcher / bet_manager** — those are
  Phase −1's territory.
- **Do tests** for your label generator and feature extractor.
  These are NEW code that needs regression coverage. Put them
  under `tests/test_scorer_v1_dataset.py`. Aim for ~5–10 tests:
  one per outcome class, one per feature group, a few
  sanity-property tests.
- **Parallel tree.** Code goes under `agents_v2/scorer/` or
  `training_v2/scorer/` — implementer's choice. **Not** in
  `agents/` or `training/`.

## Out of scope

- Training any model.
- Calibration.
- Evaluation.
- Wiring into a policy.
- Adding any feature beyond the locked list (file as future
  finding if you spot one).
- Multi-class labels.

## Useful pointers

- `data/processed/<date>.parquet` — the input data.
- `data/processed/<date>_runners.parquet` — per-runner metadata.
- `env/exchange_matcher.py::ExchangeMatcher.match_back/match_lay`
  — the matcher. Note `force_close=True` gives the relaxed path.
- `env/scalping_math.py::equal_profit_lay_stake` /
  `equal_profit_back_stake` — sizing.
- `env/betfair_env.py::_settle_current_race` (~line 2842) — the
  reference implementation of pair-outcome classification.
- `agents/ppo_trainer.py:1604–1704` — the existing pair-outcome
  backfill loop, for cross-checking your label classifier.

## Estimate

3–5 hours.

- 30 min: read context.
- 30 min: feature extractor.
- 60 min: label generator.
- 30–90 min: dataset generation wall time.
- 15 min: sanity checks.
- 30 min: tests.
- 30 min: findings writeup.

If you're past 6 hours, stop and check whether scope crept (most
likely: re-implementing env logic instead of importing it, or
adding features beyond the locked list).
