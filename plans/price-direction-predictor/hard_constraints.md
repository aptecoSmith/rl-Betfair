---
plan: price-direction-predictor
---

# Hard constraints

These are invariants any session must satisfy. A session that
violates one is wrong even if its results look good — the result
is contaminated and must be excluded from the scoreboard.

These are written so that an autonomous run can self-check.
Violations should fail loudly, not silently degrade.

## §1 — Pre-off only

Training and evaluation use only ticks where `in_play == False`
AND `timestamp < market_start_time`. In-play has fundamentally
different price dynamics and would dominate the loss if mixed in.
The labelling pipeline must reject in-play ticks at extraction
time; a guard test verifies zero in-play rows in the persisted
dataset.

## §2 — Self-supervised labels, no simulator

Labels come straight from future `LastTradedPrice` in the parquet.
Never from a simulator-replayed order. Labels via
`env.exchange_matcher` or `oracle_scan` are out of scope here.
A session that imports from the env's matcher in its labelling
path is invalid.

## §3 — Date-based train/val/test split, no random shuffle

Splits are by calendar date and live in
`scripts/predictor/splits.py`. NO within-day shuffle. Every
candidate uses the same split — sessions that hand-edit dates are
invalid. The test split is sealed: a session that opens the test
parquets before S09 is invalid and any model that touched test
data is excluded.

## §4 — Frozen handoff to RL

When the predictor is wired into RL observations (S11), its
weights are frozen (`requires_grad_(False)`) and the optimiser
does not see them. Joint optimisation breaks the whole point of
this plan. A future plan can revisit fine-tuning, but not this
one.

## §5 — Test set is touched ONCE per candidate

The test date range is reserved for S09 (the final test pass).
Hyperparameter search, architecture sweeps, and loss-shape
decisions are made on the val set. A candidate evaluated on the
test set more than once must be re-trained on a fresh held-out
range before any number is quoted.

## §6 — Quantile output is the comparison surface

Even when a candidate uses a parametric output (Gaussian,
Student-t) or classification, the metric pipeline reads at
minimum q10/q50/q90 of Δprice per (runner, horizon). Parametric
candidates derive their quantiles from the fitted distribution;
classification candidates derive them from the cumulative bin
probabilities. A median-only candidate that cannot produce
quantiles is invalid.

## §7 — Param-count cap per architecture

Architecture sweeps run at THREE sizes per family (small / medium
/ large — see `master_todo.md` S03 for the matrix). The LARGE
size has ≤ 1M trainable parameters (GBM caps: ≤ 500 trees, depth
≤ 6). Small and medium sizes are intentionally below the cap —
the point is to span the capacity range within a family so we
can read the per-family scaling curve. Sessions that:

- Breach the cap at LARGE size, or
- Add sizes outside small/medium/large without updating
  `master_todo.md`, or
- Compare a LARGE-of-one-family to a SMALL-of-another and call
  the difference a "family" effect

…are invalid. The cap exists so capacity is held roughly
constant across the LARGE row; cross-family comparisons must
match sizes.

## §8 — Calibration is a first-class metric

A candidate with great MAE but a calibration gap > 10pp on any
horizon is not a usable predictor — the operator's decision rule
depends on the quantile spread MEANING the stated coverage. Every
model card includes a calibration plot per horizon and the
calibration gap is a hard-stop column in the scoreboard.

## §9 — TVL feature path zero-handles missing data

When TVL is missing (pre-2026-04-26 data, post-off rows where
the ladder collapsed, or rows where the runner is dormant), the
feature path must zero-fill or mask, never crash and never
silently default to a non-zero. A NaN-poisoned tensor reaching
the optimiser is a hard fail. A guard test in the harness asserts
this on a known no-TVL row.

## §10 — Model card per candidate, no exceptions

Every candidate that gets compared in the scoreboard ships a
model card under `plans/price-direction-predictor/models/
{experiment_id}.md` with: architecture, parameter count,
training-data date range, val/test metrics (per horizon),
training time, inference time per tick, calibration plot path,
backtest result. A scoreboard row without a corresponding model
card is invalid.

## §11 — RL handoff is opt-in per-cohort, not a global default

When the predictor is exposed to RL training (S11), it ships as
an opt-in observation feature behind
`config.observations.use_price_direction_predictor: false`. A
cohort without the flag set has byte-identical observations to
today's runs. We do NOT change the default observation shape
until the predictor has demonstrated a measured RL win in a
follow-on plan.

## §12 — Scoreboard is append-only

Once a row is written it is never edited. If a candidate is
re-trained, it gets a new `experiment_id`. Hand-editing existing
rows to "fix" a metric is forbidden and any session caught doing
so is invalid. Re-runs are detected by experiment_id collision
and refused unless `--rebuild` is set explicitly.

## §13 — Determinism within a candidate run

Every config sets a seed. The seed plus the config plus the
dataset shard hashes uniquely determine `experiment_id`. Re-running
the same config produces the same experiment_id and is a no-op
unless `--rebuild`. This is what makes the matrix replayable.

## §14 — Three seeds per cell

Sweep cells (architecture, feature variant, output formulation,
horizon set, smoothing) are run with at least 3 seeds. Single-seed
results are noisy enough on small val sets that a winner could
flip on a different seed; 3 seeds let us report median +
inter-seed range and detect that case.

## §15 — Operator-overridable axes are documented

The matrix configs in `configs/predictor/SXX/` are checked in.
The operator can edit them between sessions to add/remove
candidates. A session that hard-codes axes outside the YAML is
invalid — autonomous execution depends on the YAML being the
truth.
