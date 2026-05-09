---
plan: price-direction-predictor
---

# Hard constraints

These are invariants any session in this plan must satisfy. A
session that violates one of these is wrong even if its results
look good — the result is contaminated.

## §1 — Pre-off only

Training and evaluation use only ticks where `in_play == False`
AND `timestamp < market_start_time`. In-play has fundamentally
different price dynamics (one resolution per tick from
race-position information) and would dominate the loss if mixed
in. A session that includes in-play ticks in its training set is
invalid.

## §2 — Self-supervised labels, no simulator

Labels are read directly from future `LastTradedPrice` in the
parquet — never from a simulator-replayed order. The whole point
of this plan is to escape the sim/reality coupling that
`fill_prob` and `mature_prob` are stuck with. Labels via
`env.exchange_matcher` or `oracle_scan` are out of scope here.

## §3 — Date-based train/val/test split, no random shuffle

Splits are by calendar date (e.g. 2026-04-06 → 2026-04-30 train,
2026-05-01 → 2026-05-03 val, 2026-05-04 → 2026-05-06 test). No
within-day shuffle that would leak future ticks into earlier
training rows. A model evaluated on a randomly-shuffled split is
overstating performance and the result must not be quoted.

## §4 — Frozen handoff to RL

When the predictor is wired into RL observations, its weights are
frozen (`requires_grad_(False)`) and the optimiser does not see
them. The whole point of separating the predictor from the RL
loop is to evaluate it on its own terms — joint optimisation
breaks that and is explicitly forbidden in v1. A future plan can
revisit fine-tuning, but not this one.

## §5 — Held-out test set is touched once

The test date range is reserved for the FINAL run of each
candidate model. Hyperparameter search, architecture sweeps, and
loss-shape decisions are made on the val set. A session that
peeks at the test set during development is invalid and the
candidate must be re-trained on a fresh held-out range.

## §6 — Quantile output, not point estimate

The model emits at minimum 10th / 50th / 90th percentile of
Δprice per (runner, horizon) — never just a mean. A median-only
model has no notion of confidence and the operator-stated decision
rule ("act when I see consistent strong-confidence predictions")
cannot be expressed against it. Pinball loss for training; spread
between quantiles is the confidence signal.

## §7 — Multiple horizons share one backbone

The first-cut model emits all horizons (1 / 3 / 7 min, or
whatever set is chosen) from a single shared backbone with
horizon-specific output heads. Per-horizon separate models are
allowed only as an explicit ablation arm; production model is
multi-horizon.

## §8 — Calibration is a first-class metric

A model with great MAE but terrible calibration is not a usable
predictor — the operator's decision rule depends on the quantile
spread MEANING the stated coverage. Every model card includes a
calibration plot (predicted vs realised quantile coverage per
horizon) and the headline acceptance number is the calibration
gap, not just MAE.

## §9 — `TradedVolumeLadder` features must include zero-handling

When TVL is missing (pre-2026-04-26 data, or post-off rows where
the ladder collapsed), the feature path must zero-fill or mask,
not crash and not silently default to a non-zero. A model that
trained with NaN-poisoned features is not trustable on the
held-out test range. Sessions must include a guard test on a
no-TVL row.

## §10 — Model card per candidate, no exceptions

Every candidate model that gets compared in the evaluation matrix
ships a model card with: architecture, parameter count,
training-data date range, val/test metrics (MAE, calibration gap,
directional accuracy at threshold, stability), training time,
inference time per tick. Candidates without a model card are not
considered for the win.

## §11 — RL handoff is opt-in per-cohort, not global

When the predictor is exposed to RL training (later), it ships as
an opt-in observation feature behind a config flag. A cohort
without the flag set has byte-identical observations to today's
runs. We do NOT change the default observation shape until the
predictor has demonstrated a measured RL win.
