---
plan: price-direction-predictor
status: draft
created: 2026-05-09
revised: 2026-05-09 (matrix exploration restructure)
motivated_by: discussion 2026-05-09 — split RL training from market prediction
related:
  - plans/fill-prob-in-actor/ (in-policy aux head, the precedent we're stepping back from)
  - plans/per-runner-credit/findings.md (root cause: fill_prob label conflated force-close with maturation)
  - plans/arb-curriculum/session_prompts/01_oracle_scan.md (existing labelling pipeline shape)
note: this plan is STANDALONE — it does not depend on, supersede, or
  coordinate with `plans/rewrite/phase-16-ensemble-market-state/`.
  That phase upgrades the in-policy aux-head pipeline; this plan
  builds a separate, frozen, supervised model OUTSIDE the RL loop.
  Both can coexist; neither is a prerequisite for the other.
---

# Purpose — train a standalone price-direction predictor, then feed it to RL

## End goal — the deliverable in one sentence

A signal that says, on a specific horse at a specific tick:

> "the price for runner R is going to move IN/OUT by approximately X
> ticks (with confidence Y%) over the next N minutes."

Per-runner, per-tick, multiple horizons N, quantile-style confidence,
**pre-off only**, computed from polled-feed observations.

## Observation that motivated this plan

The current policy stack has three auxiliary supervised heads
(`fill_prob_head`, `mature_prob_head`, `risk_head`) jointly trained
with the PPO actor on per-cohort RL data. Each predicts a different
flavour of "what will happen if I trade this runner" and feeds its
output back into `actor_head`. This is an **advisor architecture**
grafted onto an RL loop, with two structural problems:

1. **Label provenance is awkward.** `fill_prob` and `mature_prob`
   labels require the simulator to run hypothetical orders on real
   ladders (the oracle-scan pipeline). Sim/reality gap is a confound
   and labels can only be generated for ticks the simulator can
   replay.
2. **Co-training confounds two distinct ML problems.** A predictor
   that helps PPO converge faster is not the same as a predictor
   that is well-calibrated on its own task. We can't measure either
   cleanly because the gradient pathway entangles them.

A separately-trained, frozen, well-evaluated predictor sidesteps
both, and is also useful as a trading signal in its own right
without needing an RL agent on top.

## Operating principle for this plan — explore, don't lock in

The operator's standing instruction (2026-05-09):

> "make sure that in your plan you are trying different things and
> generating enough data to make decisions — don't just lock yourself
> into one path"

This plan therefore runs as a **matrix of experiments anchored on a
shared scoreboard**, not as a linear "build the v1" pipeline. Every
session emits scoreboard rows; downselection happens late and is
based on data, not premature commitment. The scoreboard is the
durable artefact that informs every decision.

Concretely:

- Multiple **architectures** are tried in parallel (MLP baseline,
  GBM baseline, LSTM, small Transformer, 1D conv) — picking one
  upfront would foreclose options the data hasn't ruled out.
- Multiple **output formulations** are tried (quantile regression,
  parametric distributional, signed-magnitude classification).
- Multiple **feature variants** are tried (top-3 ladder only,
  + TVL, + cross-runner z-scores, + market-state aggregates).
- Multiple **horizon sets** are tried ({3m,7m,15m},
  {1m,3m,7m}, {7m only}, {3m,7m,15m,30m}).
- Multiple **smoothing strategies** are tried (raw, post-hoc EMA,
  temporal-consistency loss).

Each cross is evaluated on the same val set with the same metric
suite, so candidates are directly comparable. The held-out test
date range is touched ONCE per candidate at the end (§5).

## Scope — what this plan builds, and what it does not

### Builds

1. A multi-horizon, multi-feature-variant labelling pipeline.
   Self-supervised: labels are read straight from future
   `LastTradedPrice` in the parquet. Persisted to disk so all
   experiments share the same (features, labels) data.
2. An experiment harness with a scoreboard CSV. Config-driven.
   One config file → one row in the scoreboard.
3. A matrix of training experiments across the axes above, all
   landing in the scoreboard.
4. A non-RL backtest harness that runs candidate predictors
   through a simple decision rule and reports realised P&L
   on val dates. The operator's safety net — even if RL handoff
   later fails, the predictor may already be a usable signal.
5. A visualisation tool: per-race, per-runner prediction
   trajectory over time. Catches the "oscillates wildly" failure
   mode by inspection in addition to the stability metric.
6. A frozen-handoff path: the winning predictor exposed as
   per-runner observation features behind a config flag for the
   RL env (opt-in, behind a flag — no global default change).

### Does not

- Change PPO training. Cohort runs continue unchanged on the
  existing aux-head architecture during this plan's duration.
- Deprecate `fill_prob_head` / `mature_prob_head` / `risk_head`.
  Those decisions wait until the standalone predictor has
  measured its win on its own terms AND the RL handoff has shown
  measurable improvement.
- Touch live-feed inference. The `ai-betfair` repo can consume
  the frozen model later via a separate handoff plan.

## Hard scope decisions (operator confirmed 2026-05-09)

These are NOT search axes — they're fixed.

- **Target type:** signed Δprice in Betfair price ticks, per
  (runner, tick), at multiple horizons. No fill-prob. No
  mature-prob. No locked-P&L variance.
- **Pre-off only:** `in_play=False` AND
  `timestamp < market_start_time`. In-play is a different problem.
- **Output is quantile-style:** even when we explore parametric
  distributional outputs, every candidate emits at minimum the
  10th / 50th / 90th percentile of Δprice per (runner, horizon)
  for like-for-like comparison.
- **`TradedVolumeLadder` mandatory in at least one feature
  variant.** Available from 2026-04-26 onwards (10 days of
  training corpus).
- **Frozen handoff to RL.** When wired in, predictor weights are
  fixed. No joint optimisation.
- **Test date range is touched ONCE per candidate.** Hyperparameter
  search and architecture comparison happen on the val set.

## Search axes (these ARE explored, not pre-decided)

1. **Architecture:** MLP, GBM (XGBoost / LightGBM), LSTM,
   Transformer (ctx 32–64), 1D conv. Each family run at THREE
   sizes (small / medium / large) so we can read the
   per-family scaling curve as well as the cross-family
   comparison. Large-size cap is ~1M trainable parameters
   (GBM cap 500 trees, depth 6) to keep training cheap and to
   avoid one giant model winning by capacity alone.
2. **Output formulation:** quantile regression with pinball loss
   (3-quantile and 5-quantile variants), parametric Gaussian
   (mean+log-var with NLL), parametric Student-t (heavier tail),
   signed-magnitude classification (drift-large/drift-small/flat/
   shorten-small/shorten-large bins).
3. **Feature variants:**
   - V1: top-3 ladder (back+lay) + LTP + time-to-off only
   - V2: V1 + per-runner trailing window (last 32 ticks of LTP,
     ladder mid, total volume)
   - V3: V2 + `TradedVolumeLadder` features (cumulative volume
     bucketed by ticks-from-LTP)
   - V4: V3 + cross-runner features (rank, share, z-score within
     market) — borrowed conceptually from phase-16's session 03
     but built independently
   - V5: V4 + market-state features (vol estimate, depth, spread)
4. **Horizon sets:** {3m,7m,15m} (recommended default),
   {1m,3m,7m}, {7m only}, {3m,7m,15m,30m}.
5. **Smoothing:** raw output, post-hoc EMA on quantile outputs,
   temporal-consistency loss baked into training.
6. **Training-data scope:** TVL-required (10 days post-2026-04-26)
   vs TVL-mask (29 days, mask the TVL features when missing).

The matrix is large but most cells are cheap. We do NOT run every
combination — Sessions 03–07 stage the search so that each session's
sweep is informed by the previous session's scoreboard rows.

## Success criteria — how a candidate "wins"

A candidate is a viable predictor if all hold on the held-out test
date range:

1. **Calibration.** 80% of realised Δprice values fall within the
   10th–90th percentile band (target ±5pp). Equally-bad failure
   modes: over- and under-confident.
2. **Directional accuracy at confidence threshold.** When `q50 ≥ K
   ticks shorten AND q10 ≥ 0`, realised Δprice has the same sign
   ≥ 70% of the time at a frequency where the threshold actually
   fires. Both halves matter — the model has to be confident often
   enough AND right when confident.
3. **Stability.** Lag-1 prediction autocorrelation ≥ 0.7. Catches
   the "oscillates wildly" failure mode the operator flagged.
4. **Backtest sanity.** A naïve "act on top-K confident
   predictions" rule has positive realised P&L on val dates after
   commission. Sanity check, not a tuning target.

A candidate that passes (1)–(3) but fails (4) is recorded as a
"calibrated-but-not-actionable" result and kept on the bench — a
later decision rule may extract value the naïve rule cannot.

## Why this matters even before any RL handoff

Two operating modes the predictor enables, in order of immediacy:

1. **Trading signal in its own right.** A calibrated multi-horizon
   price-direction predictor + a simple decision rule is itself a
   trading strategy. We can paper-trade it the day it's frozen.
2. **Richer observation features for the RL loop.** Once frozen,
   the predictor's quantile outputs become per-runner observation
   features. The RL agent gets to learn against a richer state
   without us ever co-training a supervised head with PPO again.

## Open questions resolved by experiments, not by argument

- Does TVL meaningfully help, or is the top-3 ladder sufficient?
- Are sequence models worth the complexity vs a strong tabular
  GBM on engineered features?
- Is a single multi-horizon backbone better than one model per
  horizon?
- Where is the calibration vs. accuracy tradeoff sitting?
- Does the operator-flagged "oscillation" failure mode actually
  appear in our raw outputs, or is it solved for free by a
  sequence model with built-in memory?

These are answered by scoreboard rows, not by debate.
