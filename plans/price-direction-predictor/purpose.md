---
plan: price-direction-predictor
status: draft
created: 2026-05-09
motivated_by: discussion 2026-05-09 — split RL training from market prediction
related:
  - plans/fill-prob-in-actor/ (in-policy aux head, the precedent we're stepping back from)
  - plans/per-runner-credit/findings.md (root cause: fill_prob label conflated force-close with maturation)
  - plans/arb-curriculum/session_prompts/01_oracle_scan.md (existing labelling pipeline shape)
---

# Purpose — train a standalone price-direction predictor, then feed it to RL

## Observation that motivated this plan

The current policy stack has three auxiliary supervised heads
(`fill_prob_head`, `mature_prob_head`, `risk_head`) jointly trained
with the PPO actor on per-cohort RL data. Each predicts a different
flavour of "what will happen if I trade this runner" and feeds its
output back into `actor_head`. This is an **advisor architecture**
grafted onto an RL loop:

- The aux heads are supervised forecasters.
- Their outputs become per-runner features for the policy.
- They co-train with the policy on the same small RL dataset.

Two problems with the joint formulation:

1. **Label provenance is awkward.** `fill_prob` and `mature_prob`
   labels require the simulator to run hypothetical orders on real
   ladders (the oracle-scan pipeline), then BCE against the simulator
   output. Sim/reality gap is a confound; labels can only be generated
   for ticks the simulator can replay.
2. **Co-training confounds two distinct ML problems.** A predictor that
   helps PPO converge faster is not the same as a predictor that is
   well-calibrated on its own task. We can't measure either cleanly
   because the gradient pathway entangles them.

A separately-trained, frozen, well-evaluated predictor sidesteps both.

## What this plan builds

A standalone supervised model that predicts **signed Δprice in
Betfair price ticks over horizon N**, per runner, per tick, on
**pre-off** snapshots only. Output is a quantile prediction
(10th/50th/90th) so the consumer has confidence directly. Inputs come
straight from parquet — top-3 ladder, LTP, `TradedVolumeLadder`,
time-to-off, plus a short window of recent history for sequence
context. No simulator in the labelling loop.

Once trained, the predictor is:
- **Frozen** — its weights are fixed; no joint optimisation.
- **Reusable** — RL agents read it as additional observation features,
  but a non-RL decision rule (open if median-Δprice ≥ k AND 10th-
  percentile-Δprice ≥ 0) is also a valid consumer.
- **Evaluable on its own terms** — MAE/calibration/directional
  accuracy on a held-out date range, independent of any RL run.

## Why this is the right cut

1. **Self-supervised label.** Δprice over N ticks is read directly
   from `LastTradedPrice` at tick `t+N`. No simulator, no oracle, no
   BCE-against-counterfactual. Cheap to relabel any time we want a
   different horizon.
2. **Cheap iteration.** The 29 days of parquet data we already have
   gives ~5,000 markets × ~150 pre-off ticks × ~7 runners ≈ 5M label
   examples. A small sequence model trains in minutes on that.
3. **Independent evaluability.** Held-out dates → train/val/test
   splits with no simulator coupling. A bad predictor fails MAE on
   the val set; we know immediately, without burning an RL run.
4. **Composable handoff.** The frozen predictor's outputs are extra
   observation features. The RL agent doesn't change shape; it just
   sees richer state. Pre-existing aux-head wiring can be deprecated
   gradually.
5. **Useful even without RL.** A calibrated price-direction predictor
   is a trading signal end-to-end — it does not require an RL agent
   to extract value. RL remains the long-term automation goal but is
   no longer on the critical path for proving the predictor works.

## Hard scope decisions (operator confirmed 2026-05-09)

- **Single target type:** Δprice over N ticks. No fill-prob, no
  mature-prob, no locked-P&L variance — those stay on the in-policy
  heads (and are not addressed by this plan).
- **Multiple horizons:** model emits predictions at several horizons
  (recommended: 1 min, 3 min, 7 min) sharing a backbone. Cost is
  ~3× the output dim, ~zero extra training cost.
- **Quantile output:** model emits 10th / 50th / 90th percentile of
  Δprice per (runner, horizon). Quantile regression with pinball
  loss, three quantiles per output. Confidence is read off the
  quantile spread (90th − 10th).
- **`TradedVolumeLadder` is mandatory input.** Per-runner cumulative
  traded volume per price level, available from 2026-04-26 onwards
  (10 days of training corpus). The 19 earlier days remain available
  for an ablation arm without TVL features.
- **Pre-off only.** `in_play=False` AND `timestamp < market_start_time`.
  In-play prediction is a different problem.
- **Build a matrix, evaluate, pick.** Recommend trying:
  - Architectures: small LSTM, small Transformer (ctx 32–64), 1D conv
    over recent ticks, bag-of-features MLP baseline.
  - Smoothing: raw output vs. EMA-smoothed output vs. temporal-
    consistency loss. Raw + post-hoc EMA is the operator-friendly
    default; the loss-baked variant is the principled comparator.
  - With/without TVL features (10-day vs 29-day corpus tradeoff).

## What this plan does NOT do

- No changes to PPO training. The current cohort runs continue
  unchanged on the existing aux-head architecture during this plan's
  duration; this plan is a parallel build, not a refactor.
- No deprecation of `fill_prob_head` / `mature_prob_head` / `risk_head`.
  Those decisions wait until the standalone predictor has a measured
  win on its own terms.
- No live-feed integration. The `ai-betfair` repo can consume the
  frozen model later via a separate handoff plan.

## Success criteria (predictor stage)

A predictor passes if all of the following hold on a held-out test
date range:

1. **Calibration.** 80 % of realised Δprice values fall within the
   model's 10th–90th percentile band (target ±5 pp). Over- or under-
   confident predictions are equally bad.
2. **Directional accuracy at confidence threshold.** When the median
   prediction is ≥ K ticks shorten AND the 10th percentile is ≥ 0,
   the realised Δprice is the same sign ≥ 70 % of the time at the
   matching frequency where the threshold actually fires (ie: the
   model has to BOTH be confident often enough AND right when
   confident).
3. **Stability.** Cross-correlation of predictions at consecutive
   ticks is ≥ 0.7. (Catches the "oscillates wildly" failure mode the
   operator flagged.) A sequence model should pass this trivially;
   a stateless MLP probably will not.
4. **Backtest sanity.** A naïve strategy "open back when median
   prediction ≥ +5 ticks AND 10th percentile ≥ 0; close at horizon"
   has positive realised P&L on the held-out test dates. This is a
   sanity check, not a tuning target.

## Open questions for the design phase

- Quantile spacing — are 10/50/90 enough, or should we predict more
  (5/25/50/75/95) to support different decision rules?
- Per-runner vs. per-market normalisation — Δprice in ticks is
  already comparable across runners (the tick ladder normalises),
  but is there a runner-specific volatility scaling that helps?
- How long is "recent history" — 16, 32, or 64 ticks?
- One model for all horizons (multi-head) or one model per horizon
  (separate training runs)? Multi-head is the recommendation, but
  worth confirming on a small ablation.

These get resolved by the labelling-pipeline prototype + first
model-training pass (Session 02).
