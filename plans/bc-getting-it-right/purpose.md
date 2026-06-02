# BC getting-it-right — purpose

**Created 2026-05-31.** A dedicated plan to make behavioural cloning
actually work for the imitation-first thesis. Read this, then
`hard_constraints.md` (locked decisions — especially the METRIC), then
`master_todo.md` (the measured experiment grid). NOT yet started.

## One-paragraph thesis

The imitation-first diagnostics PROVED the opportunity exists
(`plans/imitation-first/findings.md` Step 0.5: the maturation-conditioned
oracle is ~breakeven on holdout, locked +£559) AND that maturation is
LEARNABLE (Step 1: LightGBM holdout AUC **0.76**, top-decile lift
2.45×). BC is the bridge from "the signal is in the features" to "a
policy acts on it." But the first BC attempts FAILED to produce a
selective, maturation-aware policy — and crucially, they failed in a way
that the crude knob I reached for (down-weighting NOOP negatives) cannot
fix. This plan gets BC right by **fixing the measurement first** (a clean,
confound-free metric) and then testing the **principled levers**
systematically, instead of chasing an unstable knob.

## How we got here (so a fresh session doesn't relitigate)

From the imitation-first session (2026-05-30/31):

- **Architecture is capable.** Overfit test (`_step1/bc_overfit_diag.py`):
  the v2 `DiscreteLSTMPolicy` with input-norm ON reaches **77% train
  accuracy** on a fixed batch of oracle positives in 300 steps, and never
  predicts NOOP when trained on positives only. So obs→which-runner IS
  learnable by this architecture.
- **Full BC collapsed.** With the standard negatives (random non-opens)
  at equal weight → policy collapses to NOOP (opens 0). The 768k NOOP
  negatives are one concentrated action; the OPEN signal is diffused
  across ~14 runner slots, so NOOP's per-action gradient is ~14× stronger.
- **`neg_weight` is a knife-edge, not a dial.** Down-weighting:
  `0.1 → 849 opens @ 4.1% mat%` (4× the ~1% random baseline, but not
  selective, still −£1513/7d); `0.3 → 6 opens @ 0%` (near-collapse). There
  is NO stable selective regime between them. Sweeping it is the wrong
  direction.

## What's actually wrong (the deeper read)

`neg_weight` instability is a symptom. Three load-bearing root causes:

1. **The negatives teach the wrong discrimination.** `NegativeOracleSample`
   is a *random* (tick, runner) not in the positive set — overwhelmingly
   "no profitable spread here." So BC learns *opportunity vs. non-
   opportunity* (easy). It never learns the HARD, decisive discrimination:
   among profitable spreads, **which will MATURE vs. force-close.** That
   maturation discrimination IS the AUC-0.76 signal; the force-closing
   spreads (which look nearly identical to maturing ones) are never shown
   as negatives. The hard negatives already exist in the
   `maturation_label_out` machinery (`training_v2/arb_oracle.py`).

2. **The natural selectivity mechanism is unused.** The policy already has
   a `mature_prob_head` built to forecast per-runner maturation and gate
   opens (`mature_prob_open_threshold`). It is the calibrated, principled
   selectivity lever. The failed BC left it untrained and tried to
   manufacture selectivity from a knob on the actor instead. The right
   move is to SUPERVISE `mature_prob_head` (BCE on maturation labels) —
   literally porting the LightGBM 0.76 signal INTO the policy — and select
   by thresholding it.

3. **We measured with a confounded proxy.** Rollout `mat%` blends runner
   selection + when-to-open + the env fill model + per-race budget + a
   train/eval recurrence mismatch (BC trains on 1-tick zero-hidden
   samples; eval runs the full LSTM). The CLEAN metric is the policy's
   *held-out maturation AUC*, directly comparable to LightGBM's 0.76.

## The plan in one line

**Step A** fix the metric (held-out maturation AUC + a precision/recall
curve) → **Step B** the lead lever (hard negatives + `mature_prob` BCE
supervision; does the policy's head reach ~0.76 held-out AUC?) → **Steps
C–D** secondary levers (target structure; ctx=1 vs sequence recurrence)
ONLY if B underperforms → **Step E** selectivity tuning + one honest
fully-hedged rollout → **Step F** gate to BC→PPO.

Each step is measured against the AUC north-star, not rollout noise.

## Relationship to other plans

- `plans/imitation-first/` is the parent (gates 0/0.5/1 PASS; this plan
  is the "make BC work" follow-on its Step 1c/1d exposed). Its unblockers
  are landed + tested: policy `input_norm` (opt-in) and
  `maturation_reward_mode` env wiring.
- Reuses cohort machinery where possible: `mature_prob_head` +
  `mature_prob_loss_weight` + `mature_prob_open_threshold`,
  `fill_prob_head`, `DiscreteBCPretrainer`, `RolloutCollector`.
- Downstream: a working, SELECTIVE BC is the warm-start for the
  reward-aware BC→PPO step (imitation-first Step 2 proper).

## Success bar (what "BC working" means)

Defined precisely in `hard_constraints.md §8`. In short: the policy's
held-out maturation AUC approaches LightGBM's 0.76 AND there is a
threshold where fully-hedged holdout rollout shows mat% well above the
~1% random baseline with locked positive — a clearly better warm-start
than the imitation-first BC (4% mat%, −£1513/7d). Only then is BC→PPO
warranted.
