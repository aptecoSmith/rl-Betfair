---
plan: rewrite/phase-0-supervised-scorer
status: design-locked
opened: 2026-04-26
---

# Phase 0 — standalone supervised scorer

## Purpose

Build a frozen `P(mature | features)` classifier trained on historical
data. **No RL is touched in this phase.** The deliverable is a model
file plus a calibration plot that says "given this opportunity at
this tick, the probability the open will mature without needing
force-close is X, calibrated to ±10 %".

Phase 1 wires the scorer's outputs into the new actor as features.
Phase 0 is purely supervised: train, calibrate, eval, persist.

## Why supervised, not RL

The `plans/per-runner-credit/findings.md` H1 finding showed the joint-
trained `fill_prob_head` learned a wrong-target label (force-closes
lumped with matures). Joint training is fragile: the head's training
data is whatever the policy happens to do, the labels feedback into
the policy's input next epoch, and the head's coverage is policy-
dependent.

A standalone scorer trained on historical data sees the universe of
opportunities, not just the policy's chosen path. The label is
defined once, carefully, against simulated outcomes. The model is
frozen before the policy ever sees it. None of the H1 failure modes
apply.

## Label definition (locked)

Per opportunity = `(date, market_id, runner_idx, tick_idx, side)`:

```
label = 1.0 iff the agent could open at this tick on this side AND the
        opposite-side passive would naturally fill before the close
        window expires (matured) OR the agent's close_signal would
        successfully cross out of the pair (agent-closed)

label = 0.0 iff the agent could open AND the open would either go
        naked (passive never fills, no close attempted) OR be
        force-closed at T−N seconds

label = NaN (mask out of training) iff the open isn't even feasible
        (no LTP / book empty / hard cap exceeded / budget exhausted)
```

This is the **strict mature label** — the same label the post-H1
`mature_prob_head` uses (CLAUDE.md "mature_prob_head feeds
actor_head"), but applied to the universe of opportunities, not just
the ones the policy chose.

The label is derivable post-hoc by simulating "what if I opened here"
against the historical price book, using the same matcher / sizer /
force-close logic the env uses at training time.

## Feature set (locked at design time, expandable)

**Per-opportunity features** (computed at the candidate open tick):

- Price features: best_back, best_lay, LTP, spread (lay − back),
  spread_in_ticks, mid_price.
- Book depth: top-3-level sizes on both sides, total volume on
  both sides.
- Time features: time_to_off_seconds, time_since_last_trade_seconds.
- Velocity: traded_volume_last_30s, ltp_change_last_30s,
  spread_change_last_30s.
- Side: one-hot back / lay.
- Runner attributes: favourite_rank (1 = lowest LTP), sort_priority,
  ltp_rank_change_last_60s.
- Market attributes: n_active_runners, market_type (one-hot),
  total_market_volume, total_market_volume_velocity.

**Total feature count: ~25–35.** Low enough for a tree model to
handle without dimensionality reduction; high enough to give signal.

Feature engineering iterations (adding LTP-momentum windows, deeper
book features, etc.) happen AFTER Phase 0 ships if calibration is
disappointing.

## Model class (locked: gradient-boosted trees)

**Choice: LightGBM (or XGBoost — interchangeable for this purpose).**

Rationale:

- **Tabular features → trees are the strong baseline.** No
  representation-learning argument here; the features are already
  semantically meaningful.
- **Calibration is a known problem with simple solutions.** Platt
  scaling or isotonic regression on a held-out fold gives well-
  calibrated probabilities.
- **Training is minutes, not hours.** Iteration speed for
  feature/label experiments is critical.
- **Interpretable.** Feature importance plots tell us what's actually
  predictive — useful for Phase 1's actor design.
- **Persistence is trivial.** A `.lgb` or `.json` model file, ~MB-
  scale.

Move to a small NN later only if the tree model plateaus below the
success bar AND we have a concrete reason to believe a NN would do
better.

## Training data shape

Walk every (date, market_id, runner_idx, tick_idx, side) opportunity
in the historical data and emit one row per opportunity that:

- Has a feasible open (LTP exists, book non-empty, side priceable).
- Has resolved within the historical record (we can compute the
  label).

Skip opportunities where the second leg's outcome is censored by
end-of-data.

**Expected scale:** ~12 days × ~300 markets × ~14 runners × ~1000
ticks × 2 sides × ~50 % feasible ≈ 50M rows. Plenty for tree
modelling. If labelling is slow, sub-sample by tick (e.g. every 5th
tick) — the per-tick state changes slowly relative to the close
window so this loses little signal.

**Train / val / test split: chronological, NOT random.**

- Train: earliest 60 % of dates.
- Val: middle 20 % of dates (for early stopping + calibration).
- Test: latest 20 % of dates (for the final eval and the success
  bar).

Random splits leak across-time information and inflate metrics.
Chronological is the only honest split for time-series.

## Success bar (locked)

Phase 0 ships iff ALL of:

1. **AUC on test set ≥ 0.70.** Random guessing is 0.5. The
   `mature_prob_head` post-H1 didn't ship a standalone AUC number,
   but if the supervised problem isn't comfortably above 0.7 the
   whole rewrite premise is wrong.
2. **Calibration within ±10 % across 10-bin probability range.**
   Predicted P=0.3 should match observed mature rate ≈ 0.3, with the
   biggest bin error ≤ ±10 percentage points.
3. **Held-out P&L sanity check.** Apply a greedy threshold policy
   (open whenever predicted P × spread > naked_loss_estimate) and
   simulate against the test days. The sanity check is "is this
   policy non-catastrophic on raw cash P&L?" Not "does it win
   money" (that's Phase 3's bar). Just "doesn't lose £1000s per
   day".

If 1 fails: the data doesn't have predictive signal at the tick
level. Stop the rewrite, reconsider. (This would be a major
finding — it would mean RL was never going to find selectivity
either, and the problem is fundamentally not solvable from this
feature set.)

If 2 fails: tractable. Try isotonic regression on the val set;
re-eval on test. If still failing, file as a follow-on.

If 3 fails: the model is "predictive but the strategy doesn't
work". Surfaces a Phase 1 design question (is greedy-on-score the
right policy class?) — does NOT block Phase 0 shipping.

## Deliverables (Phase 0 closeout)

A new directory `models/scorer_v1/` with:

- `model.lgb` — the trained LightGBM / XGBoost model file.
- `feature_spec.json` — list of feature names, dtypes, and the
  function(s) that compute them from raw env state. **The actor in
  Phase 1 must compute features the same way; this file is the
  contract.**
- `calibration_curve.png` — predicted vs observed mature rate, 10
  bins.
- `feature_importance.png` — top 20 features by gain.
- `eval_summary.json` — AUC, log-loss, calibration error per bin,
  greedy-threshold P&L summary on test days.
- `training_log.txt` — hyperparameters, training time, dataset row
  counts, train/val/test date splits.

A short writeup at `plans/rewrite/phase-0-supervised-scorer/
findings.md` that says: success bars hit / not hit, top predictive
features, surprises (if any), and any Phase 1 design implications
that fall out of the feature-importance plot.

## Out of scope

- Wiring the scorer into a policy (Phase 1).
- Building the new trainer (Phase 2).
- Replacing the existing `mature_prob_head` (it stays in v1 code
  alongside; v2 just doesn't use it).
- Comparing this scorer to `mature_prob_head`'s outputs as a sanity
  check — they're trained differently, comparison is muddy. The bar
  is absolute (AUC ≥ 0.7), not relative to v1.
- Multi-class outputs (e.g. predicting matured vs agent-closed vs
  force-closed separately). Single binary for now; revisit only if
  Phase 1 says it needs the finer granularity.
- Real-time inference performance — we'll measure latency in Phase 1.

## Sessions

1. `01_label_and_feature_design.md` — produce the labelled dataset.
   Build the label generator (walk historical data, simulate "what
   if I opened here" against the matcher, emit row per opportunity
   + outcome). Build the feature extractors. Persist a parquet
   dataset. **No model training in this session.**
2. `02_train_and_evaluate.md` — train the LightGBM model, calibrate,
   evaluate against the success bars, persist artefacts, write
   findings. **No feature engineering in this session.** If the
   model fails the bar, don't hack features — stop and discuss.

Each session is independently re-runnable. The dataset from session
1 is the input to session 2; session 2 doesn't re-generate the
dataset.
