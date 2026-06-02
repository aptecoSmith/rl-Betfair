# BC getting-it-right — master TODO (measured experiment grid)

**START HERE (fresh session):** read `purpose.md` + `hard_constraints.md`
(especially §1 THE METRIC). Do **Step A first** — the metric harness
gates everything. Then **Step B** (the lead lever). Steps C–D only if B
underperforms. Each step is judged on held-out maturation AUC (§1), not
rollout mat%.

GPU: default `--device cuda`. Caches already scanned. Iterate standalone
(`plans/imitation-first/_step1/` scripts), not the cohort runner (§7).

---

## Step A — Fix the measurement (the metric harness). DO FIRST.

**Why first:** without a clean metric we can't tell a good lever from a
lucky rollout. The whole point of this plan.

- [ ] **A1 — Build the labelled maturation dataset.** For train + holdout
  days, `scan_day(..., maturation_label_out=labels)` (already built) →
  every spread-placeable candidate's full predictor obs + matured/not
  flag. Cache it (these are big; reuse the imitation-first scan where
  possible). This is the SAME data the LightGBM probe used — so the
  policy's AUC is directly comparable to 0.76.
- [ ] **A2 — Held-out maturation-AUC evaluator.** Given a policy, run its
  `mature_prob_head` over the holdout candidates and compute per-(tick,
  runner) ROC-AUC vs the matured flag. This is the PRIMARY metric (§1).
- [ ] **A3 — Precision/recall curve.** Sweep the `mature_prob` open
  threshold; for each, report holdout mat% (precision) vs opens-count.
- [ ] **A4 — Baselines on the metric:** (i) LightGBM 0.76 (the ceiling
  reference), (ii) an untrained policy's head (chance, ~0.5), (iii) the
  imitation-first BC's head (probably ~chance — it was unsupervised).

**GATE:** the harness reproduces LightGBM's ~0.76 when handed the same
features (sanity check the metric itself), and cleanly separates an
untrained head (~0.5) from the reference. THEN the metric is trustworthy.

---

## Step B — Lead lever: hard negatives + mature_prob BCE supervision.

**Hypothesis:** supervising `mature_prob_head` with BCE on (matured vs
force-closing-spread) installs the AUC-0.76 signal INTO the policy; the
actor then opens selectively by thresholding it.

- [ ] **B1 — Train `mature_prob_head` (BCE)** on the hard split (§3):
  positive = matured candidate, hard-negative = not-matured-but-placeable.
  input_norm ON. Full-network or head-only — test both; head-only over a
  trained-or-frozen backbone is the cheap first cut.
- [ ] **B2 — Measure held-out maturation AUC** (A2). Compare to 0.76.
- [ ] **B3 — Couple to the actor.** With `mature_prob` calibrated, wire
  the open decision to the `mature_prob_open_threshold` gate; produce the
  precision/recall curve (A3) + one fully-hedged rollout (§1 tertiary).

**GATE (§8):**
- Head AUC ≥ ~0.70 → the signal is in the policy → **proceed to Step E**
  (tune the threshold, confirm rollout).
- Head AUC ≈ 0.60 (well below LightGBM) → the pooled-LSTM actor backbone
  is losing signal that a tree keeps. That's an **architecture finding** →
  Step C/D (or feed the per-runner feature slice to the head directly, as
  the direction-head redesign did — CLAUDE.md phase-15). Do NOT paper over
  it with knobs.

---

## Step C — Lever: target structure (only if B underperforms).

- [ ] Compare the actor's open-decision quality under single-action-per-
  tick CE (current) vs **per-runner binary** (each runner: open / don't).
  The per-runner target matches the actor's per-runner outputs and the
  "a tick can have several maturing runners" reality. Measure on A2/A3.

## Step D — Lever: recurrence (only if B underperforms).

- [ ] BC currently trains on **1-tick zero-hidden samples**; eval runs the
  full LSTM (a train/eval mismatch). Test **short-sequence BC** (train the
  LSTM on real consecutive-tick windows). Measure whether held-out AUC /
  rollout coherence improves. (May matter more for the actor than the
  feed-forward mature_prob head.)

---

## Step E — Selectivity tuning + honest rollout.

- [ ] Take the best config; sweep the `mature_prob` threshold; pick the
  knee of the precision/recall curve on a TRAIN-day sweep (not holdout).
- [ ] Report ONE fully-hedged holdout rollout at that threshold:
  mat% / locked / day_pnl / fc% / opens, fc=120 + close_walk=10.
- [ ] Compare to: imitation-first BC (4% mat%, −£1513/7d) and the random
  baseline (~1% mat%).

---

## Step F — Gate to BC→PPO (imitation-first Step 2 proper).

- [ ] If §8 passes (AUC ≥ ~0.70 AND a threshold gives mat% ≫ baseline,
  locked positive): BC is a sound warm-start. Hand off to the reward-aware
  BC→PPO step (`maturation_reward_mode` + `open_cost`, both already wired)
  — warm-start from this BC's weights. That's a SEPARATE run, operator-
  gated.
- [ ] If §8 fails: write the finding (architecture or data is the limit;
  e.g. the actor bottleneck, or features insufficient → deeper book /
  StreamRecorder work). Do NOT proceed to PPO (it can't fix a signal the
  policy can't represent).

---

## Carry-over (state at plan creation, 2026-05-31)

**Landed + tested (imitation-first):** maturation-conditioned oracle
(`scan_day(maturation_conditioned=...)` + `maturation_label_out`), policy
`input_norm` (opt-in), `maturation_reward_mode` env wiring. Gates 0/0.5/1
PASS (LightGBM holdout AUC 0.76).

**BC diagnosis:** architecture capable (overfit 77%); collapse was the
random negatives + the actor knob; `neg_weight` is a knife-edge
(0.1→849 opens@4% mat%, 0.3→6 opens).

**Artifacts:** `plans/imitation-first/_step1/` — `bc_fullnet_canary.py`
(full-network BC + rollout eval), `bc_overfit_diag.py`,
`maturation_predictability_probe.py` (the LightGBM 0.76 reference),
`obs_audit.py`. Reuse/extend these.

## Verification (every experiment)
- [ ] Reported on held-out maturation AUC (§1), holdout split (§2), input
  norm ON (§5).
- [ ] Hard negatives = force-closing spreads, not random non-opps (§3).
- [ ] No PPO until §8 passes (§9).
