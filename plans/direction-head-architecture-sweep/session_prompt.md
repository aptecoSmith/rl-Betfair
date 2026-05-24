# Session prompt — direction-head architecture sweep

Self-contained brief for a fresh session to investigate whether
the **v1 shared frozen direction head** (`models/direction_head/
v1_2026-05-24/`) is the best architecture, or whether a small
sweep finds something materially better.

The predictor architecture is a separate concern — noted at the
bottom; this sweep is JUST the head.

---

## TL;DR — what you're doing

The v1 head is a 2-layer MLP (LayerNorm → Linear(23, 64) → ReLU
→ Linear(64, 2)) trained on 1.03M (per-runner obs, direction
label) pairs across 16 cohort training days. Held-out eval on
2026-04-10 gives:

```
                  Pearson(pred,label)   ROC AUC   unwgt BCE descent
back              +0.231                0.673     -5.4%
lay               +0.306                0.734     -12.1%
```

Good ranking signal (AUC ~0.7), but uniformly over-confident
(reliability table shows predicted prob ≈ 2× empirical rate per
bucket). Cause: pos_weight class-balancing during training.

**Question this sweep answers:** can a different architecture
and/or training-loss variant deliver materially better metrics?
Specifically: (a) better AUC / Pearson → more discriminative
signal for the agent; (b) better calibration → predicted probs
that mean what they say.

## Where to pick up

1. **Read first** (~10 min):
   * `plans/shared-direction-head/purpose.md` + `findings.md` — why
     the head exists, what we expect from it
   * `scripts/train_direction_head.py` — current training script
   * `scripts/evaluate_direction_head.py` — current evaluator
   * `models/direction_head/v1_2026-05-24/manifest.json` — baseline
     manifest

2. **Verify state of the world** (~2 min):
   * Master HEAD should be ≥ `09a488e` (commit lands the v1 head
     + tools).
   * `data/oracle_cache_v2/<training_day>/` exists for all 16
     training days with obs_dim=574 and predictor columns
     populated (re-scanned today with --use-direction-predictor).
   * `data/oracle_cache_v2/2026-04-10/` also re-scanned with the
     predictor flag (so the eval works). For more eval days,
     repeat the scan command in §"Reference commands" below.

## The five (six) candidates

| Variant | Goal | Architecture / training change |
|---|---|---|
| **C0 baseline** | Reference point | The committed v1 head (LayerNorm → Linear(23, 64) → ReLU → Linear(64, 2), pos_weight=balanced) |
| **C1 wider** | More capacity in 1 layer | Linear(23, 256) → ReLU → Linear(256, 2). 4× the hidden dim. |
| **C2 deeper** | More layers, less width | Linear(23, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 2). |
| **C3 no-pos-weight** | Address over-confidence | C0 architecture; train with `pos_weight=1` (unweighted BCE). Predicted probs should be calibrated to the 18% positive rate. Expect lower BCE but possibly lower AUC. |
| **C4 dropout + batchnorm** | Reduce overfit | LayerNorm → Linear(23, 128) → BatchNorm → ReLU → Dropout(0.2) → Linear(128, 2). Regularised wider variant. |
| **C5 full obs input (stretch)** | Test the architecture-vs-input-features question | Same architecture as C0, but input is the FULL 574-dim obs, not just the 23 per-runner block. Requires changing the policy's `direction_prob_head` call site (it currently slices per-runner). This is a more invasive change — DEFER unless C0-C4 leave clear performance on the table. |

Pick the 5 winners by the metric described in §"Acceptance".

## How to run the sweep

For each candidate, do:

1. **Modify `scripts/train_direction_head.py`'s `DirectionHead` class**
   (or branch the file as `train_direction_head_<candidate>.py`)
   to use the candidate's architecture. Add the relevant flags
   to control pos_weight, dropout, etc.

2. **Train on the SAME training days** as the baseline
   (the 16 dates in `models/direction_head/v1_2026-05-24/
   manifest.json::training.training_dates`). Same hyperparameters
   otherwise — `--epochs 50 --lr 1e-3 --batch-size 4096 --patience
   5 --seed 42` — so the architecture is the variable.

3. **Save to `models/direction_head/sweep_<candidate>/`** with a
   manifest.

4. **Evaluate on a panel of held-out eval days.** Per
   `hard_constraints.md §1` from shared-direction-head, NO eval day
   can be in the training set. Use multiple eval days to reduce
   noise — recommend the cohort's 10 eval days (2026-04-07, 04-10,
   04-14, 04-17, 04-21, 04-23, 04-25, 05-01, 05-03, 05-06). Each
   needs its oracle cache re-scanned with --use-direction-predictor
   FIRST (see Reference commands).

5. **Tabulate**, for each candidate × each eval day:
   * Pearson(pred, label_back), Pearson(pred, label_lay)
   * ROC AUC back, ROC AUC lay
   * Unweighted BCE back, lay
   * (Optional) Brier score, reliability diagonal-distance
     summary (mean |predicted_prob_bucket_midpoint - empirical_rate|
     across the 10 reliability buckets)

   Then average across eval days for each candidate's
   "out-of-sample" score.

## Acceptance criterion (pick the winner)

The candidate is "better" if it improves Pearson and ROC AUC on
held-out eval AND doesn't make calibration meaningfully worse. The
exact ranking rule:

1. Primary: **mean Pearson(pred, label) averaged across both sides
   and all eval days.** Higher is better.
2. Tie-break: **mean ROC AUC, same average.**
3. Sanity: **mean Brier score must not regress by > 10 % relative
   vs baseline.** If a candidate has higher Pearson but much worse
   Brier, the calibration regression matters and disqualifies it.

The "winner" gets a writeup: which candidate, by how much, why
(speculate based on the architecture difference), and a
recommendation to promote it to the next cohort's
`--direction-head-manifest`.

If C0 (baseline) wins, write that up too — knowing the simple
2-layer MLP is already best is useful information.

## Expected wall time

* Each training: ~5-10 min on GPU
* Each evaluation across 10 days: ~30 seconds total
* 5 candidates × (train + eval) ≈ 1-2 hours total
* Plus the eval-day re-scans (one-off, ~5 min for all 10 days)

Should fit in a single 2-3 hour session.

## What success looks like

By end of session, the plan dir has:

```
plans/direction-head-architecture-sweep/
    session_prompt.md       (this file)
    findings.md             (full results table + winner)
    candidate_c0_results.md  ← optional per-candidate detail
    candidate_c1_results.md
    ...
```

Plus `models/direction_head/sweep_<winner>/` if it's not C0.

The 12 × 3 full cohort relaunch (which is currently waiting on
go-ahead) should then use whichever head wins, via
`--direction-head-manifest models/direction_head/<winner>/`.

## Reference commands

Re-scan eval-day oracle caches with predictor injection (idempotent
— skip days already done):

```
python -m training_v2.oracle_cli scan \
  --dates 2026-04-07,2026-04-14,2026-04-17,2026-04-21,2026-04-23,\
2026-04-25,2026-05-01,2026-05-03,2026-05-06 \
  --predictor-lean-obs \
  --predictor-bundle-manifests \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/race-outcome-ranker/manifest.json \
    C:/Users/jsmit/source/repos/betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor
```

(2026-04-10 is already done.)

Scan direction labels for any missing eval day:

```
python -m training_v2.direction_label_cli scan \
  --date 2026-04-XX --horizon-ticks 60 --threshold-ticks 5 \
  --force-close-before-off-seconds 60
```

Train a variant (after modifying the architecture):

```
python -m scripts.train_direction_head \
  --training-dates 2026-04-06,2026-04-08,2026-04-09,2026-04-11,\
2026-04-12,2026-04-13,2026-04-15,2026-04-16,2026-04-19,2026-04-20,\
2026-04-22,2026-04-24,2026-04-26,2026-05-02,2026-05-04,2026-05-05 \
  --output-dir models/direction_head/sweep_<candidate> \
  --experiment-id <candidate> \
  --hidden <N> --epochs 50 --lr 1e-3 --batch-size 4096 --patience 5
```

Evaluate:

```
python -m scripts.evaluate_direction_head \
  --manifest models/direction_head/sweep_<candidate> \
  --eval-dates 2026-04-07,2026-04-10,2026-04-14,... (all 10 eval days)
```

## Constraints

* **Held-out invariant.** No eval day enters any candidate's
  training set. The train script's pre-flight enforces this — DO
  NOT disable it.
* **Same training days for every candidate.** The "all else equal"
  variable being tested is architecture, not data.
* **No new data sources.** This sweep doesn't pull additional
  features; it varies only the head architecture and training-loss
  variant.
* **Manifest discipline.** Every saved candidate gets a manifest
  with `commit_sha`, `training_dates`, val metrics. So the result
  is reproducible.

---

## Out of scope: predictor architecture sweep

A parallel investigation of the upstream `betfair-predictors`
Conv1D direction model would also be interesting. **Not for this
session.** Reasons:

* Lives in a separate repo: `C:/Users/jsmit/source/repos/
  betfair-predictors/`. Different conventions, different training
  infrastructure (`scripts/predictor/train_one.py` and friends).
* Training cost per variant is hours not minutes.
* The predictor's outputs are consumed at the env's
  `_compute_tick_predictor_outputs` — any predictor change
  requires re-scanning every cache (train + eval + monitor) so
  the cohort sees the new outputs.
* Re-running every cohort downstream to revalidate.

Queue as a separate plan if the head sweep shows the head is
already extracting most of the available signal — at that point
the bottleneck shifts to the predictor's quality.

Concretely the predictor's candidate architectures to consider
would be:

* Current (Conv1D k=3, 4 layers, 64 channels)
* Deeper Conv1D
* Conv1D + attention
* Small Transformer encoder
* LSTM
* MLP on flattened ladder window

But all of those need to be run inside `betfair-predictors` with
its training pipeline (`scripts/predictor/run_all_sessions_neural.py`
is the entry point). Out of scope here.

---

## State of the world at plan creation

* Master HEAD: `09a488e`
* Last cohort: stopped (`_predictor_SCALPING_full_features_cohort_
  1779622853`) — partial gen-1 data on disk.
* Smoke probe with frozen v1 head: PASSED (registry/
  `_smoke_shared_head_1779635753`)
* The next full cohort (12 × 3) is WAITING on go-ahead AND on
  this architecture sweep, IF the operator wants to use whichever
  variant wins.

Sanity checks before any training run in this session:

* `nvidia-smi` shows GPU free.
* No `cohort.runner` process is running.
* `git status` is clean (or the operator has explicit pending
  edits they want to keep).
