# Session prompt — Phase 0, Session 02: train, calibrate, evaluate

Use this prompt to open a new session in a fresh context. The prompt
is self-contained — it briefs you on the task, the design decisions
already locked, and the constraints. Do not require any context from
the session that scaffolded this prompt.

---

## The task

Train a LightGBM (or XGBoost) classifier on the dataset produced by
Session 01, calibrate its probabilities, evaluate against the
locked success bars, and persist artefacts. **No feature
engineering in this session.** If the model fails the success bars,
stop and discuss — don't hack new features in.

Output: `models/scorer_v1/` directory with:

- `model.lgb` (or `.json`).
- `feature_spec.json` (copied / re-generated from Session 01's
  spec to lock the contract).
- `calibration_curve.png`.
- `feature_importance.png`.
- `eval_summary.json`.
- `training_log.txt`.

Plus a writeup at `plans/rewrite/phase-0-supervised-scorer/
findings.md` covering success bars, top features, and Phase 1
implications.

## What you need to read first

1. `plans/rewrite/phase-0-supervised-scorer/purpose.md` — locked
   model class (LightGBM/XGBoost), success bars, deliverables.
   **The success bars are not up for debate; if the model fails,
   stop and discuss with the operator.**
2. `plans/rewrite/phase-0-supervised-scorer/session_01_findings.md`
   — Session 01's writeup. Pay attention to any feature
   definition judgement calls or label-rate observations.
3. The feature dataset itself: `data/scorer_v1/dataset.parquet`
   (or sharded equivalent).
4. `data/scorer_v1/feature_spec.json` — feature ordering /
   dtypes. Phase 1 will read the same file.

## What to do

### 1. Load the dataset and apply the chronological split (~10 min)

```python
import pandas as pd
df = pd.read_parquet("data/scorer_v1/dataset.parquet")
df = df.dropna(subset=["label"])  # drop infeasible rows

dates = sorted(df["date"].unique())
n = len(dates)
train_dates = dates[: int(n * 0.6)]
val_dates   = dates[int(n * 0.6): int(n * 0.8)]
test_dates  = dates[int(n * 0.8):]

train = df[df["date"].isin(train_dates)]
val   = df[df["date"].isin(val_dates)]
test  = df[df["date"].isin(test_dates)]
```

Print row counts per split. Print label balance per split. **If
test set has < 100k rows or label balance < 5 % minority class,
stop** — the dataset is too small for the success bar's AUC
threshold to mean anything.

### 2. Train the model (~20 min)

Hyperparameters — start with defensible defaults, NOT a search:

```
learning_rate    = 0.05
num_leaves       = 63
max_depth        = -1   (let leaves bound it)
min_data_in_leaf = 1000  (we have lots of data; protect against overfitting)
feature_fraction = 0.9
bagging_fraction = 0.9
bagging_freq     = 5
n_estimators     = 5000  (with early stopping)
early_stopping_rounds = 100  on val_set log-loss
```

If LightGBM, use `objective='binary'`, `metric='binary_logloss'`.

Log:

- Train + val log-loss curves (per round).
- Final n_estimators after early stopping.
- Wall time.
- Train + val AUC.

If val AUC < 0.65 or train AUC < 0.70 → the model is underfitting
or the data is fundamentally low-signal. **Stop and discuss before
calibrating.** Don't try to tune your way out — the success bar is
absolute and re-tuning rarely moves AUC by more than 0.02.

### 3. Calibrate (~15 min)

Trees produce raw probabilities that are NOT well calibrated by
default. Two options:

- **Platt scaling** — fit a logistic regression on val-set
  predictions to map raw → calibrated probabilities. Simple but
  less flexible.
- **Isotonic regression** — fit a step function on val-set
  predictions. More flexible, handles non-monotone miscalibration.
  Better default for trees.

Use `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`.
Fit on val. Apply to test predictions.

Plot calibration curve: bin test-set raw and calibrated predictions
into 10 equal-width bins, plot bin midpoint (predicted) vs observed
mature rate. Save as `calibration_curve.png` with both curves
overlaid.

### 4. Evaluate against the success bars (~15 min)

**Bar 1: Test AUC ≥ 0.70.**

```python
from sklearn.metrics import roc_auc_score
test_auc = roc_auc_score(test["label"], calibrated_test_predictions)
```

**Bar 2: Calibration error within ±10 % across 10 bins.**

For each bin, compute `|predicted_mid_bin − observed_mature_rate|`.
Max bin error must be ≤ 0.10. Bins with < 100 samples should be
flagged separately (they're just noisy) but don't count against the
bar.

**Bar 3: Greedy-threshold P&L sanity check.**

Apply this policy to the test days, simulating against the
historical book using the same env machinery the dataset used:

```
for each (date, market_id, runner_idx, tick_idx, side):
    p = scorer.predict(features)
    expected_spread = (best_lay - best_back)  # at the open tick
    expected_naked_loss = ~£1 (or learn from data — see below)
    EV = p * expected_spread - (1 - p) * expected_naked_loss
    if EV > 0 AND budget_remaining > MIN_BET_STAKE:
        open the pair
```

Tune the threshold to maximise raw P&L on val days, then evaluate
on test days. The bar is "non-catastrophic" — total P&L per day on
test should be in the −£100 to +£100 range, not −£1000s. We are
NOT requiring profitability; we're requiring "the predictions are
consistent enough that a naive policy doesn't blow up".

If estimating naked loss from data: compute mean(naked_loss) per
side from the training set's label = 0 / outcome = naked rows.
This is `~spread × stake × directional_drift` in expectation.

### 5. Persist artefacts (~15 min)

- `models/scorer_v1/model.lgb` (or `.json`) — the trained model.
- `models/scorer_v1/feature_spec.json` — copied from
  `data/scorer_v1/feature_spec.json` so the model directory is
  self-contained.
- `models/scorer_v1/calibration_curve.png`.
- `models/scorer_v1/feature_importance.png` — top 20 features by
  gain. Use LightGBM's built-in plot.
- `models/scorer_v1/eval_summary.json`:

  ```json
  {
    "test_auc": 0.7XX,
    "test_log_loss": 0.XXX,
    "calibration": {
      "max_bin_error": 0.0XX,
      "bin_errors": [0.0X, ...],
      "bin_counts": [XXXXX, ...]
    },
    "greedy_pnl_test": {
      "per_day": {"2026-04-XX": +12.34, ...},
      "total": +XX.XX,
      "threshold_used": 0.XX
    },
    "n_train_rows": XXXXXXX,
    "n_val_rows": XXXXXXX,
    "n_test_rows": XXXXXXX,
    "n_estimators_after_early_stop": XXX
  }
  ```

- `models/scorer_v1/training_log.txt` — hyperparameters used,
  training wall time, environment versions (`lightgbm.__version__`,
  `sklearn.__version__`, etc.).

### 6. Tests (~30 min)

Under `tests/test_scorer_v1_inference.py`:

- Loading the model file produces a working `predict` callable.
- `predict` on a known feature vector produces a deterministic
  probability (within float tolerance).
- Calibrated predictions are in [0, 1].
- The feature_spec.json has the same names / dtypes that the
  model expects.

These are the regression guards that protect Phase 1 — the actor
will load this same model file and rely on the same feature
contract.

### 7. Write up (~30 min)

`plans/rewrite/phase-0-supervised-scorer/findings.md`:

- Success bar status: Bar 1 (AUC), Bar 2 (calibration), Bar 3
  (P&L sanity). PASS / FAIL each, with the actual numbers.
- Top 20 features by importance. Surprises (if any).
- Verdict:
  - **GREEN** — all three bars pass, Phase 1 unblocked.
  - **AMBER** — Bar 1 / Bar 2 pass, Bar 3 fails (model is
    predictive but greedy-threshold strategy doesn't work).
    Phase 1 still proceeds; the scorer is a useful feature
    even if greedy-on-score isn't the right policy class.
  - **RED** — Bar 1 fails. Stop. The data doesn't support the
    rewrite premise; escalate to operator.
- Phase 1 implications. Specifically: which features
  were most important? Does that suggest the actor needs
  particular state inputs we don't currently surface? Are
  there features that look load-bearing but only available
  late (e.g. velocities computed over the last 30s) — Phase 1
  needs to provide them online.

## Stop conditions

- All three bars pass → write `findings.md` GREEN, message
  operator "Phase 0 GREEN, ready for Phase 1", **stop**.
- Bar 1 fails → write `findings.md` RED, message operator
  "Phase 0 RED — data lacks predictive signal", **stop**. Do not
  iterate.
- Bar 2 fails despite isotonic calibration → file as a finding,
  try Platt scaling once, if still failing write `findings.md`
  AMBER, message operator, **stop**. Don't redesign.
- Bar 3 fails alone → write `findings.md` AMBER (the scorer is
  fine; the greedy strategy is the wrong policy class — that's
  Phase 1's problem to solve). Phase 1 proceeds.

## Hard constraints

- **No feature engineering in this session.** If you find yourself
  wanting to add a feature, that's a Session 01 follow-on — file
  the request and stop.
- **No model class change in this session.** If LightGBM/XGBoost
  fails the bar, the answer is NOT "try a NN" in this session —
  that's a separate follow-on.
- **No hyperparameter search.** The defaults in step 2 are
  defensible. If the bar fails, the diagnosis is "the data /
  features don't support the bar", not "we used the wrong LR".
  Don't burn a session trying to tune your way to 0.70 AUC if
  you got 0.65.
- **Parallel tree.** Code goes under `agents_v2/scorer/` or
  `training_v2/scorer/`. Tests go under `tests/test_scorer_v1_*`.

## Out of scope

- Wiring into a policy (Phase 1).
- Multi-class outputs.
- Real-time inference latency profiling.
- Training a larger / different model class.

## Useful pointers

- `data/scorer_v1/dataset.parquet` — Session 01's output.
- `data/scorer_v1/feature_spec.json` — feature contract.
- `plans/rewrite/phase-0-supervised-scorer/purpose.md` —
  locked design decisions.
- LightGBM docs: https://lightgbm.readthedocs.io/
  (`lightgbm.train`, `lightgbm.plot_importance`).
- sklearn calibration: `sklearn.isotonic.IsotonicRegression`,
  `sklearn.calibration.calibration_curve`.

## Estimate

2–4 hours.

- 10 min: load + split.
- 20 min: train + log curves.
- 15 min: calibrate.
- 15 min: evaluate.
- 30 min: greedy-threshold P&L sim.
- 15 min: persist artefacts.
- 30 min: tests.
- 30 min: writeup.

If you're past 5 hours, stop and check scope. The most likely
overrun is the greedy-threshold sim — if it's hard to wire up the
env-replay machinery, file the simplification as a finding and
ship Bar 3 with whatever you have, AMBER if needed.
