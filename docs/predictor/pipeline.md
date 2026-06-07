# Price Mover Model — Training Pipeline

The price mover model predicts whether a Betfair runner's price will shorten
or drift over the next N minutes, using only ladder microstructure data
(prices, sizes, traded volume, timing). It is blind to form, jockey, trainer,
or any external race data — it reads the market itself.

This document describes the full training pipeline: what runs, in what order,
what each script does, and how to retrain from scratch.

---

## What the model learns

Each example is a snapshot of the Betfair ladder at one tick (one polling
interval, ~1 second). The model predicts the price change in Betfair ticks
at future horizons (1 minute, 3 minutes, 7 minutes from now). It outputs
quantile predictions (q10, q50, q90), not point estimates, so the confidence
interval is part of the signal.

A **signal fires** when the model is confident enough:
- **Drift signal:** q50 ≥ +5 ticks AND q10 ≥ 0 (model is sure price will rise)
- **Shorten signal:** q50 ≤ −5 ticks AND q90 ≤ 0 (model is sure price will fall)

Only ~0.4% of ticks produce a signal. On those ticks, the current champion
achieves 78.8% directional accuracy on unseen test data.

---

## Data flow

```
data/processed/{date}.parquet          ← raw Betfair ladder snapshots
        ↓  build_dataset.py
data/predictor_dataset/{date}.parquet  ← feature-engineered + labelled rows
        ↓  train_one.py / run_matrix.py
registry/predictor/{experiment_id}.pt  ← trained model weights
registry/predictor_scoreboard_neural.csv  ← all results
        ↓  run_s09.py
registry/s09_test_results.csv          ← sealed test evaluation
        ↓  (manual selection)
registry/predictor/production/manifest.json  ← champion model record
```

---

## Scripts reference

| Script | Purpose |
|---|---|
| `scripts/predictor/build_dataset.py` | Reads `data/processed/{date}.parquet`, computes V1–V5 features and multi-horizon labels, writes `data/predictor_dataset/{date}.parquet`. Idempotent. |
| `scripts/predictor/splits.py` | Defines train/val/test date boundaries. Single source of truth — never bypass. |
| `scripts/predictor/datasets.py` | Feature column definitions (V1–V5), `load_split()`, Dataset classes for tabular and sequence models. |
| `scripts/predictor/models.py` | Model definitions: MLP, LSTM, Transformer, Conv1D. `build_model()` is the factory. |
| `scripts/predictor/eval_metrics.py` | All evaluation functions: MAE, pinball loss, directional accuracy, naive backtest P&L. |
| `scripts/predictor/train_one.py` | Trains one config (YAML) → one scoreboard row + one `.pt` weights file. Idempotent on experiment ID. |
| `scripts/predictor/run_matrix.py` | Runs `train_one.py` in subprocess per config in a directory. Skips already-done experiments. |
| `scripts/predictor/run_all_sessions_neural.py` | Autonomous orchestrator: S04 → S05 → S06 → S07 (skip) → S08, then halts for S09 sign-off. |
| `scripts/predictor/run_s09.py` | ONE-SHOT sealed test evaluation on chosen experiment IDs. Writes `registry/s09_test_results.csv`. |

---

## Session structure

The sweep is organised into sessions. Each session narrows the search space
based on the previous session's results.

| Session | What it sweeps | Downselect criterion |
|---|---|---|
| S01 | Build labelled dataset | — |
| S02 | Implement training harness, smoke-test all architectures | — |
| S03 | Architecture family × variant | median MAE across seeds |
| S04 | Feature variants (V1–V5) on best S03 cell | median MAE |
| S05 | Output formulation (pinball3 vs pinball5) on best S04 cell | median MAE |
| S06 | Horizon sets (7 combinations of 1m/3m/7m/15m) | median MAE |
| S07 | Smoothing (ema_post, temporal_loss) | *skipped — not yet implemented* |
| S08 | Backtest summary across all runs; produces candidate shortlist | — |
| S09 | **ONE-SHOT** sealed test evaluation on operator-chosen candidates | — |

S08 and S09 do not train — they evaluate existing model weights.

---

## Feature variants

| Variant | Features added | Notes |
|---|---|---|
| V1 | Ladder (3 back/lay levels), LTP, traded volume, num runners, time to off | Baseline |
| V2 | + LTP lags (1/5/10/30 ticks), 32-tick window stats (mean/std/min/max) | Best general performer |
| V3 | + TradedVolumeLadder (total, at-LTP, near-LTP bands, level count) | Requires TVL data (from 2026-04-26) |
| V4 | + Cross-runner features: rank, LTP share, LTP z-score, volume share/z-score | Strong on MAE; good fires |
| V5 | + Market-state: total traded volume, avg spread ticks, depth total, active runners | Not consistently better than V4 |

V2 and V4 are the sweet spot. V1 is a safe fallback. V3/V5 add marginal signal at
the cost of TVL data dependency (only available from 2026-04-26 onward).

---

## Train / val / test split

| Split | Dates | Rows | Notes |
|---|---|---|---|
| Train | 2026-04-06 to 2026-04-30 | 1,647,901 | 25 days (2 missing: Apr 18, Apr 27) |
| Val | 2026-05-01 to 2026-05-03 | 223,036 | Used for early stopping and model selection |
| Test | 2026-05-04 to 2026-05-06 | 214,836 | **Sealed until S09. Never use for selection.** |

The test split is defined in `scripts/predictor/splits.py` and enforced there.
Do not read `TEST_DATES` outside of `run_s09.py`.

---

## How to retrain from scratch

```bash
# 1. Build/update labelled dataset for any new dates
python scripts/predictor/build_dataset.py --dates 2026-05-07,2026-05-08,...

# 2. Run the full neural sweep (S04 → S08, then stop for S09 sign-off)
python scripts/predictor/run_all_sessions_neural.py

# 3. Review registry/backtest_summary_neural.csv, pick top candidates

# 4. Run sealed test evaluation (ONE-SHOT — pick candidates carefully)
python scripts/predictor/run_s09.py <experiment_id_1> <experiment_id_2> ...
```

The orchestrator survives session exit if launched detached:

```powershell
Start-Process -FilePath "C:\Python314\python.exe" `
    -ArgumentList "-u scripts\predictor\run_all_sessions_neural.py" `
    -WorkingDirectory (Get-Location) `
    -WindowStyle Hidden `
    -RedirectStandardOutput logs\out.log `
    -RedirectStandardError logs\err.log
```

To resume a crashed run mid-sweep:

```bash
python scripts/predictor/run_all_sessions_neural.py --start-from S06_neural
```

`run_matrix.py` is idempotent — already-trained experiments are skipped automatically.

---

## Adding new training data

When new processed parquets arrive in `data/processed/`:

1. Run `build_dataset.py --dates <new_dates>` to generate labelled parquets.
2. Update `splits.py` if you want to extend the train window.
3. Retrain from S04 (the sweep configs will regenerate; existing rows are skipped).
4. A new test split must be chosen and sealed *before* the new sweep begins.

---

## Current champion

See [`docs/predictor/champion.md`](champion.md) for the current production model
and [`registry/predictor/production/manifest.json`](../../registry/predictor/production/manifest.json)
for the machine-readable record.
