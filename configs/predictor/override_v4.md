# Override: S05/S06 target V4 instead of MAE auto-downselect winner (V2)

**Date:** 2026-05-09 ~15:30
**Operator decision:** user override mid-run (PID 6124 running, cannot modify loaded bytecode)

## Why

S04_neural downselect by median mean_mae returned `conv1d/k3/V2/tvl_mask_29d` (mean_mae 2.973)
over `conv1d/k3/V4/tvl_mask_29d` (mean_mae 2.981). The gap is 0.008 — within noise.

But V4 is dramatically better as a **trading candidate**:

| Cell | mean_mae | dir_acc_k5_7m | fires | backtest_pnl_k5 |
|---|---|---|---|---|
| V2 (MAE winner) | 2.973 | 73.2% | 24 | £23.55 |
| V4 (override) | 2.981 | 72.6% | **504** | **£413.98** |

V4 adds cross-runner features (V4_EXTRA on top of V3). With 504 median fires vs 24, V4 produces
a real signal at k=5 confidence threshold. £413 backtest P&L vs £24 makes V4 the actionable cell.
MAE-blind downselect missed it because MAE is measured on all rows (including non-firing ones)
and the cross-runner features add variance that slightly raises MAE without hurting direction quality.

## Mechanism

Running process (PID 6124) has `run_all_sessions_neural.py` compiled into bytecode — editing the
source file has no effect. Override implemented by **pre-writing override configs into the target
directories before the orchestrator reaches those steps**.

- `configs/predictor/S05_neural/conv1d_k3_V4_tvl_mask_29d_3m_7m_15m_pinball5_s{0,1,2}.yaml`
  → 3 configs, pre-written. Auto-downselect will also write V2 configs (different filenames:
  `conv1d_k3_V2_...`). Both coexist. `run_matrix` trains both. Scoreboard gets both.

- `configs/predictor/S06_neural/conv1d_k3_V4_tvl_mask_29d_*_pinball5_s{0,1,2}.yaml`
  → 21 configs (7 horizon sets × 3 seeds), pre-written. Auto-downselect will also write
  auto-selected V2 configs (different filenames). Both coexist. `run_matrix` trains both.

## What S08 will see

S08 reads all S04_neural/S05_neural/S06_neural rows and reports top-3 by backtest_pnl_k5_7m.
V4 results will be in the pool. Given V4's 504-fire / £414 S04 baseline, V4 S06 results
are expected to dominate S08.

## Note on S06 formulation for V4

Pre-wrote V4 pinball5 for S06. The V4 pinball3 S04 result (default horizons, mean_mae 2.981)
serves as the pinball3 reference. If pinball3 turns out better for V4, the user should
manually run a V4 pinball3 horizon sweep after reviewing the S08 summary.
