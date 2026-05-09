---
plan: price-direction-predictor
session: S03 — architecture sweep
date: 2026-05-09
status: complete (45/45 rows in scoreboard)
---

# S03 — architecture sweep findings

## What was run

5 families × 3 family-distinctive variants × 3 seeds = 45 candidates.
All trained on V3 (TVL-required) features, `tvl_required_10d`
corpus, horizons {3m, 7m, 15m}, pinball-3 quantile output, raw
smoothing. Train wall-clock ~54 min total on a single RTX 3090.

The variant axis per family was the family's distinctive
hyperparameter, per `master_todo.md` S03:
- `mlp` depth ∈ {2, 3, 4}
- `gbm` (n_trees, max_depth) ∈ {(100, 4), (300, 5), (500, 6)}
- `lstm` time_window ∈ {16, 32, 64}
- `transformer` depth ∈ {2, 4, 6}
- `conv1d` kernel ∈ {3, 5, 7}

## Median (across 3 seeds) leaderboard

Sorted by mean MAE across the three horizons (lower better):

| Cell | MAE | Calib gap | Dir-acc 7m@k=5 | Fires 7m@k=5 | Stab 7m | Params | Train s |
|---|---|---|---|---|---|---|---|
| **gbm_t100d4** | 3.038 | 0.034 | NaN | 0 | 0.97 | 496 | 4 |
| **gbm_t300d5** | 3.040 | 0.038 | NaN | 0 | 0.92 | 832 | 6 |
| gbm_t500d6 | 3.041 | 0.043 | NaN | 0 | 1.00 | 891 | 8 |
| conv1d_k3 | 3.051 | 0.028 | 0.60 | 105 | 0.48 | 48K | 49 |
| lstm_tw16 | 3.052 | 0.028 | NaN | 0 | 0.89 | 64K | 49 |
| mlp_d2 | 3.052 | 0.059 | 0.55 | 445 | 0.69 | 22K | 47 |
| **mlp_d4** | 3.053 | 0.034 | **0.74** | **214** | 0.62 | 55K | 57 |
| lstm_tw64 | 3.054 | 0.030 | NaN | 0 | 0.96 | 64K | 52 |
| mlp_d3 | 3.056 | 0.030 | 0.63 | 576 | 0.68 | 39K | 62 |
| conv1d_k7 | 3.059 | 0.046 | 0.56 | 340 | 0.64 | 106K | 54 |
| lstm_tw32 | 3.074 | 0.026 | NaN | 0 | 0.93 | 64K | 55 |
| conv1d_k5 | 3.094 | 0.050 | 0.59 | 547 | 0.58 | 77K | 55 |
| transformer_L6 | 3.461 | 0.033 | 0.52 | 88 | 0.98 | 305K | 288 |
| transformer_L4 | 3.634 | 0.033 | 0.46 | 217 | 0.99 | 205K | 95 |
| transformer_L2 | 3.789 | 0.041 | 0.69 | 54 | 0.99 | 105K | 100 |

## Headline findings

### 1. MAE alone is a very flat ranking

The top 12 of 15 cells are within **0.06 ticks of MAE** of each
other (3.04–3.09). The pinball loss / MAE objective on
mean-zero-median-near-zero labels is almost saturated by a
constant predictor — most cells learn to predict ~0 for q50 and a
calibrated spread for q10/q90. The MAE difference between GBM
and LSTM is statistical noise, not a real architecture effect.

**Implication:** the original downselect criterion ("median val
MAE across horizons; top 2 cells") would pick gbm_t100d4 and
gbm_t300d5 — but those candidates have **zero usable signal at the
operator's decision rule** (k=5 fires=0). That cannot be the
right downselect for S04.

### 2. The real signal lives in directional-accuracy-at-fires

Sorted by `dir_acc_k5_7m × log(fires_7m + 1)` (composite that
rewards being right AND firing):

1. **mlp_d4**: 74% accuracy on 214 fires — **clear winner**
2. **mlp_d3**: 63% on 576 fires — high firing rate, decent acc
3. **transformer_L2**: 69% on 54 fires — small fires, but high acc
4. **conv1d_k3**: 60% on 105 fires
5. **mlp_d2**: 55% on 445 fires
6. **conv1d_k5**: 59% on 547 fires
7. **conv1d_k7**: 56% on 340 fires
8. **transformer_L6**: 52% on 88 fires
9. **transformer_L4**: 46% on 217 fires (worse than coin flip!)
10. **lstm_*** (all variants): **0 fires** — under-confident

`mlp_d4` produces real, actionable predictions: ~71 fires per val
day, 74% accuracy. That's a tradeable signal even before any
downstream feature improvements.

### 3. LSTM and GBM are calibrated but too cautious to fire

LSTM and GBM both have well-calibrated predictions
(coverage near 0.8 nominal) but their q50 medians stay close to
zero and q10/q90 spreads stay tight enough that the
`q50 ≥ 5 AND q10 ≥ 0` decision rule never fires. They might still
be useful at lower thresholds (k=2 or k=3) and for raw quantile
outputs feeding a downstream rule that's not threshold-based.

S04/S05 should retest with `k_ticks ∈ {2, 3, 5}` to see whether
LSTM/GBM are usable at less aggressive thresholds.

### 4. Transformer is the loser

Transformer L2/L4/L6 all have MAE 0.4 ticks worse than every
other family, and L4's directional accuracy (46%) is below the
random-coin-flip 50%. The transformer also takes 2-5x the
wall-clock of every other family. Three possible explanations:

- LR sensitivity (already dropped from 1e-3 to 5e-4 in S03)
- Position embedding randomly initialised may not be picking up
  the per-tick temporal structure with this small data
- Training on only 242K examples may be insufficient for
  attention-based models which usually want >1M

The cleanest follow-up is to compare against `tvl_mask_29d`
(1.65M rows) in S04 — if transformer recovers there, it's a data
issue, not architecture. If not, transformer should be dropped
from later sweeps.

### 5. Stability target: only LSTM, GBM, Transformer pass

Lag-1 autocorrelation ≥ 0.7 (the operator's "no wild
oscillation" target) is met by:

- gbm: 0.92–1.0 ✅
- lstm: 0.87–0.97 ✅
- transformer: 0.98–0.99 ✅
- mlp: 0.62–0.69 ❌
- conv1d: 0.48–0.64 ❌

Both top performers on directional accuracy (mlp_d4, mlp_d3) FAIL
the stability target. This is exactly the operator-flagged
"oscillation" failure mode — the MLP looks at each tick
independently with no temporal smoothing, so its quantiles flip
between fire and no-fire signals tick-to-tick. **S07 (smoothing
sweep) is now load-bearing for the MLP candidates** — they need
EMA or a temporal-consistency loss to be usable.

## S04 downselection — recommended

The master_todo S03 acceptance ("top 2 by val MAE") would pick
GBM. That's the wrong call given the data. Recommended composite
for S04:

> Take the top 2 cells by **directional accuracy at k=5 (7m
> horizon) × log(fires + 1)**, breaking ties by lower
> calibration gap.

Under that composite:
- **Cell 1: `mlp_d4`** — 74% acc, 214 fires, calib 0.034
- **Cell 2: `mlp_d3`** — 63% acc, 576 fires, calib 0.030

Both are MLP. That's a thin family choice. If we want S04 to
retain genuine architectural diversity, an alternative is:

> Top 1 cell per family; drop any family below `dir_acc_k5_7m =
> 0.55` AND `fires_7m > 50`. Keep failing families on the bench
> for S07 smoothing diagnosis.

Under that alternative:
- **mlp_d4** (74%, 214 fires) — keeper
- **conv1d_k3** (60%, 105 fires) — keeper
- transformer_L2 (69%, 54 fires) — borderline; drop for S04
- **lstm_*** (0 fires) — bench until k=2/3 sweep
- **gbm_*** (0 fires) — bench until k=2/3 sweep

I lean toward the second framing — a 2-family S04 (mlp + conv1d)
keeps the comparison meaningful. The first framing puts both eggs
in the MLP basket.

**Decision needed from operator before S04 fires.** Marking the
composite criterion in `master_todo.md` would make this
auditable.

## Known issue: LSTM teardown crash

All 9 LSTM runs in S03 exited with Windows status 0xC0000409
(STATUS_STACK_BUFFER_OVERRUN) AFTER the scoreboard row was
written. Specifically, the crash happens during process teardown
when CUDA/cuDNN frees LSTM weights. Workarounds:

1. Add `torch.cuda.empty_cache()` and `gc.collect()` before
   `sys.exit()` in `train_one.py`. Not yet applied — punted.
2. Run LSTM on CPU (slower but stable).
3. Disable cuDNN for LSTM: `torch.backends.cudnn.enabled = False`.

The matrix runner counted these as failures (exit-1) but the
data is good. Future runs that filter S03 for LSTM should NOT
re-run them with `--rebuild` on the same code — the crash will
recur. Better to apply (1) first.

## Files of interest

- Scoreboard: `registry/predictor_scoreboard.csv` (45 S03 rows +
  5 smoke rows; filter by `session=='S03'`)
- Per-candidate model cards:
  `plans/price-direction-predictor/models/{experiment_id}.md`
- Weights: `registry/predictor/{experiment_id}.{pt,joblib}`
- Sweep log:
  `C:\Users\jsmit\AppData\Local\Temp\claude\C--Users-jsmit-source-repos-rl-betfair--claude-worktrees-affectionate-proskuriakova-108942\1b104bb3-9746-4f6a-a4bc-e388f9764dd7\tasks\b9lus6qhq.output`
