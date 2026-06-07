---
id: 01KTJ0EVNKR5HEQ4FZ92XM5DJ1
type: project
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-042412]
aliases: [direction-head-sweep, c11-sweep]
---

# Direction-head architecture sweep (C0–C20)

Four-round sweep (2026-05-24) over 20 variants of the per-runner direction-prediction head, run against 10 held-out eval days. Result: **C11 promoted** as the new shared-head manifest.

## Goals
- Find the head architecture that maximises held-out Pearson while keeping Brier calibrated.
- Determine whether width / depth / activation / loss-recipe changes transfer in this regime.

## Status
**Done, C11 promoted.** Master HEAD `8878e98` (no commits made by this sweep). 19 new head dirs under `models/direction_head/sweep_c{1..20}/`. Sweep evaluator: `scripts/sweep_eval_direction_heads.py`.

C11 = `LayerNorm → Linear(23, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 2)`, trained with **pos_weight=1** (unweighted [[bce]]). Combines round-2's wider+deeper winner (C9) with round-1's calibration finding (C3/C8).

| metric | C11 | C0 (baseline) | delta |
|---|---|---|---|
| mean Pearson | +0.2921 | +0.2719 | +0.0202 (+7.4 %) |
| mean ROC [[auc]] | 0.7098 | 0.6976 | +0.0122 |
| mean Brier | **0.1433** | 0.2282 | **−0.0849 (−37.2 %)** |

The headline is the Brier improvement (calibration), not the Pearson lift. C11 is consistent across all 10 eval days — leads or ties on 9, loses to C13 by 0.0023 on 2026-04-25.

## Inputs
- 23-d per-runner input vector.
- Trained on [[gpu]] (RTX 3090 reference; ~30-90s per variant). Operating regime: [[scalping]] cohort downstream consumer; head exposes "[[win]]" probability that downstream BC ([[bc]]) and gating reads. The earlier `direction_prob_loss_weight` and `bc_direction_target_weight` flags are mutually-exclusive with this manifest.
- 10 held-out eval days (2026-04-07 → 2026-05-06).
- Default recipe `--epochs 50 --lr 1e-3 --batch-size 4096 --patience 5 --seed 42` `--optimizer adam`.

## Key findings (each is its own concept note)
- [[pos-weight-balanced-harms-calibration]] — the dominant design lever.
- [[width-diminishing-returns]] — width helps to 256, then sated.
- [[depth-plateaus-at-2]] — `[W, W/2]` is the optimal shape.
- [[c15-pairwise-overfit]] — input ceiling is generalisation, not expressiveness.
- C18 200-epoch overfit — covered inline by [[c15-pairwise-overfit]] (same finding from opposite direction); `--patience 5` early-stop catches the generalisation peak.
- [[recipe-ablations-all-hurt]] — AdamW / GELU / label-smoothing / focal loss all regressed.

## Recommendation (next launch)
```
--direction-head-manifest models/direction_head/sweep_c11
```
Drop the mutually-exclusive flags: `--enable-gene direction_prob_loss_weight` and `--enable-gene bc_direction_target_weight` (per `plans/shared-direction-head/hard_constraints.md §4`).

## Follow-ons (priority)
1. **Highest:** smoke-validate C11 on a 5-day probe before committing GPU-hours to a 12×3.
2. **Medium:** consider C11+C9 ensemble at inference (averaging, no retrain) — C9 has highest AUC.
3. **Low:** revisit deferred C5 (full 574-d obs) only with strong regularisation; C15's result lowers its expected value.

[[shared/index|hub]]
