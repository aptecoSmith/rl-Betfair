---
id: 01KTG90VS0GHJAF37FMB3HS0F7
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [LayerNorm raw obs, sigmoid saturation, per-runner head input norm, RUNNER_DIM LayerNorm]
---

# LayerNorm is mandatory for a per-runner head fed raw obs

When the direction head was rewired to read the runner's **raw** RUNNER_KEYS slice directly
([[direction-head-feature-slice]]), its sigmoid saturated against truth — the first smoke returned
direction BCE 4–12 (vs phase-14's 1.04 baseline), i.e. confidently-wrong predictions across the board.
Prepending `nn.LayerNorm(RUNNER_DIM)` fixed it.

## What it is

The other v2 heads (fill / mature / risk / value) read `lstm_last`, which is post-`input_proj`'s learned
scaling, so they never see the raw heavy-tail features. The direction head, reading raw obs, saw
`vol_delta_60` in [10², 10³]; kaiming-init weights × those magnitudes pushed pre-activations into the
thousands and the sigmoid saturated (`p ≈ exp(−12.4) ≈ 4e-6`). LayerNorm normalises each example to zero
mean / unit std across the feature dim — the same squash the supervised probe got from per-day `pd.std`,
but without dataset-stats bookkeeping. Re-smoke: BCE drops to [1.05, 1.12] — back to the phase-14
baseline, no longer saturated, KL healthy, full PPO budget runs.

## Why it matters

The general rule (the same concern full-obs BC/PPO carries — the v2 policy ships with no input-norm):
**don't rely on the optimiser to eat a feature-scale mismatch** — kaiming-init plus [10², 10³]-scale inputs saturates
the first layer faster than gradients can recover. Getting BCE back to baseline only un-broke the input
pathway; making the head actually *learn* needed the freeze pipeline ([[freeze-bc-head-post-pretrain]]),
because [[adam-ratios-away-aux-loss-weight]] meant no aux-weight setting could drive convergence.

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
