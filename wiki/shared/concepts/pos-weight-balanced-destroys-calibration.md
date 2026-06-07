---
id: 01KTFTFE339WVS45NY8XRAKSNA
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-042412]
aliases: [pos_weight balanced, unweighted BCE, calibration lever]
---

# pos_weight=balanced destroys calibration

The dominant design lever in the direction-head sweep: training with class-balanced `pos_weight`
(common practice for imbalanced classification) is **actively harmful** here — switch to unweighted
BCE (`pos_weight=1`).

## What it is

At the ~18% empirical positive rate, balanced `pos_weight` pulls the output mean to ~0.46 (a uniform
decision boundary), which doesn't help ranking and **destroys calibration**. Switching to unweighted
BCE consistently: improves mean Pearson slightly (+0.001 to +0.006 across every architecture), **cuts
mean Brier by 35-37%**, and produces calibrated outputs (predicted mean tracks the empirical positive
rate within 1-2 points per eval day). Held across [64], [256], [256,128], [256,128,64] — it's
architecture-independent. This is what makes [[direction-head-c11]]'s output usable as a probability.

## Why it matters

The single highest-leverage knob for a predictor whose output an actor consumes as a probability: a
ranking-only metric (Pearson/AUC) hides that balanced training is 2.5x over-confident. Always check
calibration (Brier / predicted-vs-empirical mean), not just ranking.

## Sources
- `src-042412` findings.md (js_desktop:present)
