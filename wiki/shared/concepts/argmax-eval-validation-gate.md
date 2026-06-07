---
id: 01KTG1DSM7RSDBGNVEEJG89YW2
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0d0f55]
aliases: [argmax validation gate, reproducibility variance rank-correlation]
---

# Argmax-eval validation gate

The three-part pass condition for [[argmax-eval]]: **reproducibility, variance reduction, and rank
correlation** — "lower variance, similar ranking".

## What it is

(1) **Reproducibility:** two argmax-eval rollouts of the same trained policy on the same eval day
produce bit-identical day_pnl/locked_pnl/naked_pnl (a single integer-comparison test; if it fails,
there's another stochastic source — e.g. env-side passive matching — to audit). (2) **Variance
reduction:** the identical-gene lineage's single-day spread is ≥3× smaller under argmax than the £185
stochastic spread (if it fails, the spread is driven by *training* stochasticity, not eval sampling —
useful either way). (3) **Rank correlation:** Spearman ρ ≥ 0.7 between the 3-day-stochastic-mean
ranking and the 1-day-argmax ranking over all agents — argmax need not be the *same* signal, just a
**strongly correlated cheaper proxy**. The instructive failure mode: poor rank correlation means the
argmax-best agents differ from sampled-best — the policies are too uncertain at the argmax decision
points, and the noise they make under sampling **reveals real structure that argmax hides** → don't
ship as default.

## Why it matters

A template for validating a noise-reduction change: prove it's reproducible, that it actually cuts the
variance you targeted, and that it preserves the ranking you care about — with pre-committed
interpretations for each failure. Validates [[argmax-eval]] against [[eval-sampling-variance-dominates]].

## Sources
- `src-0d0f55` purpose.md (js_desktop:present)
