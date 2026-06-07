---
id: 01KTG90VRXBJ7J9S8D85TP3595
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-0fd276]
aliases: [Adam scale-invariance, aux-loss weight no-op, BCE weight doesn't converge, optimiser eats the weight]
---

# Adam ratios away aux-loss-weight magnitude

A 30× increase in `direction_prob_loss_weight` (0.1 → 3.0) produced **essentially identical** BCE (agent
1 dir_bce_back 1.1193 vs 1.1215 — Δ=0.002). The plumbing was verified (weight reaches the loss, loss
reaches the head, gradient norm 0.29 on a synthetic batch), yet 30× weight changed nothing. The cause is
the optimiser, not the wiring.

## What it is

PPO uses Adam, whose per-parameter update is ≈ `lr × m / sqrt(v)` (first/second-moment EMAs). This is
**scale-invariant in the gradient magnitude**: multiplying every gradient by 30× scales `m` and
`sqrt(v)` by 30× each, so `m/sqrt(v)` stays the same. So weight=0.1 and weight=3.0 produce the same
effective update trajectory on the head — they differ only in Adam's variance-EMA warmup, minor over a
364-update window. End-of-day BCE 1.05 ≈ `−log(0.35)`, the balanced no-skill baseline; the probe reached
0.4–0.6 in 600 dedicated SGD steps, but Adam at 3e-4 can't escape the no-skill basin in 364 mixed
PPO+BCE updates regardless of BCE weight.

## Why it matters

Two false leads were chased first — a "gate self-reference loop" (fixed with a `.detach()` on the
direction probs before `actor_input`; no change) — before Adam was identified as the real cause.
**Lesson: don't expect aux-loss weight to drive convergence speed when the optimiser is Adam.** Weight
sets the BCE-vs-PPO *direction* of the trade-off, not the *magnitude* of progress. To make the head
converge: run far more steps, use a separate optimiser for the head, or BC-pretrain it
([[freeze-bc-head-post-pretrain]]) — the analogue of the probe's dedicated-SGD regime.

## Sources
- `src-0fd276` lessons_learnt.md (js_desktop:present)
