---
id: 01KTGC1SK97QSBX09G5MJ3R4QS
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [rotating eval pool, contiguous eval window overfit, monitor days, GA memorises eval window]
---

# Rotating non-contiguous eval pool (overfit prevention)

A GA-overfit fix triggered by a catastrophic **+£72/d → −£200/d swing** between two *adjacent* 7-day
held-out windows (2026-05-07..13 vs 05-14..20), six agents all swinging the same direction — regime-fit
to the original eval window, not noise.

## What it is

Five sources of overfit were identified in the GA setup: a **single contiguous eval window** (the GA
memorises specific races over generations), the **same eval window across all generations**,
**naked-tail amplification** (a few lucky long-shots add £100+/pair to fitness), a **tiny effective
sample** (10 days × ~80 races, selecting ~12 agents over 5–8 gens), and **early-stop on the same signal**
(improvement on a memorised window is indistinguishable from progress).

The fix is three additive flags on the cohort runner: `--cohort-eval-days` (explicit non-contiguous eval
pool spanning multiple weeks), `--training-days-explicit`, `--monitor-days` (observe-only, disjoint from
train/eval, reserved for the honest post-cohort deployment number), and `--rotating-eval-sample N`
(per-generation deterministic sample of N days from the pool). The legacy chronological path is unchanged
when no new flag is set (byte-identical).

## Why it matters

The honest deployment number is a **monitor set never seen during training or selection** — if the
rotating-eval design works, the post-cohort monitor result should not regress vs the in-training eval
metric. This is the structural complement to "always eval on held-out": held-out catches overfit
*after*, rotating-eval + monitor days prevent the GA from *manufacturing* it. Closely related to the
`select_days` data-dir-dependence leak risk and [[fc0-insample-mirage]] (the in-sample → held-out
collapse this prevents).

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
