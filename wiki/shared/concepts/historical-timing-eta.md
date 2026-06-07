---
id: 01KTGF67BR1Q6ECWYFCN51MTXT
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-15d1e2]
aliases: [historical timing ETA, drop the hardcoded estimate, 12s/agent/day, real timing varies 3-5x]
---

# ETA from historical timing, not a hardcoded constant

The training wizard's run estimate was a hardcoded **12s/agent/day** with an arbitrary 60/40 train/eval
split — "wildly off" because real timing varies by **3–5×**. The fix: estimate from the last completed
run's actual timing, falling back to 12s only when no history exists.

## What it is

A hardcoded per-unit constant can't capture the real cost structure (which the speedup work measured at
~867s/agent-train-day batched vs ~70s/agent-eval-day serial), so any single constant with a fixed
train/eval split is off by multiples. Using historical timing from the last run grounds the ETA in
measured reality and self-corrects as the machine / config changes; the 12s constant survives only as a
cold-start fallback.

## Why it matters

The user-facing twin of the [[cost-model-from-per-phase-walls]] lesson: don't predict run cost from a
uniform hardcoded average — measure per-phase from real runs. A wrong ETA isn't just cosmetic; it drives
how the operator sizes and schedules cohorts. Pairs with [[eta-bar-labels-mislead]] (the other half of
the ETA overhaul).

## Sources
- `src-15d1e2` purpose.md (js_desktop:present)
