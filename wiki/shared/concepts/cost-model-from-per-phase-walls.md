---
id: 01KTGC1SK3CG1QRKC34XFK4FAP
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [cost model backwards, per-phase wall timers, what does per-agent divide by, uniform average hides structure]
---

# Cost-model from per-phase walls, not a uniform average

A planning lesson banked from the bc-to-ppo cohort-1 misfire: a **uniform per-agent-day cost average
hides the phase structure** and leads you to cut the cheap thing while maxing the expensive thing —
exactly backwards.

## What it is

Cohort 1 was sized from a pre-launch uniform estimate of ~76 s/agent-day. The c1 scoreboard then showed
**training is the expensive lever (~867 s/agent-train-day**, batched ~10× via hidden-size clusters) while
**eval is cheap (~70 s/agent-eval-day**, serial). The uniform average had blended these, so the run cut
eval days to 2 (the cheap thing) and kept 25 train days (the expensive thing) — `gen_wall ≈
n_train×0.64h + n_eval×n_agents×70s`. Worse, 2 eval days starved the `locked_per_std` selection metric
(σ over n=2 is meaningless → noisy fitness → no clean GA gradient). Fix in c2: 7 eval days for a usable
σ, train days cut 25→12 (BC warm-start means PPO doesn't need 25 to express the reward genes).

A second instance of the same trap: the "867 s/agent-day" number itself is a *cluster* wall written into
every agent's `wall_time_sec` and cohort-averaged — true per-agent marginal ≈ 80 s. "Always check what a
'per-agent' wall number divides by before optimising against it."

## Why it matters

Two banked rules: (1) cost-model a run from measured **per-phase** wall times, not a uniform per-agent-day
average; (2) match the eval-day count to the **selection metric's variance needs** before launch — a
metric with a σ denominator needs enough eval days to estimate σ. The same "diff the two call sites"
vigilance that catches forked-path silent drops ([[batched-path-silent-drops]]) applies to wall-time
attribution.

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
