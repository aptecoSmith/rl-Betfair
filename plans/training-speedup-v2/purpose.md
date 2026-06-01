# training-speedup-v2 — purpose

## Problem

Training is the binding constraint on every experiment we run. Measured
cost (Cohort 1, 2026-06-01): **~867s per agent-train-day**; a
20-agent × 12-day × 4-gen cohort is **~1.5 days** wall. The machine is
**CPU-bound** — 99.7% CPU across 20 cores while the GPU sits idle at ~35%,
because env stepping is serial and per-agent on the CPU and the GPU waits
on it. As training data grows this only worsens. We want **1-2 orders of
magnitude** so a 4-gen cohort runs in hours, not days — that changes which
experiments are even possible, not just how long they take.

## Supersedes

`plans/cohort_training_speedup/` (Phase 3, 2026-05-23) shipped mixed-device
(+31%) and a per-date `feature_cache` (+63% on env-build) — but on the
**lean-obs** path. We have since moved to **full obs (2254-d) + 3 predictor
bundles + `--batched`**, a config that plan never profiled. Its
`deferred.md` flagged the biggest upside (`--batched`, 2-5×) *and* a
landmine we then sailed into: **`--batched` silently drops `bc_pretrain`
and `per_transition_credit`** (the warning fired 3× in c1's log; both
cohorts trained with no BC warm-start).

## The non-negotiable spine: bit-identical validation

Every speedup — cheap or transformational — must reproduce the **current
env's** behaviour (trajectories, rewards, values, bet-state, settle P&L)
**bit-for-bit** (or within a declared, justified float tolerance) on a
battery of golden cases **before** it is trusted for a single result. The
harness (Step 1) is the deliverable that converts the operator's stated
fear — "two weeks in, a basic error you sailed into" — into "the divergence
trips the golden-value gate the instant it appears." It is precisely what
makes the high-risk env-core rewrite (Stage 3C) safe to attempt, and it is
the same bit-identical property that makes Steps 1 and 2 trustworthy.

## Approach: profile, then stage from cheap+safe to transformational

0. **Re-profile the REAL config** (full-obs + predictors + batched). No more
   optimizing an assumed or stale breakdown.
1. **Bit-identical regression harness** — golden trajectories + comparator.
2. **Hot-path vectorization** (low-risk, bit-identical) — the prior plan's
   deferred Option B-big plus the obs-build hot spots Step 0 surfaces.
3. **Staged GPU vectorization** — the transformational bucket:
   - **3A** true batched policy forward (stack N agents → one GPU forward;
     uses the idle GPU, amortizes kernel-launch overhead).
   - **3B** vectorize the obs/market path across agents.
   - **3C** vectorize the env core (matching / bets / settlement) — the big
     multiplier, gated by a feasibility spike + the harness.

**Realistic ceiling.** 100× is the physics-sim ideal (uniform tensor ops,
Brax/Isaac). Ours is a market-*replay* sim: obs build, market features, and
the policy forward vectorize cleanly, but order-matching and settlement are
data-dependent branching that resists it. Expect **10-100×**; Step 0 plus a
3C feasibility spike size where we actually land. Even the low end is
transformational, and the staged order captures the easy multipliers first
so a stall at 3C still ships 2+3A+3B.

## BC: unproven, not assumed

The operator recalls "great success" with BC. The record (checked
2026-06-01) says otherwise: BC's documented success is the **supervised
maturation-AUC probe** (0.745, "maturation is learnable"), *not* a
PPO-warm-start that improved P&L — the imitation-first BC *policy* lost
£1513/7d (a non-BC approach beat it at −£418/7d), and BC has been silently
ignored in every recent batched cohort while c1 still found high-locked
structural scalpers from scratch. So the prior is **"BC may not be
load-bearing."** Step 4 settles it with a clean BC-vs-from-scratch ablation
on held-out; the speedup architecture is **not** constrained to preserve BC
until that ablation justifies it. Whatever we decide, **no feature is ever
dropped silently again** (hard constraint #2).
