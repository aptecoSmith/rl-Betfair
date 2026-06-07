---
id: 01KTGF3ZB43123N3WD2B642TB1
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-15a08d]
aliases: [garaged models breeding, external parents, parent-only vs survivor-slot, cross-run warm genes]
---

# Mixing garaged models into the breeding pool

A GA-design question raised in the breeding-pool-scope investigation: garaged (previously-proven) models
are currently re-evaluated only AFTER all generations complete and **never enter the breeding pool** —
but seeding them in could accelerate convergence by giving new generations access to optimised
hyperparameters from previous runs.

## What it is

The key design decision is whether external (garaged) parents **occupy survivor slots** — competitive and
biologically accurate, but they displace freshly-bred agents — or are **parent-only** — they contribute
genes without taking a slot, which is more predictable and safer. The recommendation: "Parent-only is
probably better to start with." This is a way to carry forward optimised hyperparameters across runs
rather than rediscovering them each cohort from scratch.

## Why it matters

An early sketch of cross-run warm-starting that the later PBT promotion-ladder breeding made concrete
(warm-start weight-threading + a forward-match gate). The parent-only-vs-survivor-slot tradeoff is the
same competitive-vs-safe choice that recurs whenever external genes are injected into an evolving
population. Companion to [[zero-children-bred-anomaly]] (the other breeding-pool-scope finding).

## Sources
- `src-15a08d` lessons_learnt.md (js_desktop:present)
