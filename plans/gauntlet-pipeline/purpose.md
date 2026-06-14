---
plan: gauntlet-pipeline
status: proposed — design agreed, not built
created: 2026-06-13
---

# Purpose — gauntlet-pipeline

Re-architect the lockstep cohort from a coupled in-loop march into a
**decoupled pipeline**: a uniform "run a batch of agents through one tranche"
executor, fed by **per-tranche queues**, with selection + breeding as a
separate stage. The scientific model is unchanged — we are still searching for a
single **gene recipe** that, cooked consistently from scratch, yields a good
model — but the *execution* is reorganised so it scales as data accumulates.

## The problem this solves

The current `--breeding lockstep` path (`plans/lockstep-cohort/`) couples
execution and breeding in one loop: generation = tranche, with selection +
mutation + **mutant catch-up all in one generation**. Each new mutant replays
*every* earlier tranche from scratch inside a single generation, so:

- **Runtime is O(N²)** across an N-tranche march and the **final generation is
  the heaviest** (full-history catch-up). Observed live on `tt_tock_004`: gen 4
  ran ~hours with the box at 100% CPU and no registry output (the parent blocked
  on the worker pool through one giant generation).
- **Per-run cost is wildly uneven** (cheap early generations, brutal late ones),
  which makes scheduling and memory headroom unpredictable as data grows.

As we add days, both get worse. The fix is **not** to reduce the work (see the
non-negotiable below) — it is to **reshape it into uniform, bounded runs**.

## The non-negotiable: recipe purity (why we do NOT warm-start mutants)

The cheap fix — warm-start a mutant from its parent survivor's weights and only
train the new tranche — is **rejected on principle** (operator, 2026-06-13). A
warm-started mutant is cooked under recipe A for tranche 1, then recipe B
afterwards: its final weights are a **chimera of recipes**, and the gene config
we'd report as "the winner" would **not reproduce that model** if run clean from
T1. That is not a recipe; it is a cooking trajectory, and it breaks attribution.

Therefore **every model must be cooked under ONE gene config, from T1, warm-
starting only its OWN weights between tranches.** Survivors already satisfy this
(same genes throughout). New recipes (fresh blood + mutants) must **climb the
gauntlet from T1**. The catch-up cost is the **intentional price** of recipe
purity, not a bug to optimise away.

## What "good" looks like

- **Uniform per-run cost + bounded memory** — every run is `batch × one
  tranche`, regardless of how deep the gauntlet is. No heavy tail → no
  OOM-by-tail; the machine stays evenly fed (only the agent spec changes between
  runs).
- **Recipe purity preserved** — no chimeric models; a reported recipe
  reproduces.
- **Scales with data** — more days → a longer gauntlet → more rounds of
  selection → a stronger filter (a feature, not a cost to avoid).
- **Distributable + resumable** — uniform runs are trivially farmed across
  machines; the queue/ledger *is* the state.

## Scope

This plan PROPOSES + builds the new execution architecture and validates it
A/B against the current lockstep before any cutover. It also seeds the **gene
register** (coverage map of explored gene space) that later replaces random
fresh blood with gap-targeted sampling. It does NOT change the reward path, the
env, or the held-out-selection regime (those carry over verbatim:
`plans/maturation-raising/holdout-selection.md`).
