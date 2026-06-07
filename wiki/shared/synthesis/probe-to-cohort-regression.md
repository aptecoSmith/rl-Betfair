---
id: 01KTGC1SKBKQ7S84M0EZJGZ6H0
type: synthesis
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-106b41]
aliases: [probe-scale to full-cohort regression, GA can't amplify probe signal, Sortino retired, R3 retired]
---

# Probe-scale promise → full-cohort regression

A recurring trap: a mechanism looks promising at probe scale (5 agents, few gens) and then **regresses**
when escalated to a full 12×8-generation cohort under GA breeding. Two independent mechanisms — R3
(quadratic naked-loss reward penalty) and Sortino (downside-aware selector) — both followed this exact
path and were retired.

## What it is

- **R3 β=0.01** at probe scale showed locked-floor lift + worst-day truncation + retained tailwind
  response — "the most promising add-on we've seen." The full R3+E3 cohort fell behind the E3 cohort by
  £8/d (fc=120) and £13/d (fc=0). **REGRESSION; R3 retired.**
- **Sortino** selector at probe scale targeted the bounded-worst-day phenotype; the full E3+Sortino
  cohort bred weaker-upside lineages and lost to `day_pnl_per_std` by £45/d at fc=0 — because Sortino's
  downside-aversion breeds fewer/safer pairs, killing the naked-tailwind upside that drives fc=0 cash.
  **Sortino retired.**

Both interventions targeted naked variance; both bred lower-mean lineages because the naked tailwind is
too large a part of the recipe's edge to penalise without collateral damage. The pattern: **probe-scale →
full-cohort regressions are real, and the GA can't be trusted to amplify a probe-scale signal under
selection breeding.** At probe scale (n=5, no breeding) you see a high-variance per-agent population
averaged; the breeding dynamics over generations are a different system.

## Why it matters

A direct caution for the [[probe-before-cohort-budget]] methodology: a probe bite is necessary but NOT
sufficient — it must be confirmed at cohort scale before deployment. Combined with
[[selection-vs-measurement-signal]] (selection can't fix reward problems) and
[[gradient-delivered-ppo-unresponsive]] (shaped gradients don't bite), the surviving lesson is that only
env-side priors that REMOVE bad decisions ([[close-feasibility-open-gate]]) have scaled cleanly — naked-
variance reward-shaping and selector changes are "off the menu unless we find a mechanism that doesn't
sacrifice mean for variance."

## Sources
- `src-106b41` EXPERIMENTS.md (js_desktop:present)
