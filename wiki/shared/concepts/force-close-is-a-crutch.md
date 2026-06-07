---
id: 01KTGJG33X808VGJVCY7YNT97G
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1663dd]
aliases: [force-close is a crutch, fc rate 0.809, mechanics not coefficients, the crutch masks the failure]
---

# Force-close is a crutch (the operator reframe)

The reframe that redirected the rewrite's force-close work from coefficient tuning to mechanics: the
AMBER v2 baseline measured **mean force-close rate = 0.809** — slightly *worse* than the v1 baseline
(~0.75) the rewrite was supposed to improve on, so the rewrite's central claim (per-runner credit reduces
fc rate) is not supported.

## What it is

The operator's framing: "force close is a crutch we put in because the models weren't closing trades —
perhaps that itself points to an architectural issue?" A human scalper actively closes trades that aren't
going their way; ~80% of pairs ending via env-initiated bail-out means the policy never learned to close.
The reframe says don't tune the existing shaping coefficients — fix the underlying mechanics; the original
ablation tree (matured_arb_bonus → naked_loss_anneal → mark_to_market) is deferred indefinitely because
those terms shape incentives on top of mechanics that may themselves be wrong.

The plan's single mechanics-level hypothesis: give the policy a first-class £-target per open
([[first-class-pnl-target-per-open]]) AND a projected-loss stop-close
([[env-stop-close-not-agent-learned]]), and it will learn closes on its own — fc rate falling below 0.30
(vs 0.809). Force-close stays on as a T−N backstop throughout; the goal is the policy stops *needing* it.

## Why it matters

A methodological pivot: when a safety net is doing the work, "improve the metric" can mean "remove the
need for the net," not "tune the net." It reframes [[force-close-population-cost]] from a cost to optimise
into a symptom of a missing capability. The smell that exposed it is [[fc-rate-correlates-with-pnl-smell]].

## Sources
- `src-1663dd` purpose.md (js_desktop:present)
