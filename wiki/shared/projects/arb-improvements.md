---
id: 01KTGF1J130AB6PYDHNG2M26GZ
type: project
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research]
sources: [src-1340a0]
aliases: [arb-improvements plan, make arbs happen, why agents don't arb]
---

# Arb Improvements (make arbs *happen*)

The plan that follows forced-arbitrage: the scalping machinery shipped (paired orders, arb_spread action,
commission-aware locked P&L, scalping reward terms) but in practice **agents almost never take arbs**.
"The forced-arbitrage plan made arbs *possible*. This plan makes them *happen*."

## What it is

Evidence from run `90fcb25f` (2026-04-14): arb counts per episode like 6/124, 0/18, 3/16, then 0 for
most of episodes 4–18 (P&L +£0.00, near-zero loss = the policy stopped betting). Design review found
**three distinct problems stacking on top of each other**: (1) catastrophic value-function collapse that
pushes the policy into a no-bet corner ([[value-collapse-dont-bet-corner]]); (2) arbs are invisible in
the observation — the policy must derive a lockable arb from raw prices ([[arb-as-observation-feature]]);
(3) no warm start for a sparse-reward skill — random exploration of "place aggressive + passive with
N-tick offset" is lottery-ticket learning, even though the episode data already contains the ground truth
of every arb moment.

Success criteria: no epoch-1 collapse (loss < 10⁷, entropy above a floor, bet-rate positive across all 18
episodes); arbs as first-class observation signal; an opt-in BC warm start (default off); a head-to-head
beating the `90fcb25f` baseline; and all new knobs default to no-op (byte-identical otherwise).

## Why it matters

The origin of the BC-warm-start + arb-feature direction that the later imitation-first and bc-to-ppo
campaigns built on ([[freeze-bc-head-post-pretrain]]); the value-collapse it diagnoses is the early form
of the [[noop-absorbing-state]] that recurs throughout the scalping work. Its BC oracle must obey the
matcher ([[bc-oracle-must-respect-matcher]]). Substrate: [[forced-arbitrage-scalping-mode]].

## Sources
- `src-1340a0` purpose.md (js_desktop:present)
