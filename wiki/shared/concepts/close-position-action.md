---
id: 01KTFBH2KD44FJ3P2QFW9M2FFQ
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-009382]
links: [{to: equal-profit-sizing, type: derived-from}, {to: locked-pnl-per-pair-definition, type: supports}]
aliases: [close_signal action, close position on runner]
---

# Close-position action

A dedicated env action that closes any open position on a runner at market.
The env sizes the hedge analytically (the equal-profit / asymmetric-stake
formula, clamped to ladder depth); the agent only chooses *whether* and
*when* to close.

## What it is

The legacy discrete stake head could not pick non-grid stakes like £41.67,
so it could never construct a properly-sized hedge on its own — every
agent-driven close was equal-stake by construction. The close-position
action removes that limitation by delegating sizing to the env: the agent
emits a categorical "close X" signal and the env computes the lay (or back)
stake needed to lock profit at the current ladder.

Mirrors how a human scalper actually trades — pick the runner and the
moment, let the platform compute the hedge.

## Why it matters

Without this action, [[locked-pnl-per-pair-definition]] and
[[worst-case-improvement-shaping]] would give the agent a reward signal it
literally couldn't earn — the stake grid has no value that produces a
locked floor at most price pairs. The close-position action is the
**capability** half of the asymmetric-hedging plan; the reward changes are
the **incentive** half. Both must land for the policy to learn proper
scalping.

## Sources
- `src-009382` purpose.md (js_desktop:present)
