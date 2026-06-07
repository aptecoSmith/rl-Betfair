---
id: 01KTHZTN02WRDMHS7SX1VRQMXW
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-009382]
aliases: [close-signal-action]
---

# Close-position action

Dedicated env action `"close open position on runner X at market"` — the agent picks *whether* and *when* to close; the env computes the hedge stake from [[equal-profit-pair-sizing]] (clamped to [[ladder]] depth).

## What it is

Per the source: the discrete stake head can't pick £41.67, so the agent has no way to act on a reward signal that says "size £41.67". The close-position action removes the stake choice from the agent and gives the env the freedom to size the hedge correctly. Mirrors how a live human scalper actually operates — they decide direction and timing; the platform handles the exact stake.

In the deployed env this became the `close_signal` action; the shaped `CLOSE_SIGNAL_BONUS` is zeroed (closing learns from raw cash only — see the repo CLAUDE.md §"CLOSE_SIGNAL_BONUS = 0.0"). The matched-arb bonus excludes agent-closed pairs (they crossed at market, not scalped).

## Why it matters

Step 4 of the [[scalping-asymmetric-hedging]] plan: without this action the fixed reward signal has no policy lever the agent can pull. The other three changes (locked-pnl redefinition, worst-case shaping, badge classification) instrument the problem; this one gives the agent the tool to act on it.

## Links
- [[scalping-asymmetric-hedging]] — the plan.
- [[equal-profit-pair-sizing]] — the sizing rule the env uses for the hedge.
- [[shared/index|hub]]
