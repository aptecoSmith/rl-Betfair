---
id: 01KTGJG33SVPPP8JPXPSHS67XG
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1663dd]
aliases: [symmetric vs asymmetric MTM, loss-cutting gradient, MTM wants asymmetry]
---

# Symmetric MTM vs the asymmetric gradient trading wants

An observation from the force-close-architecture code read: the mark-to-market shaping term is
**symmetric** (a gradient on every tick) where trading wants it **asymmetric** — a much stronger gradient
when exposure is bleeding past a tolerable loss than when it's in modest profit.

## What it is

The `close_signal` closes at the current top opposite-side price with no target and no stop-loss anchor —
the agent fires and takes whatever the book offers — and the reward shaping doesn't push the policy to fire
on a projected-loss threshold. A symmetric MTM term redistributes realised P&L evenly across ticks but
gives no extra pressure to cut a losing pair; loss-cutting behaviour needs the gradient to spike on the
loss side specifically.

## Why it matters

The design rationale behind the stop-close mechanism ([[env-stop-close-not-agent-learned]]) — rather than
try to bend the symmetric MTM term into an asymmetric one purely via reward weights, the plan makes
loss-cutting structural in the env. It also predicts why the later **E2 asymmetric-MTM probe** (drawdowns
weighted 3× gains) was tried — and why it was NO BITE at probe scale
([[gradient-delivered-ppo-unresponsive]]): the asymmetry is the right intuition but a shaped gradient
couldn't carry it. Same symmetry discipline as the zero-mean reward terms, inverted here on purpose.

## Sources
- `src-1663dd` purpose.md (js_desktop:present)
