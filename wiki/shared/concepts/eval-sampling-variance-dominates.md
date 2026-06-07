---
id: 01KTG1DSM9J7K8YA404AFF9X2N
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, lessons]
sources: [src-0d0f55]
aliases: [eval sampling variance, lucky dice, action-sampling noise]
---

# Eval action-sampling variance dominates day-P&L

Action-sampling stochasticity at **eval** time produces ~£200+ of day-pnl swing on the *exact same
trained policy on the exact same eval day* — a single different RNG roll.

## What it is

Concrete: agent 658a7f72 swung +£178.32 → +£55.43 (£123) and 81c80d76 swung −£0.19 → −£336 between two
sampling seeds on identical weights+day. Three causes, by magnitude: per-tick **categorical action
sampling** (`action_dist.sample()` → different open/close decisions → different naked exposure into
settle); per-tick **Beta stake sampling** (`stake_dist.sample()` magnifies/shrinks each decision); and
**naked-pnl amplification** (small differences in which pairs end up naked get multiplied by race
outcomes — the loudest noise channel, see [[raw-vs-shaped-reward]]). The effect **dominates any
cross-agent or cross-architecture P&L comparison**, so **even agents that look profitable are mostly
riding lucky dice**.

## Why it matters

The motivation for [[argmax-eval]]: a cash signal drowning in eval-sampling noise can't rank agents.
This is the measurement-side counterpart to getting selection right — and the core reason argmax-vs-
stochastic is the dominant deploy-time unknown.

## Sources
- `src-0d0f55` purpose.md (js_desktop:present)
