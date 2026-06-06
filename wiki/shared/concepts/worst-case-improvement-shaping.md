---
id: 01KTFBH2KF8RX5VFGCN5M4FN79
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research, reward, shaped]
sources: [src-009382]
links: [{to: locked-pnl-per-pair-definition, type: builds-on}, {to: raw-vs-shaped-reward, type: refines}]
aliases: [worst-case-improvement term, Δ worst_case shaping]
---

# Worst-case-improvement shaping term

Per closing leg, add `Δ worst_case = worst_case_after − worst_case_before`
to `shaped_bonus`. This is a dense per-step gradient that rewards moves
which lift the pair's guaranteed floor.

## What it is

`worst_case` is the per-runner pair P&L minimised over race outcomes (the
quantity clipped in [[locked-pnl-per-pair-definition]]). Each time the
agent places a closing leg, the term fires once with the delta.

Example (Joyeuse, back £20 @ 12.5 → worst_case = −£20):
- Lay £20 @ 6.0 → worst_case = £0; Δ = **+£20** (equal-stake hedge).
- Lay £41.67 @ 6.0 → worst_case = +£21.67; Δ = **+£41.67** (properly sized).

The proper hedge collects more than twice the shaped reward of the bad
hedge — gradient that specifically rewards sizing, not just closing.

## Why it matters

The legacy [[close-signal-bonus-legacy]] was a flat per-close payout that
fired regardless of whether the close actually locked anything. Worst-case-
improvement replaces "you closed" with "you closed *well*" and arrives
before settlement, so the credit lands while the action is still recent.
Sits in the shaped bucket of [[raw-vs-shaped-reward]].

## Sources
- `src-009382` purpose.md (js_desktop:present)
