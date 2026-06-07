---
id: 01KTHZTN05Z2YVC8RJ1T9PAT48
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work]
sources: [src-009382]
aliases: [bet-explorer-badge]
---

# Pair classification badge (locked / neutral / directional / naked)

[[ui|UI]] / diagnostic taxonomy that labels each pair by its **worst-case floor**, not its realised P&L — so the operator can tell skill from luck at a glance.

## What it is

Classification rules per pair:

| Floor | Label | Colour |
|---|---|---|
| `min(win_pnl, lose_pnl) > 0` | locked | green |
| `min(win_pnl, lose_pnl) = 0` | neutral pair | grey |
| `min(win_pnl, lose_pnl) < 0` | directional | amber |
| unpaired matched order | naked | red |

Pure diagnostic — no training impact. Lets the operator watch [[locked-pnl-min-over-outcomes]] and [[worst-case-improvement-shaping]] teach the agent before committing to the [[close-position-action]].

## Why it matters

[[bet-explorer|Bet Explorer]]'s pre-fix tinting was based on realised P&L, which mis-reads outcome luck as scalping skill. The badge taxonomy makes that distinction visible without changing reward — a precondition for honest evaluation of the [[scalping-asymmetric-hedging]] plan's effect. Aligns with the broader reporting principle that selection should look at structural metrics, not luck-blended ones. [[pnl]].

## Links
- [[scalping-asymmetric-hedging]] — the plan that introduced it.
- [[shared/index|hub]]
