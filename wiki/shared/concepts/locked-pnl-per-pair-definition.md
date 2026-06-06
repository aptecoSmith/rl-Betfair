---
id: 01KTFBH2KEFJE01E2EFFPXHCSS
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-009382]
links: [{to: equal-profit-sizing, type: supports}, {to: raw-vs-shaped-reward, type: refines}, {to: reward-shaping-supersessions, type: part-of}]
aliases: [scalping_locked_pnl per-pair, locked floor per pair]
---

# Locked-PnL per-pair definition

The scalping reward's locked accumulator is defined **per pair** as
`max(0, min(win_pnl, lose_pnl))` — the guaranteed floor across the two race
outcomes, never less than zero.

## What it is

For a back/lay pair on the same runner, evaluate the pair's P&L under each
race outcome (runner wins / runner loses). The pair contributes its
**worst-case profit** — the `min` over outcomes — to `scalping_locked_pnl`,
clipped at zero so a directional pair contributes nothing rather than a
negative number.

An equal-stake pair on a price that has moved 12.5 → 6.0 (Joyeuse, Aintree
2026-04-10) yields `win=+£130`, `lose=£0`; `min=£0`; locked contribution =
**£0**. A properly-sized pair at the same prices yields `win≈+£21.65`,
`lose≈+£21.67`; locked contribution ≈ **£21.66** — guaranteed profit from
the price move.

## Why it matters

Without the `min`-over-outcomes definition, the locked accumulator would
absorb whichever realised P&L the pair happened to produce — so a lucky
back-side hit on an equal-stake pair would launder into the "locked" bucket
even though the pair locked nothing. The per-pair `min` is what makes
[[equal-profit-sizing]] the only sizing rule that earns locked reward; it
is also the mathematical underpinning of the locked / directional / naked
classification surfaced in the Bet Explorer UI. Sits in the raw bucket of
[[raw-vs-shaped-reward]].

## Sources
- `src-009382` purpose.md (js_desktop:present)
