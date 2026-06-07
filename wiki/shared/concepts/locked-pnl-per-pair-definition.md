---
id: 01KTFQ0JCZ7ZMGENY4BAQJBTGG
type: concept
cloud: shared
status: draft
created: 2026-06-06
updated: 2026-06-06
tags: [research]
sources: [src-009382]
aliases: [locked_pnl, scalping_locked_pnl, locked pnl per pair]
---

# Locked-PnL per-pair definition

`scalping_locked_pnl` per pair = **`max(0, min over outcomes of pair P&L)`** — the guaranteed
worst-case floor of a back/lay pair, not its realised P&L.

## What it is

An equal-stake pair has `min(win_pnl, lose_pnl) = 0`, so it locks nothing and contributes £0 to the
locked bucket regardless of which outcome fires — even if it realised a large profit (that profit came
from the race outcome, i.e. luck, not from trading the price move). A properly-sized pair has
`min > 0` and contributes its guaranteed floor. This definition is what stops the reward crediting
"accidental pairs" where the agent got lucky on outcome. It is the foundation the sizing work builds
on: [[equal-exposure-sizing]] (the original asymmetric formula) sized for it, and
[[equal-profit-sizing]] later made `min(win, lose)` the *true* lock after commission.

## Why it matters

It is the structural core of the locked half of [[raw-vs-shaped-reward]] — "locked" means
worst-case-guaranteed, so the GA selects trading skill rather than directional luck. Introduced with
the scalping-asymmetric-hedging work (origin of the reward redefinition).

## Sources
- `src-009382` purpose.md (js_desktop:present)
