---
id: 01KTHZTN03PK60FTSD89MB26BP
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-009382]
aliases: [asymmetric-hedge-sizing, equal-profit-sizing]
---

# Equal-profit pair sizing

Sizing rule that picks the **lay stake from the back stake** (or vice-versa) so the net P&L on the WIN and LOSE outcomes of a back/lay pair are equal — locking profit regardless of which side fires.

## What it is

For a back leg @ `back_price` with stake `back_stake`, the equal-profit lay stake is:

```
lay_stake = back_stake × back_price / lay_price
```

Joyeuse / Aintree 2026-04-10 12:45 worked example (price 12.5 → 6.0):

- Equal stakes — back £20 @ 12.5, lay £20 @ 6.0: win = +£130, lose = £0, **locked floor £0**.
- Asymmetric — back £20 @ 12.5, lay £41.67 @ 6.0: [[win]] = +£21.65, lose = +£21.67, **locked floor ~£21.66**.

This is the gross-of-commission form. The commission-aware variant landed later (see the repo CLAUDE.md §"Pair sizing: equal-profit (not equal-exposure)"). [[pnl]].

## Why it matters

The pre-fix scalping reward credited *realised* pair P&L. Equal-stake pairs that happened to win banked £130 as if it were a scalp — but the floor was £0 (pure outcome luck). Without this rule the agent has no reward gradient toward proper scalping; it just places correlated pairs and hopes. The rule is the load-bearing primitive behind [[locked-pnl-min-over-outcomes]] and the [[worst-case-improvement-shaping]] term.

## Links
- [[scalping-asymmetric-hedging]] — the plan that introduced it.
- [[locked-pnl-min-over-outcomes]] — the reward redefinition that depends on it.
- [[shared/index|hub]]
