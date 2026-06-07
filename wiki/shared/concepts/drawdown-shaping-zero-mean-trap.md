---
id: 01KTGJS2NE3QFXRE384NNW83MN
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [drawdown shaping trap, non-positive shaping, reflection symmetry zero-mean, peak_0 trough_0 zero]
---

# Drawdown shaping: the obvious formulation is a zero-mean trap

A reward-design trap: the obvious drawdown term `shaped = −ε·max(peak − current, 0)/budget` is **strictly
non-positive**, so a random policy accumulates negative shaped reward and "bet less" looks strictly better
than "bet more" — the same asymmetric-shaping bug the phantom-profit investigation spent a whole session
fixing.

## What it is

The fix is **reflection symmetry as the cheapest zero-mean tool**: Option D `(2·day_pnl − peak − trough)/
budget` is zero-mean not by lucky integration but because every path and its sign-flipped reflection
cancel *algebraically* (under `X → −X`, `peak ↔ −trough`, so the term maps to its own negation). That
admits a much stronger guard than a statistical "N=1000 within 2 SE" test — an exact "reflection pairs
cancel to 1e-9" test. The cancellation **requires `peak_0 = trough_0 = 0`**: initialising peak/trough to
±inf breaks the reflection proof. The discipline: if a future session adds a shaped term, write the
zero-mean proof into the design pass **before** touching `_settle_current_race`.

## Why it matters

The general rule behind the project's symmetry-around-random-betting invariant: any new shaped term must be
zero-mean for a random policy or it teaches the agent to bet (or not bet) for the reward rather than the
outcome — proven before code, not patched after. Reflection symmetry gives both the construction and an
exact-cancellation test. Pairs with [[isolate-term-for-zero-mean-test]] (how to test it without commission
drift) and the [[raw-vs-shaped-reward]] split.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
