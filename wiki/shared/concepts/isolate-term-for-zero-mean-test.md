---
id: 01KTGJS2NHNZ09K40X96DEA8ZQ
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-184d90]
aliases: [isolate the term, commission=0 in zero-mean test, extract formula to helper, 4-dp rounding tolerance]
---

# Isolate the reward term to test its zero-mean property

A testing-method lesson: to verify a shaping term's zero-mean property, **extract the formula into a
helper and drive it with synthetic trajectories** — don't run it through the full env, where unrelated
effects (commission, matcher behaviour) contaminate the math.

## What it is

The drawdown formula originally lived inline in `_settle_current_race`; testing it there would require the
full bet-matching pipeline, couple the invariant test to `ExchangeMatcher`/`BetManager`, and pick up
commission drift forcing a looser tolerance. Pulling it into `_update_drawdown_shaping()` let tests drive
arbitrary synthetic `_day_pnl` trajectories directly. And **commission drift matters once you isolate the
term**: the `early_pick_bonus`/`precision_bonus` zero-mean claims are "zero-mean modulo commission" — small
enough to hide in a mixed-term test but, isolated at 5% commission over N=1000, large enough to blow past
2 SE, so the fixture sets `commission: 0.0` to isolate the mathematical property. Related tolerance trap:
the episode log writes `round(v, 4)`, capping `|total − (raw+shaped)|` at ~1.5e-4 — so a 1e-6 invariant
threshold would fail on rounding, not real bugs.

## Why it matters

Two reusable rules: isolate a numerical invariant from its environment to test it cleanly (helper + zero
commission), and **size the tolerance to the logging precision** (4-dp rounding ⇒ ~1e-4 floor, not 1e-6).
The companion to [[drawdown-shaping-zero-mean-trap]] (the property) and a concrete application of the
[[raw-vs-shaped-reward]] invariant's "raw + shaped ≈ total" check.

## Sources
- `src-184d90` lessons_learnt.md (js_desktop:present)
