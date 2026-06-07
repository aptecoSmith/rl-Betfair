---
id: 01KTGF1J0WXFN47AS1HXBQK1BD
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [research, lessons]
sources: [src-1340a0]
aliases: [arb as observation, surface the opportunity, arb_lock_profit_pct, don't make the policy derive it]
---

# Make the arb a first-class observation feature

The second of the three arb-blocking problems: the policy can't act on an opportunity it can't see
cheaply. It currently sees only raw `back_price_1`, `lay_price_1`, `spread`, `spread_pct` per runner and
must **derive** "there is a lockable arb here right now, worth £X after commission" from those — a
non-trivial transform for an LSTM/transformer to stabilise on within 3 generations.

## What it is

The fix surfaces the opportunity directly as observation columns: `arb_lock_profit_pct`,
`arb_spread_ticks`, `arb_fill_time_norm` per runner, and `arb_opportunity_density_60s` globally. "Arbs
are invisible in the observation" — and doubly hard to learn when the [[value-collapse-dont-bet-corner]]
kills exploration before the representation can form. The feature functions live in `env/features.py` and
are written to be **vendorable into `ai-betfair` unchanged**. Hard constraints: the post-commission arb
profit must use the one imported commission constant (same number the settlement uses), and any
RUNNER_KEYS / MARKET_KEYS change bumps the obs schema version (old checkpoints refuse to load; silent
zero-padding forbidden).

## Why it matters

A general representation-learning lever: when a skill depends on a derived quantity the network struggles
to compute, compute it deterministically and feed it in — don't make the policy rediscover it under a
noisy gradient. The same philosophy as the per-runner aux heads (fill_prob / mature_prob) feeding the
actor, and the direction-head feature slice. One of the three stacked problems in [[arb-improvements]].

## Sources
- `src-1340a0` purpose.md (js_desktop:present)
