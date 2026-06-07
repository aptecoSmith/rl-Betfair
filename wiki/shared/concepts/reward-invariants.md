---
id: 01KTJ039CMQ8TWYEHXDJZG4Z23
type: concept
cloud: shared
status: draft
created: 2026-06-07
updated: 2026-06-07
tags: [work, research]
sources: [src-032073]
aliases: [raw-plus-shaped-invariant]
---

# Reward invariants (raw + shaped ≈ total)

The three load-bearing invariants every new reward term must respect, and the test contract that catches drift.

## What it is

1. **`raw + shaped ≈ total_reward` every episode.** Both accumulators must add to the total — if a new term lands outside an accumulator, the master invariant breaks. Classify every new reward term as raw or shaped and bucket it correctly in `env/betfair_env.py::_settle_current_race`.
2. **Zero-mean for random policies.** No asymmetric positive-per-bet bonuses (the prior `early_pick_bonus` was made symmetric for exactly this reason).
3. **Authoritative day P&L.** `info["day_pnl"]` is the day's truth; `info["realised_pnl"]` is last-race-only and exists for backward compat only — reading the wrong one is how the phantom-profit bug hid.

## Why it matters

Together these are what makes shaped-reward changes *safe* — they let a new term redistribute training signal without secretly inflating raw cash, so cohort runs across reward changes stay comparable on [[locked-pnl-min-over-outcomes|locked_pnl]] / `raw_pnl_reward`. Every plan in this folder (`plans/`) leans on these — violating one will quietly corrupt selection scoreboards across cohorts.

## Links
- [[arch-exploration]] — the plan that calls these out as hard constraints.
- [[exchange-matcher]] — paired load-bearing contract on the matching side.
- [[shared/index|hub]]
