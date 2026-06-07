---
id: 01KTJ0K80KN2HJ4J2A2BWK1ESM
type: concept
cloud: shared
status: stable
created: 2026-06-07
updated: 2026-06-07
tags: [work, research, lessons]
sources: [src-04294a]
aliases: [naked-pnl-asymmetry, per-pair-naked-fix]
---

# Naked-PnL asymmetry — per-pair aggregation fix

Replace `min(0, sum(naked_pnls))` with `sum(min(0, per_pair_naked_pnl))` in `env/betfair_env.py::_settle_current_race` so each individual naked loss costs reward and lucky naked wins can no longer cancel them.

## What it is

Pre-fix the asymmetric naked term computed `min(0, scalping_naked_pnl)` over the **aggregate** race naked PnL. Pathology:

```
race naked book: +£100 winning naked + (−£80) losing naked
aggregate naked_pnl = +£20
min(0, +£20)         = 0
penalty for nakeds   = £0
```

Every losing naked is cancelled by any unrelated lucky naked in the same race. The asymmetric intent ("naked losses cost real reward, naked windfalls excluded") only fired when the aggregate was already negative — exactly the case where it was redundant with the loss-direction signal `day_pnl` already provides.

The fix adds a small accessor: `BetManager.get_naked_per_pair_pnls(market_id) -> list[float]` returning the realised PnL of each unfilled-paired aggressive leg. Then the raw reward branch sums `min(0, p)` over those.

## Why it matters

Pre-fix, "place lots of nakeds and hope for lucky aggregate" was a positive-EV strategy. The selection-degeneracy symptom on the activation-A-baseline run (2026-04-17/18): best_fitness=0.338 frozen for three generations, mean degrading −0.016 → −0.024. The [[close-position-action|close_signal]] mechanism was being used (some agents 400+ closes per 15 episodes) but those agents sat at the BOTTOM of the GA ranking — the env was rewarding low-volume naked-roulette over volume-with-closes.

This is a **reward-scale change** under the `raw + shaped ≈ total_reward` invariant ([[reward-invariants]]). Post-fix scoreboards are NOT comparable to pre-fix ones on raw reward magnitudes — call out loudly in commit messages.

## What does NOT change
- [[locked-pnl-min-over-outcomes|locked_pnl]] handling.
- `early_lock_bonus` gate, commission-aware tick floor, `close_signal` mechanism, `naked_penalty_weight` shaped term — all unchanged. Only the raw asymmetric PnL term is rewritten.

## Tests
Pre-existing `raw + shaped == total_reward` invariant must stay green; new cases cover two-naked-pair race (one wins, one loses) so penalty does NOT cancel; single-loser unchanged; single-winner zero penalty; all-completed zero contribution.

## Links
- [[scalping-asymmetric-hedging]] — the upstream plan that introduced `min(0, naked_pnl)`.
- [[close-position-action]] — the mechanic this fix makes uniformly rational.
- [[reward-invariants]] — the contract the change must respect.
- [[shared/index|hub]]
