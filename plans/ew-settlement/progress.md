# Progress — EW Settlement

One entry per completed session.

---

## Session 01 — BetManager EW settlement logic

**Date:** 2026-04-11

Added `each_way_divisor` and `winner_selection_id` parameters to
`BetManager.settle_race()`. When `each_way_divisor` is not None, stake
is split into two half-legs internally. Winner collects both legs
(win + place), placed-only runner loses win half-stake but collects
place leg, unplaced loses full stake. Commission applied per-leg on
gross profit. Lay bets mirror the back logic (layer pays both legs
for winner, wins win leg / loses place leg for placed).

`bet.pnl` reflects total net P&L of both legs combined. `bet.outcome`
set to WON/LOST based on net pnl sign (short-odds placed runners can
still be LOST).

Non-EW path (`each_way_divisor=None`) is completely unchanged.

**Tests added:** `TestEachWaySettlementCorrected` — 10 tests covering
back/lay × winner/placed/unplaced, divisor=None fallback, short-odds
placed-back-loses, and commission=0 per-leg verification.

**All 68 tests pass.**

---

## Session 02 — Environment integration

**Date:** 2026-04-11

Updated `_settle_current_race()` in `betfair_env.py` to pass
`each_way_divisor=race.each_way_divisor` and
`winner_selection_id=race.winner_selection_id` to `bm.settle_race()`.

Verified: `bet.matched_stake` remains the original full stake (no
split), so shaped reward components use the correct value.
`raw + shaped ≈ total_reward` invariant holds.

**Tests added:** `TestEachWayEnvIntegration` — 3 tests: placed-back
uses place fraction, raw+shaped invariant on EW race, mixed WIN+EW
day accumulates day_pnl correctly.

**1725 tests pass** (excluding 2 pre-existing failures: e2e WebSocket
timeout, `past_races_json` integration test).

---

## Session 03 — Fix incorrect comments + episode builder audit

**Date:** 2026-04-11

Fixed 4 incorrect EW comments:
1. `data/episode_builder.py:622-623` — replaced "already reflect the
   place fraction" with correct reference to `BetManager.settle_race()`.
2. `data/feature_engineer.py:581-584` — replaced "already quote the
   place-adjusted price" with correct description.
3. `env/bet_manager.py` docstring — already corrected in Session 01.
4. `data/episode_builder.py:197-198` — replaced "all of them pay out
   on a back bet" with correct EW settlement reference.

Grep for "already quote", "already include the place", "already reflect
the place" returns zero results outside of plan docs.

**Race field audit:** All three fields (`winner_selection_id`,
`winning_selection_ids`, `each_way_divisor`) correctly populated from
tick data and Parquet columns. No fixes needed.

**All 196 core tests pass.**

---

## Session 04 — Historical P&L comparison

**Date:** 2026-04-11

Created `scripts/ew_pnl_comparison.py`. Simulates back-the-favourite
strategy on all 145 EW races across 11 days of training data. Computes
old (full odds) vs new (place fraction) P&L for both BACK and LAY.

Results: mean absolute delta £7,127 per bet at £10 stake. Total delta
+£52,944. 0% sign changes. PLACED runners account for 84% of the
distortion. Full details in `lessons_learnt.md` and
`scripts/ew_pnl_comparison_results.csv`.

**All sessions complete.**
