# Purpose — Market Simulation Improvement

## Why this work exists

The 2026-04-22 Betfair market-model spec
([docs/betfair_market_model.md](../../docs/betfair_market_model.md))
audited the rl-betfair order-matching / passive-fill simulator against
Betfair's documented behaviour. The audit closed one known bug (commit
`4ee9fb5`, the crossability gate) and surfaced a handful of residual
drifts between the simulator and the real exchange. This plan
prioritises the drifts that affect **training-signal correctness** and
fixes them; drifts that only matter for live trading (`Keep` / `Persist`
semantics, non-runner reduction-factor auto-cancels, cross-matching)
are deliberately out of scope.

Ranked by expected impact on training signal:

1. **`tv = backers stake × 2` units audit.** Betfair's traded-volume
   field on the stream API is reported as *backer stake × 2* — the
   doubling accounts for both sides of the matched trade. The
   simulator uses `RunnerSnap.total_matched` deltas to decrement the
   `queue_ahead_at_placement` counter that governs passive fills. If
   those two quantities are in different units — one doubled, one
   one-sided — every passive order fills at 2× the correct rate (or
   half; whichever way the mismatch lands). An invisible factor-of-2
   on the rate at which passive orders mature would materially warp
   arb counts, locked-pnl per race, and the matured-arb bonus. This
   is a one-hour diagnostic; if the units already line up no code
   change is needed, but the fact must be established and written
   down.

2. **LTP as trade-price proxy is lossy on volatile ticks.** The
   crossability gate added in commit `4ee9fb5` uses each tick's LTP
   as a single-price stand-in for "where trades happened on this
   tick". On a tick containing trades at many prices (fast market,
   horse drifting / steaming pre-off) the gate silently drops
   accumulation for every resting order sitting on the wrong side of
   the picked LTP — even though some trades *did* cross those
   orders. The fix is per-price traded-volume deltas:
   `RunnerSnap` carries `traded_delta_by_price: dict[price, volume]`
   already decoded from the stream's `trd` array, and
   `PassiveOrderBook.on_tick` sums only the deltas at prices that
   would cross each order's price. More faithful to Betfair, costs
   one extra dict-lookup per open order per tick.

3. **`MIN_BET_STAKE = 2.00` is stale.** Betfair's exchange minimum
   has been £1 since February 2022. The constant is in
   `env/bet_manager.py`. Update is a two-character code change; the
   load-bearing question is whether any downstream test or training
   invariant is frozen against the £2 value.

Items deferred:

- Queue-ahead frozen at placement (simulator is conservative — biases
  toward slower fills, acceptable).
- Cross-matching (minor for horse markets, material only for
  small-field binaries not in training data).
- Unmatched-bet `Keep` / `Persist` (live-system concern, not training).
- Out-of-band passive cancel-on-drift (real Betfair also doesn't
  auto-cancel these).

See [docs/betfair_market_model.md §7](../../docs/betfair_market_model.md)
for the full open-questions list this plan was derived from.

## Out of scope

- Any change to `ExchangeMatcher` (aggressive-match path) beyond what
  Session 02 needs. The single-price / no-walking contract is
  load-bearing and governed by CLAUDE.md "Order matching" — leave it
  alone.
- Any change to reward-shape terms (matured-arb, MTM, naked clip).
  Those live in other plans.
- Any attempt to model cross-matching, Keep/Persist, or auto-cancel-
  on-non-runner. Out by explicit scope.

## Success criteria

- **Session 01 (tv audit).** A written answer in `progress.md` stating
  whether `RunnerSnap.total_matched` is in doubled (backers×2) or
  one-sided units as consumed by `PassiveOrderBook.on_tick`. If a
  units mismatch exists, a code fix plus a regression test replaying
  a known passive-fill count before-vs-after. If units already line
  up, a targeted unit test that locks the current convention in
  place so a future refactor can't silently flip it.
- **Session 02 (per-price crossability).** `RunnerSnap` carries a
  per-price traded-delta dict (already-decoded or newly-extracted
  from the stream/historical data). `PassiveOrderBook.on_tick`
  Phase 1 sums per-price deltas at crossable prices only. A new
  unit test constructs a synthetic tick with trades at 1.29 AND 1.52
  and asserts the 1.29 lay fills from the 1.29 volume, not from the
  1.52 volume. No regression on existing passive-fill tests.
- **Session 03 (minimum-stake update).** `MIN_BET_STAKE = 1.00`.
  Every test that references the constant still passes. No
  production training run is in flight when this change lands.
- **Validation.** One smoke run on a known date after each session
  to verify arb counts / locked P&L don't shift catastrophically.
  Expected direction of shift is known per-session (see
  `session_prompts/*.md`) and any deviation from that direction is a
  signal that the fix is wrong.
