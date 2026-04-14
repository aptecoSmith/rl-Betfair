# Progress — Forced Arbitrage

## Session 1 — Environment mechanics (2026-04-14)

Implemented:

- `env/tick_ladder.py` — Betfair non-linear tick ladder with
  `snap_to_tick`, `tick_offset(price, n_ticks, direction)`, and
  `ticks_between` utilities. Dependency-free stdlib only so the module
  can be vendored into `ai-betfair` live-inference alongside
  `exchange_matcher.py`.
- `Bet.pair_id` and `PassiveOrder.pair_id` fields link an aggressive
  fill to its auto-generated passive counter-order.
- `PassiveOrderBook.place()` gained `price=` and `pair_id=` keyword
  args. With an explicit price the order rests at that exact ladder
  level (still subject to the junk filter) and records queue-ahead of
  zero when the level is empty. `place_back`/`place_lay` accept
  `pair_id=` so the aggressive leg can be tagged.
- `BetManager.get_paired_positions()` groups matched bets by pair_id
  and computes `locked_pnl` (commission-deducted spread) for completed
  pairs. `BetManager.get_naked_exposure()` sums the worst-case loss on
  unpaired matched bets.
- `BetfairEnv.__init__` accepts `scalping_mode` (also from
  `config["training"]["scalping_mode"]`). When on:
  - Per-runner action dim bumps from 4 to 5 (`arb_spread` maps
    [-1, 1] → [MIN_ARB_TICKS, MAX_ARB_TICKS] = [1, 15] ticks).
  - Observation gains 2 per-runner features (`has_open_arb`,
    `passive_fill_proximity`) and 2 global (`locked_pnl_frac`,
    `naked_exposure_frac`).
  - After every successful aggressive fill, `_maybe_place_paired`
    auto-places the opposite-side passive counter-order at
    `fill_price ± arb_ticks` using the real ladder.
- `config.yaml` — `training.scalping_mode: false` added, documented.
- `tests/test_forced_arbitrage.py` — 25 new tests covering tick ladder
  math (band transitions, clamping), pair helpers (locked PnL,
  naked exposure, commission), and env integration (action/obs space
  shapes, paired placement, backward-compat when off).

Backward compatibility verified: all 1811 pre-existing tests still
pass. When `scalping_mode=False` the action space (56), observation
space, and step behaviour are byte-identical to pre-session code.
