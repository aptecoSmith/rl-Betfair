# Session 4 — Pure arb feature functions

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 2, Session 4.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — "stdlib only,
  vendorable into ai-betfair, zero not NaN when unpriceable".
- `plans/arb-improvements/progress.md` — read the Phase 1 summary.
  Phase 1 must be green before this session starts.
- Reference implementation: `env/features.py` already contains six
  pure feature functions. Match the style exactly.

## Goal

Add four pure feature functions that make arb opportunities directly
observable to the policy. No env wiring this session — just the
functions and their tests. Wiring happens in Session 5.

## Scope

**In scope:** add to `env/features.py`:

- `compute_arb_lock_profit_pct(back_levels, lay_levels, ltp_fallback,
  commission_rate) -> float`
  - Returns the after-commission lockable profit (positive number)
    expressed as a fraction of the mid-price.
  - When the book is crossed (best back > best lay), this is
    positive and represents real lockable P&L.
  - When uncrossed (best back < best lay — the normal state), returns
    a *negative* number representing the cost of crossing the
    spread. This is the key signal: a less-negative value is closer
    to a lockable arb.
  - Returns `0.0` if either side has no valid level after junk
    filter, or if `ltp_fallback` is non-positive and no book exists.

- `compute_arb_spread_ticks(back_levels, lay_levels, ltp_fallback,
  max_ticks) -> float`
  - Betfair-tick-aware version of the above. Number of ticks
    between best back and best lay. Positive when crossed,
    negative when uncrossed. Clamped to `[-max_ticks, +max_ticks]`.
  - Uses `betfair_tick_size` from the same module.
  - Returns `0.0` when unpriceable.

- `compute_arb_fill_time_norm(passive_size, traded_delta, max_norm)
  -> float`
  - Heuristic: `passive_size / max(traded_delta, eps)` — "how many
    seconds of recent volume before the passive leg fills at
    current size". Normalised into `[0, 1]` by dividing by
    `max_norm` and clamping.
  - Lower values = faster fills. Higher = slow market.
  - Returns `max_norm` (i.e. 1.0 after normalisation) when
    `traded_delta` is zero or negative.

- `compute_arb_opportunity_density(history, window_seconds, now_ts)
  -> float`
  - Fraction of entries in `history` within the last
    `window_seconds` where `arb_lock_profit_pct > 0` was observable
    across any runner.
  - Returns `0.0` on empty history or when no entries fall in the
    window.
  - `history` is an iterable of `(timestamp_s, any_arb_available:
    bool)` tuples. Keep the format this simple — the computation
    of `any_arb_available` per tick happens upstream of this
    function so the function stays pure.

- Commission constant import: define `BETFAIR_COMMISSION_RATE` in
  one place (the cleanest home is `env/bet_manager.py` next to the
  existing settlement math, but if it already lives there, just
  import — don't duplicate). Feature functions accept commission
  as an arg so they remain pure; callers pass the module constant.

**Out of scope:**

- Wiring any of this into `BetfairEnv` (Session 5).
- Computing the `any_arb_available` flag per tick (that's a
  Session 5 concern — one-pass across all runners).
- Any policy / training changes.

## Exact code path

Everything is inside `env/features.py`. Follow the docstring and
style of the existing `compute_microprice` /
`compute_obi` / `compute_mid_drift` functions:

- Full docstring with Parameters / Returns / Raises sections.
- Type hints using `|` unions (Python 3.12).
- No imports beyond stdlib and the existing `betfair_tick_size`
  helper already in the module.
- Duck-typed `PriceLevel` inputs (anything with `.price` and
  `.size`).

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_arb_features.py`:

1. **Uncrossed book → negative profit_pct.** Back at 5.0, lay at
   5.2. Computed value is negative (cost to cross), value is
   hand-computable from the definition.

2. **Crossed book → positive profit_pct.** Back at 5.2, lay at 5.0.
   Positive value equals `(5.2 - 5.0) / mid * (1 - commission_rate)`
   within float tolerance.

3. **Unpriceable → zero.** One-sided book, no LTP → returns `0.0`
   (not NaN, not negative).

4. **`compute_arb_spread_ticks` across ladder transitions.**
   Back at 2.02, lay at 1.99 spans the 2.00 tick-size change.
   Assert the tick count is computed using the correct band per
   side.

5. **`compute_arb_spread_ticks` clamps to `max_ticks`.** Wildly
   crossed book → returns `+max_ticks`, not the raw count.

6. **`compute_arb_fill_time_norm` clamped to `[0, 1]`** and returns
   `1.0` when `traded_delta <= 0`.

7. **`compute_arb_opportunity_density` on synthetic history.** Build
   a 120-tick history with 30 ticks flagged as arb-available in the
   last 60 seconds; assert density ≈ 0.5 within float tolerance.

8. **Density on empty / all-old history → 0.0.**

9. **Commission constant is a single source.** Assert the value
   used by the feature function equals the value used by
   `BetManager.get_paired_positions` (introspect the attribute or
   assert equality against the constant directly).

10. **Vendorable.** Import `env/features.py` in isolation — no
    imports of `env.betfair_env`, `env.bet_manager`, etc. should
    be required. (Mechanical: the module itself must only import
    stdlib + its own helpers.)

## Session exit criteria

- All 10 tests pass: `pytest tests/arb_improvements/test_arb_features.py -x`.
- Existing tests still pass.
- `progress.md` Session 4 entry written.
- `lessons_learnt.md` updated if anything surprising came up.
- Commit: `feat(features): pure arb-opportunity feature functions`.
- `git push all`.

## Do not

- Do not wire the features into `BetfairEnv` this session. That's
  Session 5's problem — splitting keeps the diff reviewable.
- Do not clip `arb_lock_profit_pct` to non-negative. Negative
  values are signal, not noise.
- Do not import anything from `env/betfair_env.py` or
  `env/bet_manager.py` at module load. Commission constant is
  fine to import from bet_manager if that's where it lives.
- Do not add GPU tests.
