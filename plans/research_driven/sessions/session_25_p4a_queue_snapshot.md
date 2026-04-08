# Session 25 — P4a: queue-snapshot bookkeeping (state only, no fills)

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraints 4, 5, 7 (matcher
  single-price, LTP junk filter, matcher stays simulation-only
  and vendorable)** all apply. Read them.
- `../analysis.md` §2 — the research's rough queue estimator.
- `../proposals.md` P4
- `../master_todo.md` Phase 2
- `../open_questions.md` Q1 — this session assumes the operator
  answered Q1 with **B** (execution-aware) or
  **B-lite-was-not-enough** (tried P1+P2, gains insufficient).
  If Q1 is still A or unresolved, STOP.
- `../progress.md` — confirm sessions 18 and 22 decision gate
  have landed.
- `../downstream_knockon.md` §3 — context on why the live side
  is different (uses real order stream, not this estimator).
- `../lessons_learnt.md`
- `env/bet_manager.py`, `env/exchange_matcher.py`,
  `env/betfair_env.py`.

## Goal

Introduce the **data structures** for passive-order queue
position into `BetManager` (or a sibling class), **without**
implementing any fill logic or user-visible behaviour yet. After
this session, placing a passive order records its queue-ahead
snapshot and keeps it across ticks, but the order never
matches — fills land in session 26.

This is the "build the container before pouring the liquid"
session. It exists separately so the data-structure change is
reviewed in isolation from the fill logic.

## Inputs — constraints to obey

1. **Matcher stays stateless.** Queue bookkeeping lives on
   `BetManager` (or a new sibling `PassiveOrderBook` class
   owned by `BetManager`). The matcher does not gain new
   state. Its existing aggressive code path is unchanged.
2. **No action-space change yet.** This session does not touch
   the action vector. The only way to place a passive order
   after this session is via a new internal method on
   `BetManager` — call it from a test, not from the policy.
   The action-space change comes in session 28.
3. **Resting orders do not affect budget yet.** Budget
   reservation for passive orders comes in session 26 (when
   they can actually fill and consume budget). This session
   records the order and its queue position; that's all.
4. **Race reset is explicit.** When `BetManager` is recreated
   per race (the env already does this), the passive order
   book is empty on construction. Test this.

## Steps

1. **Decide structure.** Two options:
    - **(A)** New fields on `BetManager`:
      `_passive_orders: list[PassiveOrder]` plus accessors.
    - **(B)** New class `PassiveOrderBook` owned by
      `BetManager` as `self.passive_book`.
   **Recommended: (B).** It gives the fill logic in session 26
   a clean surface to attach to, and makes it obvious to
   reviewers which code is aggressive (on `BetManager`
   directly) and which is passive (on `passive_book`).
   Document the choice in `progress.md`.

2. **Add `PassiveOrder` dataclass.** Fields at minimum:
    - `selection_id: int`
    - `side: BetSide`
    - `price: float` (the price the order rests at)
    - `requested_stake: float`
    - `queue_ahead_at_placement: float`
    - `traded_volume_since_placement: float = 0.0`
    - `placed_tick_index: int`
    - `market_id: str`
    - `matched_stake: float = 0.0`  # reserved for session 26
    - `cancelled: bool = False`      # reserved for session 29

3. **Add placement method** (e.g. `PassiveOrderBook.place(
   runner, stake, side, market_id, tick_index)`) that:
    - Snapshots `queue_ahead_at_placement` from the own-side
      top level's `size` (the level the order would rest
      behind — for a passive back, it's the best back
      price; for a passive lay, it's the best lay price).
    - Refuses the order if the own-side top is junk-filtered
      out under the same ±`max_price_deviation_pct` rule the
      matcher uses. A passive order placed into filtered-out
      levels must not silently succeed — it returns `None`.
    - Appends the `PassiveOrder` to the book and returns it.

4. **Add per-tick update method** (e.g.
   `PassiveOrderBook.on_tick(tick)`) that currently **does
   nothing** to the orders except accumulating
   `traded_volume_since_placement` from the tick's traded-
   volume delta at the relevant price. No fill logic. No
   cancellation logic. Just the accumulator.

   Source of the traded-volume delta: per `open_questions.md`
   Q4, the default is "compute at runtime by snapshotting at
   placement and subtracting". Implement that here. If Q4 was
   answered differently, follow the operator's decision.

5. **Wire `on_tick` into the env's per-tick loop.** Every tick,
   after the ladder is updated and before the action is
   processed, call `self.bet_manager.passive_book.on_tick(tick)`.
   Place the call at the same point in the tick where the
   live version would be refreshed by an order-stream event.

6. **Expose for inspection.** `info["passive_orders"]` returns
   a list of `PassiveOrder` dicts (serialised) for the current
   race. Used by the tests and by the replay UI later.

## Tests to add

Create `tests/research_driven/test_p4a_queue_snapshot.py`:

1. **Place a passive back order.** Known top-of-back size at
   placement → `queue_ahead_at_placement` equals that size.
2. **Place a passive lay order.** Mirror of (1).
3. **Placement into junk-filtered level refused.** If the
   own-side top is outside the LTP tolerance, `place` returns
   `None` and the book is empty afterwards.
4. **Traded volume accumulates across ticks.** Place an order,
   advance the fixture through K ticks with known traded
   volume at the relevant price, assert
   `traded_volume_since_placement` equals the sum.
5. **Traded volume at other prices is ignored.** A tick with
   traded volume at a *different* price does not contribute.
6. **Race reset empties the book.** Start a second race,
   assert `passive_book` is empty on construction.
7. **No aggressive regression.** Place a mix of aggressive and
   passive orders in the same race. Aggressive bets still go
   through `bets` exactly as before. The existing matcher and
   bet-manager tests all still pass.
8. **Budget is unaffected by passive placement.** Place a
   passive order; assert `available_budget` is unchanged.
   (Budget reservation is session 26's job; this test pins
   that it is deliberately not happening yet.)

All CPU, all fast.

## Manual tests

None. This session has no user-visible behaviour. Session 26
is the first one the operator can see.

## Session exit criteria

- All 8 new tests pass.
- All existing tests pass.
- Structure choice (A) vs (B) documented in `progress.md` with
  reasoning.
- `progress.md` Session 25 entry.
- `master_todo.md` Phase 2 P4 box gets a sub-tick for P4a
  (add sub-bullets if not already present).
- Commit.

## Do not

- Do not implement fill logic. Session 26.
- Do not implement cancellation logic. Session 29.
- Do not change the action space. Session 28.
- Do not reserve budget for passive orders yet. Session 26.
- Do not touch `ExchangeMatcher`. All queue bookkeeping is on
  `BetManager` / `PassiveOrderBook`.
- Do not skip the "race reset empties the book" test. The
  whole class of "state leaks across races" bugs we've
  already fixed once (see CLAUDE.md `realised_pnl` note)
  starts with exactly this kind of forgotten reset.
