# Session 26 — P4b: passive-fill triggering + budget reservation

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 2 (raw vs shaped
  bucketing), 4 (single-price), 7 (matcher stays vendorable),
  11 (live wins).
- `../bugs.md` R-2 — self-depletion. **Passive fills are
  affected too**: if the agent has two passive orders at the
  same price, the second one's fill decision must deduct the
  first one's already-filled stake. Session 18 handled this
  for aggressive fills; session 26 extends the same pattern
  to passive fills.
- `../analysis.md` §2
- `../proposals.md` P4
- `../master_todo.md` Phase 2
- `../progress.md` — confirm session 25 has landed and the
  `PassiveOrderBook` structure is in place.
- `../lessons_learnt.md`
- `env/bet_manager.py`, and whatever `PassiveOrderBook` class /
  fields session 25 landed.

## Goal

Implement the **fill logic** for passive orders. After this
session, a passive order placed at price P rests until
accumulated traded volume at P since placement ≥ queue-ahead,
at which point it matches and becomes a real `Bet` that
settles with the rest of the race.

Also: reserve budget at placement time, release/convert it at
fill time.

## Inputs — constraints to obey

1. **Fill conversion produces a normal `Bet`.** A passive
   order that fills becomes a `Bet` object in `BetManager.bets`
   identical to what an aggressive fill at the same price would
   have produced. Downstream settlement, P&L, reward shaping,
   and eval logs all work unchanged.
2. **Fill price is the queue price, not the opposite-side
   price.** This is the whole point of passive orders — the
   agent joined the queue at a price *better than* the current
   opposite-side top. Test this.
3. **Budget is reserved at placement, released on cancel, and
   converted on fill.** A placed-but-unfilled passive order
   reduces `available_budget` by its requested stake (or
   liability, for lay orders) exactly as an aggressive fill
   would. On fill, the reservation becomes the fill's real
   budget consumption (no double-counting).
4. **Self-depletion applies to passive fills too.** If two
   passive back orders rest at the same price on the same
   selection, they share the queue-ahead. The second one's
   fill threshold is `queue_ahead + first_order_size`, not
   `queue_ahead`. Equivalently, the same
   `_matched_at_level`-style accumulator on
   `PassiveOrderBook` that session 18 introduced on
   `BetManager`.
5. **LTP junk filter still applies.** If at some later tick the
   price the order rests at has become junk (e.g. the LTP
   moved away and the old rest price is now outside the
   tolerance), the order is NOT auto-cancelled — that's
   session 27's job (race-off cleanup and policy-driven
   cancel). For this session, a junk-filtered rest price
   means the order simply does not fill on that tick; it is
   re-evaluated next tick.

## Steps

1. **Extend `PassiveOrderBook.on_tick`** (the no-op from
   session 25) with fill logic:
    - For each open passive order, check if
      `traded_volume_since_placement >= queue_ahead_at_placement
      + self_depletion_at_this_price`.
    - If yes, mark as filled: set `matched_stake = requested_stake`
      (partial fills come later if at all), create a `Bet`
      with the queue price as `average_price`, append to
      `BetManager.bets`, remove from the open passive list.

2. **Add placement-time budget reservation.** When
   `PassiveOrderBook.place` is called, subtract the requested
   stake (for back) or liability (for lay) from
   `BetManager.budget`. If the reservation would exceed
   `available_budget`, return `None` (do not place).

3. **On fill, do not double-subtract.** The budget was already
   reserved at placement; the fill conversion is a no-op on
   budget. Add a test that pins this.

4. **Extend the passive self-depletion accumulator.** Analogous
   to `_matched_at_level` from session 18 but keyed on *own-
   side* price levels (since passive orders rest on the own
   side, aggressive bets match on the opposite side). The two
   accumulators are distinct. Document the distinction in a
   comment above each.

5. **Emit a fill event** in `info["passive_fills"]` per tick —
   list of selection_id / price / filled_stake tuples for any
   orders that converted on this tick. Used by the replay UI
   and the manual test.

## Tests to add

Create `tests/research_driven/test_p4b_passive_fill.py`:

1. **Rest and fill.** Place passive back at P with
   queue-ahead £200. Advance ticks with £150 traded →
   unfilled. Then £60 more traded → filled. The new `Bet`
   has `average_price == P`, not the opposite-side top.
2. **Rest and not-fill.** Same but only £150 total traded
   across all ticks → still unfilled at the end of the
   fixture. The passive order is still in the open list.
3. **Budget reservation at placement.** Place passive back of
   £50; assert `available_budget` drops by 50 immediately,
   before any traded volume.
4. **No double-subtraction on fill.** Place passive back of
   £50; let it fill. Assert `available_budget` is the same
   *after* the fill as it was *immediately before* the fill.
   The fill only moves the accounting, not the amount.
5. **Passive self-depletion.** Place two passive backs at the
   same price on the same selection, £10 each, queue-ahead
   £15. After £15 traded → first fills (queue-ahead met),
   second still resting. After £10 more traded → second
   fills.
6. **Fill price is the queue price, not the opposite-side
   top.** This is the key invariant that distinguishes P4
   from the aggressive path. Assert the `Bet.average_price`
   equals the price the passive order was placed at, *not*
   the best opposite-side price at fill time.
7. **Filled passives settle with the race.** A filled passive
   appears in `BetManager.bets`, contributes to `day_pnl`
   settlement, and appears in eval bet logs identically to
   an aggressive bet at the same price.
8. **Junk-filtered rest price does not fill.** A passive
   order whose rest price has drifted outside the LTP
   tolerance on a later tick does not match on that tick.
   (It might match again if the LTP drifts back.)
9. **Aggressive regression.** Mixed aggressive + passive
   placements in one race — all existing aggressive matcher
   tests still pass.
10. **Invariant.** `raw + shaped ≈ total` holds across all of
    the above.

All CPU, all fast.

## Manual tests

- **Open a race in the replay UI with a P4b-enabled fixture**
  and confirm the operator can *see* a passive order rest
  for several ticks before filling. If the filler converts
  immediately on every order, the fill threshold is wrong.
- **Spot-check "phantom passive."** Three races, confirm that
  every passive fill had sufficient traded volume between
  placement and fill to cross the queue-ahead threshold. This
  is the sim-side analogue of R-1; it should never happen.

## Session exit criteria

- All 10 new tests pass.
- All existing tests pass (including session 25).
- `raw + shaped ≈ total_reward` invariant holds.
- `progress.md` Session 26 entry with at least one numeric
  example of a rest-then-fill sequence from the manual test.
- `ui_additions.md` row for visualising resting passive
  orders and fill events ticked / filed.
- `master_todo.md` Phase 2 P4 sub-tick for P4b.
- Commit.

## Do not

- Do not implement the cancel action. Session 29.
- Do not implement race-off cleanup. Session 27.
- Do not add partial fills. A passive order fills all-or-
  nothing in this session — partial fills are a future
  refinement that nobody has asked for yet.
- Do not change the fill logic for aggressive orders. Only
  the passive code path is new.
- Do not auto-cancel passive orders whose rest price drifts
  into junk territory. They stay open and fail to match each
  tick until either (a) the price drifts back, or (b)
  race-off cleanup in session 27 closes them out.
- Do not let the policy place passive orders yet. The only
  placement entry point is the test fixture. Session 28 is
  where the action space opens up.
