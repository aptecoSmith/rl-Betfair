# Session 27 — P4c: race-off cleanup for un-cleared passive orders

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — constraints 2, 4, 11.
- `../proposals.md` P4
- `../master_todo.md` Phase 2
- `../progress.md` — confirm sessions 25 and 26 have landed.
- `../lessons_learnt.md`
- `env/bet_manager.py`, `PassiveOrderBook` from sessions 25/26.
- `env/betfair_env.py::_settle_current_race` and whatever the
  env uses to detect "race has gone in-play" or "race is
  settling".

## Goal

When a race ends (goes in-play or the fixture's last pre-race
tick is reached), any passive orders still unfilled must be
cancelled cleanly:

- Budget reservation released.
- Order removed from the open passive list.
- `PnL = 0` for that order (it never matched, no exposure).
- Visible in the replay UI as "cancelled at race-off".

After this session, a race with any passive orders ends in a
fully-settled state with no leaked reservations and no
exceptions.

## Inputs — constraints to obey

1. **Zero P&L impact.** A cancelled-at-race-off passive order
   contributes exactly nothing to `day_pnl`, `race_pnl`, any
   shaped term, or any eval metric. It is as if it never
   existed from an accounting perspective — only the
   efficiency_penalty (if configured) charges for the
   placement attempt, because that friction is real (it's an
   API call in live).
2. **Deterministic cleanup order.** Cleanup runs once per
   race, at a well-defined point in the settlement path. No
   "maybe it fires during `step`, maybe during
   `_settle_current_race`" ambiguity. Pick one.
3. **Idempotent.** Running cleanup twice produces the same
   state as running it once. Test this.
4. **No hidden side-effects on next race.** After cleanup,
   the `PassiveOrderBook` is empty for the current race; the
   next race's fresh `BetManager` starts with its own empty
   book anyway (covered by session 25's test), but a test
   here also pins that the cleanup didn't mutate anything
   that persists across races.

## Steps

1. **Pick the cleanup hook point.** Two candidates:
    - **(A)** At the top of `_settle_current_race`, before
      race settlement runs.
    - **(B)** At the end of the race's last pre-race tick,
      as part of the `step` that transitions into settlement.
   **Recommended: (A).** It keeps the cleanup next to the
   race-settlement code, which is where the operator's
   mental model already expects "end of race" logic to live.
   Document the choice in `progress.md`.

2. **Add `PassiveOrderBook.cancel_all(reason)` method.** For
   each open passive order:
    - Release its budget reservation back to `BetManager.budget`.
    - Mark the order as cancelled (set `cancelled=True`,
      leave in a history list if useful for the replay UI,
      but remove from the "open" iterator).
    - Record a reason string ("race-off cleanup") for the
      replay UI.

3. **Call `cancel_all("race-off")` from the chosen hook
   point.**

4. **Emit cancellation events** in `info["passive_cancels"]`
   for the tick on which cleanup runs. Includes selection_id,
   price, requested_stake, reason.

5. **Efficiency penalty interaction.** Decide whether
   cancelled-at-race-off passive orders still count toward
   `efficiency_penalty × bet_count`. Arguments:
    - *Yes*: in live, placing the order cost an API call, so
      the friction should be recorded.
    - *No*: the bet never matched, so it shouldn't appear in
      the "bet count" at all.
   Pick one, document the choice in `progress.md` with
   reasoning, and pin it with a test. Recommended:
   **yes, it counts** — the friction is real and ignoring it
   would let passive-heavy policies look artificially
   efficient.

## Tests to add

Create `tests/research_driven/test_p4c_race_off_cleanup.py`:

1. **Unfilled passive is cancelled at race-off.** Place a
   passive that cannot fill given the fixture's traded volume.
   Let the race settle. Assert the order is no longer in the
   open list, budget is fully released, and no exception
   fires.
2. **Cancelled passive contributes zero P&L.** Same fixture;
   assert `day_pnl` for this race equals the P&L from
   non-passive bets only (compute the expected number, don't
   rely on "zero").
3. **Budget fully restored.** After cleanup, `available_budget`
   equals what it would be if the cancelled passive had
   never been placed.
4. **Idempotent cleanup.** Call `cancel_all` twice; assert
   the state after the second call equals the state after
   the first.
5. **Cleanup does not touch filled passives.** Place two
   passives; fill one mid-race, leave the other open. At
   race-off cleanup, the filled one is still in
   `BetManager.bets` contributing to P&L, the unfilled one
   is cancelled.
6. **Race reset isolation.** After cleanup for race A, start
   race B. Place a passive in B and let it fill normally;
   confirm nothing from race A's cleanup affected race B's
   state.
7. **Efficiency penalty interaction.** Whichever decision
   you took in step 5, pin it. A passive placed and
   cancelled at race-off contributes (or does not contribute)
   to `bet_count` for the efficiency-penalty term.
8. **Invariant.** `raw + shaped ≈ total` holds across all
   the above.
9. **Aggressive + passive mixed run.** A race with both
   kinds of bets settles correctly end-to-end.

All CPU, all fast.

## Manual tests

- **Open a race in the replay UI with a fixture where a
  passive order is known not to fill.** Confirm the replay
  UI shows the order as cancelled at race-off with a sensible
  reason string.
- **Check a race with partial coverage.** Two passives, one
  fills, one doesn't. Confirm the filled one shows as a
  normal bet and the unfilled one shows as cancelled.

## Session exit criteria

- All 9 new tests pass.
- All existing tests (including sessions 25 and 26) pass.
- `raw + shaped ≈ total_reward` invariant holds.
- Hook-point choice (A vs B) documented in `progress.md`.
- Efficiency-penalty interaction documented in `progress.md`
  with the reasoning.
- `ui_additions.md` row for cancellation visibility in the
  replay UI ticked.
- `master_todo.md` Phase 2 P4 sub-tick for P4c. P4 as a
  whole is complete at this point.
- Commit.

## Do not

- Do not implement the policy-driven cancel action in this
  session. Session 29.
- Do not change how filled passives settle — that's session
  26's code and it is done.
- Do not leak cleaned-up orders into next race's state. The
  "race reset isolation" test exists to pin this.
- Do not ignore the efficiency-penalty question. Pick an
  answer with reasoning; silently defaulting is how future
  sessions accidentally re-litigate decisions.
- Do not couple cleanup to real-time wall-clock. It's
  triggered by the *race's* end-of-pre-race state transition,
  not by any real timer.
