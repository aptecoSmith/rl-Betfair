# Session 29 — P3b: cancel action

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 9 (cancel ships with
  passive orders or not at all)** is the whole reason this
  session exists in the same branch as session 28. Also 10
  (no `modify` action).
- `../proposals.md` P3
- `../master_todo.md` Phase 2
- `../progress.md` — confirm session 28 has landed. **Session
  29 immediately follows session 28 with no intervening
  session** unless there's an explicit hand-off note.
- `../design_decisions.md`
- `../not_doing.md` ND-1 (no `modify` action)
- `env/betfair_env.py`, `PassiveOrderBook`.

## Goal

Add a **cancel** action verb so the policy can withdraw a
resting passive order on a later tick. After this session, the
three-way research-driven decision (join / cross / cancel) is
fully available to the policy, sessions 28 + 29 are complete,
and session 30 can run the fresh re-train.

## Inputs — constraints to obey

1. **Cancel is its own action verb.** It is NOT a special
   value of the existing aggression flag. Choosing to cancel
   is semantically different from choosing to place, and
   mixing them would create exploration pathologies (the
   policy would have to emit a "cancel" signal alongside a
   "place" signal in the same slot, which is confusing).
2. **Cancel releases the budget reservation.** Same code
   path as race-off cleanup from session 27. Reuse the
   existing `cancel_all(reason)` mechanism where possible.
3. **Cancel is policy-targetable.** The policy must be able
   to say *which* resting order to cancel, not just "cancel
   everything". Simplest useful version: "cancel the oldest
   open passive order on this runner" — a per-slot cancel
   flag. No per-order IDs in the action space (that would
   explode the action dim).
4. **No `modify` action.** Per `not_doing.md` ND-1. Price
   moves = cancel + new place. Reject any review comment
   asking to add `modify`.
5. **Cancel idempotency.** Cancelling a runner with no
   resting passive orders is a no-op, not an error. The
   policy will often emit cancel signals that don't
   correspond to anything; treating them as errors would
   poison training.

## Steps

1. **Extend the per-slot action.** The new action shape is
   `[signal, stake, aggression, cancel]` — or however the
   existing structure encodes flags. `cancel` is a boolean
   threshold on the output, same pattern as `signal` uses
   `_BACK_THRESHOLD` / `_LAY_THRESHOLD`.

2. **Bump the action schema version again.** Or — if this
   session lands in the same release as session 28 — do one
   bump for both sessions and document in `progress.md` that
   the bump is the combined 28+29 change. Either way, the
   loader refuses pre-P3 checkpoints loudly.

3. **Add `PassiveOrderBook.cancel_oldest_for(
   selection_id, reason="policy cancel")`.** Finds the
   oldest open passive order on that selection, cancels it
   via the same internal path as `cancel_all`, returns the
   cancelled order (or None if nothing was open).

4. **Dispatch in `_process_action`.** Read the cancel slot.
   If above threshold, call `cancel_oldest_for(sid,
   "policy cancel")`. Idempotent: if nothing was cancelled,
   continue.

5. **Cancel + place in the same tick.** A slot emitting
   both "cancel" and "place" in the same tick runs cancel
   *first*, then place. This gives the policy an atomic
   "move this rest to a new price" behaviour without needing
   a `modify` verb. Test this sequencing.

6. **Expose cancel events** in `info["action_debug"]` per
   slot (alongside the passive_placed / aggressive_placed
   fields from session 28).

## Tests to add

Create `tests/research_driven/test_p3b_cancel.py`:

1. **Cancel of a resting order releases budget.** Place a
   passive; on a later tick, emit cancel; assert the order
   is no longer in the open list, budget is fully restored,
   no exception.
2. **Cancel with nothing to cancel is a no-op.** Emit
   cancel on a slot whose selection has no resting passive;
   no exception, no state change.
3. **Cancel oldest.** Place two passives at different prices
   on the same selection; cancel; assert the *older* one
   is cancelled and the newer one is still resting.
4. **Cancel + place in same tick.** Place passive at P1; on
   a later tick, cancel and place at P2; assert the resulting
   state has one passive at P2 and none at P1, and the
   budget is consistent (not double-reserved).
5. **Cancelled passive contributes zero P&L.** Same as race-
   off cleanup test — a policy-cancelled passive settles
   with zero P&L.
6. **Efficiency-penalty interaction.** Whichever decision
   session 27 took for race-off cancels, apply the same to
   policy cancels. Pin with a test.
7. **Cancel does not affect aggressive bets.** A race with
   both aggressive bets and cancel signals — the aggressive
   bets are unaffected by cancel dispatch.
8. **Schema-bump loader refuses pre-P3b.** If session 29
   does its own bump (rather than sharing with session 28),
   the loader refuses session-28-only checkpoints.
9. **Invariant.** `raw + shaped ≈ total` holds.

All CPU, all fast.

## Manual tests (this is the deferred manual test from
session 28)

- **Watch one race in the replay UI with a P3-emitting
  policy.** Confirm the operator can visually distinguish:
  - Passive orders resting on the book before filling.
  - Aggressive orders filling instantly.
  - Passive orders the policy later cancels.
- **Spot-check one cancel event.** Pick a cancelled passive
  order; confirm the event shows in the bet log as a
  distinct row (not a silent disappearance), and the
  reason string is "policy cancel" (not "race-off").

## Session exit criteria

- All 9 new tests pass.
- Manual tests passed and noted in `progress.md`.
- All existing tests (28 and earlier) pass.
- `raw + shaped ≈ total_reward` invariant holds.
- `progress.md` Session 29 entry — and explicitly marks the
  28+29 bundle as complete.
- `ui_additions.md` row for cancel events in the bet log
  ticked.
- `master_todo.md` Phase 2 P3 sub-tick for P3b. P3 is
  complete at the end of this session (both sub-bullets
  ticked); the P3 parent box may be ticked.
- **Do not mark the Phase 2 decision gate complete.** That's
  session 30.
- Commit.

## Do not

- Do not add a `modify` action. It stays parked (ND-1).
  Reject any reviewer comment asking to add it; cite
  `design_decisions.md` and `not_doing.md`.
- Do not make cancel require an explicit order ID in the
  action space. "Cancel oldest on this runner" is the
  contract.
- Do not treat cancel-of-nothing as an error. The policy
  will often emit spurious cancels; that's fine.
- Do not start the retrain in this session. Session 30.
- Do not ship session 29 in a release that does not also
  include session 28. They are bundled.
- Do not use cancel internally for race-off cleanup
  implementation (session 27's code). They should share
  the underlying cancellation helper but have *separate*
  entry points, because the operator needs to distinguish
  the two reasons in the replay UI.
