# Session 28 — P3a: aggression flag in action space

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — **constraint 9 (cancel ships with
  passive orders or not at all)** means session 28 cannot land
  without session 29 in the same branch or immediately after.
  These two sessions are bundled by decision, even though
  they're split into two session files for reviewability.
  Also constraints 13 (schema bumps refuse old checkpoints) and
  7 (matcher stays vendorable).
- `../proposals.md` P3
- `../analysis.md` §1 and §2
- `../master_todo.md` Phase 2
- `../open_questions.md` Q1 and Q2 — confirm Q1 is B/B-lite
  and Q2 permits a fresh re-train (checkpoint invalidation is
  inevitable here).
- `../progress.md` — confirm sessions 25/26/27 (P4) have landed.
  **P4 is a prerequisite for P3** — the matcher needs
  somewhere to put passive orders before the action space can
  emit them.
- `../design_decisions.md`
- `env/betfair_env.py::_process_action`

## Goal

Extend the per-slot action vector with an **aggression flag**
that the policy uses to choose between "cross the spread"
(existing aggressive path) and "join the queue at my-side
best" (the passive path built in P4).

After this session, the policy can emit passive orders, they
rest on the `PassiveOrderBook` from session 25/26, and they
fill or get cleaned up exactly as the unit tests in sessions
26/27 established. The only new thing is the action-space
plumbing.

Cancel is **not** part of this session — see session 29. But
session 28 cannot be released on its own; sessions 28 and 29
ship together.

## Inputs — constraints to obey

1. **Fresh re-train is expected.** Action-space change
   invalidates all existing checkpoints. The loader refuses
   them loudly. Document this in `progress.md` and note that
   all Phase 1 comparison data (from session 22) is still
   valid as a baseline, it just won't be the starting
   weights.
2. **Aggressive behaviour must be reproducible.** If the
   aggression flag is forced to "always cross" (via a config
   override), the policy's behaviour must be byte-identical
   to the pre-session-28 aggressive-only policy. This is the
   regression backstop.
3. **Action design is discrete, not continuous.** Options
   considered in `proposals.md` P3 included
   `aggression ∈ [0, 1]` and an explicit `limit_price_offset`.
   **Use a discrete passive/aggressive flag.** Two values.
   Reasoning: a continuous aggression is much harder for the
   policy to explore, because the interior of `[0, 1]` has
   no meaningful interpretation until queue-offset pricing
   is added (which we deliberately parked as out of scope).
   A discrete flag has a clean semantic: 0 = passive, 1 =
   aggressive. Document this choice in `design_decisions.md`.
4. **Per-slot, not global.** The flag is per-runner-slot so
   the policy can be aggressive on some selections and
   passive on others within the same tick.
5. **Matcher stays stateless and vendorable.** The new action
   verb dispatches to `PassiveOrderBook.place` (for passive)
   or `BetManager.place_back`/`place_lay` (for aggressive),
   both of which already exist. The matcher itself gains
   nothing.

## Steps

1. **Add a design-decisions entry** recording the discrete-
   vs-continuous choice with reasoning. Commit before code.

2. **Bump the action-space definition** in
   `env/betfair_env.py`. The new per-slot action shape is
   `[signal, stake, aggression]` — or whatever the existing
   action-space structure dictates. Check how the existing
   `[signal, stake]` is assembled and extend consistently.

3. **Bump the action schema version.** Loader refuses
   pre-P3 checkpoints with a clear error. Same pattern as
   sessions 19–21 but on the action side.

4. **Dispatch in `_process_action`.** Read the aggression
   slot value. If above the threshold, call the aggressive
   path (unchanged). If below, call the passive path (new
   dispatch to `PassiveOrderBook.place`).

5. **Config override for "always cross".** Add
   `actions.force_aggressive: false` (default false). When
   true, the dispatch ignores the aggression flag and always
   uses the aggressive path. This is the regression
   backstop from constraint 2 — used by tests and by
   operators who want to reproduce the pre-P3 policy.

6. **Expose the decision in `info["action_debug"]`.** For
   each slot, a dict `{passive_placed, aggressive_placed,
   skipped_reason}`. Used by the manual test.

## Tests to add

Create `tests/research_driven/test_p3a_aggression_flag.py`:

1. **Aggressive dispatch reproduces pre-P3.** With
   `actions.force_aggressive=true`, run a fixture race and
   assert the resulting `BetManager.bets` and `day_pnl` are
   byte-identical to a reference recorded before the session
   started.
2. **Passive dispatch routes to `PassiveOrderBook`.** Emit a
   policy action with `aggression=0` for one slot; assert a
   `PassiveOrder` appears in `passive_book` and no new entry
   appears in `bets`.
3. **Mixed per-slot dispatch.** In one tick, slot 0
   aggressive, slot 1 passive, slot 2 no-bet. All three are
   routed correctly.
4. **Schema-bump loader refuses pre-P3 checkpoints.**
   Including P1/P2 checkpoints — any checkpoint pre-dating
   the action schema bump is refused loudly.
5. **Force-aggressive + passive signal doesn't crash.**
   With `force_aggressive=true`, emit an aggression=0
   signal; the dispatch cleanly overrides to aggressive and
   places a normal aggressive bet.
6. **Aggressive regression suite.** All matcher, bet-
   manager, and session-25/26/27 passive tests still pass.

All CPU, all fast.

## Manual tests

- **Deferred until session 29** — the manual test "watch a
  race with a P3 policy using both regimes" only makes sense
  once cancel exists, because without cancel the policy has
  only two of the three research-driven action verbs. Note
  this deferral in `progress.md`.

## Session exit criteria

- All 6 new tests pass.
- All existing tests pass, especially the passive tests from
  sessions 25–27.
- `design_decisions.md` entry for the discrete-vs-continuous
  choice.
- `progress.md` Session 28 entry including the note that
  sessions 28 and 29 must ship as a pair — this session
  alone is not a valid production release.
- `master_todo.md` Phase 2 P3 sub-tick for P3a.
- **Do not mark P3 complete yet** — P3 completes at the end
  of session 29.
- Commit.

## Do not

- Do not implement the cancel action. Session 29. Placing
  it here would make the review unwieldy and would also
  delay the "can the policy use both regimes" manual test
  until *after* cancel, which has the same net effect but a
  less reviewable diff.
- Do not let sessions 28 and 29 drift apart in time.
  Whoever runs session 28 should also run session 29 next,
  or explicitly flag that they're handing off mid-bundle.
- Do not start the retrain in this session. Session 30.
- Do not try to preserve existing checkpoints by padding
  the new action dim with a default. The loader refuses
  them loudly; a fresh re-train is the deliberate cost.
- Do not add a continuous-aggression interpretation "just in
  case". The parking in `not_doing.md` (or equivalent) is
  deliberate. Revisit only when a concrete eval result
  justifies it.
