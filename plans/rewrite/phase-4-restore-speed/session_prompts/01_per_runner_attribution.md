# Session prompt — Phase 4 Session 01: per-runner attribution incremental tracking

Use this prompt to open a new session in a fresh context. Self-
contained — does not require context from the session that
scaffolded it.

---

## The task

`RolloutCollector._attribute_step_reward`
(`training_v2/discrete_ppo/rollout.py` lines 346–426) walks
**every bet ever placed in the episode, every tick**:

```python
all_bets = list(env.all_settled_bets) + list(live_bets)
for bet in all_bets:
    bet_id = id(bet)
    prev_pnl = prev_pnl_by_id.get(bet_id, 0.0)
    cur_pnl = float(bet.pnl)
    delta = cur_pnl - prev_pnl
    if delta == 0.0:
        prev_pnl_by_id[bet_id] = cur_pnl
        continue
    ...
```

This is **O(bets-so-far) per tick** = **O(n²) per episode**. By
tick 11 000 of a 12 k-tick day the loop touches hundreds-to-
thousands of bets, 99 % of which haven't changed since last
tick. It also rebuilds two Python lists every tick
(`list(...) + list(...)`) and does a dict lookup per bet.

This is a v2 feature (per-runner attribution didn't exist in v1)
and the **most likely single explanation for v2's 9.6 ms/tick
vs v1's 2.94 ms/tick**.

**Replace with incremental tracking** — maintain a "pending" set
of bet objects whose `pnl` may still mutate. A bet enters when
first observed (typically pnl == 0); leaves once its pnl is
final (post-settle, no further mutation can produce a non-zero
delta). Most ticks scan zero bets. Same per-runner attribution,
same invariant, same numbers.

End-of-session bar:

1. **CPU bit-identity preserved.** Pre-/post-change rollout on a
   fixed seed and fixed day produces bit-identical
   `per_runner_reward` arrays for every tick. Strict equality.
2. **Pending-set invariant tested.** A no-bet tick scans zero
   bet objects (regression guard for the O(n²) failure mode
   re-emerging).
3. **Attribution invariant still holds.** The
   `np.isclose(total, step_reward, ...)` assert in the modified
   `_attribute_step_reward` continues to pass on every tick —
   the algebra of the attribution is unchanged.
4. **All pre-existing v2 tests pass on CPU.**
5. **ms/tick measurement** on a 1-day single-episode CPU run
   (same day as Phase 3 Session 01b: 2026-04-23 from
   `data/processed_amber_v2_window/`). Logged in
   `findings.md` (create the file with this session's row).
6. **Verdict** logged as one of:
   - **GREEN**: ms/tick drops by ≥ 30 % AND bit-identity
     preserved AND tests pass.
   - **PARTIAL**: bit-identity preserved AND tests pass but
     speedup < 30 % — surprising; document and proceed to
     Session 02 (the work landed correctly, just smaller win
     than expected).
   - **FAIL**: bit-identity broken or invariant assertion fires.
     Stop and triage.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` — phase goal,
   per-session contract, hard constraints.
2. `training_v2/discrete_ppo/rollout.py` — read all of
   `RolloutCollector` (lines 59–426). Pay special attention to:
   - The `prev_pnl_by_id` dict (line 137).
   - `all_bets` construction at line 382.
   - The `attributed_total += delta` loop body.
   - The invariant assert at lines 412–424.
3. `env/bet_manager.py` — read enough to understand the `Bet`
   lifecycle:
   - When does `bet.pnl` change? (Settlement is the typical
     mutation; check whether MTM or partial-fill paths also
     mutate `bet.pnl`.)
   - Does any code path mutate `bet.pnl` AFTER the race the bet
     belongs to has settled? (If yes, the "remove from pending
     post-settle" rule needs adjusting.)
4. `env/betfair_env.py::_settle_current_race` — the per-race
   settle path that finalises bet pnl. Understand which fields
   on `Bet` are written here and whether they're written more
   than once.
5. `tests/test_v2_rollout_per_runner_attribution.py` (or the
   nearest existing test for the attribution path — search for
   `_attribute_step_reward` callers in `tests/`). The new tests
   should sit alongside or extend the existing ones.

## Implementation sketch (one of several valid shapes)

```python
class RolloutCollector:
    def _collect(self) -> list[Transition]:
        ...
        # Pending-pnl set: bets whose pnl may still mutate.
        # Bet objects are hashable by id; use a dict keyed by
        # id(bet) → bet to keep references alive (otherwise gc
        # may collect unreferenced bet objects between ticks,
        # breaking id-stability — though in practice the env
        # holds them in all_settled_bets / bm.bets).
        pending_bets: dict[int, "Bet"] = {}
        prev_pnl_by_id: dict[int, float] = {}
        seen_bet_ids: set[int] = set()  # all bets ever observed

        # Per-tick: instead of walking all_bets, walk the
        # symmetric difference (new bets that just appeared,
        # plus the pending set whose pnl may have changed).
        ...
```

The ENTRY rule: scan `env.all_settled_bets[len(seen):]` and
`env.bet_manager.bets` (current race's live bets) for *new*
bet objects each tick; add to `pending_bets`. This list-slice
is O(new_bets_this_tick), typically 0–3 per tick.

The EXIT rule: a bet leaves `pending_bets` once its pnl is
final. Two practical signals (verify against `bet_manager.py`
in step 3 above):

- The bet is in `env.all_settled_bets` AND its `outcome` field
  is not `UNSETTLED` AND the current `bet.pnl` matches the last
  recorded `prev_pnl_by_id[id(bet)]` (delta = 0 implies stable).
- OR the bet's race has fully settled (the env's `_race_idx`
  has advanced past the bet's market) AND outcome != UNSETTLED.

Pick one signal that's verifiable from inside the collector
without env edits. Document the choice in the file's docstring.

The PER-TICK loop: scan `pending_bets.values()` only, compute
deltas, update `prev_pnl_by_id`, mark stable bets for removal,
remove after the loop.

## Tests to add

In `tests/test_v2_rollout_per_runner_attribution.py` (create
if absent; otherwise extend):

1. `test_attribution_bit_identical_to_pre_session_01_on_fixed_seed`
   — capture per-tick `per_runner_reward` arrays from a 1-day
   CPU rollout pre-change (seed 42, 2026-04-23). Re-run post-
   change and assert byte-for-byte equality on every array.
   Use `np.testing.assert_array_equal` (NOT `assert_allclose` —
   the algebra is unchanged; bits should match exactly).

2. `test_pending_set_scans_zero_on_no_bet_tick` — patch the
   collector to count `len(pending_bets)` at the entry of
   `_attribute_step_reward`. Assert that for a tick where no
   new bets appeared and no pnl changed, the iteration count is
   zero. Catches a future regression where someone puts the
   "all bets" walk back in.

3. `test_attribution_invariant_assert_still_holds` — exercise
   a settle-step (highest-mutation tick of the episode) and
   assert the `np.isclose(total, step_reward)` invariant
   passes. Same shape as Session 01's existing assert; just
   pinned for regression.

4. `test_pending_set_size_bounded_across_episode` — assert
   `len(pending_bets)` never exceeds (e.g.) 50 across a full
   1-day rollout — catches a memory leak where bets are added
   but never removed.

## Hard constraints

In addition to all Phase 4 hard constraints
(`plans/rewrite/phase-4-restore-speed/purpose.md`
§"Hard constraints"):

1. **Bit-identity is the load-bearing correctness guard.** If
   the post-change attribution differs from pre-change by even
   1 bit on any tick, the change is wrong. Reformulate, don't
   loosen the test.
2. **No env edits.** The pending-set heuristic for "bet pnl is
   final" must be observable from inside `_attribute_step_reward`
   alone, using only fields already on `Bet` and on the env.
3. **Don't drop the invariant assert.** Even if it's slow.
   Session 05 will deal with the assert's per-tick cost; this
   session keeps it as-is to verify the new code path is correct.
4. **No restructuring of the surrounding rollout loop.** This
   session changes only `_attribute_step_reward` and its inputs
   (the pending-set bookkeeping in `_collect`). Sessions 02–06
   own the other inefficiencies; bundling them here makes the
   bit-identity test ambiguous.

## Deliverables

- `training_v2/discrete_ppo/rollout.py` — incremental
  attribution implementation.
- `tests/test_v2_rollout_per_runner_attribution.py` (new or
  extended) with the four tests above.
- `plans/rewrite/phase-4-restore-speed/findings.md` (new file)
  with the cumulative-ms/tick table from `purpose.md` §
  "Session 99" populated for Session 01.
- A commit per the per-session contract:
  `feat(rewrite): phase-4 S01 (GREEN|PARTIAL) - incremental per-
  runner attribution` with the cumulative ms/tick in the body.

## Estimate

~2.5 h. If past 3.5 h, stop — the pending-set invariant is
trickier than expected; document the surprise in findings.md
and decide whether to push to next session.
