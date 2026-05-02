# Session prompt — Phase 4 Session 02: eliminate obs / mask double-copy

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

Every tick of the rollout, `obs` and `mask` are each
materialised TWICE:

```python
# (rollout.py lines 185-193)
obs_buffer.copy_(
    torch.from_numpy(np.asarray(obs, dtype=np.float32))
    .unsqueeze(0)
)
mask_np = shim.get_action_mask()
mask_buffer.copy_(
    torch.from_numpy(np.asarray(mask_np, dtype=bool))
    .unsqueeze(0)
)
```

```python
# (rollout.py lines 281-283)
per_tick_obs.append(np.asarray(obs, dtype=np.float32))
per_tick_mask.append(np.asarray(mask_np, dtype=bool))
```

Per tick: TWO `np.asarray` allocations, TWO `from_numpy` /
`unsqueeze` view constructions, TWO buffer copies (one to the
device buffer, one into the per-tick CPU list). At 12 k
ticks/episode this is 48 k unnecessary allocations and 24 k
unnecessary copies.

**Replace with a single pre-allocated `(n_steps_max, obs_dim)` /
`(n_steps_max, action_n)` numpy buffer per episode.** Each
tick writes obs and mask into a slice view ONCE; the device
buffer is filled from that slice via a single
`torch.from_numpy(slice).unsqueeze(0)` (which is a view, not a
copy, when the underlying buffer is contiguous).

The `n_steps_max` upper bound is knowable from the day's
`day.races` cumulative tick count (the env's episode loop is
deterministic given the day). If that's awkward to compute
ahead of time, allocate a generous initial buffer (e.g.
20 000 rows) and grow with `np.resize` if exceeded. The growth
path should fire once-per-episode at most (or never, in
practice).

End-of-session bar:

1. **CPU bit-identity preserved** vs Session 01's output (i.e.
   the rollout collector returns transitions whose `obs` and
   `mask` arrays are byte-equal to pre-Session-02).
2. **Allocation-count test passes** (see Tests below) — the
   per-tick `np.asarray` calls drop from 4-per-tick to ≤ 1.
3. **All pre-existing v2 tests pass on CPU.**
4. **ms/tick measurement** vs Session 01's baseline; logged in
   `findings.md`.
5. **Verdict** logged as one of GREEN / PARTIAL / FAIL per the
   same shape as Session 01.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `plans/rewrite/phase-4-restore-speed/findings.md` — phase
   contract and Session 01's measured baseline (so you know the
   numerator for "ms/tick before this session").
2. `training_v2/discrete_ppo/rollout.py` lines 139–293 — the
   `_collect` body. Focus on:
   - The pre-allocated `obs_buffer` and `mask_buffer`
     (lines 146–151).
   - The two `per_tick_*.append(np.asarray(...))` calls
     (lines 281–283).
   - How `Transition.obs` and `Transition.mask` are consumed in
     the end-of-episode list comprehension (lines 320–334).
3. `training_v2/discrete_ppo/transition.py` — confirm `obs` and
   `mask` types are `np.ndarray`. The PPO update path
   (`trainer.py::_ppo_update`) stacks these into batched torch
   tensors; that consumer-side stack is what your buffer must
   feed into without a per-transition copy.
4. `agents_v2/env_shim.py::get_action_mask` — does it return a
   fresh array per call, or a view into env state? If it returns
   the same object every call, copying once into your per-step
   buffer is straightforward. If it returns a fresh allocation,
   the copy IS the materialisation — the question is whether to
   copy into the buffer or use the returned array directly via a
   `__setitem__` slice write.

## Implementation sketch

```python
def _collect(self) -> list[Transition]:
    ...
    # Estimate upper-bound n_steps from the day's race tick
    # counts; fall back to a generous default + grow on overflow.
    n_steps_estimate = self._estimate_max_steps(env)
    obs_arr = np.empty((n_steps_estimate, obs_dim), dtype=np.float32)
    mask_arr = np.empty((n_steps_estimate, action_n), dtype=bool)
    n_steps = 0

    while not done:
        if n_steps >= obs_arr.shape[0]:
            # Once-per-episode growth path; logged as a warning
            # so the estimate can be tuned if it fires.
            obs_arr = np.resize(obs_arr, (obs_arr.shape[0] * 2, obs_dim))
            mask_arr = np.resize(mask_arr, (mask_arr.shape[0] * 2, action_n))

        # Single materialisation per tick.
        obs_arr[n_steps] = obs
        mask_np = shim.get_action_mask()
        mask_arr[n_steps] = mask_np

        # Device buffer copy from the slice (one copy total per tick).
        obs_buffer.copy_(
            torch.from_numpy(obs_arr[n_steps]).unsqueeze(0)
        )
        mask_buffer.copy_(
            torch.from_numpy(mask_arr[n_steps]).unsqueeze(0)
        )
        ...
```

The Transition build at end-of-episode then reads
`obs=obs_arr[i]` (a view into the contiguous buffer) instead of
the per-tick `np.asarray` copy.

⚠️ **View vs copy semantics.** The rows of a contiguous numpy
buffer are views, not copies. Stashing a view into a Transition
that the PPO update mutates would alias across all transitions.
Verify the PPO update never mutates `Transition.obs` /
`Transition.mask` in place; if it does, take a `.copy()` at
Transition build time (still one copy total, end-of-episode,
not per-tick).

## Tests to add

In `tests/test_v2_rollout_buffer_reuse.py` (new file):

1. `test_obs_mask_buffers_bit_identical_to_pre_session_02_on_fixed_seed`
   — capture the full `[Transition.obs for t in transitions]`
   stack from a 1-day CPU rollout pre-change. Re-run post-change
   and assert byte-for-byte equality on every row. Same for
   `mask`. Use `np.testing.assert_array_equal`.

2. `test_obs_buffer_allocated_once_per_episode` — patch
   `np.empty` (or whatever the new code uses for the buffer
   allocation) and assert it's called at most twice per episode
   (once for obs, once for mask). Catches a future regression
   where someone re-introduces per-tick allocation.

3. `test_mask_buffer_allocated_once_per_episode` — symmetric.

4. `test_buffer_grow_path_warns_and_continues` — synthesise an
   episode that exceeds the estimate (or shrink the initial
   estimate via test injection) and verify the resize path fires
   and produces the same final transitions.

5. `test_transition_obs_not_aliased_after_buffer_grow` — after
   a buffer grow, the transitions stored from before the grow
   must still hold valid data (the grow can re-alloc; views
   into the old buffer would be invalid).

## Hard constraints

In addition to all Phase 4 hard constraints:

1. **Bit-identity is the load-bearing correctness guard.** Same
   strict-equality bar as Session 01.
2. **Don't change the Transition shape.** `obs` stays
   `np.ndarray`; `mask` stays `np.ndarray`. Session 06 owns the
   broader Transition restructure; this session just stops
   double-copying.
3. **Don't touch `env_shim.get_action_mask`.** If it returns a
   stale view that breaks the bit-identity test, document and
   stop — env_shim edits are explicitly out of scope per
   `purpose.md` §"Hard constraints" §9.
4. **Buffer grow path must work.** Even if it never fires in
   practice, the test for it is mandatory — silent
   index-out-of-range is the worst kind of regression.

## Deliverables

- `training_v2/discrete_ppo/rollout.py` — single-allocation
  obs / mask buffers with grow-on-overflow.
- `tests/test_v2_rollout_buffer_reuse.py` (new) with the five
  tests above.
- `findings.md` updated with this session's row.
- Commit: `feat(rewrite): phase-4 S02 (GREEN|PARTIAL) -
  single-allocation obs/mask rollout buffers`.

## Estimate

~1.5 h. If past 2.5 h, stop — likely the view-vs-copy semantics
in the PPO consumer are subtler than expected; document and
decide.
