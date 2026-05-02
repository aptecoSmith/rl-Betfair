# Session prompt — Phase 4 Session 04: pre-allocate hidden-state capture buffer

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

The recurrent-PPO contract (see CLAUDE.md §"Recurrent PPO:
hidden-state protocol on update") requires capturing the hidden
state that was passed INTO each tick's forward pass, so the PPO
update can reproduce rollout-time log-probs:

```python
# (rollout.py lines 207-209)
hidden_in_t = tuple(
    t.detach().clone() for t in hidden_state
)
```

`hidden_state` for the LSTM family is `(h, c)`, each shape
`(num_layers=1, batch=1, hidden_size)`. So per tick, this code
allocates 2 tensors and does 2 `clone()` copies. At 12 k
ticks/episode that's 24 k tensor allocations + 24 k memcopies,
churning the allocator.

The clones are LOAD-BEARING — subsequent LSTM forwards mutate
the rolling hidden state in place, so we can't store a view.
But the per-tick allocation is unnecessary: pre-allocate ONE
`(n_steps_max, num_layers, batch, hidden_size)` buffer pair at
episode start, and write each tick's clone into a slice view.

This is the same pattern as Session 02's obs/mask buffer. The
trade-off is the same: one big upfront allocation vs many
small per-tick allocations.

End-of-session bar:

1. **CPU bit-identity preserved.** The PPO update's
   `_ppo_update` consumes the same numerical hidden states that
   it consumed pre-Session-04. Verify by capturing
   `pack_hidden_states(per_tick_hidden_in)` pre/post and
   asserting strict equality on a fixed-seed 1-day rollout.
2. **All pre-existing v2 tests pass on CPU.** Especially
   `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
   (the load-bearing recurrent-state-through-update guard from
   CLAUDE.md).
3. **CUDA self-parity test passes** (the existing
   `tests/test_v2_gpu_parity.py` trio).
4. **Allocation-count test passes** — per-tick clone count
   drops from 2 to 0 (the buffer slice copy replaces both).
5. **ms/tick measurement** vs Session 03's baseline.
6. **Verdict** GREEN / PARTIAL / FAIL.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `findings.md` (Sessions 01–03 rows).
2. CLAUDE.md §"Recurrent PPO: hidden-state protocol on update"
   — the load-bearing contract this session is preserving.
3. `training_v2/discrete_ppo/rollout.py` lines 130–212 — the
   `init_hidden` call, the per-tick clone, the way
   `hidden_state` is updated by the policy forward.
4. `agents_v2/discrete_policy.py::DiscreteLSTMPolicy.init_hidden`
   (line 246) and `pack_hidden_states` (line 253) — the
   architecture-specific shape and the consumer-side batching
   contract.
5. `training_v2/discrete_ppo/trainer.py::_ppo_update` — find
   where it calls `policy.pack_hidden_states(...)` on the per-
   transition hidden states. The shape it expects is the
   contract you must preserve.
6. `tests/test_ppo_trainer.py::TestRecurrentStateThroughPpoUpdate`
   — read all four tests. These are the load-bearing guards;
   any change you make must not break them.

## Implementation sketch

```python
def _collect(self) -> list[Transition]:
    ...
    # Pre-allocate a (n_steps_max, num_layers, batch, hidden) buffer
    # pair for the LSTM hidden state. Shape mirrors init_hidden's
    # output, with an extra leading dim for the per-tick slice.
    h0_shape = hidden_state[0].shape   # (num_layers, 1, hidden)
    c0_shape = hidden_state[1].shape
    n_steps_estimate = self._estimate_max_steps(env)

    h_arr = torch.empty(
        (n_steps_estimate, *h0_shape),
        dtype=hidden_state[0].dtype,
        device=hidden_state[0].device,
    )
    c_arr = torch.empty(
        (n_steps_estimate, *c0_shape),
        dtype=hidden_state[1].dtype,
        device=hidden_state[1].device,
    )

    while not done:
        if n_steps >= h_arr.shape[0]:
            # Once-per-episode growth path; same shape as Session 02.
            h_arr = torch.cat([h_arr, torch.empty_like(h_arr)], dim=0)
            c_arr = torch.cat([c_arr, torch.empty_like(c_arr)], dim=0)

        # Single copy into the buffer slice; preserves the
        # detach()-then-clone() semantic.
        h_arr[n_steps].copy_(hidden_state[0].detach())
        c_arr[n_steps].copy_(hidden_state[1].detach())

        # Reference into the buffer is the captured state.
        # Note: the SLICE is a view; subsequent LSTM forwards
        # mutate `hidden_state[0]` in place — but they don't
        # mutate `h_arr[n_steps]`, which is a separate memory
        # region (the .copy_() did the actual data transfer).
        hidden_in_t = (h_arr[n_steps], c_arr[n_steps])
        ...

    # End-of-episode: hand the trainer pre-stacked tensors
    # instead of a list-of-tuples for a future Session 06 win.
    # This session keeps the per-tick tuple in per_tick_hidden_in
    # so the consumer (Transition list comprehension) is unchanged.
    ...
```

⚠️ **The view-vs-copy semantics matter.** If the LSTM mutates
`hidden_state[0]` in place between ticks (it doesn't — it
returns a NEW `(h, c)` tuple from `nn.LSTM.forward`), then the
buffer slice would alias and corrupt earlier captures. Verify
this by reading `nn.LSTM.forward`'s return contract: a new tuple
is returned, the input is unchanged. So the `.copy_()` into
`h_arr[n_steps]` snapshots the value at THIS tick, and
`hidden_state` being reassigned to `out.new_hidden_state`
(line 212) doesn't touch the previous slice. Confirm with a
test.

## Tests to add

In `tests/test_v2_rollout_hidden_state_buffer.py` (new file):

1. `test_hidden_state_packed_bit_identical_to_pre_session_04`
   — capture `pack_hidden_states([t.hidden_state_in for t in
   transitions])` from a 1-day CPU rollout pre-change. Re-run
   post-change and assert byte-for-byte equality on each tensor
   in the packed tuple.

2. `test_hidden_state_buffer_allocated_once_per_episode` —
   patch `torch.empty` (or whatever the new code uses) and
   assert it's called at most twice per episode for the hidden
   state. Catches per-tick allocation re-introduction.

3. `test_hidden_state_slice_independent_of_subsequent_ticks`
   — synthesise a 5-tick rollout, capture `hidden_in_t` at
   tick 0, advance the rollout, and assert that the captured
   `hidden_in_t[0]` STILL equals what was passed to the policy
   at tick 0 (i.e. the buffer slice didn't alias the rolling
   hidden state).

4. `test_per_tick_clone_count_zero` — patch `torch.Tensor.clone`
   on the rollout module and count calls per episode. Should be
   0 after this session (was 2 × n_steps before).

5. `test_recurrent_ppo_kl_still_small` — run a real
   `_ppo_update` post-change and assert `approx_kl < 1.0` on
   the first epoch. This is the regression guard for the
   recurrent-state-through-PPO contract from
   `tests/test_ppo_trainer.py::test_ppo_update_approx_kl_small_
   on_first_epoch_lstm` — the same shape, just pinned to this
   session.

## Hard constraints

1. **Bit-identity on the packed hidden state**, not just per-
   tick. The PPO update consumes
   `pack_hidden_states(per_tick_hidden_in)` and the byte
   identity of THAT result is what the test must guard.
2. **The CUDA self-parity test from Phase 3 must keep passing.**
   Hidden-state buffer churn is a real CUDA cost; the test
   numerically validates that the post-fix path is consistent
   across CUDA runs, which is the load-bearing correctness
   foundation.
3. **Don't drop the `.detach()`.** It strips autograd; the
   clone (now `.copy_()` into a slice) is what duplicates the
   tensor. Both are load-bearing.
4. **Don't restructure `pack_hidden_states` / `slice_hidden_
   states`.** Those are policy-side helpers (Phase 1 contract);
   their signatures stay the same.

## Deliverables

- `training_v2/discrete_ppo/rollout.py` — pre-allocated hidden-
  state buffer pair with grow-on-overflow.
- `tests/test_v2_rollout_hidden_state_buffer.py` (new) with the
  five tests above.
- `findings.md` updated.
- Commit: `feat(rewrite): phase-4 S04 (GREEN|PARTIAL) - pre-
  allocate hidden-state capture buffer`.

## Estimate

~1.5 h. If past 2.5 h, stop — likely the LSTM-output-vs-input
aliasing is subtler than expected (e.g. on CUDA the in-place
LSTM kernel can mutate the input under certain conditions).
Document and decide.
