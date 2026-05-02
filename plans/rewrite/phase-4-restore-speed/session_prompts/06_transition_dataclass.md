# Session prompt — Phase 4 Session 06: replace Transition list with aligned RolloutBatch

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

End-of-episode in `_collect`, the rollout loop builds 12 k
frozen `Transition` dataclass instances:

```python
# (rollout.py lines 320-334)
transitions: list[Transition] = [
    Transition(
        obs=per_tick_obs[i],
        hidden_state_in=per_tick_hidden_in[i],
        mask=per_tick_mask[i],
        action_idx=per_tick_action_idx[i],
        stake_unit=per_tick_stake_unit[i],
        log_prob_action=float(log_prob_action_arr[i]),
        log_prob_stake=float(log_prob_stake_arr[i]),
        value_per_runner=value_per_runner_arr[i],
        per_runner_reward=per_tick_per_runner_reward[i],
        done=per_tick_done[i],
    )
    for i in range(n_steps)
]
```

12 k dataclass instantiations + 24 k `float()` conversions on
log-prob entries. The PPO update then immediately stacks these
back into aligned tensor batches via
`torch.stack([t.value_per_runner for t in transitions])` (and
similar). The dataclass round-trip serves no PPO-update
purpose — it's purely an organisational artefact.

**Replace with `RolloutBatch`** — a namedtuple of pre-stacked
tensors / arrays returned directly from `_collect`. The trainer's
`_ppo_update` consumes the batch directly, skipping the per-
transition stacking.

This is the larger change in Phase 4 — it touches the trainer
(`_ppo_update` consumer-side) plus the rollout collector. Two
correctness bars:

1. The PPO update consumes the same numerical inputs (just in
   a different shape).
2. Sessions 02 and 04's pre-allocated buffers slot directly
   into `RolloutBatch` fields without the end-of-episode list-
   to-tensor stacking.

End-of-session bar:

1. **PPO update output bit-identical.** Run a 1-day single-
   episode rollout pre/post change and assert that the post-
   `_ppo_update` policy state-dict is byte-identical to the
   pre-change run. This is the strictest end-to-end test of
   the change.
2. **All pre-existing v2 trainer / rollout / collector / cohort
   tests pass on CPU.** The trainer's tests need updating to
   consume `RolloutBatch`; the existing tests should be the
   acceptance check that the consumer-side change is correct.
3. **CUDA self-parity test passes** (the existing
   `tests/test_v2_gpu_parity.py` trio).
4. **`Transition` import-count test** — assert that
   `_ppo_update` no longer imports / constructs `Transition`
   instances on the hot path. Catches a partial refactor.
5. **ms/tick measurement** vs Session 05's baseline.
6. **Verdict** GREEN / PARTIAL / FAIL.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `findings.md`.
2. `training_v2/discrete_ppo/transition.py` — current
   `Transition` definition, fields, and the `action_uses_stake`
   helper.
3. `training_v2/discrete_ppo/rollout.py` lines 167–335 — the
   per-tick CPU lists and the end-of-episode dataclass build.
4. `training_v2/discrete_ppo/trainer.py::_ppo_update` — read
   end-to-end. Find every site that:
   - Reads a per-transition field (e.g.
     `t.log_prob_action for t in transitions`).
   - Stacks transitions into batched tensors.
   - Iterates the transition list.
   The `RolloutBatch` shape must serve all these consumers.
5. `training_v2/discrete_ppo/trainer.py::_compute_advantages`
   and `_bootstrap_value` — these read `transitions[-1]`'s
   hidden state and possibly other terminal fields. Verify
   `RolloutBatch` exposes them.
6. `training_v2/discrete_ppo/batched_rollout.py` — Session 02
   of throughput-fix introduced a parallel batched collector;
   confirm it returns transitions in the same shape, and decide
   whether `RolloutBatch` lands in both collectors or just the
   sequential one. (Recommendation: just the sequential one for
   this session; `BatchedRolloutCollector` is opt-in and rarely
   used today.)
7. `tests/test_discrete_ppo_trainer.py` and
   `tests/test_discrete_ppo_rollout.py` — the consumer-side
   tests. Many will need updating to construct `RolloutBatch`
   instead of `Transition` lists.

## Implementation sketch

In `training_v2/discrete_ppo/transition.py`:

```python
from typing import NamedTuple

class RolloutBatch(NamedTuple):
    """Aligned per-tick rollout outputs, ready for PPO update."""
    obs: np.ndarray              # (n_steps, obs_dim)
    hidden_state_in: tuple[torch.Tensor, ...]
    # For LSTM: ((n_steps, num_layers, 1, hidden), same for c)
    mask: np.ndarray             # (n_steps, action_n) bool
    action_idx: np.ndarray       # (n_steps,) int
    stake_unit: np.ndarray       # (n_steps,) float
    log_prob_action: np.ndarray  # (n_steps,) float
    log_prob_stake: np.ndarray   # (n_steps,) float
    value_per_runner: np.ndarray # (n_steps, max_runners)
    per_runner_reward: np.ndarray  # (n_steps, max_runners)
    done: np.ndarray             # (n_steps,) bool
    n_steps: int

# Keep Transition around for tests / backward-compat that
# explicitly want it; mark with deprecation notice.
@dataclass(frozen=True)
class Transition:
    ...
```

In `training_v2/discrete_ppo/rollout.py::_collect`:

```python
def _collect(self) -> RolloutBatch:
    ...
    return RolloutBatch(
        obs=obs_arr[:n_steps],   # view from Session 02 buffer
        hidden_state_in=(h_arr[:n_steps], c_arr[:n_steps]),  # Session 04
        mask=mask_arr[:n_steps], # view from Session 02 buffer
        action_idx=np.asarray(per_tick_action_idx, dtype=np.int64),
        stake_unit=np.asarray(per_tick_stake_unit, dtype=np.float32),
        log_prob_action=log_prob_action_arr,
        log_prob_stake=log_prob_stake_arr,
        value_per_runner=value_per_runner_arr,
        per_runner_reward=np.stack(per_tick_per_runner_reward, axis=0),
        done=np.asarray(per_tick_done, dtype=bool),
        n_steps=n_steps,
    )
```

In `trainer.py::_ppo_update`, replace
`torch.stack([t.field for t in transitions])` with
`torch.from_numpy(rollout_batch.field)` (or equivalent move-to-
device call). The hidden-state pack call already operates on a
per-transition tuple; with `RolloutBatch` storing the pre-
stacked `(n_steps, ...)` tensor, you can skip
`pack_hidden_states` entirely when n_agents == 1 (which is
always true for sequential rollouts). For the batched collector,
keep the current per-transition pack.

⚠️ **Public API.** `collect_episode` returning `RolloutBatch`
instead of `list[Transition]` is a public API change.
Downstream consumers in `training_v2/cohort/*` need updating.
Audit all callers of `collect_episode` and update each.

## Tests to add

In `tests/test_v2_rollout_batch.py` (new file):

1. `test_rollout_batch_fields_match_transition_list_pre_session_06`
   — capture the full transition list pre-change. Re-run
   post-change collecting `RolloutBatch`. For each field, assert
   `np.array_equal(batch.field, np.asarray([t.field for t in
   transitions]))`. Strict equality.

2. `test_ppo_update_state_dict_byte_identical_to_pre_session_06`
   — run a 1-day single-episode rollout + 1 PPO update on a
   fresh-init policy at fixed seed. Capture
   `policy.state_dict()` post-update. Compare pre/post: every
   tensor must be byte-equal. This is the load-bearing test for
   this session.

3. `test_transition_dataclass_not_constructed_on_hot_path` —
   patch `Transition.__init__` and run a 1-day rollout. Assert
   call count is 0 (or whatever the new sequential path
   reports).

4. `test_rollout_batch_views_remain_valid_after_collect_returns`
   — after `collect_episode` returns, write to (e.g.)
   `obs_arr[0]` of the underlying buffer (via a held reference)
   and assert that `batch.obs[0]` reflects the change. This
   verifies the view semantics are intentional. If you ship
   COPIES instead, flip the assertion direction.

5. `test_existing_trainer_tests_still_pass_with_rollout_batch`
   — meta-test that runs a representative subset of
   `tests/test_discrete_ppo_trainer.py` after the consumer-side
   refactor.

## Hard constraints

1. **PPO update output is bit-identical end-to-end.** This is
   not a per-tick test — it's a state-dict-after-update test.
   Anything less and you're shipping a behavioural change.
2. **Don't restructure `pack_hidden_states` / `slice_hidden_
   states`.** Their signatures stay the same; they just operate
   on a `RolloutBatch.hidden_state_in` slice instead of a list.
3. **Keep `Transition` around for tests.** Don't delete the
   dataclass — tests that explicitly construct synthetic
   transitions are still valid. Mark with a deprecation note;
   removal is a separate plan.
4. **Audit every `collect_episode` caller.** Cohort code,
   batched collector, smoke driver. Each needs the consumer-
   side update.
5. **Sequential collector only this session.** Leave
   `BatchedRolloutCollector` returning per-agent transition
   lists; converting it to `RolloutBatch` is mechanical but
   bundling here makes the bit-identity test ambiguous.

## Deliverables

- `training_v2/discrete_ppo/transition.py` — `RolloutBatch`
  added; `Transition` deprecation note.
- `training_v2/discrete_ppo/rollout.py` — `_collect` returns
  `RolloutBatch`; hot-path stacking gone.
- `training_v2/discrete_ppo/trainer.py` — `_ppo_update`,
  `_compute_advantages`, `_bootstrap_value` consume
  `RolloutBatch`.
- `training_v2/cohort/worker.py` — call-site update for
  `collect_episode` if needed.
- `tests/test_v2_rollout_batch.py` (new) with the five tests
  above.
- Updates to `tests/test_discrete_ppo_trainer.py` and
  `tests/test_discrete_ppo_rollout.py` to consume
  `RolloutBatch` where appropriate.
- `findings.md` updated.
- Commit: `feat(rewrite): phase-4 S06 (GREEN|PARTIAL) -
  RolloutBatch replaces per-transition dataclass list`.

## Estimate

~2 h. If past 3 h, stop — most likely the trainer-side consumer
audit found a non-trivial caller that needs more thought;
document and decide whether to split into Sessions 06a (rollout
side) and 06b (trainer side) with a temporary adapter.
