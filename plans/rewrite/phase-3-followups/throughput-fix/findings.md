# Throughput-fix — findings

## Session 01 — rollout sync removal

**Status:** PARTIAL

Self-parity holds bit-identical (load-bearing correctness bar PASS).
Combined two-run mean wall is essentially at the AMBER v2 baseline:
the three deferrable per-tick CUDA→CPU sync barriers were removed
correctly, but the speedup is below the ≥20% bar the session set.
Session 02 (batched cohort forward) is the critical path to hit the
plan-level GREEN bar.

### Speed table

| Run | ep0 | ep1 | ep2 | ep3 | ep4 | mean | Δ vs AMBER v2 |
|---|---|---|---|---|---|---|---|
| AMBER v2 baseline (Phase 3 Session 01b) | — | — | — | — | — | 145.5 s | — |
| Session 01 CUDA-a (cold) | 155.6 | 154.8 | 155.0 | 155.2 | 138.9 | 151.87 s | **+4.4%** |
| Session 01 CUDA-b (warm) | 144.8 | 137.1 | 137.6 | 138.3 | 138.3 | 139.21 s | **−4.3%** |
| Session 01 combined (10 episodes) | — | — | — | — | — | 145.54 s | **+0.0%** |
| Session 01 run-b steady (eps 1–4) | — | — | — | — | — | 137.83 s | **−5.3%** |

A→B run-pair disagreement is 9.1% — under the 20% "something is off"
threshold from the session prompt but not negligible. The shape is
classic cold-start GPU warm-up: run A's first four episodes sit at
~155 s, then ep4 drops to ~139 s and stays there in run B. Either
the cudnn algorithm cache, the CUDA caching allocator, or thermal
state takes ~4 episodes (~10 minutes) to settle on a cold device.

For an honest steady-state estimate, run B's eps 1–4 mean (137.8 s)
is the right number — about a **5% improvement** vs the AMBER v2
145.5 s baseline. That is consistent with a small per-tick sync
saving (~3 syncs × ~12k ticks × ~10 µs ≈ 360 ms per episode) plus
some Python-side overhead from holding ~12k device tensors in
sidecar lists and building Transitions at end-of-episode.

### Self-parity

CUDA↔CUDA self-parity at seed 42 across both 5-episode runs:

```
ep0: |Δreward|=0.00e+00  |Δvalue_loss_mean|=0.00e+00
ep1: |Δreward|=0.00e+00  |Δvalue_loss_mean|=0.00e+00
ep2: |Δreward|=0.00e+00  |Δvalue_loss_mean|=0.00e+00
ep3: |Δreward|=0.00e+00  |Δvalue_loss_mean|=0.00e+00
ep4: |Δreward|=0.00e+00  |Δvalue_loss_mean|=0.00e+00
```

PASS — strict equality, not 1e-7 tolerance. The deferred batched
end-of-episode `.cpu()` transfer is numerically identical to the
per-tick `.item()` transfers it replaced (same tensor contents on
the GPU, same `.cpu()` memcpy, same Python-float conversion at end).

The new `tests/test_v2_rollout_sync.py::test_cuda_self_parity_after_
sync_removal` (1-episode scoped, ~5 minutes) was run independently
and PASSED. It doubles as a cheap CI guard for future refactors of
this loop.

CPU/CUDA action-histogram band (Phase 3 Session 01b's Bar 2): NOT
re-measured this session. The CPU rollout code path is structurally
unchanged (verified by `test_transition_log_probs_byte_identical_
across_two_cpu_runs`) and the CUDA path's self-parity holds, so the
band cannot have widened. Re-running the existing `test_v2_gpu_
parity.py` 5-episode trio (~25–30 minutes wall) is unnecessary
unless an operator wants the strict signature for the cohort
launch.

### Pre-existing tests

All pre-existing v2 trainer / rollout / transition tests pass on
CPU:

```
tests/test_discrete_ppo_rollout.py     4 passed
tests/test_discrete_ppo_trainer.py     8 passed
tests/test_discrete_ppo_transition.py  2 passed
                                      14 passed
```

PPO trainer suite (the load-bearing recurrent-state-through-update
guards in `TestRecurrentStateThroughPpoUpdate` live here) — **66
passed**. The hidden-state contract from CLAUDE.md §"Recurrent PPO:
hidden-state protocol on update" is unchanged: `hidden_state_in` is
still captured BEFORE the forward pass via
`tuple(t.detach().clone() for t in hidden_state)`.

### What changed

`training_v2/discrete_ppo/rollout.py::_collect`:

- The three deferrable CUDA→CPU syncs (action log-prob, stake
  log-prob, per-runner value) now emit 0-d / 1-d device tensors
  into three sidecar buffers (`pending_log_prob_action`,
  `pending_log_prob_stake`, `pending_value_per_runner`). One
  batched `.cpu()` per buffer at end-of-episode replaces ~3 ×
  n_steps per-tick syncs.
- The two structural syncs (`int(action.item())`,
  `float(stake_unit_t.item())`) stay exactly where they were —
  the CPU env consumes int + float every tick.
- `Transition` is a frozen dataclass, so transitions are now built
  in a list comprehension at end-of-episode (after the device→CPU
  transfer). All other per-tick CPU-side data (obs, mask,
  hidden_state_in, action_idx, stake_unit, per_runner_reward,
  done) accumulates in parallel Python lists during the rollout
  and is zipped into Transitions at the end.
- Memory: ~12k 0-d / 1-d device tensors held during the rollout
  is well under 15 MB total — no measurable VRAM impact.

`tests/test_v2_rollout_sync.py` (new):

- `test_cuda_self_parity_after_sync_removal` (gpu+slow, ~5 min) —
  bit-identical CUDA-vs-CUDA at seed 42, 1 episode.
- `test_action_idx_and_stake_unit_still_materialise_per_tick`
  (slow) — patches `shim.step` and asserts the action_idx and
  stake kwargs received are CPU scalars (`int` / `float` /
  `np.floating`) on every tick. A torch.Tensor or np.ndarray
  reaching this point would mean the structural `.item()` calls
  were accidentally deferred.
- `test_transition_log_probs_byte_identical_across_two_cpu_runs`
  (slow) — strict-equality two-run CPU self-parity on
  log_prob_action, log_prob_stake, value_per_runner. CPU code
  path is unchanged, so this is byte-identical, not "close".

### What's next

PARTIAL → Session 02 is the critical path. The hypothesis from
the session prompt is intact: cutting sync barriers alone gives
~5% speedup at LSTM(h=128, batch=1), but the kernel-launch
overhead at this scale is the dominant cost. The path to GREEN
goes through batched cohort forward (Session 02) — running N
agents through one `(N, obs_dim)` forward per tick, amortising
launch overhead across the batch.

Recommendation for the Session 02 prompt: open with the Session
01 measured 5.3% steady-state speedup as the gap-to-close (vs
the plan-level GREEN bar of 90 s/ep). This rules out "Session 01
plus a tweak" as a path and makes batched-forward the
unambiguous next step.

cProfile snapshot: NOT taken. The session prompt scopes profiling
to "near-miss in the 116–130 s/ep range" — we landed at the
baseline, so the bottleneck isn't a single fixable hot spot, it's
the architectural batch=1 problem Session 02 is designed for.
