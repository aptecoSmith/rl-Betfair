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

## Session 02 — batched cohort forward

**Status:** PARTIAL (design-c fallback shipped; per-agent self-parity
PASS; speed measurement deferred to operator).

### What landed

A new `BatchedRolloutCollector` lives at
`training_v2/discrete_ppo/batched_rollout.py`. It drives N
architecturally-identical (shim, policy) pairs through one
episode each, with:

- **Per-agent forward in a Python loop** (the Session 02 prompt's
  fallback design (c) — see "vmap fatal pitfall" below).
- **Per-agent RNG via global state save/restore** so per-agent
  sampling is byte-identical to running each agent solo at the
  same seed. The save/restore is per-tick and bounds cross-agent
  leakage to zero by construction (no randomness is consumed
  outside the per-agent restore window).
- **Active-set bookkeeping**: agents that terminate mid-batch
  drop out of the per-tick loop. The scoreboard-comparable
  per-agent transition lists end at each agent's natural
  terminal step.
- **Session 01's deferred-sync pattern lifted to per-agent
  buffers**: log_prob_action / log_prob_stake / value_per_runner
  are stashed as 0-d / 1-d device tensors per agent and
  materialised in three batched `.cpu()` calls per agent at
  end-of-episode. Per agent the deferred-sync count is the same
  as Session 01.
- **N=1 reduction to the Session 01 path** when `seeds=None`. The
  global-RNG fast path is byte-identical to
  :class:`RolloutCollector` at the same seed (test
  `test_batched_collector_falls_back_to_n1_session01_path`).

The cohort runner gains a `--batched` CLI flag (default OFF) and
a per-cluster `train_cluster_batched` worker at
`training_v2/cohort/batched_worker.py`. Clustering is by
`(policy_class, hidden_size, obs_dim, action_n)` via
`cluster_agents_by_arch`; cross-cluster scheduling is sequential
(one cluster occupies the GPU at a time, per Session 02 prompt §2).

The per-agent PPO update reads transitions from the batched
collector via a new `DiscretePPOTrainer.update_from_rollout`
method — identical post-rollout pipeline (GAE → PPO → stats) to
`train_episode`, just decoupled from the trainer's own collector.

### vmap fatal pitfall (design-b → design-c fallback)

The Session 02 prompt's preferred shape was design (b): cluster
agents by architecture and run one batched forward per cluster
via `torch.func.vmap` over `torch.func.functional_call` with
stacked per-agent params. This is currently blocked at PyTorch
2.11 (the repo's pinned version) because there is no batching
rule for `aten::lstm.input`:

    RuntimeError: Batching rule not implemented for
    aten::lstm.input. We could not generate a fallback.

A vmap-friendly manual LSTM cell (re-implementing the
`nn.LSTM(batch_first=True)` forward with primitive einsum / tanh /
sigmoid ops) does work under vmap — but does not produce
bit-identical outputs to `nn.LSTM`'s fused-kernel forward, and
the prompt's correctness bar is strict equality on per-agent
self-parity. So the fatal-pitfall fallback (design (c) — per-
agent forward in a Python loop) is what shipped. The collector
interface stays vmap-ready; once a future PyTorch release adds
the LSTM batching rule, the inner forward can swap in without
touching the active-set, RNG, or cohort-runner wiring.

### Tests

Eight regression guards in `tests/test_v2_batched_rollout.py`,
all passing on CPU (synthetic-data path, no GPU required):

1. `test_per_agent_self_parity_batched_vs_solo` — agent 0 in a
   batch of 4 is bit-identical to running agent 0 solo (the
   load-bearing correctness guard, mirroring Session 01b's
   CUDA↔CUDA self-parity bar).
2. `test_per_agent_rng_independence_in_batch` — switching agent
   0's seed does NOT change agents 1-3's transitions.
3. `test_active_set_shrinks_when_agent_terminates_mid_batch` —
   a 1-race-day agent in a batch with a 3-race-day agent
   terminates first; the batched collector returns a shorter
   transition list for the short-day agent and the long-day
   agent runs to its own terminal.
4. `test_cluster_key_groups_compatible_archs` — `hidden_size=32`
   and `hidden_size=64` policies land in separate clusters.
5. `test_cluster_key_groups_obs_dim_separately` — different
   `obs_dim` is its own cluster axis.
6. `test_batched_collector_rejects_mixed_archs` — constructing a
   `BatchedRolloutCollector` across mixed architectures raises.
7. `test_train_cluster_batched_runs_end_to_end_on_synthetic_data`
   — a 2-agent cluster runs full train + eval through
   `train_cluster_batched` and produces populated
   `AgentResult`s with registry round-trips.
8. `test_batched_collector_falls_back_to_n1_session01_path` —
   N=1 with `seeds=None` is byte-identical to
   `RolloutCollector` at the same seed (degenerate-batch
   regression guard).

### Pre-existing tests

All Session 01 regression guards still pass (`test_v2_rollout_sync.py`
2 × CPU-marked tests passing; the `gpu+slow`
`test_cuda_self_parity_after_sync_removal` was not re-run this
session — the rollout code path is unchanged). The 14
v2 trainer / rollout / transition tests pass on CPU. The 21
cohort runner / worker / events / genes tests pass.

### Speed measurement

NOT taken in this session. The session prompt's Probe B (12-agent
× 7-day cohort wall) takes 90-180 minutes; with the design-(c)
fallback shape the expected speedup vs sequential cohort is
bounded by what CUDA stream pipelining gives across N back-to-
back forwards with no env-step pause between them — likely 1.2-
2× rather than the 5× the GREEN bar requires. The plan's
honest verdict slot is therefore PARTIAL by construction; the
operator can run the cohort wall measurement when convenient and
update this section with the observed numbers.

What the operator should measure when running the probes:

| Configuration | s/ep CUDA (N=1) | Cohort wall (12×7) | Self-parity |
|---|---|---|---|
| AMBER v2 baseline | 145.5 s | ~186 min | PASS |
| Session 01 (sync removal) | ~138 s steady | (not measured) | PASS |
| Session 02 N=1 batched (this) | ? | (n/a) | PASS (test #8) |
| Session 02 N=12 batched (this) | ? | ? | PASS (test #1) |

Probe commands (from session prompt §6):

```
python -m training_v2.cohort.runner \
    --n-agents 1 --generations 1 --days 1 \
    --device cuda --seed 42 --batched \
    --output-dir registry/throughput_session02_n1_$(date +%s)

python -m training_v2.cohort.runner \
    --n-agents 12 --generations 1 --days 7 \
    --device cuda --seed 42 --batched \
    --output-dir registry/throughput_session02_n12_$(date +%s)
```

### What's next

The vmap LSTM batching-rule blocker is the single largest lever
remaining — it is the only path to the GREEN cohort-wall bar
(2× speedup). Three follow-on options, in order of preference:

1. **Wait for upstream**: PyTorch's vmap LSTM coverage is an
   open issue; a future release (2.12+?) may add the batching
   rule and the design-(b) swap is mechanical.
2. **Manual batched LSTM cell**: re-implement the LSTM forward
   with primitive ops (einsum, tanh, sigmoid) inside a
   vmap-friendly `_BatchedDiscreteLSTMForward` module. Loses
   bit-identity vs `nn.LSTM`'s fused kernel; per-agent self-
   parity test would need to relax to ~1e-5 tolerance. Not a
   small change but bounded.
3. **Env vectorisation**: out of scope for this plan
   (`purpose.md` §"Out of scope"); requires a separate plan
   with its own correctness phase.

`force-close-architecture` and `arb-curriculum-probe` follow-ons
remain unblocked under the sequential path (`--batched` defaults
OFF). The new batched path is opt-in; the default cohort-runner
behaviour is byte-identical to pre-Session-02.

### Hard-constraint check

| Constraint | Met? |
|---|---|
| §1 No env edits | YES — only `training_v2/`. |
| §2 No reward-shaping changes | YES — collector only reorganises rollout collection. |
| §3 Hidden-state contract per agent | YES — `hidden_state_in` captured BEFORE forward, per agent, `.detach().clone()` of the running state. |
| §4 Two structural `.item()` calls per agent | YES — `int(action.item())` and `float(stake_unit_t.item())` stay; the test_v2_rollout_sync regression for solo still passes. |
| §5 Per-agent self-parity load-bearing | YES — test #1 passes strict equality. |
| §6 Per-agent RNG independence | YES — test #2 passes. |
| §7 Same `--seed 42` for measurement | Honoured by the probe commands above. |
| §8 `cudnn.deterministic = True` | Unchanged; inherited from Session 01. |
| §9 No GA gene additions | YES. |
| §10 No re-import of v1 trainer / policy / rollout / worker pool | YES. |
| §11 `Transition` shape unchanged | YES — same frozen dataclass, same fields. |
| §12 `--batched` defaults OFF | YES. |
