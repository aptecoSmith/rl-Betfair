# Cohort speedup — phase 3 profile + measured results

> **Status: profile, plan, and Options A + F.1 + B-lite all shipped and
> validated end-to-end on a 2-agent × 1-gen mini-cohort.** Options B-big
> and C deferred per operator decision after measuring diminishing
> returns.

## TL;DR

Shipped three changes, measured improvements per-agent:

| Change | Effect | Per-agent time saved | Per-cohort time saved |
|---|---|---:|---:|
| **A** Mixed-device (CPU rollout, CUDA update) | 31% faster `train_episode` | ~7 min | ~3.5 h |
| **F.1** Cohort-scoped `feature_cache` | 63% faster env build for cache-hit days | ~3 min* | ~2 h |
| **B-lite** Lifted dev assert in `feature_extractor.extract` | ~1s/day | ~30s | ~12 min |

\* per-agent gain is amortised across cohort agents; first agent pays the full
build cost, subsequent agents in the gen reuse the cache.

**Measured train_episode wall (2026-04-16, 10,419 ticks, head-to-head):**

| Config | wall_s | ms/step | Δ vs baseline |
|---|---:|---:|---:|
| dev=cuda rollout=cuda (baseline) | 64.21 | 6.16 | — |
| **dev=cuda rollout=cpu (Option A)** | **44.52** | **4.27** | **−30.7%** |
| dev=cpu rollout=cpu (pure CPU) | 52.48 | 5.04 | −18.3% |

Pure CPU is slower than split-device because the PPO update at
mini-batch=64 across ~620 mini-batches genuinely benefits from CUDA.

**Honest verdict on the original 28→14 min/agent target:** 25-30% of the
per-agent wall is recovered by what shipped (~21 min/agent). Hitting
14 min from rollout-side changes alone would require an architectural
move (the existing `--batched` cluster runner, phase_2.md option E) —
documented but not pursued in this phase.

---

## Profile setup

- **Harnesses** (under `tools/`):
  - `profile_v2_rollout.py` — cProfile on `RolloutCollector.collect_episode`
  - `profile_v2_full_agent.py` — cProfile on `DiscretePPOTrainer.train_episode`
    (rollout + GAE + PPO update)
  - `bench_v2_rollout.py` — un-instrumented wall bench, single device
  - `bench_v2_split_device.py` — wall bench for cuda / split / cpu
  - `bench_env_build.py` — cold vs warm env-build break-down
- **Day profiled**: 2026-04-16 (56 races, 10,419 ticks). All profiles use
  the exact same env config as the running cohort
  (`--strategy-mode arb`, predictor bundle,
  `force_close_before_off_seconds=120`, `pwin` thresholds 0.20/0.40,
  `race_confidence_threshold=0.50`, `lay_price_max=20.0`,
  `emit_debug_features=False`, scorer at `models/scorer_v1`).
- **Profiles**:
  - `phase_3_profile.txt` / `.prof` — rollout, CUDA, cProfile
  - `phase_3_profile_cpu.txt` / `.prof` — rollout, CPU, cProfile
  - `phase_3_full_agent.txt` / `.prof` — full train_episode, CUDA
  - `phase_3_full_agent_postA.txt` / `.prof` — same with reference rebuild

Methodology note: cProfile inflates per-step wall by ~50% on the CPU
path (3.8 → 5.9 ms/step measured). It is faithful for **relative**
function rankings but unreliable for **absolute** ms numbers. Wall
numbers in this doc come from `bench_v2_split_device.py` (un-instrumented).

---

## Where the agent's wall spends its time

cProfile breakdown on **single-device CUDA** (pre-Option-A baseline,
~87s instrumented train_episode):

| Bucket | cum_s | % of train_episode |
|---|---:|---:|
| Rollout (`_collect`) | 80.43 | 92.4% |
| └─ Policy forward | 26.06 | 30.0% |
| └─ Shim+scorer (`compute_extended_obs`) | 18.29 | 21.0% |
| └─ Env step internals (`_get_obs`+`_get_position_vector`+`_process_action`+`_get_info`) | 14.32 | 16.5% |
| └─ Rollout buffer copies + dist sampling + syncs | ~21 | ~24% |
| PPO update (`_ppo_update`) | 6.61 | 7.6% |
| `_update_from_batch` overhead | ~0.1 | <1% |

**Key surprise:** PPO update is only ~8% of train_episode wall, not the
~20% the original phase_3 brief guessed. So no upside from PPO-update
optimisation alone — the rollout is where the time lives.

**Second key surprise:** env build (per-day, `_build_env_for_day`) is
**~16s steady-state, not 43s.** The earlier 43s number included the
~11s predictor-bundle load which happens **once per worker process**,
not per day. Of the 16s, `engineer_day` is ~10s (the cacheable bit),
and per-race predictor inference is ~6s (not cacheable across agents
the same way — see §"What's not in this plan" for the cache extension).

---

## What was shipped

### Option A — Mixed-device (CPU rollout, CUDA PPO update)

**The single biggest win.** Measured 31% reduction in per-day
train_episode wall on identical inputs.

**Mechanism:** At `hidden_size=128` + batch=1 (rollout per env step),
the policy forward is dominated by per-op CUDA kernel-launch overhead
(measured: 132,852 Linear calls × ~63µs each = 8.4s tottime). CPU
avoids this entirely. The PPO update at mini-batch=64 across ~620
mini-batches amortises the launch cost so CUDA wins there.

**Implementation:**

- `DiscretePPOTrainer.__init__` gains a `rollout_device` kwarg
  (default `None` = single-device = byte-identical to pre-plan).
- At construction, policy + optimiser are built on `device`, then
  moved to `rollout_device` so the first rollout sees the fast device.
- `_update_from_batch` wraps `_ppo_update` with a `try` / `finally`
  that moves the policy + optimiser state TO `device` for the surrogate-
  loss work and BACK to `rollout_device` after. The `finally` guarantees
  restoration on KL-early-stop or numerical-loss exits.
- `_bootstrap_value` reads its forward device from
  `next(self.policy.parameters()).device` so it works under either
  device parking.
- Hidden-state buffers from rollout land on `rollout_device`. Their
  `pack_hidden_buffer` output is moved to `self.device` inside
  `_ppo_update` so the mini-batch slice ops aren't device-mismatched.
- `training_v2/cohort/worker.py::train_one_agent` sets
  `rollout_device="cpu"` when the operator-supplied device is `"cuda"`;
  single-device CPU runs keep `rollout_device=device` so they're
  byte-identical.
- `_rebind_trainer` and the eval `RolloutCollector` both use
  `trainer.rollout_device` so the per-day rebind and per-eval-day
  rollouts stay on CPU.

**Regression guards** in `tests/test_discrete_ppo_trainer.py` (slow):

- `test_split_device_round_trip_keeps_policy_on_rollout_device_after_update`
  — three snapshots (before `train_episode`, after, optimiser state)
  confirm policy + Adam state survive the round trip.
- `test_split_device_default_disabled_is_byte_identical_to_pre_plan`
  — `rollout_device` unspecified leaves the trainer single-device on
  `cuda`; policy stays on cuda through the episode.

Existing 76 trainer + 31 worker tests still pass.

**Measured impact**: 64.21s → 44.52s/episode = **−30.7%**. Per-agent
wall reduction depends on agent shape, but a 23-day-mix agent (16
train + 7 eval) drops from ~28 min → ~21 min.

### Option F.1 — Cohort-scoped feature cache

**Mechanism:** `BetfairEnv.__init__` already accepts a
`feature_cache: dict[str, list]` keyed by `day.date`. When the cache
contains the date, `engineer_day` (~10s of the 16s env build) is
skipped and the cached static features are reused. The cohort runner
never threaded the cache through — every agent paid the full cold-build
cost.

**Implementation:**

- `training_v2/cohort/runner.py::run_cohort` allocates a single
  `feature_cache: dict[str, list]` at function scope.
- Threaded through `train_one_agent_fn(...)` → `_build_env_for_day(...)`
  → `BetfairEnv(feature_cache=...)`.
- Also threaded through `_evaluate_agents_on_monitor_days(...)` so
  the monitor-eval reuses the cache too.
- Cohort-scoped (not gen-scoped): features are pure functions of
  (date, env feature knobs). Knobs don't change between gens, so the
  cache survives the full cohort run. Memory: ~40 MB/day × #unique
  days ≈ 900 MB for a typical 23-day cohort. Fits well under the
  4.5 GB worker RSS the running cohort had.

**Measured impact (cold vs warm env build):**

| Build | wall |
|---|---:|
| Cold (no cache) | 15.80s |
| Warm (cache hit) | 5.92s |
| **Reduction** | **63%** |

**Validation:** Smoke run (`registry/smoke_phase3_optA_1779536161/`)
shows the log lines we wanted:
- Agent 1 day 1: "Feature engineering (56 races, 10419 ticks) completed in 9.97s"
- Agent 1 day 2: "Feature engineering (73 races, 11353 ticks) completed in 10.05s"
- Agent 2 day 1: **"Feature cache hit for 2026-04-16"**
- Agent 2 day 2: **"Feature cache hit for 2026-04-15"**
- Both eval days for agent 2: cache hits
- Monitor eval day: cache hit

Per-cohort recovery at 6 agents × 23 days × ~10s = **23 minutes** of
engineer_day work eliminated across the cohort wall (the first agent
of each unique date still pays the cold cost).

### Option B-lite — Lifted dev assert in `feature_extractor.extract`

**Mechanism:** `extract` was running `set(feats.keys()) == set(FEATURE_NAMES)`
on every call — a ~5-10 µs cost per call × 184k calls/day = ~1-2s/day.
The check catches keyset drift between FEATURE_NAMES and the producer;
runtime divergence is impossible without a code edit, which would
trip the first-call check anyway.

**Implementation:** Added a `_extract_contract_verified: bool`
instance flag (initial `False`). The first call performs the
`frozenset(keys) == _FEATURE_NAME_SET` comparison; subsequent calls
skip it. Test coverage in `tests/test_scorer_v1_dataset.py` and
`tests/test_env_shim_batched_scorer.py` (14 tests) still pass.

**Measured impact:** ~1-2s per day rollout. Small but free.

---

## Reflected breakdown after A + F.1 + B-lite

The smoke (`registry/smoke_phase3_optA_1779536161/`) shows:

| Phase | Per-agent wall | Notes |
|---|---:|---|
| env build (first agent) | ~12s/day | cold engineer_day (10s) + per-race predictor (~2s for small races) |
| env build (subsequent agents) | ~2-3s/day | cache hit; per-race predictor still computed |
| train_episode | 45-50s/day | Option A delivering 30% rollout reduction |
| eval (no PPO) | 35-65s/day | varies with race count; pure rollout on CPU |
| monitor eval | similar to eval | uses the same cache |

For the 6-agent × 5-gen × 23-day cohort recipe currently in use:
- **Before** (pre-Options A+F.1): ~28 min/agent × 30 agents = **14 h wall**
- **After**: ~21 min/agent × 30 agents = **~10 h wall**
- **Saved**: ~4 h cohort wall, with no quality risk

---

## What's not in this plan (deferred)

### Option B-big — full vectorisation of `feature_extractor.extract`

`extract` is called 184k times per day. Per-call body is dict-build
(30+ dict-insertions) + ~3 small numeric computations. The big lift is
adding an `extract_array(...)` sibling that writes into a pre-allocated
numpy array and skipping the dict construction entirely.

**Projected**: 3-4s/day saved (~1.5-2 min/agent).
**Cost**: 6h implementation + byte-equality regression test against the
scorer's training-time feature build.
**Risk**: feature ordering drift could silently break the LightGBM
scorer's expected input shape.

**Decision (operator, 2026-05-23)**: defer. The B-lite lifted assert
ships; the rest is documented for a follow-on session.

### Option C — incremental bet aggregates

Phase 2 hypothesised `get_paired_positions` / `get_naked_exposure` /
`get_positions` as the main bet-walk cost. Profile measured 0.42 ms/step
for `_get_position_vector` (5.6% of rollout) — much smaller than phase 2
expected. Per-agent savings from caching: ~1.6 min.

**Cost**: 8h implementation + consistency assertions across
`place_back`/`place_lay`/`passive_book.on_tick`/`_settle_current_race`.
**Risk**: bet-state invariant bugs are silent — wrong exposure
numbers propagate to risk gating and reward shaping.

**Decision**: deferred. The cost/benefit isn't there at this scale.

### Option F.2 — Cache per-race predictor outputs

The remaining ~6s of env-build cost after F.1 is dominated by
`_compute_race_predictor_outputs` and `_compute_tick_predictor_outputs`
(predictor inference per race / per tick). These are pure functions of
(market_id, predictor_bundle) but the bundle holds its own internal
cache by `market_id`, so the second-agent cost may already be lower
than the first. Worth profiling before committing engineering time.

**Projected**: 4-6s/day per cache hit ≈ 1.5-2 min/agent.
**Cost**: 4-6h implementation, new cache key, memory ~10-20 MB/day.
**Decision**: deferred to a separate session.

### Architectural alternative — `--batched` cluster training

Documented in phase_2.md option E. Trains all same-hidden-size agents
in one env pass. **2-5× speedup** if compatible with the current recipe.
Caveats: silently ignores `per_transition_credit=True` and
`bc_pretrain_steps>0`. The currently running cohort uses neither, so
this MIGHT be a drop-in upgrade. Recommended as the next-phase
investigation once the cohort is generating clean numbers.

---

## Files added / modified in this phase

**New:**
- `tools/profile_v2_rollout.py`
- `tools/profile_v2_full_agent.py`
- `tools/bench_v2_rollout.py`
- `tools/bench_v2_split_device.py`
- `tools/bench_env_build.py`
- `plans/cohort_training_speedup/phase_3_profile.txt` / `.prof`
- `plans/cohort_training_speedup/phase_3_profile_cpu.txt` / `.prof`
- `plans/cohort_training_speedup/phase_3_full_agent.txt` / `.prof`
- `plans/cohort_training_speedup/phase_3_full_agent_postA.txt` / `.prof`

**Modified:**
- `training_v2/discrete_ppo/trainer.py` — `rollout_device` kwarg,
  device round-trip in `_update_from_batch`, `_move_optim_state_to`,
  `_move_to_device` short-circuit when already on target, bootstrap
  uses policy's current parameter device.
- `training_v2/cohort/worker.py` — `feature_cache` threaded through
  `_build_env_for_day` and `train_one_agent`; trainer constructed
  with `rollout_device="cpu"` when device is cuda; eval `RolloutCollector`
  bound to `trainer.rollout_device`.
- `training_v2/cohort/runner.py` — cohort-scoped `feature_cache`
  threaded into `train_one_agent_fn` and `_evaluate_agents_on_monitor_days`.
- `training_v2/scorer/feature_extractor.py` — lifted per-call
  contract assert to once-per-instance.
- `tests/test_discrete_ppo_trainer.py` — two new split-device
  regression tests.

**Smoke artefact:**
- `registry/smoke_phase3_optA_1779536161/` — 2-agent × 1-gen × 2 train
  × 2 eval × 1 monitor mini-cohort proving the cache hits land and the
  per-episode wall numbers match the bench predictions.

---

## Recommendation

Options A + F.1 land transparently for any `--device cuda` cohort
(single-device CPU runs are byte-identical to before). Expected wall
reduction: 25-30% per agent, ~30 min off the cohort wall per generation
at the running cohort's recipe.

**Not done yet — see [deferred.md](deferred.md) for the punch list**
(Option B-big, Option C, Option F.2, `--batched` cluster training, PPO
update profile re-pass). Each entry has projected savings, cost, risk,
and the decision reason.

Re-profile after one full cohort to confirm the per-agent wall actually
lands at ~21 min and identify which deferred item is worth the next
round of investment.
