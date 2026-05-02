---
plan: rewrite/phase-4-restore-speed
status: design-locked
opened: 2026-05-02
depends_on: rewrite/phase-3-cohort, rewrite/phase-3-followups/throughput-fix
---

# Phase 4 — restore-speed: cut v2's per-tick rollout overhead

## Purpose

v2 is **3× slower per tick than v1** on equivalent architectures.
Measured from `logs/discrete_ppo_v2/run_cpu_post_sync_fix.jsonl`
(median 9.60 ms/tick) and `logs/training/episodes.plan-A-diverged-
20260422T055217Z.jsonl` (median 2.94 ms/tick for `ppo_lstm_v1`,
5.86 ms/tick for `ppo_transformer_v1`):

| System | Architecture | ms/tick (median) |
|---|---|---|
| v1 | `ppo_lstm_v1` | **2.94** |
| v1 | `ppo_transformer_v1` | 5.86 |
| v2 | LSTM h=128, **CPU** | **9.60** |
| v2 | LSTM h=128, **CUDA** | 11.88–12.79 |

This is the opposite of what the architecture would predict — v2's
policy is *smaller* than v1's. So the slowdown isn't model compute,
it's per-tick overhead the rewrite introduced into the rollout path.
Six concrete sites in `training_v2/discrete_ppo/rollout.py` plus
several smaller ones in `agents_v2/discrete_policy.py`. Each is
recoverable while keeping every v2 advancement (per-runner reward
attribution, per-runner value head, recurrent-state-through-PPO
contract, masked categorical sampling, deferred-sync rollout).

**The objective is "same numbers, faster execution."** No
behavioural change, no new shaping, no architectural rework.

## Why this is its own phase, not a follow-on

`phase-3-followups/throughput-fix/` was scoped around the v2-vs-v2
question (CUDA vs CPU on v2). It successfully landed Sessions 01
(per-tick CUDA→CPU sync removal) and 02 (batched cohort forward
shell), but its premise was "GPU isn't saturated"; the answer it
converged on was "we need vmap or worker-pool parallelism." That
was the wrong frame.

This phase reframes around the v2-vs-v1 question. **Per-tick
Python and tensor-allocation overhead in the rollout loop is the
dominant cost**, not GPU underutilisation. A worker pool of 6
agents each running an O(n²) attribution loop just multiplies the
waste. Cut the overhead first; revisit GPU saturation if and only
if there's still a gap once each tick is back to v1's ~3 ms.

It's a phase, not a sub-followup, because the work here is the
critical path to "training like we used to":

1. **Sequential cohort runs become viable again** at v1's
   per-tick rate (~35 s/ep × 12 agents × 7 days = ~50 min vs
   today's 3.1 h).
2. **Multi-generation GA cycles unblock** without needing a
   concurrency rewrite. 4 generations at 50 min/gen = ~3.5 h, a
   normal training run.
3. **Phase-4-then-scale-up** is the rewrite endgame — 66-agent
   cohorts only become practical once each agent is fast.

Every rewrite-overall hard constraint applies (parallel tree, no
env edits, no v1 imports, no new shaped rewards, no entropy /
KL gymnastics). The work is concentrated in
`training_v2/discrete_ppo/rollout.py` (Sessions 01–06) and
`agents_v2/discrete_policy.py` (Sessions 07+).

## What's locked

### Same numbers, faster execution

Every session ships with a **bit-identical-on-CPU parity test**
against the pre-session code, on the same seed and the same day.
Numpy / torch fp32 operations are deterministic on CPU at fixed
seed; if the pre/post outputs differ at all, the change is wrong.
CUDA self-parity (Phase 3 Session 01b's load-bearing bar) is the
secondary guard.

The only allowed deviations from bit-identity are:

- Floating-point order-of-operations changes that are documented
  in the session prompt and verified at fp32 epsilon (~1e-7
  relative) on the same fixed seed.
- Sessions 06 and 07+ which restructure data layout but preserve
  the same numerical inputs to the PPO update.

### No env edits

The env (`env/betfair_env.py`, `env/bet_manager.py`,
`env/exchange_matcher.py`) is off-limits. Inherited from
`plans/rewrite/README.md` hard constraint §1. If a session's
finding traces a real overhead into env code, file it as a
follow-on plan and stop.

### No new GPU work, no concurrency rewrite

This phase is single-process, single-agent, sequential cohort.
The throughput-fix plan's vmap / worker-pool / batched-forward
work stays out of scope here. Phase 4's whole point is to verify
that v2 *can* be fast on the existing shape before any
concurrency layer is added.

### Same protocol as `phase-3-cohort`

Each per-session speed measurement uses the same single-day
1-episode baseline as Phase 3 Session 01b (2026-04-23 on
`data/processed_amber_v2_window/`). `--seed 42` mandatory.
Cohort-wall measurements use the same 12-agent / 1-gen / 7-day
shape as `phase-3-cohort` Session 04.

### One inefficiency per session, with tests and a commit

The user contract for this phase: each session lands **one** fix,
**adds tests for it**, and **commits before moving on**. This is
identical to the per-session discipline used by
`force-close-architecture` (one mechanics change per cohort) but
applied to the throughput axis. A session that touches more than
one item or skips the test guard does not ship.

## Success bar

The phase ships GREEN iff cumulative across Sessions 01–06:

1. **CPU ms/tick ≤ 4.0** on the same single-day measurement
   (vs today's 9.6 → ≥ 2.4× single-tick speedup; gets within
   striking distance of v1's 2.94).
2. **All session-level bit-identity tests pass** (per-session
   outputs match pre-change at fp32 epsilon on a fixed seed).
3. **CUDA↔CUDA self-parity from Phase 3 Session 01b still holds**
   (bit-identical `total_reward` and `value_loss_mean` on two
   CUDA runs at seed 42).
4. **All pre-existing v2 trainer / rollout / collector / cohort
   tests pass** on CPU.
5. **No behavioural drift on the 12-agent cohort scoreboard.**
   Re-running the AMBER v2 baseline protocol post-Phase-4 produces
   per-agent eval P&L within fp32-aggregation tolerance of the
   pre-Phase-4 baseline (`registry/v2_amber_v2_baseline_1777577990/
   scoreboard.jsonl`).

GREEN-with-stretch if Sessions 07+ also land and CPU ms/tick
drops to ≤ 3.0 (v1 parity or better).

If after Sessions 01–06 the CPU ms/tick is still ≥ 6.0, the
remaining overhead lives outside the rollout collector — likely
in the policy forward (Session 07+) or the env step itself
(out of scope, file separately).

## Sessions

### Session 01 — per-runner attribution incremental tracking

**The single biggest win.** `_attribute_step_reward`
(`training_v2/discrete_ppo/rollout.py` lines 346–426) walks
**every bet ever placed in the episode, every tick** —
`all_bets = list(env.all_settled_bets) + list(live_bets)` is
O(bets-so-far) per call, O(n²) per episode. By tick 11 000 the
loop touches hundreds of bets, 99 % of which haven't changed.

Replace with incremental tracking: maintain a "pending-pnl" set
of bet-ids whose `pnl` may still mutate. A bet enters when first
seen; leaves once its pnl is final (post-settle, no further
mutation). Most ticks scan zero bets. Same per-runner
attribution, same invariant assert, same numbers.

Session prompt: `session_prompts/01_per_runner_attribution.md`.

### Session 02 — eliminate obs / mask double-copy

`obs` and `mask` are each materialised twice per tick in
`rollout.py`: once into the device buffer for the forward pass
(lines 185–193), once into the per-tick CPU list for the
Transition (lines 281–283). Two `np.asarray(...)` allocations
and two copies per tick where one suffices.

Replace with a single pre-allocated `(n_steps_max, obs_dim)` /
`(n_steps_max, action_n)` numpy buffer per episode, written into
once per tick with a slice view.

Session prompt: `session_prompts/02_obs_mask_double_copy.md`.

### Session 03 — drop per-tick distribution wrapper construction

`torch.distributions.Beta(out.stake_alpha, out.stake_beta)`
constructs a fresh `Beta` per tick (line 230) for one
`.sample()` and one `.log_prob()` call. The wrapper's
`__init__` does parameter validation, broadcasting checks, and
pre-allocates internal tensors — measurable Python overhead at
12 k/episode.

Replace with a direct functional sampling path on the rollout
hot path (compute the gamma-ratio sample formula inline OR
disable validation via `Beta.set_default_validate_args(False)`
and reuse a singleton-pattern wrapper). The PPO update keeps the
distribution wrapper because it needs `.log_prob` on stored
tensors. Bit-identity: yes, when the underlying `_standard_gamma`
calls consume the same RNG sequence; preserve seed order.

Session prompt: `session_prompts/03_distribution_objects.md`.

### Session 04 — pre-allocate hidden-state capture buffer

`hidden_in_t = tuple(t.detach().clone() for t in hidden_state)`
per tick (lines 207–209) is the recurrent-PPO contract — load-
bearing, can't drop. But the per-tick tuple-of-clones allocates
2 × `(num_layers, 1, hidden)` tensors every tick, churning the
allocator. Pre-allocate a `(n_steps_max, num_layers, 1, hidden)`
buffer pair at episode start and write into a slice view per
tick. Same numerical contents flow into the PPO update; far less
allocator churn.

Session prompt: `session_prompts/04_hidden_state_allocator.md`.

### Session 05 — make attribution invariant assert opt-in / sampled

`_attribute_step_reward` runs `np.isclose(total, step_reward, ...)`
every tick and raises on drift (lines 412–424). Cheap
individually but compounds at 12 k/episode. Make it a one-in-N
sample-check by default with the option to switch to per-tick
under a debug flag (`PHASE4_STRICT_ATTRIBUTION=1`). Session 01's
implementation must obey the same flag.

Session prompt: `session_prompts/05_invariant_assert.md`.

### Session 06 — replace Transition list with aligned RolloutBatch

End-of-episode list comprehension at lines 320–334 builds 12 k
frozen `Transition` dataclass instances and `float()`-converts
each log-prob entry. The PPO update doesn't actually consume a
list of dataclasses — it consumes aligned tensor batches that
the trainer immediately stacks back up.

Replace with a `RolloutBatch` namedtuple of pre-stacked tensors
returned directly from `_collect`. The trainer's `_ppo_update`
consumes the batch directly without an intermediate stacking
pass. Same shapes, same numbers, no dataclass round-trip.

Session prompt: `session_prompts/06_transition_dataclass.md`.

### Session 07 — policy forward mask-path cleanup

The post-read findings from `agents_v2/discrete_policy.py`:

- `_apply_mask` builds a fresh `torch.tensor(float("-inf"),
  ...)` per call (lines 355–357). Pure waste — replace with
  `logits.masked_fill(~mask, float("-inf"))`. Bit-identical;
  small per-tick win.
- `Categorical(logits=...)` per tick (line 308) — wrapper
  construction overhead. **Already covered by Session 03's
  global `set_default_validate_args(False)` toggle**; no
  separate session needed.
- Four separate `nn.Linear` heads on `lstm_last`
  (`logits_head`, `stake_alpha_head`, `stake_beta_head`,
  `value_head`). Fusing them changes weight-init RNG
  consumption order, which breaks bit-identity-on-init.
  **Skip** — the speed win (one matmul vs four) is small at
  batch=1 and CPU, and the bit-identity cost is real.

Session prompt: `session_prompts/07_policy_mask_path.md`.

This is the only policy-side session in Phase 4. The env_shim
investigation surfaced a much larger candidate; see "Phase 4b
candidates" below.

## Phase 4b candidates (NOT IN SCOPE for this phase)

Findings discovered while scoping Phase 4 that warrant their
own follow-on plan rather than bundling here.

### env_shim per-tick scorer overhead

`agents_v2/env_shim.py::compute_extended_obs` (called by
`step` on every non-terminal tick, line 263) runs the Phase 0
LightGBM scorer per active runner per side per tick — up to
**~28 `booster.predict()` calls per tick** for a 14-runner
race. Plus a per-slot `next(j for j, r in enumerate(tick.
runners) if r.selection_id == sid)` scan (lines 351–355) that's
O(N) per slot per side, i.e. O(N²) per tick.

This is potentially a **larger single-source overhead than
anything in the rollout collector** — at ~50–100 µs per
booster.predict (typical for a small LightGBM model called via
the Python API, no batching), 28 calls = **1.4–2.8 ms per
tick**, which is 15–30 % of the current 9.6 ms/tick CPU
budget.

**Why this is Phase 4b, not Phase 4:**

1. **env_shim edits are out of scope** per Phase 4 hard
   constraint §9 — the shim is co-equal with the env in the
   "untouched parallel-tree boundary" the rewrite established
   in Phase 1.
2. The fix likely needs **batched booster prediction** (one
   `booster.predict(stack_of_28_feature_vectors)` per tick
   instead of 28 separate calls) AND a **slot-to-tick-runner-
   index cache** built once per race rather than per tick. Both
   are correctness-sensitive (wrong batching = wrong scorer
   inputs = different obs vector = behavioural change).
3. The investigation needs **profiling first** to confirm the
   28 × predict call count, the per-call wall, and that the
   scorer output is bit-identical pre/post batching. That's
   its own session.

**Suggested Phase 4b shape:**

- Session 01 — Profile `compute_extended_obs` end-to-end on
  a representative tick. Measure the per-call wall of
  `booster.predict`, `calibrator.predict`, `FeatureExtractor.
  extract`, and the slot-lookup scan. Establish the actual
  per-tick budget for each.
- Session 02 — Slot-to-tick-runner-index cache (built at
  race start, invalidated on race transition). Bit-identical;
  pure index-lookup speedup.
- Session 03 — Batched booster prediction (28 calls → 1).
  Bit-identity verification against per-call outputs at fp32
  epsilon (LightGBM's per-tree path may not be byte-identical
  under batching).
- Session 04 — Verdict + measurement.

Open `plans/rewrite/phase-4b-env-shim-scorer/` after Phase 4
completes, with this section as the founding finding.

### Session 99 — verdict + writeup

`plans/rewrite/phase-4-restore-speed/findings.md` with the
per-session ms/tick table:

| Session | Cumulative ms/tick CPU | Δ vs prev | Bit-identity | Tests added |
|---|---|---|---|---|
| Pre-Phase-4 baseline | 9.60 | — | — | — |
| + S01 (attribution) | ? | ? | ? | ? |
| + S02 (obs/mask) | ? | ? | ? | ? |
| + S03 (distributions) | ? | ? | ? | ? |
| + S04 (hidden state) | ? | ? | ? | ? |
| + S05 (assert) | ? | ? | ? | ? |
| + S06 (RolloutBatch) | ? | ? | ? | ? |
| + S07 (mask path) | ? | ? | ? | ? |
| v1 reference | 2.94 | — | — | — |

Plus the success-bar verdict (GREEN / GREEN-with-stretch /
PARTIAL / RED) and the next step (Phase-5 scale-up if GREEN;
file remaining gap as a follow-on plan if not).

Session prompt: NOT YET WRITTEN. Trivial — gated on Sessions
01–06 (and 07+ if any).

## Per-session contract

Every session, in order:

1. **Read** the relevant code path end-to-end (rollout.py for
   01–06, discrete_policy.py for 07+) and the pre-session
   measurement.
2. **Land the change** in `training_v2/` only.
3. **Add tests** under `tests/test_v2_*.py`. At minimum:
   - One **bit-identity test** comparing pre-/post-change output
     on a fixed seed for the same single-day rollout. Strict
     equality on CPU; fp32 epsilon documented if order-of-ops
     changed.
   - One **structural test** asserting the new behaviour (e.g.
     "pending-pnl set scans zero bets on a no-bet tick", "obs
     buffer is allocated once per episode not per tick").
   - All pre-existing tests in the touched file pass on CPU.
4. **Measure**: ms/tick on the same single-day baseline (one
   1-episode CPU run is sufficient at this level — full 5-episode
   parity runs from Phase 3 are reserved for the verdict
   session).
5. **Commit** with `feat(rewrite): phase-4 SXX (PARTIAL|GREEN) -
   <one-line description>`. Include the cumulative ms/tick in
   the commit body.
6. **Update** `findings.md` (creating it on Session 01) with the
   row for this session in the cumulative table.

## Hard constraints

In addition to all rewrite hard constraints
(`plans/rewrite/README.md` §"Hard constraints") and inherited
from `phase-3-cohort` and `phase-3-followups/throughput-fix`:

1. **No env edits.** Same as throughput-fix §1.
2. **No reward-shaping changes.** Same as throughput-fix §2.
3. **No GA gene additions.** Same as throughput-fix §3.
4. **No `cudnn.benchmark = True`.** Determinism stays on; the
   load-bearing CUDA self-parity bar from
   `phase-3-cohort/findings.md` Session 01b is preserved.
5. **CPU bit-identity is the load-bearing per-session
   correctness guard**, not just CUDA self-parity. Phase 3 used
   self-parity because device drift was the failure mode being
   guarded against; in Phase 4 the failure mode is
   "optimisation accidentally changed a number," which CPU
   bit-identity catches more cleanly.
6. **Same `--seed 42` for every measurement.**
7. **No re-import of v1 trainer / policy / rollout / worker pool.**
   Phase 2 / 3 hard constraint inherited verbatim. v1 ms/tick
   numbers are the *target*; the v1 *code path* is not used.
8. **One fix per session, tested and committed before the next
   starts.** This is the user contract that defines this phase's
   workflow shape; it is the discipline that makes per-session
   bit-identity tests meaningful.
9. **Don't restructure the env shim** (`agents_v2/env_shim.py`)
   in this phase. If a session's finding traces overhead into
   the shim's `step` or `get_action_mask`, write it up as a
   Phase-4b candidate and stop — don't bundle it.
10. **PPO update path stays untouched** unless a session's
    deliverable explicitly requires changing the consumer side
    (Session 06 does; the others don't). Same correctness
    rationale: unrelated edits silently break the bit-identity
    test bar.

## Out of scope

- Multi-GPU, AMP / autocast, env vectorisation, multi-process
  workers — all inherited verbatim from throughput-fix's
  out-of-scope list. Phase 4 is single-process, single-GPU,
  bit-identity-preserving micro-optimisation.
- Architectural changes to the policy (head fusion that breaks
  init-time RNG consumption, switching from `nn.LSTM` to a
  manual cell, removing the per-runner value head, etc.). These
  are Phase-5 questions if ever.
- 66-agent scale-up (gated on this plan's verdict + any
  surviving GPU concurrency work).
- v1 deletion (still gated on rewrite-overall PASS).
- Reward-shape iteration (`no-betting-collapse` /
  `force-close-architecture` own those).
- Frontend-event throughput (queue drops on backpressure are
  by-design; not on the hot path).
- Throughput-fix's batched cohort path — the Session 02 shell
  in `BatchedRolloutCollector` stays opt-in (`--batched=False`
  default) and is not touched in this phase. Re-evaluate its
  necessity once Phase 4 completes.

## Useful pointers

- Per-tick rollout loop:
  [`training_v2/discrete_ppo/rollout.py`](../../../training_v2/discrete_ppo/rollout.py)
  lines 114–342.
- Per-runner attribution (Session 01 target):
  [`training_v2/discrete_ppo/rollout.py`](../../../training_v2/discrete_ppo/rollout.py)
  lines 346–426.
- Policy forward (Session 07+ target):
  [`agents_v2/discrete_policy.py`](../../../agents_v2/discrete_policy.py)
  lines 276–358.
- Transition definition:
  [`training_v2/discrete_ppo/transition.py`](../../../training_v2/discrete_ppo/transition.py).
- PPO update consumer:
  [`training_v2/discrete_ppo/trainer.py`](../../../training_v2/discrete_ppo/trainer.py)
  `_ppo_update` (search for `pack_hidden_states`).
- v1 ms/tick reference data:
  `logs/training/episodes.plan-A-diverged-20260422T055217Z.jsonl`
  (compute via successive `timestamp` deltas / `n_steps`).
- v2 ms/tick baseline data:
  `logs/discrete_ppo_v2/run_cpu_post_sync_fix.jsonl`,
  `logs/discrete_ppo_v2/run_cuda_post_sync_fix*.jsonl`.
- AMBER v2 cohort scoreboard (behavioural-drift floor):
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- CUDA self-parity test:
  [`tests/test_v2_gpu_parity.py`](../../../tests/test_v2_gpu_parity.py).
- Phase 3 GPU pathway baseline:
  `plans/rewrite/phase-3-cohort/findings.md` §"Session 01b".

## Estimate

Per session, including the per-session contract:

- Session 01: ~2.5 h (1 h refactor + 0.5 h tests + 0.5 h
  measurement + 0.5 h findings/commit). Highest-impact session;
  bias the time toward correctness tests over micro-optimisation.
- Sessions 02–05: ~1.5 h each. Mechanical, well-contained.
- Session 06: ~2 h (data-flow change touches the trainer's
  consumer side; PPO-update path needs verification).
- Sessions 07+: TBD after the read-through; ballpark ~1 h each
  for cached `neg_inf` and `masked_fill`, ~2 h for fused-heads
  if pursued.
- Session 99: ~1 h verdict writeup.

Total budget: **~12–14 h** for Sessions 01–06 + verdict, **~16
h** with all 07+ sessions. If past 4 h on any single session,
stop and check scope.
