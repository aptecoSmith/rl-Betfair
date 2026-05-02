---
plan: rewrite/phase-3-followups/throughput-fix
status: design-locked
opened: 2026-05-02
depends_on: rewrite/phase-3-cohort (Sessions 01/01b GPU pathway, 2026-04-27;
            AMBER v2 baseline at 145 s/ep CUDA — Bar 4 FAIL documented but
            unaddressed)
---

# Throughput-fix follow-on — saturate the GPU

## Purpose

Phase 3's GPU pathway is wired correctly (Session 01 + 01b in
`plans/rewrite/phase-3-cohort/findings.md`) but does not actually
saturate the device. Single-agent CUDA runs at **~145 s/episode**
vs **~113 s/episode** on CPU — a **1.28× regression** on the very
device the rewrite was supposed to lean on. The first 12-agent
cohort took **3.1 h** (sequential, one agent at a time on one GPU);
extrapolating to 4 generations would have been ~13 h, which is
why the Phase 3 operator scoped back to one generation and shipped
AMBER on a thinner-than-planned baseline.

Phase 3 Session 04 named this gap and proposed
`plans/rewrite/phase-3-followups/throughput-fix/` as a separate
workstream:

> A SEPARATE workstream (parallel to (a)) addresses throughput:
> `plans/rewrite/phase-3-followups/throughput-fix/` — vectorised
> env / worker pool / batched LSTM forward, so future cohort runs
> take 90 min not 13 hours. Not in the critical path of the
> no-betting-collapse decision but required before any 66-agent
> scale-up.
> (`plans/rewrite/phase-3-cohort/findings.md` line 753–759)

The plan was named, scoped, and never written. This is that plan.

## Why now

Two consumers force the issue:

1. **`force-close-architecture` is consuming GPU sequentially.**
   Each cohort takes ~3.5 h. Sessions 01 + 02 + (potentially)
   stacked Session 03 = 7–11 h GPU. Throughput-fix shrinks that
   envelope; the mechanics-iteration cycle gets faster.
2. **66-agent scale-up is gated on this.** Phase 3's purpose.md
   lists scale-up as the Phase-4 entry. At 145 s/ep × 66 agents
   × 7 days × 4 generations = ~52 GPU-hours per cohort — not
   shippable. The rewrite cannot reach its endgame without this
   fix.

## What's locked

### The mechanics changes from `force-close-architecture` are NOT touched

This plan is purely about throughput. The pair-placement path,
close-leg path, force-close logic, equal-profit math, and reward
shaping are byte-identical pre/post each session of this plan.
Cross-cohort comparison against the AMBER v2 baseline
(`registry/v2_amber_v2_baseline_1777577990/`) and against any
future `force-close-architecture` baselines is the load-bearing
mechanism for both correctness AND speed verdicts; mechanics
changes from one plan must not silently piggyback on the other.

### Correctness is non-negotiable; speed is the goal

Every session of this plan ships with a CUDA-self-parity test
(bit-identical `total_reward` and `value_loss_mean` across two
runs at the same seed) BEFORE the speed measurement. Session 01b
of `phase-3-cohort` set the precedent: speed work that breaks
self-parity is not shipped, no matter how fast it makes the run.
The CPU/CUDA action-histogram band (±5 %) is the secondary guard
— if the band loosens we've changed the policy, not the runtime.

### The Phase-3 cohort protocol stays locked

Same `select_days(seed=42)` / 7+1 day split / 12 random-init
agents / 1 generation / no other shaping. Each speed-bar cohort
re-runs the AMBER v2 protocol so Δ-wall is measured against a
fixed baseline. **`--seed 42` is mandatory** for every cohort in
this plan; differing seeds invalidate the comparison.

### One mechanics layer per session

Sessions don't stack until each individually has a verdict.
This is the same constraint as `no-betting-collapse` and
`force-close-architecture`; stacked changes produce no per-layer
signal and a perf regression in any one layer can't be isolated.

### No env edits

The env (`env/betfair_env.py`, `env/bet_manager.py`,
`env/exchange_matcher.py`) is the slowest non-policy component
in the rollout; it dominates wall time after the GPU sync
barriers are cut. But touching the env breaks every other plan's
cross-cohort comparison floor and re-opens correctness questions
already-closed by Phase 0/1/2. **Throughput work happens in
`training_v2/` only**, not in `env/`. If env vectorisation turns
out to be the only path to saturation, that's a SEPARATE plan
with its own correctness re-validation phase.

### `cudnn.deterministic` stays ON

Yes, it costs us ~5–10 % vs `cudnn.benchmark=True`. No, we are
not turning it off in this plan. The CUDA↔CUDA self-parity bar
from `phase-3-cohort/findings.md` Session 01b is the load-bearing
correctness foundation for every other rewrite plan that
cross-compares cohort runs at fixed seeds. A speed plan that
silently breaks bit-reproducibility is the worst kind of debt.
A SEPARATE follow-on with its own parity-band reformulation
could turn `cudnn.benchmark` on; it's not this plan.

## Success bar

The plan ships GREEN iff at least one session produces:

1. **CUDA wall ≤ 90 s/episode** on single-agent (vs AMBER v2
   145 s/ep → 1.6× speedup, vs CPU 113 s/ep → 0.80× = CUDA
   finally beats CPU on this workload), AND
2. **CUDA↔CUDA self-parity holds** (bit-identical total_reward
   and value_loss_mean at fixed seed across two CUDA runs), AND
3. **CPU/CUDA action-histogram band stays ≤ ±5 %** on the same
   12-day eval slice as Phase 3 Session 01b.

The plan ships GREEN-with-stretch if a session ALSO produces:

4. **12-agent cohort wall ≤ 90 min** (vs AMBER v2 ~3.1 h → ~2×
   speedup), measured on the same protocol.

If single-agent goal (1) is hit but cohort goal (4) is not, the
plan is GREEN on Phase-3-followups blockers, AMBER on
Phase-4-readiness. Document and decide whether Session 02
(parallel cohort) is worth the complexity.

If neither single-agent nor cohort goals are hit after both
sessions: **RED**. The architecture's combined "single-agent
GPU + sequential cohort" shape is not viable for the rewrite's
endgame; Phase-4 scale-up is blocked pending a more fundamental
rethink (env-side vectorisation, multi-process workers across
multiple GPUs, or a CPU-only retreat).

## Sessions

### Session 01 — single-agent rollout sync removal

Cut the per-tick CUDA→CPU sync barriers in
`training_v2/discrete_ppo/rollout.py` that aren't structurally
required by the env step. The env runs on CPU and consumes
`action_idx` (int) and `stake_pounds` (float) every tick — those
two `.item()` calls are unavoidable. Everything else stored in
`Transition` (`log_prob_action`, `log_prob_stake`,
`value_per_runner`) does NOT have to materialise to CPU per-tick;
it can be deferred to end-of-episode batched transfer.

Hypothesis (from `plans/rewrite/phase-3-cohort/findings.md`
Session 01b §"Bar 4 — Speed (FAIL)"): the per-tick syncs are the
dominant cost on hidden_size ≤ 256, batch=1; cutting from
~5 syncs/tick to ~2 syncs/tick should halve the sync overhead.
Honest range: 10–40 % single-agent speedup. If the lower bound
holds, single-agent goal (1) doesn't trip alone — Session 02 is
needed to get the rest of the way.

Session prompt: `session_prompts/01_rollout_sync_removal.md`.

### Session 02 — parallel-cohort batched forward

If Session 01 alone doesn't hit the speed bar (likely), step up
to running multiple agents through one batched policy forward
per tick. **Each agent runs its own independent env**; the
forward pass batches them as `(N_agents, obs_dim)` instead of
`(1, obs_dim)`. LSTM(h=128, batch=N) at N=12 amortises kernel
launch overhead across agents — the regime where GPUs actually
beat CPUs. Per-tick env steps still happen sequentially in
Python (the env is not thread-safe), but the dominant cost
(forward pass + backward pass) becomes batched.

This is a `training_v2/cohort/` change: a new
`BatchedRolloutCollector` that holds N agents' obs / hidden
states / transition lists in parallel and steps each env after
a single batched forward. The PPO update path stays per-agent
(each agent has its own optimiser + trajectory); only the
rollout loop is batched.

Hard constraints on this session in particular:

- **Self-parity per agent.** Running agent A in a batch of 12
  must produce bit-identical transitions to running agent A
  alone. The batch dim is the only dimension that changes;
  the per-agent forward must select its own slice with no
  cross-agent leakage. This is the single most likely
  correctness bug.
- **Per-agent RNG independence.** Each agent's
  `Categorical.sample()` and `Beta.sample()` must draw from a
  generator seeded by its agent_id, not from the shared
  default generator. Otherwise sampling becomes order-dependent
  inside the batch.
- **Transition shapes match Session 01's.** The PPO update
  doesn't know batched-rollout exists; it consumes per-agent
  transition lists from the collector.

Session prompt: NOT YET WRITTEN. Scaffold once Session 01's
verdict is in. Prompt opens with the Session 01 actual measured
speedup as the gap-to-close.

### Session 03 — verdict + writeup

`plans/rewrite/phase-3-followups/throughput-fix/findings.md`
with the speedup table:

| Configuration | s/ep CUDA | s/ep CPU | Cohort wall (12 agents × 7 days) | Self-parity |
|---|---|---|---|---|
| AMBER v2 baseline | 145 | 113 | 11187 s | PASS |
| + Session 01 (sync removal) | ? | (n/a) | ? | ? |
| + Session 02 (batched forward) | ? | (n/a) | ? | ? |

Plus the success-bar verdict (GREEN / GREEN-with-stretch / RED)
and the next step (proceed to scale-up if GREEN; document
blocker if RED). Update
`plans/rewrite/README.md` Phase-3-followups status table.

Session prompt: NOT YET WRITTEN. Trivial — gated on Session 01/02.

## Deliverables

### Session 01

- `training_v2/discrete_ppo/rollout.py` — rollout loop emits
  device-resident tensors into the transition buffer; the
  CPU materialisation moves to a single end-of-episode
  pass (or, equivalently, lives on `Transition` as a deferred
  `.cpu()` view that the trainer's `_ppo_update` realises in
  bulk).
- `training_v2/discrete_ppo/transition.py` — `Transition`
  fields `log_prob_action`, `log_prob_stake`,
  `value_per_runner` accept torch tensors (CUDA or CPU) AND
  the existing numpy/float types for byte-compat with
  Sessions 01/01b. Or: a new `BatchedTransitionStore` holds
  the deferred device tensors and yields per-step records on
  demand.
- `tests/test_v2_rollout_sync.py` — three guards:
  1. `test_cuda_self_parity_after_sync_removal` — same seed,
     two CUDA runs → bit-identical `total_reward` and
     `value_loss_mean`. Same shape as Session 01b's bar 1.
  2. `test_action_idx_and_stake_unit_still_materialise_per_tick`
     — these two `.item()` calls remain in the rollout (they
     have to; the CPU env consumes them). Catches a regression
     where a refactor accidentally batches the env step too.
  3. `test_transition_log_probs_match_pre_plan` — running with
     the sync-removal flag OFF (or pre-plan code) and ON
     produces transitions whose log-prob fields are equal
     within fp32 epsilon.
- A 5-episode CUDA wall-time measurement on the same day used
  by `phase-3-cohort` Session 01b (2026-04-23) vs AMBER v2's
  145 s/ep. Logged in
  `plans/rewrite/phase-3-followups/throughput-fix/findings.md`.

### Session 02

(Skip for now; sketched in §"Sessions" above.)

### Session 03

- Findings table + verdict.
- `plans/rewrite/README.md` update.
- Decision on Phase-4 scale-up gate.

## Hard constraints

In addition to all rewrite hard constraints
(`plans/rewrite/README.md` §"Hard constraints"), phase-3-cohort
hard constraints, and inherited from `no-betting-collapse` and
`force-close-architecture`:

1. **No env edits.** Throughput work happens in `training_v2/`
   only. `env/betfair_env.py`, `env/bet_manager.py`,
   `env/exchange_matcher.py` are off-limits.
2. **No reward-shaping changes.** This plan does not touch
   `reward.*` config, the matured-arb bonus, naked-loss
   handling, mark-to-market shaping, or the open-cost shaping
   from `selective-open-shaping`. Cross-cohort comparison
   against AMBER v2 + any `force-close-architecture` baseline
   must be apples-to-apples on reward.
3. **No GA gene additions.** Same as every other rewrite
   plan; per-agent throughput tuning is not a gene.
4. **No `cudnn.benchmark = True`.** Determinism stays on. See
   §"What's locked" above.
5. **CUDA↔CUDA self-parity is the load-bearing correctness
   guard.** Every session ships with the parity test passing
   BEFORE the speed measurement. A 50 % speedup that breaks
   self-parity is not shipped.
6. **Same `--seed 42` for every cohort.** Cross-cohort wall-time
   comparison invariant.
7. **NEW output dirs for every cohort run.** Don't overwrite
   AMBER v2 or `force-close-architecture` baselines.
8. **No re-import of v1 trainer / policy / rollout.** Phase 2 /
   3 hard constraint inherited verbatim. The v1 GPU pinning
   pattern is read for reference, not imported.

## Out of scope

- Multi-GPU training (one machine, one GPU; multi-GPU is a
  Phase-5 question if it exists at all).
- AMP / autocast (Phase 3 ships fp32; mixed-precision is
  another correctness layer to validate, separate plan).
- Env vectorisation (would require touching `env/`; explicitly
  excluded — see hard constraint §1).
- 66-agent scale-up (gated on this plan's verdict).
- v1 deletion (gated on rewrite-overall PASS).
- Reward-shape iteration (`no-betting-collapse` /
  `force-close-architecture` own those).
- BC pretrain.
- New cohort-protocol design (the same protocol used by
  `phase-3-cohort` is reused verbatim).
- Frontend-event throughput (the websocket emitter is not on
  the hot path; queue drops on backpressure are tolerated by
  design).

## Phase-3-cohort hand-offs

From `plans/rewrite/phase-3-cohort/findings.md` Sessions 01/01b
(GPU pathway):

1. **AMBER v2 single-agent baseline:** 145 s/ep CUDA, 113 s/ep
   CPU. The 1.28× CUDA-slowdown is the gap to close.
2. **Per-tick sync barriers identified** in `rollout.py`
   (Session 01b §"Bar 4 — Speed"):
   - `action.item()` (line 191; STRUCTURALLY REQUIRED — env
     consumes int).
   - `stake_unit_t.item()` (line 204; STRUCTURALLY REQUIRED —
     env consumes float).
   - `out.action_dist.log_prob(action).item()` (line 193;
     DEFERRABLE).
   - `stake_dist.log_prob(stake_unit_t).item()` (line 217;
     DEFERRABLE).
   - `out.value_per_runner.detach().squeeze(0).cpu().numpy()`
     (lines 222–225; DEFERRABLE).
3. **CUDA↔CUDA self-parity test** (`tests/test_v2_gpu_parity.py`)
   is the load-bearing correctness foundation. Every change in
   this plan must keep it passing.
4. **`select_days(seed=42)` order is deterministic** per agent;
   re-running on the same seed against the same `data_dir` gives
   the same training-day order.

From `plans/rewrite/phase-3-followups/no-betting-collapse/`:

5. **AMBER v2 baseline cohort dir:**
   `registry/v2_amber_v2_baseline_1777577990/`. Comparison
   floor for any cohort-wall measurement in this plan.

## Useful pointers

- Per-tick sync sites:
  [`training_v2/discrete_ppo/rollout.py`](../../../../training_v2/discrete_ppo/rollout.py)
  lines 190–225.
- v1 GPU pinning reference (read, do not import):
  [`agents/ppo_trainer.py`](../../../../agents/ppo_trainer.py)
  lines 2131–2174.
- v1 obs buffer reuse:
  [`agents/ppo_trainer.py`](../../../../agents/ppo_trainer.py)
  lines 1384–1390.
- Phase 3 CUDA pathway baseline:
  `plans/rewrite/phase-3-cohort/findings.md` §"Session 01b".
- AMBER v2 cohort scoreboard:
  `registry/v2_amber_v2_baseline_1777577990/scoreboard.jsonl`.
- CUDA self-parity test:
  [`tests/test_v2_gpu_parity.py`](../../../../tests/test_v2_gpu_parity.py).
- Transition definition:
  [`training_v2/discrete_ppo/transition.py`](../../../../training_v2/discrete_ppo/transition.py).

## Estimate

Per session:

- Session 01: ~3 h (1.5 h refactor + 0.5 h tests + 1 h
  measurement run on 5-episode wall × 2 for self-parity).
- Session 02: ~6 h (4 h refactor — batched collector is a real
  refactor, plus per-agent RNG plumbing — + 1 h tests + 1 h
  cohort run if measurement targets the cohort wall).
- Session 03: ~1 h writeup.

Best case (Session 01 GREEN alone): ~4 h.
Worst case (Session 01 + 02, neither alone hits): ~10 h +
verdict.

If past 5 h on Session 01 excluding cohort-wall measurement,
stop and check scope — something other than the rollout loop
is taking the time.
