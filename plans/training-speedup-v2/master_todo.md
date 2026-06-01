# training-speedup-v2 — master todo

Legend: `[ ]` todo · `[~]` in progress · `[x]` done. Each step states its
DELIVERABLE and its VALIDATION GATE. Nothing is "done" until its gate passes.

---

## Step 0 — Re-profile the real config  `[x]`  (DONE 2026-06-01 — see step0_profile.md)

Profile a single agent-train-day at the **production** config: full obs
(2254-d) + 3 predictor bundles + `--batched`, on a representative day
(~125 races / ~13.5k ticks).

- Multiprocess-aware: plain `cProfile` misses worker/child processes (which
  is where the CPU saturation lives). Use per-phase wall timers around
  {env build; per-tick: obs-build / policy-forward / env-step / matching /
  settle / MTM; PPO update}, or attach to the worker process directly.
- DELIVERABLE: a per-phase breakdown table that **supersedes** the Phase-3
  lean-obs numbers; the true hot path on OUR config; the policy-independent
  vs per-agent split.
- GATE: the breakdown reconciles with the measured 867s/agent-day (±15%).
  If it doesn't, the profile is wrong — find out why before proceeding.

**RESULT (`tools/profile_v2_batched_breakdown.py`, full detail in
`step0_profile.md`):**
- GATE PASSES: N=11 extrapolation of day-1 (~1040s) matches c1 log's
  measured day-1 cluster wall (1101s) within 6%; 867s is the cohort
  average over smaller days (1040 × 11k/13.5k ≈ 846s).
- Rollout ≈ 75% of cluster-day wall, env build ≈ 22%, PPO update ≈ 6%.
- Rollout split: **policy_forward 35% + sampling 15% + rng 2% ≈ 52%
  (batch=1 GPU, Step 3A lever)**; scorer_obs 13% + base_obs 5% ≈ 18.5%
  (Steps 2/3B); collector residual ~22%; **env-core matching+settle ≈ 3%
  (Step 3C is the SMALLEST lever)**.
- Forward is **kernel-launch-bound, not FLOP-bound** (hidden=128 ≈
  hidden=256 wall) → batch=N forward should amortise hard.
- **CORRECTNESS FINDING (HC#2):** the `--batched` path silently drops
  predictors, feature_cache, AND input_norm (on top of BC /
  per_transition_credit). c1+c2 ran predictor-less. Operator decision
  needed — see step0_profile.md + lessons_learnt.md.
- **Steering:** lead with 3A (true batch=N forward) + the cheap
  byte-identical feature_cache re-wire; then 2 + 3B; DEFER 3C.

---

## Step 1 — Bit-identical regression harness  `[ ]`  (the spine)

Capture golden streams from the **current** env and build the comparator
that every later step is gated on.

- Golden capture: for a battery of cases, dump per-tick `(obs, action,
  reward, value, hidden_state, bet-state snapshot, per-pair lifecycle,
  settle P&L)` to disk. Cases MUST cover: a normal day; force-close at
  T−N; naked settle; a multi-pair race; stop-loss fire; a predictor-gated
  day; an empty/all-refused-open day; ≥2 seeds; ≥2 hidden_sizes.
- Comparator: run a candidate env on identical inputs (same RNG, same policy
  weights) and assert per-quantity matches per hard-constraint #1 (exact on
  discrete, declared tolerance on continuous).
- DELIVERABLE: `tests/test_env_golden_parity.py` (+ golden fixtures under the
  plan dir or `tests/fixtures/`).
- GATE: (a) the current env passes against its own golden (sanity); (b) the
  harness CATCHES a deliberately-injected 1-tick perturbation (proves it
  actually discriminates).

**RESULT (DONE 2026-06-01 — both gates PASS, 9/9 tests).**
- Modules: `training_v2/golden_parity.py` (GoldenStream + capture + per-quantity
  comparator + npz/json (de)serialise), `training_v2/golden_cases.py` (7-case
  battery, predictors-ON, sliced days), `tools/gen_golden_fixtures.py`
  (regenerate + coverage validator). Fixtures in `tests/fixtures/golden/`.
- Battery covers every required class: normal / naked / multipair (base),
  force-close (13 fired), stop-loss (1 fired), predictor-gated (race-conf
  gating cut pairs 15→10), empty/all-refused (0 bets), seeds {1,2,3},
  hidden {64,128,256}. Predictors verified ON (race_conf=0.999 → 0 bets
  vs base → 17).
- GATE (a): current env reproduces all 7 fixtures bit-for-bit (discrete
  exact; continuous within declared per-quantity tol — GPU-touched 1e-4,
  CPU-env tight).
- GATE (b): an injected 1-tick matcher price shift (`ExchangeMatcher._match`)
  is caught end-to-end; control confirms no false alarm.
- Harness immediately earned its cost: first capture flagged `pair_id`
  (`uuid4().hex[:12]`) as nondeterministic → comparator now compares
  pairing STRUCTURE, not the opaque string. See lessons_learnt.md.

---

## Step 2 — Hot-path vectorization (low-risk)  `[ ]`

From Step 0's hot path, vectorize the worst pure-Python per-tick functions.

- Prime candidate (prior plan's deferred Option B-big): replace
  `feature_extractor.extract`'s 30-entry dict (built 184k×/day, immediately
  re-keyed into an array) with an `extract_array` writing a pre-allocated
  array via module-level indices. Plus any obs-build / position-aggregate
  hot spots Step 0 surfaces.
- Each change validated against Step 1 golden before commit.
- DELIVERABLE: vectorized hot-path functions + passing golden parity + a
  per-change measured speedup.
- GATE: bit-identical golden. Expected ~1.5-3× (prior projection), zero
  training-dynamics change.

**RESULT (DONE 2026-06-01 — `extract_array`, gated bit-identical).**
- `FeatureExtractor.extract` refactored to a single-source `_extract_into`
  (writes an indexed array); `extract` (dict, byte-identical) + new
  `extract_array` (writes a preallocated row) share it. `compute_extended_obs`
  now writes each runner-side row straight into a reused `(2N, n_feat)`
  float32 matrix — no per-call 30-key dict build, no `[d[name] for …]` re-key.
- GATES PASS: (1) byte-equality test `tests/test_extract_array_parity.py`
  (extract_array == dict re-key, NaN-inclusive, 50+ opportunities); (2) the
  golden harness — all 9 cases still reproduce bit-for-bit (the fixtures were
  captured on the OLD dict path, so this proves end-to-end byte-identity);
  (3) all 14 existing scorer tests pass.
- SPEEDUP (clean micro-bench, CPU): extract+rekey 15-19 us/call →
  extract_array ~10.6 us/call = **~32-44% faster** on that path,
  **~2-3.3 s/agent-day** (~2.5-4% of per-agent rollout). The whole-stack
  profiler was too noisy to isolate this (the untouched forward varied ±18%
  run-to-run) — micro-bench was required.

---

## Step 3 — Staged GPU vectorization (transformational)  `[ ]`

### 3A — True batched policy forward  `[~]`  FEASIBILITY SPIKE DONE 2026-06-01
Replace the per-agent **batch=1** forward loop in
`training_v2/discrete_ppo/batched_rollout.py` with a single **batch=N**
forward across active agents (stack obs/hidden/mask → one forward → scatter).
Re-evaluate Option A (CPU rollout): with a real batched forward, GPU rollout
may now win.
- GATE: identical actions/values vs the serial path on the harness (same
  RNG). Per-agent RNG ordering preserved, or its change justified + logged
  (HC #2).
- DELIVERABLE: batched-forward collector + parity + the new GPU-util number.

**FEASIBILITY SPIKE (verified, not guessed):**
- **vmap-over-LSTM STILL BLOCKED** on the installed torch 2.11.0+cu126:
  `RuntimeError: Batching rule not implemented for aten::lstm.input`. The
  collector docstring's assumption holds — the plan's "stack → one forward"
  cannot use `torch.func.vmap`.
- **Manual weight-stacking + bmm WORKS bit-identically**: stacking N agents'
  `W_ih/W_hh/b` and running the LSTM recurrence as `bmm` matched a per-agent
  `LSTMCell` loop to **1.49e-08**. So batch=N IS feasible — but it requires
  reimplementing the ENTIRE `DiscreteLSTMPolicy.forward` (input_proj, single
  LSTM step, actor + critic + 4 aux heads {fill/mature/risk/direction},
  input_norm) as stacked-over-N batched ops, bit-identical-gated. Large.
- **Approach fork for the operator** (different cost/risk):
  1. **Manual batch=N GPU forward** — uses the idle GPU on the saturated-CPU
     box; biggest lever (~52% of rollout) but the biggest single rewrite.
  2. **CPU-rollout (Option A) for the batched path** — trivial device change,
     but shifts onto the already-saturated CPU (likely wrong for a cohort;
     needs a measurement on the v2 full-obs path before trusting).
  3. **Parity-bring-up first** (feature_cache re-wire = ~14% wall byte-identical;
     predictors/input_norm/BC into the batched path = the folded-in correctness
     fix, gated against the predictors-ON golden) — lower-risk value while the
     forward-rewrite decision is made.
- **OPERATOR DECISION (2026-06-01):** check pytorch-update first → ruled out
  (structural, would break the golden); **go manual batch.** Precise gated
  build (Increments 1/2/3) in **`step3a_build_plan.md`**.

**RESULT (DONE 2026-06-01 — the achievable bit-identical wins landed + gated):**
- **KEY REALISATION:** a true batch=N GPU forward CANNOT be bit-identical on
  *discrete* quantities. The golden is captured on CPU; batching the forward
  (manual stacking/bmm or CUDA) reorders float reductions ~1e-6, which flips
  rare near-tie `Categorical.sample()` actions vs the CPU golden → violates
  HC#1 "discrete EXACT". The GPU-batch=N path is therefore a (subtle)
  dynamics change, NOT a bit-identical speedup. **Deferred** (documented in
  step3a_build_plan.md); the bit-identical lever is CPU rollout.
- **CPU rollout for the batched path** (`batched_worker.py`: `rollout_device`
  split + collector on CPU). Measured rollout **N=2: 110s CPU vs 138-154s
  CUDA (~20-28% faster)**; `collector_other` 55s→22s (no GPU sync/launch on
  sampling). **Bit-identical to the CPU golden** (same device).
- **feature_cache re-wire** into the batched path (was silently dropped).
  CUDA smoke: **8 "Feature cache hit" lines** (was 0). Byte-identical.
- **GATE:** `tests/test_env_golden_parity.py::test_batched_path_matches_golden_fixture`
  — the batched collector (N=1, CPU) reproduces the golden bit-for-bit on
  every load-bearing quantity. The only diff was the per-bet aux-head stamps
  (`fill/mature/direction_at_placement`) — a PRE-EXISTING batched-collector
  gap (it doesn't stamp them; solo does), orthogonal to these changes,
  excluded via `ignore_bet_fields` + recorded as a finding.
- Increment 3 (predictors/input_norm parity into batched) remains for a
  separate pass — it's a dynamics change (HC#8), not bundled into the speedup.

### 3B — Vectorize obs/market path across agents  `[~]`  DEFERRED w/ approach
Build the shared market-derived obs slice once per tick and broadcast;
vectorize the agent-derived slice across the N agents as tensors.
- GATE: bit-identical golden.

**DEFERRED (2026-06-01) — bit-identical-feasible, scoped, not built.**
Cleanest bit-identical win: a **cross-agent scorer cache**. The scorer
features appended by `compute_extended_obs` (the 13% `scorer_obs` phase)
are PURELY market-derived (the booster predicts on `FeatureExtractor`
runner features; NO bet/position input), so they are identical across all
cluster agents at the same `(race_idx, tick_idx)` — the batched collector
keeps active agents in lockstep, so agent 1 computes the scorer once and
agents 2..N reuse it (concatenated with their own per-agent `base_obs`).
Bit-identical by construction. NOT built this pass because: (a) it's
invasive — a shared cache threaded through the shims + `compute_extended_obs`
(the golden-gated hot path); (b) it needs an N≥2 batched gate (the N=1
`capture_golden_from_batched` doesn't exercise cross-agent sharing, so a
stale-cache bug could slip an N=1 gate); (c) the Step-2/3A wins already
landed the headline speedup. A clean follow-on, ~13% of rollout.

### 3C — Vectorize the env core  `[ ]`
- **FEASIBILITY SPIKE FIRST:** prototype matching + settlement as batched
  tensor ops on a slice; measure the realistic multiplier and the fraction
  of branching that resists vectorization. Go/no-go + realistic-ceiling
  decision comes from this spike — do not commit to the full rewrite blind.
- If go: N agents' env cores stepped as batched tensor ops, behind a
  construction-time fast-path switch (HC #6). The vectorized matcher is
  validated against the canonical single-level matcher as golden.
- GATE: bit-identical golden. This is where the harness earns its cost.

---

## Step 4 — Settle BC / per_transition_credit (compatibility decision)  `[ ]`

- ABLATION: BC-warm-start vs from-scratch, both on held-out (May 20-29), at
  equal compute. Preliminary record read (2026-06-01): BC's documented
  success was the supervised maturation-AUC probe (0.745), NOT a
  PPO-warm-start P&L win; the imitation-first BC policy lost £1513/7d; c1
  found good agents with BC silently off. Prior = "BC may not be
  load-bearing."
- DECISION: if the ablation shows BC helps → **wire it into the fast/batched
  path** (currently dropped). If not → formally retire it with a logged
  rationale + update the `bc-*` plan records.
- GATE: a decision recorded here + in EXPERIMENTS; no silent state either
  way (HC #2).

---

## Operator decisions (2026-06-01, post Step-0)

1. **Golden / speedup target config = predictors-ON (intended).** The
   Step-1 golden is captured from the CORRECT full config
   (race-outcome + direction predictors + input_norm + BC), NOT the
   predictors-OFF config c1/c2 actually ran. The 867s figure is the
   as-ran anchor only; the predictors-ON baseline is heavier (~43s env
   build + predictor inference) and is the real optimisation target.
2. **Fold batched feature-parity into this plan.** Bringing the
   `--batched` path to feature parity with `train_one_agent`
   (predictors, feature_cache, input_norm, BC) is part of Step 3A's
   HC#2 "no silent drop" bring-up, gated by the Step-1 harness. This
   couples the correctness fix to the speedup deliberately (a CORRECT
   dynamics change, harness-verified).

   Consequence for the gate: the **golden reference is the canonical
   SEQUENTIAL path** (`train_one_agent` / `RolloutCollector`, predictors
   ON) — it already wires every feature. The batched / vectorized fast
   paths must reproduce it. "Current env is golden" (HC#7) = the
   sequential predictors-ON env.

## Cross-cutting  `[ ]`

- Update `plans/EXPERIMENTS.md` with the Step-0 profile and each stage's
  measured speedup.
- **Correct the record:** the c1 / c2 EXPERIMENTS entries say "BC 500 steps"
  — it was a no-op under `--batched`. Annotate both 2026-05-31 / 06-01
  entries.
- Decide ordering with the operator: lead with the 3C env-core feasibility
  spike (to size the real ceiling before committing), or harness + 3A first
  and let the early wins inform the spike.
