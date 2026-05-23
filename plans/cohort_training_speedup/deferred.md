# cohort_training_speedup — deferred / not-done

Picked-up-cold checklist of what phase 3 (2026-05-23) measured but did
NOT ship. Each item lists what would land, the projected per-agent
savings against the post-phase-3 baseline (~21 min/agent), the
implementation cost, and the correctness risk.

## What DID ship in phase 3

- **Option A** — split-device (CPU rollout + CUDA PPO update). Measured
  31% reduction in `train_episode` wall. See [phase_3.md](phase_3.md)
  §"Option A".
- **Option F.1** — cohort-scoped `feature_cache` plumbed through
  `run_cohort` → `train_one_agent` → `_build_env_for_day` → `BetfairEnv`.
  Measured 63% reduction in cache-hit env builds.
- **Option B-lite** — lifted per-call assert in
  `feature_extractor.extract` from set-comparison every call to a
  once-per-instance check.

Per-agent: **~28 min → ~21 min** (measured on the smoke at
`registry/smoke_phase3_optA_1779536161/`).

---

## What did NOT ship

### Option B-big — vectorise `feature_extractor.extract` (deferred 2026-05-23)

`extract` runs 184k times per day-rollout. Each call builds a 30-entry
dict that is immediately re-keyed into a numpy array by
`compute_extended_obs`. Replace with an `extract_array` sibling that
writes into a pre-allocated array using module-level integer indices.

- **Projected savings**: ~3-4s/day = ~1.5-2 min/agent (~5% rollout)
- **Implementation cost**: ~6 hours (rewrite body, regression tests
  against the existing dict-API for byte-equality of float outputs)
- **Risk**: feature ordering drift could silently break the LightGBM
  scorer's expected input shape — load-bearing byte-equality test is
  the mitigation
- **Decision**: operator deferred 2026-05-23 — cost/benefit thin at
  this rollout-time share

### Option C — incremental bet aggregates (deferred 2026-05-23)

`BetManager.get_paired_positions / get_naked_exposure / get_positions`
walk `self.bets` from scratch on every env step. Measured cost is
0.42 ms/step (~5.6% of rollout) — meaningfully smaller than the
phase 2 estimate of 1+ ms/step. Maintain incremental aggregates on
`place_back` / `place_lay` / passive-fill / settle so reads are O(1).

- **Projected savings**: ~1.6 min/agent
- **Implementation cost**: ~8 hours including invariant-consistency
  asserts and tests
- **Risk**: bet-state invariants are subtle; a silent aggregate drift
  poisons risk gating and reward shaping with no obvious signal
- **Decision**: deferred — cost/benefit not there at the measured
  share. Re-evaluate if a future cohort grows the per-race bet count
  significantly (the cost scales with bets-per-race)

### Option F.2 — cache per-race predictor outputs (not measured, ~2026-05-23)

After F.1, residual env-build cost is ~6s/day dominated by
`_compute_race_predictor_outputs` and `_compute_tick_predictor_outputs`
(LightGBM + sklearn inference per race / per tick). The predictor
bundle holds its own per-(market_id) cache, so a second agent on the
same date may already get a partial speedup — needs measurement.

- **Projected savings (un-validated)**: ~1.5-2 min/agent if the
  predictor cache is per-process (not per-bundle-instance)
- **Implementation cost**: ~4-6 hours after measuring whether F.1's
  warm-build cost is dominated by predictor inference vs the
  `_static_obs` array assembly
- **Risk**: low — predictor outputs are pure functions of the inputs;
  cache invalidation is by market_id only
- **Decision**: deferred — needs a measurement session first to
  confirm whether the predictor-inference layer is actually the
  remaining bottleneck

### `--batched` cluster training (not investigated, biggest upside)

phase_2.md option E. Trains all same-`hidden_size` agents in a
generation through one env-pass each. The cohort runner already has a
batched branch (`training_v2/cohort/runner.py` § `if batched`).

- **Projected savings**: 2-5× rollout speedup if compatible with the
  current recipe
- **Known incompatibilities (from runner warnings)**:
  - `per_transition_credit=True` is ignored
  - `bc_pretrain_steps > 0` is ignored
  - Monitor-eval interaction unverified
- **Current cohort recipe**: uses neither incompatible flag → MIGHT
  be a drop-in upgrade
- **Implementation cost**: ~1-2 day spike (test 2-agent batched run,
  diff outputs vs sequential, verify monitor-eval still fires)
- **Risk**: PPO update ordering may differ subtly under batched,
  changing the gradient signal in a way the GA could amplify
- **Decision**: deferred until the post-phase-3 cohort has produced a
  clean baseline. Highest-priority follow-on if further speedup is
  wanted.

### PPO-update profile re-pass (not done, low expected upside)

PPO update measured at 7.6% of train_episode wall (~7s/day). At
~2.5 min/agent total there is little room. A focused profile of the
mini-batch loop (hidden-state slicing, aux-loss accumulation, per-runner
auxiliary BCE/NLL Python-side loops) might find ~1 min/agent.

- **Implementation cost**: 2-3 hours profile + analysis; some fraction
  ships as ~1 min/agent
- **Risk**: any change to the surrogate loss path is high risk —
  approx_kl regressions silently degrade learning
- **Decision**: deferred — share is too small to justify the risk

---

## Pre-launch checklist when picking this back up

Before launching the next cohort with the phase-3 changes in place:

1. Confirm tests pass: `pytest tests/test_discrete_ppo_trainer.py
   tests/test_v2_cohort_worker.py tests/test_v2_multi_day_train.py
   tests/test_v2_rollout_distributions.py tests/test_scorer_v1_dataset.py
   tests/test_env_shim_batched_scorer.py`
2. Confirm the split-device round-trip tests pass on CUDA:
   `pytest tests/test_discrete_ppo_trainer.py -m slow -k split_device`
3. Run the same 2-agent × 1-gen smoke and verify the
   "Feature cache hit for ..." log lines appear for the second agent.
4. Watch the first per-agent wall in the real cohort — expect
   **~21 min/agent**. If it lands at the pre-phase-3 ~28 min, something
   is wrong (likely `rollout_device` not being threaded — check the
   `Agent ...: cohort wall on first agent` line).
