# Phase 6 — A/B validation runbook (gauntlet vs lockstep)

The cutover gate (`hard_constraints.md` §"Migration safety"): **no cutover on
faith.** Run `--breeding gauntlet` and `--breeding lockstep` on the SAME data +
seed budget, judge both on the sealed-7 held-out (fc=0 select + fc=120 deploy),
and adopt the gauntlet ONLY if its held-out **locked / σ_naked_leg** matches or
beats lockstep. Also confirm the operational wins (uniform per-run wall, flat
memory).

This is a multi-hour compute run — launch detached + logged, monitor, decide.

## Matched configs

Both arms share: same `--seed`, same `--holdout-recent 7`
(sealed final-test), same `--validation-holdout-recent V` (fixed fc=0 select
set), same predictors (race-outcome + direction), `--parallel-agents 16
--device cpu`, `--enable-all-genes`, same tranche sizing
(`--pbt-train-per-rotation` / `--pbt-eval-per-rotation`), same `--n-agents`.

The ONLY difference is `--breeding gauntlet` vs `--breeding lockstep`.

Both arms pass `--gpu-policy-lane --gpu-lane-max-concurrent 2`: the env + LSTMs +
small transformers run CPU (batch=1 forward is launch-bound — `project_gpu_speedup_
decision`), but big-context transformers (`transformer_ctx_ticks >= 128`) route
their forward + PPO update to CUDA. Verified the gauntlet path threads the lane
through identically to lockstep (`run_cohort` → `_run_gauntlet_breeding` →
`TrancheExecConfig` → `_build_spec` → worker `resolve_gpu_lane`). Leaving the lane
off would unfairly exclude big transformers from BOTH tournaments.

Seed budget parity note: lockstep runs `n_generations == n_tranches`; the
gauntlet runs 1 full climb + `--generations - 1` breed rounds. To match
"selection rounds", set the gauntlet's `--generations` so its breed-round count
≈ lockstep's tranche-boundary count, OR compare at equal total agent-trains
(read `total agents trained` from each run's log). Record whichever parity basis
is used.

## Launch (see tick-tock/launch_gauntlet_ab_*.sh)

    bash tick-tock/launch_gauntlet_ab_gauntlet.sh   # arm A
    bash tick-tock/launch_gauntlet_ab_lockstep.sh   # arm B  (can run after A)

Run sequentially (each wants all 16 cores) unless the box can split.

## Judge (the great equalizer — same sealed days, same metric)

    python -m tools.cross_era_holdout_board \
        --era-dir registry/gauntlet_ab_gauntlet \
        --era-dir registry/gauntlet_ab_lockstep \
        --holdout-recent 7 --top-n 16 --rank-by locked_over_sigma \
        --device cpu --argmax-eval \
        --reeval-arg=--use-race-outcome-predictor \
        --reeval-arg=--use-direction-predictor \
        --reeval-arg=--predictor-bundle-manifests \
        --reeval-arg=../betfair-predictors/production/race-outcome/manifest.json \
        --reeval-arg=../betfair-predictors/production/race-outcome-ranker/manifest.json \
        --reeval-arg=../betfair-predictors/production/direction-predictor/manifest.json \
        --output registry/gauntlet_ab_board
        # NOTE: --reeval-arg=VALUE (with '=') is REQUIRED for values starting
        # with '--' (argparse rejects "--reeval-arg --flag" as a missing value).

Then `tools/reevaluate_cohort.py … --reward-overrides
force_close_before_off_seconds=120` on each era's champions for the fc=120
deploy estimate (per `project_force_close_train_vs_deploy`).

## Decision criteria (record in findings.md, Intention/Implementation/Result)

1. **Held-out quality (the gate):** gauntlet's top-of-board held-out
   `locked / σ_naked_leg` ≥ lockstep's, with σ_naked_leg ≤ £30 on the winner.
   fc=120 net day_pnl not worse.
2. **Operational wins:** per-tranche wall is uniform (no heavy final-generation
   tail — grep each run's per-run wall from the log); peak RAM flat across the
   run (the gauntlet's whole point). 
3. **Recipe purity intact:** spot-check that re-running a reported gauntlet
   champion's recipe clean from T1 reproduces it (the `.genehash` chain + the
   ledger lineage make this checkable).

Cut over to `--breeding gauntlet` ONLY if (1) holds. Else iterate; keep
`--breeding lockstep` as the production path.
