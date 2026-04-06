# Session 9 — Full Gen-0 GPU shakeout

## Before you start — read these

- `plans/arch-exploration/purpose.md`
- `plans/arch-exploration/master_todo.md` (Session 9 — the final one)
- `plans/arch-exploration/testing.md` — **this is the ONE session
  where GPU tests are allowed.**
- `plans/arch-exploration/progress.md` — confirm Sessions 1-8 done.
- `plans/arch-exploration/lessons_learnt.md` — read it all before
  launching. Every prior session's gotchas matter here.
- `plans/arch-exploration/ui_additions.md` — everything should be
  ticked off.
- Repo root `CLAUDE.md` — invariants still apply.

## Goal

Run a full Gen-0 population under the new planner, with all three
architectures active, all new genes live, and verify that:

1. Training doesn't crash.
2. Coverage of architectures and hyperparam buckets matches what
   the planner intended.
3. `info["raw_pnl_reward"] + info["shaped_bonus"] ≈ total_reward`
   holds across all episodes.
4. Per-agent reward shaping genes actually produced measurably
   different training trajectories (i.e. Session 1's bug did not
   regress).
5. Nothing catastrophically broken in the transformer architecture
   (no NaNs, no exploding gradients, no all-same-action collapse).

This is a shakeout, not a performance run. We are not expecting
positive P&L. We are verifying that the exploration infrastructure
produces diverse, honest agents.

## Scope

**In scope:**
- Create a `TrainingPlan` targeting:
  - Population size 21 (7 per architecture as a minimum coverage
    target).
  - All new genes active, using the ranges defined in Sessions 2-6.
  - Arch-specific LR range for transformer if Session 6's design
    chose one.
  - Fixed RNG seed so the run is reproducible.
- Launch the Gen-0 training run via the new planner path. This is
  the first GPU training of this phase.
- Monitor for crashes, NaNs, and obvious failure modes.
- Collect per-episode logs and validate invariants.
- Analysis notebook / script that:
  - Confirms architecture coverage matches the plan.
  - Confirms reward-shape gene values correlate with observed
    training signal differences (e.g. high
    `reward_efficiency_penalty` agents should have lower bet counts
    on average).
  - Checks `raw + shaped ≈ total_reward` across all episodes.
  - Surfaces any agents that diverged (NaN loss, all-zero actions,
    etc.).

**Out of scope:**
- Running Gen 1, Gen 2, or any evolution beyond the single Gen 0
  shakeout. This session is verification that the infrastructure
  works end-to-end. Running the actual exploration is a follow-up
  session after we've read the results.
- Tuning hyperparameter ranges based on Gen-0 results. That is a
  separate conversation — do not close the loop in this session by
  silently re-scoping.
- Fixing any discovered bugs beyond the obvious (NaN crash, env
  raises an exception, etc.). File issues into a fresh session
  prompt instead of hotfixing here.

## Tests allowed

- Everything that was CPU-only in earlier sessions is still
  expected to pass before launching the GPU run.
- A `@pytest.mark.gpu` smoke test that instantiates each
  architecture on CUDA, does one forward + one backward pass, and
  asserts no NaNs. Quick sanity check, not a training loop.
- The actual Gen-0 training run itself. Budget this — if it's
  going to take more than 4 hours, cut the dataset to 5-10 days
  instead of the full range. The goal is verification, not
  production training.

## Invariants to verify in the post-run analysis

1. **Every gene in the schema was sampled with variance across the
   population.** No gene should have all 21 agents with identical
   values. (Catches accidentally-still-hardcoded values.)
2. **Per-agent reward genes produced different env behaviour.**
   Correlate `reward_efficiency_penalty` vs observed per-episode
   `bet_count`. A negative correlation is expected; a zero
   correlation is a bug.
3. **Raw+shaped invariant holds everywhere.** Sum across episodes
   and assert max absolute discrepancy is below floating-point
   tolerance × episode count.
4. **Architecture coverage was honoured.** 7 agents per arch, no
   arch collapsed to zero agents.
5. **No agent's episode-1 reward is literally identical to another
   agent's episode-1 reward** (given different seeds). If two
   agents produced bitwise-identical trajectories, there's a hidden
   determinism bug.

## Session exit criteria

- Training run completed without crashes (or with crashes captured
  and documented).
- All five post-run invariants verified.
- `progress.md` Session 9 entry containing:
  - Dataset used (days, markets)
  - Final population stats (mean reward, arch breakdown)
  - Whether invariants held
  - Links to the relevant log files under `logs/training/`
- `lessons_learnt.md` — this is the single most valuable update in
  the whole plan. Everything you discover about how the real system
  differs from the CPU-test idealisation goes here.
- `master_todo.md` — mark Session 9 complete.
- Do NOT commit generated log files or checkpoints. Those go into
  `logs/` and `registry/` which are already gitignored (verify).
- Commit the plan changes (session updates, progress, lessons).

## Do not

- Do not tune anything in this session based on the results. The
  whole point is to verify infrastructure; re-scoping on the fly
  produces false-positive results and wasted GPU hours.
- Do not silently re-run with different seeds until results look
  good. Log failures honestly. A shakeout that fails is a successful
  shakeout — it told you something the CPU tests could not.
- Do not delete bad agents or cherry-pick results. The planner's
  history should contain the full run, warts and all, so future
  coverage checks can use it.
