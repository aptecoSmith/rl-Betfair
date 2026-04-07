# Session 11 — First real multi-generation exploration run

## Before you start — read these

- `../purpose.md`
- `../hard_constraints.md` — especially constraints 1-6 (reward and
  matching-engine correctness) and 9 (GPU only in dedicated sessions)
- `../master_todo.md` (you are Session 11)
- `../progress.md` — confirm Session 10 (housekeeping) landed cleanly
- `../lessons_learnt.md` + `../../arch-exploration/lessons_learnt.md`
- `../design_decisions.md`
- `../ui_additions.md`
- `../integration_testing.md` — GPU sanity checks run **before** the
  manual launch
- `../manual_testing_plan.md` — M2 (training launch) and M3 (post-run
  review) are both required here
- `../../arch-exploration/session_9_gpu_shakeout.md` — the shakeout
  that this session builds on. Read the "Invariants to verify in the
  post-run analysis" section. All five apply again, stricter.
- repo-root `CLAUDE.md`

## Goal

Run the first honest multi-generation genetic exploration with the
infrastructure built in the arch-exploration phase. Produce a
logged, reproducible result that can be used to prioritise every
subsequent session in this folder.

This is explicitly **not** a "find the winning architecture" run.
It is a "is the infrastructure giving us signal we can trust?" run.
Expectations for the actual P&L number should be low.

## Scope

**In scope:**

1. **Scope decisions, written down before launch.** Create a
   `TrainingPlan` via the UI (or the `api/routers/training_plans.py`
   backend) capturing:
   - **Dataset:** train days + held-out test days. Recommendation:
     15–20 train days, 3–5 held-out test days from the restored
     hot/cold data. Use a fixed date range so the run is
     reproducible.
   - **Population size:** 30 agents (10 per architecture minimum,
     matching the `min_arch_samples` default). If the dataset is
     small enough that 30 agents is unworkable in the GPU budget,
     drop to 21 (matching Session 9) and document the reason.
   - **Generations:** 5. Not "until convergence" — a fixed budget
     keeps the session scoped.
   - **Fitness metric:** mean held-out-day P&L across test days,
     tie-broken by Sharpe-like ratio (mean / std) across train days.
     Document the exact formula in the plan's notes field.
   - **Stop criteria:** generation count only. Do not add early
     stopping this session.
   - **Seed:** fixed RNG seed. Record it.
   - **Arch mix:** even split across `ppo_lstm_v1`,
     `ppo_time_lstm_v1`, `ppo_transformer_v1`. Use arch-specific
     LR range for the transformer (see `session_14` for widening
     this further later).
   - **Gene ranges:** current defaults. Do not tighten or widen in
     this session.

2. **Pre-flight integration tests.** Before the manual launch:
   - `pytest tests/ -m gpu` — GPU sanity checks on every architecture
     (forward + backward, no NaN).
   - Short training-loop smoke on a 1-day fixture (a handful of
     environment steps, not a full episode).
   - Coverage math check against a synthetic history.
   Record pass/fail in `progress.md` before starting the run.

3. **Launch the run.** Use the UI (preferred — exercises M2 manual
   test) or `scripts/*` if the UI path is blocked. Monitor for
   crashes, NaNs, and obvious failure modes in the first 5 minutes.
   If the first generation hasn't produced a complete episode after
   that window, abort and diagnose before continuing.

4. **Post-run analysis notebook / script.** A reusable artefact
   committed to `scripts/analysis/` or similar, that reads the run
   logs and produces:
   - **The Session 9 invariants**, repeated and tightened:
     1. Every gene in the schema was sampled with variance across
        the population.
     2. Per-agent reward genes produced measurably different env
        behaviour (correlate `reward_*` genes against observed
        per-episode metrics). Use Spearman, not Pearson, to avoid
        outlier-driven correlations.
     3. `raw + shaped ≈ total_reward` holds to floating-point
        tolerance across all episodes and all generations.
     4. Architecture coverage matches the plan across all
        generations (no arch died out between generations unless
        the elitism rules say it should have).
     5. No two agents produced bitwise-identical episode-1
        trajectories.
   - **Fitness breakdown by architecture:** mean / std / best /
     worst per arch per generation. Trends across generations.
   - **Gene vs fitness correlations:** Spearman ρ between every
     numeric gene and the fitness metric. Flag any |ρ| > 0.4.
   - **Anomaly list:** agents with NaN loss, constant action,
     all-same bet size, extreme drawdown, or zero bets placed.
     File each by agent id with a one-line diagnosis.

5. **Reprioritisation pass.** After reading the analysis output,
   update `next_steps.md` and `master_todo.md`:
   - Promote backlog items that the run gave concrete evidence for.
   - Leave items that the run gave no evidence for in the parking
     lot.
   - Add any **new** follow-ups surfaced by the run (regression
     bugs, surprising behaviours, knobs that turned out to matter
     more than expected).

**Out of scope:**

- Running Gen 6+ or extending the budget mid-run. If the 5-gen
  budget ends with "clearly needs more", that is a finding for
  the next session, not a reason to keep training.
- Tuning hyperparameter ranges mid-run based on Gen 1 results.
  That turns verification into a p-hacking exercise.
- Fixing any bugs discovered beyond "the run crashed and produced
  no data". Log them into a new backlog entry in `next_steps.md`
  for a focused follow-up session.
- Touching reward formulas, matcher logic, architectures, or the
  population manager. This session runs existing code.

## Tests to add

This session does not add unit tests. It runs the existing ones and
produces a reusable analysis script (which is a piece of
infrastructure, not a test).

Exception: if the run surfaces a reproducible bug that a CPU test
could have caught, write that test immediately and add it to
`tests/next_steps/` before continuing. The goal is to prevent the
same regression from sneaking back in.

## Manual tests

- **M2 (Training launch)** — required. Run via the UI if at all
  possible so the end-to-end dashboard update path is exercised.
- **M3 (Post-run review)** — required. Walk through every step.

## Session exit criteria

- `TrainingPlan` persisted in `registry/training_plans/` with all
  scope decisions captured in its notes field.
- Pre-flight integration tests passed; results recorded in
  `progress.md`.
- Run completed all 5 generations (or aborted cleanly with a
  recorded reason).
- Analysis script committed to the repo and rerunnable from a fresh
  clone against the persisted run logs.
- All five tightened invariants verified, with the analysis output
  captured (at least as a summary in `progress.md`).
- Anomaly list reviewed — any immediately actionable items filed
  into `next_steps.md`.
- M2 and M3 manual-test results recorded.
- `progress.md` Session 11 entry written.
- `lessons_learnt.md` updated with real-world findings — this will
  probably be the most valuable lessons entry in the whole folder.
- `master_todo.md` Session 11 ticked; Phase 3 items reprioritised.
- Do **not** commit log files or checkpoints. Those live in `logs/`
  and `registry/`, which should already be gitignored (verify).
- Commit the plan updates, analysis script, and documentation
  changes.

## Do not

- Do not extend the run budget because "it looks promising".
- Do not cherry-pick results or hide failures. Bad results are
  data; hiding them corrupts every downstream decision.
- Do not rerun with different seeds until results look good. One
  seeded run, honestly reported, is the whole point.
- Do not start writing Session 12+ prompts during this session.
  Reprioritisation is a file update, not a full prompt rewrite.
- Do not retune gene ranges in this session. If the analysis shows
  a range is badly chosen, that becomes a finding, not an in-session
  fix.
