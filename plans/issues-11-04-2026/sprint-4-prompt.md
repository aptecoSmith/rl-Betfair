# Sprint 4 — Managed Hyperparameter Search (7 sessions)

The largest piece of work. Systematic exploration of the hyperparameter
space with tracking, coverage analysis, and a dashboard.

## Before you start

- Verify Sprints 1–3 landed.
- **Important:** This sprint benefits from having multiple training runs
  already in the registry. The coverage analysis needs historical models
  to identify gaps. If you've only done 1–2 training runs, consider
  doing a few more first — the coverage dashboard will be more useful
  with data.
- Read `plans/issues-11-04-2026/order.md` for context.

## Issue 06 — Managed Hyperparameter Search (7 sessions)

Read the full plan folder: `plans/issues-11-04-2026/06-managed-hp-search/`

Start with `purpose.md` (the full design: Sobol sequences, coverage-biased
sampling, exploration log, two-phase strategy), then `hard_constraints.md`,
then work through `session_prompt.md`. Detailed session breakdowns are in
`master_todo.md`.

### Phase 1 — Infrastructure (sessions 01–04)
1. Exploration log DB table (`exploration_runs`)
2. Sobol seed point generator (quasi-random, low-discrepancy)
3. Coverage-biased seed generation (wire existing `compute_coverage()` +
   `bias_sampler()` — these are already built but never called)
4. Seed point → perturbed population (cluster initial models around seed)

### Phase 2 — Orchestration (sessions 05–06)
5. Training plan exploration strategy (random/sobol/coverage/manual)
6. Frontend strategy selector + coverage preview

### Phase 3 — Visualisation (session 07)
7. Coverage dashboard page (heat maps, seed history, suggested next point)

Key insight from analysis: `training/training_plan.py` already has ~170
lines of coverage analysis and biased sampling code that's built but
never wired into `initialise_population()`. Sessions 03–04 are mostly
integration, not greenfield.

No ai-betfair knock-on.

**Exit per session:** All tests pass, `progress.md` updated, commit.

---

## Sprint complete

After all seven sessions:
1. Full test suite green.
2. Push: `git push origin master`.
3. Run a training session using the "Sobol" strategy — verify it picks
   a different seed point from previous runs.
4. Check the coverage dashboard shows explored vs unexplored regions.
5. Run a second training session using "Coverage-biased" — verify it
   targets the gaps.
