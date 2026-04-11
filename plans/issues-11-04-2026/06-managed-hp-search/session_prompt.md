# Managed Hyperparameter Search — All Sessions (01–07)

Work through sessions sequentially. Complete each session fully
(code + tests + progress.md entry) before moving to the next.
Commit after each session.

## Before you start — read these

- `plans/issues-11-04-2026/06-managed-hp-search/purpose.md` — the
  full design: two-phase exploration, seed points, coverage maps.
- `plans/issues-11-04-2026/06-managed-hp-search/hard_constraints.md`
  — non-negotiables.
- `plans/issues-11-04-2026/06-managed-hp-search/master_todo.md` —
  session breakdown with tests.

Also read the existing (unused) coverage infrastructure:
- `training/training_plan.py` — `compute_coverage()` (lines ~479),
  `bias_sampler()` (lines ~574), `sample_with_bias()`. These are
  built but never called from `initialise_population()`.
- `agents/population_manager.py` — `initialise_population()`,
  `sample_hyperparams()`, `parse_search_ranges()`.
- `config.yaml` — full gene definitions under
  `hyperparameters.search_ranges`.
- `registry/model_store.py` — existing tables and CRUD methods.

---

Detailed session prompts omitted — follow `master_todo.md` for
each session's scope, deliverables, and tests. The master_todo
has enough detail to implement each session without ambiguity.

## Cross-session rules

- Run `pytest tests/ -x` after each session. All tests must pass.
- Do not touch the genetic operators (crossover, mutation, selection).
  The seed controls *where* we start, not *how* we evolve.
- Do not change scoring, evaluation, or reward.
- The "random" strategy must always work — it's the fallback.
- Commit after each session with a clear message.
