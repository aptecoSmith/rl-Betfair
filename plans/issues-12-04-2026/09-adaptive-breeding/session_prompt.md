# Session: Adaptive Breeding & Mutation Controls

Read `CLAUDE.md` and `plans/issues-12-04-2026/09-adaptive-breeding/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

Mutation rate is fixed at 0.3 (`config.yaml` → `population.mutation_rate`),
passed through to `population_manager.py::breed()` at line 764. There's
no way to tune it per run and no adaptive behaviour.

When a whole generation underperforms, the system breeds from bad
survivors and hopes for the best. There's no detection or response
mechanism.

## Current flow

`run_training.py::_run_generation()` lines 685-733:
1. Score all agents via scoreboard
2. Filter to this run's agents (lines 688-692)
3. Select survivors via `population_manager.select(run_scores)`
4. Breed children via `population_manager.breed(selection, generation, mutation_rate)`
5. No check on generation quality between steps 2 and 3

`population_manager.py::breed()` line 833:
```python
child_hp, deltas = self.mutate(child_hp, mutation_rate, rng)
```

`mutation_rate` is always `config["population"]["mutation_rate"]` — no
per-generation variation.

## What to do

Insert a generation quality check between scoring (step 2) and
selection (step 3). Based on the configured policy, modify the
breeding parameters before passing to `select()` and `breed()`.

### Detection

```python
best_score = max(s.composite_score for s in run_scores) if run_scores else 0.0
is_bad = best_score < bad_generation_threshold
```

### Response

- **persist**: pass through unchanged
- **boost_mutation**: `effective_mutation = min(base + increment × consecutive_bad, cap)`
- **inject_top**: expand `run_scores` to include top garaged models
  as parent-only contributors (builds on issue 08 pattern)

### Wizard

The genetics step (step 4) currently shows n_elite, selection_rate,
and mutation_rate as read-only from config. Change this:

- Make mutation_rate an editable input (override for this run)
- Add adaptive mutation toggle + increment + cap
- Add bad generation policy selector
- Add threshold input

## Key files

| File | What to change |
|------|----------------|
| `config.yaml` | New population options |
| `training/run_training.py` | Generation quality check, policy dispatch |
| `agents/population_manager.py` | breed() already accepts mutation_rate — no change needed |
| `api/schemas.py` | New fields on StartTrainingRequest |
| `training/worker.py` | Pass new fields to orchestrator |
| `frontend/src/app/training-monitor/training-monitor.html` | Wizard mutation controls |
| `frontend/src/app/training-monitor/training-monitor.ts` | New form fields, pass to API |

## Constraints

- Default behaviour must be identical to current (persist, no adaptive
  mutation, 0.3 rate).
- Adaptive mutation state (consecutive bad count) resets per training
  run — not persisted across runs.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: adaptive breeding — bad generation detection, mutation controls, wizard UI`
Push: `git push all`
