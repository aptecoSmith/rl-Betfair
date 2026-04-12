# Session: Stud Models

Read `CLAUDE.md` and `plans/issues-12-04-2026/13-stud-models/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

Breeding currently only uses survivors from the current generation as
parents (`run_training.py:688-692`). There's no way to force a specific
model into the breeding pool.

The `breed()` method (`population_manager.py:760-888`) picks parents
by random sampling from survivors:
```python
if len(survivors) >= 2:
    parent_a_id, parent_b_id = rng.sample(survivors, 2)
```

Stud models need to be injected into this sampling as guaranteed
parents for a subset of children.

## Design

For each stud, reserve one breeding slot per generation:
- Parent A = stud model (forced)
- Parent B = random survivor (normal selection)
- Crossover + mutation applied as normal

This means with 2 studs and 20 breeding slots, 2 children always
have a stud parent, 18 are normal breeding.

Stud HP needs to be loaded from ModelStore since studs aren't in the
current generation's agent list:
```python
for stud_id in stud_model_ids:
    record = store.get_model(stud_id)
    stud_hp[stud_id] = record.hyperparameters
```

## Key files

| File | What to change |
|------|----------------|
| `api/schemas.py` | Add stud_model_ids to StartTrainingRequest |
| `training/run_training.py` | Load studs, inject into breeding |
| `agents/population_manager.py` | breed() — reserve stud slots |
| `training/training_plan.py` | Add stud_model_ids to plan model |
| `training/worker.py` | Pass studs to orchestrator |
| `frontend/src/app/training-monitor/` | Wizard stud picker |
| `frontend/src/app/training-plans/` | Editor stud picker |
| `frontend/src/app/services/api.service.ts` | Model list for picker |

## Constraints

- Empty stud list = current behaviour (backward compatible).
- Studs are NOT trained or evaluated — HP only.
- Studs don't reduce n_children — they're parent-only.
- Max 5 studs enforced in UI and validated in API.
- All stud IDs must exist in registry with valid HP and weights.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: stud models — guaranteed breeding parents per generation`
Push: `git push all`
