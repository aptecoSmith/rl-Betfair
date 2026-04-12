# Session: Breeding Pool Scope

Read `CLAUDE.md` and `plans/issues-12-04-2026/08-breeding-pool-scope/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

Breeding is currently scoped to the current run's agents only. The
scoping happens at `training/run_training.py:688-692`:

```python
run_ids = {a.model_id for a in agents}
run_scores = [s for s in scores if s.model_id in run_ids]
```

The scoreboard (`registry/scoreboard.py::rank_all()`) returns ALL
active + garaged models, but only the current run's subset is passed
to `select()` and then `breed()`.

`breed()` at `agents/population_manager.py:791`:
```python
n_children = self.population_size - len(survivors)
```

If survivors >= population_size, 0 children are bred.

## Key design question

When `include_garaged` or `full_registry` mode adds external models
to the breeding pool, should those models:

**(a)** Be parent-only — contribute hyperparameters to children but
don't occupy a slot in the next generation. `n_children` is always
`population_size - run_survivors` (ignoring external parents).

**(b)** Occupy survivor slots — if a garaged model outscores run
agents, it takes a slot, reducing children. More competitive but
means the run's own agents get squeezed out by external models.

Option (a) is safer and more predictable. Option (b) is more
biologically accurate but could lead to garaged models dominating
every generation.

## Key files

| File | What to change |
|------|----------------|
| `config.yaml` | Add `population.breeding_pool` option |
| `training/run_training.py:685-733` | Pool expansion logic |
| `agents/population_manager.py:512-555` | Selection may need awareness of external parents |
| `agents/population_manager.py:760-888` | Breed — ensure external parents contribute HP |
| `api/schemas.py` | Add breeding_pool to StartTrainingRequest |
| `frontend/src/app/training-monitor/` | Wizard UI for breeding pool option |

## Constraints

- `run_only` mode must produce identical behaviour to current code.
- Garaged models must not be re-trained — only their hyperparameters
  are used for crossover.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: configurable breeding pool scope (run_only/include_garaged/full_registry)`
Push: `git push all`
