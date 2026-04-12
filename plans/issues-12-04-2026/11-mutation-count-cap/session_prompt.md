# Session: Mutation Count Cap

Read `CLAUDE.md` and `plans/issues-12-04-2026/11-mutation-count-cap/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

`population_manager.py::mutate()` (lines 646-758) iterates every gene
spec and independently flips a coin at `mutation_rate` (0.3). With ~30
genes, this produces ~9 simultaneous mutations per child on average.

The change: when `max_mutations_per_child` is set, instead of per-gene
coin flips, randomly select N genes to mutate and only mutate those.

## Current code (lines 687-701)

```python
for spec in self.hp_specs:
    name = spec.name
    if name not in hp:
        hp[name] = _default_for_spec(spec)

    if name == "architecture_name" and arch_cooldown_in > 0:
        deltas[name] = None
        continue
    if rng.random() >= mutation_rate:
        deltas[name] = None
        continue

    old_val = hp[name]
    # ... apply mutation ...
```

## New logic when cap is set

```python
# Collect eligible genes
eligible = []
for spec in self.hp_specs:
    if spec.name == "architecture_name" and arch_cooldown_in > 0:
        continue
    eligible.append(spec)

# Select which genes to mutate
n_to_mutate = min(max_mutations, len(eligible))
chosen = set(s.name for s in rng.sample(eligible, n_to_mutate))

# Mutate only chosen genes
for spec in self.hp_specs:
    name = spec.name
    if name not in hp:
        hp[name] = _default_for_spec(spec)
    if name not in chosen:
        deltas[name] = None
        continue
    # ... apply mutation as before ...
```

Note: backfill of missing keys must still happen for all genes, even
those not mutated.

## Key files

| File | What to change |
|------|----------------|
| `config.yaml` | Add `population.max_mutations_per_child` |
| `agents/population_manager.py` | mutate() — cap logic |
| `training/run_training.py` | Pass cap through to breed/mutate |
| `api/schemas.py` | Add to StartTrainingRequest |
| `training/training_plan.py` | Add to TrainingPlan model |
| `frontend/src/app/training-monitor/` | Wizard input |
| `frontend/src/app/training-plans/` | Editor input |

## Constraints

- `max_mutations_per_child: null` (default) must produce identical
  behaviour to current code — per-gene coin flip at mutation_rate.
- Architecture cooldown must still be respected.
- Backfill of missing HP keys must still happen for all genes.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: max_mutations_per_child — cap simultaneous gene changes per breeding`
Push: `git push all`
