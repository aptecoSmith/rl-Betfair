# Session 1: Mutation outcome analysis + directional bias

Read `CLAUDE.md` and `plans/issues-12-04-2026/10-directed-mutation/`
before starting. Follow session 1 of `master_todo.md`. Mark items done
as you go and update `progress.md` at the end.

## Context

The system already records detailed per-gene mutation data:

**BreedingRecord** (`population_manager.py:1085-1097`):
- `parent_a_hp`, `parent_b_hp`, `child_hp` — full HP dicts
- `inheritance: dict[str, str]` — which parent each gene came from
- `deltas: dict[str, float | None]` — per-gene mutation delta

**GeneticEventRecord** (`model_store.py:117-133`):
- Persisted to SQLite with parent IDs, mutation_delta, final_value

**Genetics log** (`population_manager.py:892-1034`):
- Human-readable per-child breakdown with parent scores

All the data is there. It's just not used to inform future mutations.

## Current mutate() logic

`population_manager.py:646-758`:
```python
def mutate(self, hp, mutation_rate=0.3, rng=None):
    for spec in self.hp_specs:
        if rng.random() >= mutation_rate:
            continue  # Skip this gene
        # For float/float_log: Gaussian noise, sigma = 10% of range
        sigma = (spec.max - spec.min) * 0.1
        delta = rng.gauss(0, sigma)  # ALWAYS centred at 0
        hp[name] = clamp(hp[name] + delta, spec.min, spec.max)
```

The key line is `rng.gauss(0, sigma)` — the mutation centre is always 0.
Directed mutation changes this to `rng.gauss(bias, sigma)` where bias
is learned from historical outcomes.

## What to build

### 1. MutationOutcome tracking

After each generation's evaluation, in `run_training.py`:
- Retrieve each child's score from the scoreboard
- Retrieve parent scores from the ranked scores list
- For each child with breeding records, compute:
  `score_delta = child_score - mean(parent_a_score, parent_b_score)`
- For each mutated gene (delta != None), record a MutationOutcome

### 2. Directional signal

Accumulate outcomes per gene across generations. After enough samples:
- Partition by sign of mutation_delta
- Compute mean score_delta per direction
- Signal = difference in means
- Confidence = min(n_positive_samples, n_negative_samples)

### 3. Biased mutation

When `directed_mutation=true` and signal is confident:
```python
bias = signal * bias_strength * sigma  # Scale bias relative to noise
delta = rng.gauss(bias, sigma)
```

This nudges mutations toward the historically successful direction
while retaining Gaussian noise for exploration.

## Key files

| File | What to change |
|------|----------------|
| `agents/population_manager.py` | mutate() bias, MutationHistory class |
| `training/run_training.py` | Outcome computation after evaluation |
| `config.yaml` | directed_mutation, bias_strength options |
| `frontend/src/app/training-monitor/` | Wizard toggle |

## Constraints

- `directed_mutation=false` (default) must produce identical behaviour
  to current code.
- Directional bias must not eliminate exploration — always retain
  Gaussian noise. The bias shifts the mean, it doesn't replace the
  distribution.
- Signals based on <5 samples per direction should not be acted on.
- MutationHistory resets per run (not across runs — the search space
  context changes between runs).
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: directed mutation — learn from mutation outcomes, bias toward successful directions`
Push: `git push all`
