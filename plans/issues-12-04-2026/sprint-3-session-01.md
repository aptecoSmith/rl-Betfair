# Sprint 3, Session 1: Mutation Count Cap + Breeding Pool Scope (Issues 11 + 08)

Two tightly-related GA improvements in one session. Read `CLAUDE.md`
first, then the issue folders listed below.

---

## Part 1: Mutation Count Cap (Issue 11)

Read `plans/issues-12-04-2026/11-mutation-count-cap/` for full context.

### Summary

Add `max_mutations_per_child` config option. Instead of each gene
independently flipping a 30% coin (~9 mutations average), randomly
select at most N genes to mutate per child.

### Key change

In `population_manager.py::mutate()`:
- When cap is set: collect eligible genes, sample N, mutate only those
- When cap is null: current coin-flip behaviour (backward compatible)

### Key files

- `config.yaml` — add `population.max_mutations_per_child`
- `agents/population_manager.py` — mutate() cap logic
- `frontend/src/app/training-monitor/` — wizard input
- `frontend/src/app/training-plans/` — editor input

---

## Part 2: Breeding Pool Scope (Issue 08)

Read `plans/issues-12-04-2026/08-breeding-pool-scope/` for full context.

### Summary

Add `breeding_pool` config option: `run_only` (default, current) /
`include_garaged` / `full_registry`. Controls whether external models
can enter the breeding pool as parent-only contributors.

### Key change

In `run_training.py::_run_generation()`:
- `run_only`: keep current scoping at lines 688-692
- `include_garaged`: expand `run_scores` to include garaged model scores
- `full_registry`: use unfiltered scoreboard scores

External models are parent-only — they don't take survivor slots.

### Key files

- `config.yaml` — add `population.breeding_pool`
- `training/run_training.py` — pool expansion logic
- `agents/population_manager.py` — breed() awareness of external parents
- `frontend/src/app/training-monitor/` — wizard selector

---

## Commits

Two separate commits:
1. `feat: max_mutations_per_child — cap simultaneous gene changes per breeding`
2. `feat: configurable breeding pool scope (run_only/include_garaged/full_registry)`

Push: `git push all`

## Verify

- `python -m pytest tests/ --timeout=120 -q` — all green
- `cd frontend && ng build` — clean
