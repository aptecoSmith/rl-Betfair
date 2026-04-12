# Progress — Managed Hyperparameter Search

One entry per completed session.

---

## Session 01 — Exploration log DB table (2026-04-12)

**What landed:**
- `ExplorationRunRecord` dataclass with fields: `id`, `run_id`, `created_at`,
  `seed_point` (JSON dict), `region_id`, `strategy`, `coverage_before` (JSON
  dict), `notes`.
- `exploration_runs` SQLite table in `_SCHEMA_SQL` with auto-increment PK,
  indexes on `run_id` and `strategy`.
- Three CRUD methods on `ModelStore`:
  - `record_exploration_run()` — insert with JSON serialisation, returns row id.
  - `get_exploration_history(strategy=None)` — list all or filter by strategy.
  - `get_exploration_run_count()` — total count (used later for Sobol skip).
- 7 new tests in `TestExplorationRuns`: insert/retrieve, JSON round-trip for
  seed_point and coverage_before, strategy filtering, run count, region_id +
  notes, ordering.

**Test results:** 1793 passed, 1 pre-existing failure (unrelated `mlp_layers`
key in `test_genetic_operators.py`), 12 skipped.

## Session 02 — Sobol seed point generator (2026-04-12)

**What landed:**
- `generate_sobol_points(hp_specs, n_points, skip)` in `training/training_plan.py`.
- Uses `torch.quasirandom.SobolEngine` (scrambled, seed=42) — no new dependency
  needed since torch is already present.
- `_unit_to_gene()` helper maps `[0,1]` to actual gene ranges:
  - `float`: linear interpolation.
  - `float_log`: log-space interpolation.
  - `int`: linear interpolation + round, clamped to bounds.
  - `int_choice` / `str_choice`: seeded uniform pick (deterministic per point
    index).
- `skip` parameter fast-forwards the Sobol engine so each training run gets
  a fresh point (wired to `get_exploration_run_count()` in Session 05).
- 4 new tests: bounds check, spacing quality, skip produces different points,
  round-trip validity.

## Session 03 — Coverage-biased seed generation (2026-04-12)

**What landed:**
- `generate_coverage_seed(hp_specs, history, seed)` in `training/training_plan.py`.
- Wires the existing `compute_coverage()` → `bias_sampler()` → `sample_with_bias()`
  pipeline into a single function that returns `(seed_point, CoverageReport)`.
- Handles all gene types: biased numeric genes via `sample_with_bias()`,
  well-covered numeric genes via vanilla sampling, choice genes via uniform pick.
- 3 new tests: empty history produces valid seed, clustered history biases away
  from cluster, CoverageReport is JSON-serialisable.

## Session 04 — Seed point → perturbed population (2026-04-12)

**What landed:**
- `perturb_from_seed(seed_point, specs, rng, sigma)` in
  `agents/population_manager.py` — Gaussian perturbation for numeric genes
  (clamped to bounds), adjacent-choice jump for choice genes.
- `initialise_population()` now accepts optional `seed_point` and `seed_sigma`
  parameters. When `seed_point` is provided, all agents are perturbed around it
  instead of being sampled uniformly. Backward compatible: no seed → random
  sampling as before.
- 5 new tests: bounds check, closeness to seed, spread proportional to sigma,
  seed_point population creates valid agents, no-seed-point unchanged.
