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

## Session 05 — Training plan exploration strategy (2026-04-12)

**What landed:**
- `TrainingPlan` now has `exploration_strategy` (default `"random"`) and
  `manual_seed_point` fields. Both round-trip through JSON/persistence. Old
  plans without these fields default to random (backward compat).
- `TrainingOrchestrator._resolve_seed_point()` wires the strategy before
  `initialise_population()`:
  - `"random"` → returns None (legacy path).
  - `"sobol"` → calls `generate_sobol_points()` with skip = exploration run count.
  - `"coverage"` → calls `generate_coverage_seed()` from model history.
  - `"manual"` → uses `plan.manual_seed_point` directly.
  - All non-random strategies record in `exploration_runs` DB table.
- API `POST /api/training-plans` accepts `exploration_strategy` and
  `manual_seed_point`.
- 6 new tests: default strategy, round-trip, manual seed round-trip, old-plan
  backward compat, API strategy acceptance, API default strategy.

## Session 06 — Frontend strategy selector + coverage preview (2026-04-12)

**What landed:**
- `TrainingPlan` and `TrainingPlanPayload` TypeScript interfaces now include
  `exploration_strategy` and `manual_seed_point` fields.
- Editor state signal `editorExplorationStrategy` wired into `savePlan()`.
- Strategy selector in the training plan editor using the same chip-button
  pattern as architecture selection — four options: Random, Sobol (systematic),
  Coverage-biased, Manual. Each shows a contextual hint below.
- Plan list card shows non-random strategies as a badge.
- Detail view shows the strategy field.
- Frontend build passes cleanly.

## Session 07 — Coverage dashboard page (2026-04-12)

**What landed:**
- New `api/routers/exploration.py` with three endpoints:
  - `GET /api/exploration/history` — all exploration runs.
  - `GET /api/exploration/coverage` — per-gene coverage report with bucket
    counts, architecture breakdown, and poorly-covered gene list.
  - `GET /api/exploration/suggested-seed` — coverage-biased next seed point.
- New Angular page `/coverage` (`coverage-dashboard` component):
  - Summary strip: total models, gene coverage %, exploration runs, poorly
    covered gene count.
  - Architecture breakdown with under-covered highlighting.
  - Per-gene coverage histograms (green=covered, yellow=sparse, red=empty)
    in a responsive grid.
  - Exploration run history table with strategy badges.
  - Suggested next seed point display with JSON preview.
- Nav link "Coverage" added to app header.
- 4 new API tests: empty history, history with runs, coverage endpoint,
  suggested seed endpoint.
- Frontend build passes cleanly.
