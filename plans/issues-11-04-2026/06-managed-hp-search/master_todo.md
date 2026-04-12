# Master TODO — Managed Hyperparameter Search

Ordered session list. Tick boxes as sessions land.

When a session completes:
1. Tick its box here.
2. Add an entry to `progress.md`.
3. Append any learnings to `lessons_learnt.md`.

---

## Phase 1 — Infrastructure: tracking + seeding

- [x] **Session 01 — Exploration log DB table**

  - `registry/model_store.py` — add `exploration_runs` table:
    `id`, `run_id`, `created_at`, `seed_point` (JSON),
    `region_id`, `strategy`, `coverage_before` (JSON), `notes`.
  - CRUD methods: `record_exploration_run()`,
    `get_exploration_history()`, `get_unused_seed_points()`.
  - Schema migration: create table if not exists (same pattern
    as existing tables).

  **Tests:**
  - Insert and retrieve an exploration run.
  - Round-trip JSON seed_point and coverage_before.

- [x] **Session 02 — Sobol seed point generator**

  - New module or add to `training/training_plan.py`:
    `generate_sobol_points(hp_specs, n_points, skip=0)`.
  - Uses `scipy.stats.qmc.Sobol` (or a lightweight implementation)
    to generate quasi-random points across the full search space.
  - Maps Sobol unit hypercube [0,1]^d to actual gene ranges:
    - `float`: linear scale min→max.
    - `float_log`: log scale.
    - `int`: discretise to nearest int in range.
    - `int_choice` / `str_choice`: map to nearest choice index.
  - `skip` parameter lets us advance past already-used points.

  **Tests:**
  - Generate 10 Sobol points → all within bounds.
  - Points are well-spaced (pairwise distances > threshold).
  - Skip=5 produces different points from skip=0.
  - Round-trip: Sobol point → gene dict → valid hyperparameters.

- [x] **Session 03 — Coverage-biased seed generation**

  Wire existing `compute_coverage()` and `bias_sampler()` into a
  unified seed generator:

  - `generate_coverage_seed(hp_specs, model_store)` →
    computes coverage from all historical models, uses
    `bias_sampler()` to generate a single seed point that targets
    the largest gap.
  - Record the `CoverageReport` snapshot in the exploration log.

  **Tests:**
  - With empty model history → seed is uniformly random.
  - With 50 models all in one region → seed is biased away.
  - Coverage report is serialisable to JSON.

- [x] **Session 04 — Seed point → perturbed population**

  - `agents/population_manager.py` — add
    `initialise_from_seed(seed_point, population_size, sigma=0.1)`:
    - For each model in population, perturb each gene by ±sigma
      of its range around the seed value.
    - `float` / `float_log`: Gaussian noise, clamped to bounds.
    - `int`: ±1 from seed (or stay at seed).
    - `int_choice` / `str_choice`: mostly stay at seed, small
      probability of jumping to adjacent choice.
  - Update `initialise_population()` to accept an optional
    `seed_point` parameter. If provided, use
    `initialise_from_seed()` instead of random sampling.

  **Tests:**
  - Population from seed → all models are close to seed point.
  - No model is outside valid bounds.
  - Spread is proportional to sigma.
  - Backward compat: no seed → random sampling (existing tests pass).

## Phase 2 — Orchestration: strategy selection

- [x] **Session 05 — Training plan exploration strategy**

  - `training/training_plan.py` — add `exploration_strategy` field
    to `TrainingPlan`: `"random"` (default), `"sobol"`, `"coverage"`,
    `"manual"`.
  - `training/run_training.py` — before `initialise_population()`,
    check the plan's strategy:
    - `"random"` → current behaviour.
    - `"sobol"` → call `generate_sobol_points()`, pick the next
      unused point (query exploration log), record it.
    - `"coverage"` → call `generate_coverage_seed()`, record it.
    - `"manual"` → use seed_point from the plan.
  - `api/routers/training.py` — accept `exploration_strategy` and
    optional `seed_point` in plan creation.

  **Tests:**
  - Plan with strategy=sobol → population seeded from Sobol point.
  - Plan with strategy=coverage → population biased to gaps.
  - Plan with strategy=random → unchanged.
  - Exploration run recorded in DB for all strategies.

- [x] **Session 06 — Frontend: strategy selector + coverage preview**

  - Training plan form — dropdown for exploration strategy:
    "Random", "Sobol (systematic)", "Coverage-biased (adaptive)",
    "Manual".
  - When Sobol selected: show "Point N of M" and a preview of the
    seed hyperparameters.
  - When Coverage selected: show the current coverage report
    summary (% covered, worst gaps).
  - When Manual selected: show a form to input seed hyperparameters
    (or load from a high-scoring model).

  **Tests:**
  - Strategy dropdown renders all options.
  - Sobol preview shows valid hyperparameters.
  - Coverage preview loads from API.

## Phase 3 — Visualisation: coverage dashboard

- [x] **Session 07 — Coverage dashboard page**

  New Angular page: `/coverage`

  - **Heat map per gene:** histogram showing explored vs unexplored
    buckets. Colour: green (well-explored), yellow (sparse), red
    (empty).
  - **Seed point history:** table of all exploration runs with
    strategy, region_id, and the best composite_score achieved.
  - **Architecture breakdown:** bar chart of model counts per
    architecture.
  - **Suggested next point:** auto-computed from coverage gaps.

  - API endpoint: `GET /api/exploration/history` and
    `GET /api/exploration/suggested-seed`.

  **Tests:**
  - Coverage dashboard loads with data.
  - Suggested seed endpoint returns valid hyperparameters.

---

## Summary

| Session | What | Phase |
|---------|------|-------|
| 01 | Exploration log DB table | 1 |
| 02 | Sobol seed point generator | 1 |
| 03 | Coverage-biased seed generation | 1 |
| 04 | Seed point → perturbed population | 1 |
| 05 | Training plan strategy selection | 2 |
| 06 | Frontend strategy selector | 2 |
| 07 | Coverage dashboard | 3 |

Total: 7 sessions. Phase 1 (sessions 1-4) builds the machinery.
Phase 2 (sessions 5-6) wires it into the workflow. Phase 3
(session 7) gives visibility into exploration progress.
