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
