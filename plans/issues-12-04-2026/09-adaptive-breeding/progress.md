# Progress — Adaptive Breeding

## Session 1 — 2026-04-14

### Backend
- `config.yaml` — `bad_generation_threshold` (0.0 = disabled),
  `bad_generation_policy` (persist|boost_mutation|inject_top),
  `adaptive_mutation`, `adaptive_mutation_increment`,
  `adaptive_mutation_cap`. Defaults preserve current behaviour.
- `api/schemas.py` — adds `mutation_rate`, the four adaptive fields
  and `bad_generation_threshold` / `_policy` to `StartTrainingRequest`.
- `api/routers/training.py` — validates ranges + policy enum, forwards
  to worker, and replays plan values on `resume`.
- `api/routers/training_plans.py` — accepts the same fields on plan
  create.
- `training/training_plan.py` — fields persisted in (de)serialisation.
- `training/ipc.py` — `make_start_cmd` carries the new fields.
- `training/worker.py::_apply_run_overrides` — layers run-level values
  onto the population config (deep-copy preserved).
- `training/run_training.py::_run_generation` — bad-gen detection
  (`max(composite) < threshold`), adaptive ramp via
  `_consecutive_bad_gens`, policy dispatch:
  - `persist`: no change.
  - `boost_mutation`: `effective = min(cap, base + increment)`.
  - `inject_top`: extends `external_ids` with top 5 garaged models
    (parent-only — same plumbing as Issue 08).
  Emits info events for triggered/recovered transitions and the
  effective mutation rate.

### Frontend
- `frontend/src/app/services/api.service.ts` — payload type extended
  with all Issue 09 fields.
- `frontend/src/app/training-monitor/training-monitor.ts` — wizard
  state for mutation override, adaptive toggle/increment/cap, bad-gen
  threshold + policy selector.
- `frontend/src/app/training-monitor/training-monitor.html` — controls
  added to step 4 alongside the existing genetics info chips.

### Tests
- `tests/test_adaptive_breeding.py` — 10 tests covering: orchestrator
  state defaults, ramp computation, cap enforcement, threshold
  triggering rules (zero/below/above), policy validity,
  StartTrainingRequest round-trip, and worker
  `_apply_run_overrides` layering.

### Verify
- `pytest tests/ --timeout=120 -q` — all green (1801 passed).
- `npx ng build` — clean.
