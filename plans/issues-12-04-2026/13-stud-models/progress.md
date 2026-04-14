# Progress — Stud Models

## Session 1 — 2026-04-14

### Backend
- `api/schemas.py` — `stud_model_ids: list[str] | None` on `StartTrainingRequest`.
- `api/routers/training.py` — validates max-5, all IDs exist with weights+HP.
  Forwarded to worker via IPC and replayed on plan resume.
- `api/routers/training_plans.py` — accepts `stud_model_ids` on plan create.
- `training/training_plan.py` — `stud_model_ids` field + (de)serialisation.
- `training/ipc.py` — `make_start_cmd(stud_model_ids=…)`.
- `training/worker.py` — passes through to `TrainingOrchestrator(...)`.
- `training/run_training.py` — orchestrator stores stud IDs; per-generation
  validates each stud still exists in the registry, then forwards to
  `breed(..., stud_parent_ids=…)`. Emits an info line listing studs.
- `agents/population_manager.py::breed()` — reserves one slot per stud
  (parent_a=stud, parent_b=random survivor / external / other stud /
  clone). Emits a `selection`/`stud` `GeneticEventRecord` per stud child.
  Warns when more studs than slots, or zero slots.

### Frontend
- `frontend/src/app/services/api.service.ts` — `stud_model_ids` (and the
  Issue 09 fields) added to `startTraining` payload.
- `frontend/src/app/training-monitor/training-monitor.ts` — picker state,
  `addStud` / `removeStud`, options sourced from `getScoreboard()`.
- `frontend/src/app/training-monitor/training-monitor.html` — wizard step 4
  picker with chips + max-5 enforcement.

### Tests
- `tests/test_stud_models.py` — 7 tests covering: each stud is parent at
  least once, studs don't take survivor slots, empty list = unchanged
  behaviour, more-studs-than-slots warning, no-slots warning, stud
  genetic-event recorded, and a smoke check for the API limit.

### Verify
- `pytest tests/ --timeout=120 -q` — all green (1801 passed).
- `npx ng build` — clean.
