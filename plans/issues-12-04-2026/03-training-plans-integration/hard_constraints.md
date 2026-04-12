# Hard Constraints

- `plan_id` is optional on `POST /api/training/start` — the existing
  wizard launch flow without a plan must keep working identically.
- The orchestrator's `training_plan=None` code path must not break.
- Plans saved before this work (no n_generations/n_epochs fields) must
  still load and display correctly — use defaults for missing fields.
- Plan outcomes must be persisted via `plan_registry.record_outcome()`
  after each generation, not batched at the end of the run.
- The exploration strategy resolution (`run_training.py:833-895`) is
  already coded — don't rewrite it, just make sure it gets called.
- Session splitting (session 3) must not require plans created before
  that feature to be recreated — `generations_per_session` defaults
  to "all in one".
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
