# Progress — Configurable Budget

One entry per completed session.

---

## Session 01 — Per-plan starting_budget override (2026-04-11)

### Changes
- `training/training_plan.py` — added `starting_budget: float | None = None`
  field to `TrainingPlan` dataclass, `to_dict()`, `from_dict()`, and
  `TrainingPlan.new()` factory. Old JSON without the field defaults to
  `None` (backward compatible).
- `training/run_training.py` — `TrainingOrchestrator.__init__` patches
  `config["training"]["starting_budget"]` when the plan provides one.
  Global config is the fallback.
- `api/schemas.py` — added `starting_budget: float | None` to
  `StartTrainingRequest` for per-run API overrides.
- `training/ipc.py` — `make_start_cmd()` now passes `starting_budget`
  in the IPC message.
- `api/routers/training.py` — passes `starting_budget` to worker;
  rejects ≤ 0 with HTTP 400.
- `training/worker.py` — `_apply_run_overrides` patches
  `config["training"]["starting_budget"]` when set.
- `api/routers/training_plans.py` — `create_plan` validates and passes
  `starting_budget` to `TrainingPlan.new()`.
- Frontend `training-plan.model.ts` — added `starting_budget` to
  `TrainingPlan` and `TrainingPlanPayload` interfaces.
- Frontend `training-plans.ts` — added `editorStartingBudget` signal,
  reset in `openEditor()`, included in `savePlan()` payload.
- Frontend `training-plans.html` — budget input in editor form, budget
  display on plan cards and detail view.

### Tests (8 new, all passing)
- Round-trip with budget, round-trip without budget (None)
- `to_dict()` includes budget, `from_dict()` backward compat
- API create with budget, API rejects negative, API rejects zero
- Orchestrator uses plan budget / falls back to global

---

## Session 02 — Record starting_budget per evaluation run (2026-04-11)

### Changes
- `registry/model_store.py` — added `starting_budget: float = 100.0` to
  both `EvaluationDayRecord` and `EvaluationBetRecord`. Schema migration
  adds `starting_budget REAL DEFAULT 100.0` column to `evaluation_days`
  table. `record_evaluation_day` INSERT includes the field.
  `get_evaluation_days` reads it (defaults 100.0 for old DBs).
  `write_bet_logs_parquet` includes `starting_budget` column.
  `get_evaluation_bets` reads it (defaults 100.0 for old parquets).
- `training/evaluator.py` — `_evaluate_day` populates
  `starting_budget=env.starting_budget` on both `EvaluationDayRecord`
  and `EvaluationBetRecord`.

### Tests (4 new, all passing)
- Day record with custom budget persists and retrieves
- Day record defaults to 100.0
- Bet parquet includes and round-trips budget
- Old parquet without budget column defaults to 100.0

---

## Session 03 — Percentage return in scoreboard (2026-04-11)

### Changes
- `registry/scoreboard.py` — `ModelScore` gains `mean_daily_return_pct`
  and `recorded_budget` fields. `compute_score()` uses the recorded
  `starting_budget` from day records (not the global config) for
  normalisation and computes `(mean_pnl / budget) * 100`.
- `api/schemas.py` — `ScoreboardEntry` gains `mean_daily_return_pct`
  and `recorded_budget`. `DayMetric` gains `starting_budget`.
- `api/routers/models.py` — passes new fields through in both
  `_score_to_entry` and `DayMetric` construction.
- Frontend `scoreboard.model.ts` — added new fields to interface.
- Frontend `scoreboard.html` — "Return %" column with green/red
  colouring. Updated empty-state colspan.
- Frontend `scoreboard.scss` — `.positive` / `.negative` classes.
- Frontend `garage.html` — same "Return %" column.
- Frontend `garage.scss` — same positive/negative classes.

### Tests (5 new, all passing)
- Return % at budget=10 (10%), budget=100 (10%), both equal
- Negative return, backward compat with default budget

---

## Session 04 — Model detail + bet explorer budget context (2026-04-11)

### Changes
- `frontend/model-detail.model.ts` — added `starting_budget` to
  `DayMetric` interface.
- `frontend/model-detail.ts` — added `recordedBudget()` and
  `meanDailyReturnPct()` computed signals.
- `frontend/model-detail.html` — "Budget: £X/race" and "Mean Daily
  Return: +X.X%" cards in metrics grid.
- `api/schemas.py` — added `starting_budget` to `BetExplorerResponse`.
- `api/routers/replay.py` — reads `starting_budget` from evaluation
  day records and includes in response.
- `frontend/bet-explorer.model.ts` — added `starting_budget` to
  `BetExplorerResponse`.
- `frontend/bet-explorer.html` — budget stat in summary bar.

### Tests
- No new backend tests (Session 04 is display-only).
  All 99 existing tests still pass.

---

## Session 05 — Percentage-based discard threshold (2026-04-11)

### Changes
- `config.yaml` — added `min_mean_return_pct: 0.0` alongside
  existing `min_mean_pnl`.
- `registry/scoreboard.py` — `check_discard_candidates()` now checks
  `min_mean_return_pct` (percentage-based) if configured. Falls back
  to `min_mean_pnl` (absolute) if `min_mean_return_pct` is not set.

### Tests (3 new, all passing)
- Profitable low-budget model survives 0% threshold
- Losing model with negative return discarded
- Backward compat: config without min_mean_return_pct uses min_mean_pnl
