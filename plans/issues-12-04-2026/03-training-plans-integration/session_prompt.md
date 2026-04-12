# Session 1: Fix save + launch a plan

Read `CLAUDE.md` and `plans/issues-12-04-2026/03-training-plans-integration/`
before starting. Follow session 1 of `master_todo.md`. Mark items done as
you go and update `progress.md` at the end.

## Context

Training plans are fully built but disconnected from training runs:
- UI: create/list/edit/validate/save all work (or should — save may be buggy)
- Persistence: `registry/training_plans/{plan_id}.json`
- Backend plumbing: `TrainingOrchestrator` accepts `training_plan=`,
  `PopulationManager.initialise_population` accepts `plan=`
- **Missing bridge**: `POST /api/training/start` has no `plan_id` field,
  worker never receives a plan, orchestrator never gets one from the API flow

## What to do

### 1. Debug save

Reproduce the bug. Check:
- `api/main.py:65-76` — `plan_registry` set on `app.state`?
- `api/routers/training_plans.py:54-58` — `_registry()` reads from state
- `frontend/src/app/training-plans/training-plans.ts:282-327` — `savePlan()`
  has error handling but early returns on validation failure might not be
  visible enough in the template
- Check the template renders `editorTopError` prominently in the editor

### 2. Wire plan_id through

**`api/schemas.py`** — add `plan_id: str | None = None` to
`StartTrainingRequest`.

**`api/routers/training.py`** — in the POST handler, if `plan_id` is set:
load plan via `request.app.state.plan_registry.load(plan_id)`, include it
in the worker start command.

**`training/worker.py`** — receive plan_id from start command, load plan,
pass to `TrainingOrchestrator(training_plan=plan, plan_registry=registry)`.

The orchestrator already handles the rest:
- `run_training.py:833-895` resolves exploration strategy from plan
- `run_training.py:768-815` records outcomes via plan_registry
- `population_manager.py` uses plan's hp_ranges, architectures, arch_mix

### 3. "Start plan" button

In `training-plans.html` detail view, add a button that:
- Calls start with `plan_id` + supplementary fields (n_generations,
  n_epochs, train/test dates — these aren't in the plan today)
- Could open a small dialog for the missing fields, or navigate to the
  training monitor wizard pre-populated from the plan
- Navigates to training monitor after successful start

**Design decision**: the plan stores *what* to train (population shape,
hp ranges, strategy) but not *how long* or *on what data* (generations,
epochs, dates). The start flow needs both. Options:
- **(a)** Add generations/epochs/dates to the plan model
- **(b)** Show a mini-dialog when starting that asks for the missing fields
- **(c)** Pre-populate the wizard from the plan, user clicks through

Option (a) is cleanest — a plan should be self-contained. Add
`n_generations`, `n_epochs` to TrainingPlan. Dates can stay in the wizard
since they depend on what data is available at launch time.

## Key files

| File | What to change |
|------|----------------|
| `api/schemas.py` | Add plan_id to StartTrainingRequest |
| `api/routers/training.py` | Load plan on start, pass to worker |
| `api/routers/training_plans.py` | Debug save endpoint |
| `training/worker.py` | Receive plan, pass to orchestrator |
| `training/training_plan.py` | Add n_generations, n_epochs fields |
| `frontend/src/app/training-plans/training-plans.html` | Fix save visibility, add Start button |
| `frontend/src/app/training-plans/training-plans.ts` | Start plan logic |
| `frontend/src/app/services/api.service.ts` | May need startTrainingWithPlan method |

## Constraints

- `plan_id` is optional on `POST /api/training/start` — existing wizard
  flow must keep working without a plan.
- Don't break the orchestrator's existing `training_plan=None` code path.
- Plans saved before this change (no n_generations/n_epochs) should still
  load — use sensible defaults.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: wire training plans into launch flow + fix save bug`
Push: `git push all`
