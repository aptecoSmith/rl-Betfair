# Master TODO — Training Plans Integration

## Session 1: Fix save + launch a plan

The critical path — make plans functional end-to-end.

### Debug save

- [ ] Reproduce the "nothing happens" bug — check browser console and
      network tab for errors when clicking Save
- [ ] Likely causes: validation error displayed but not noticed (check
      `editorTopError` rendering in template), API not running, or
      plan_registry path issue
- [ ] Fix whatever the root cause is

### Wire plan_id into training launch

- [ ] Add optional `plan_id: str | None` field to `StartTrainingRequest`
      in `api/schemas.py`
- [ ] In `POST /api/training/start` (`api/routers/training.py`): if
      plan_id is provided, load the plan from plan_registry and pass it
      to the worker start command
- [ ] In `training/worker.py`: receive plan_id, load plan via
      PlanRegistry, pass `training_plan=plan` and
      `plan_registry=registry` to `TrainingOrchestrator`
- [ ] Verify the existing orchestrator code (`run_training.py:833-895`)
      correctly consumes the plan's exploration strategy, hp_ranges,
      architectures, starting_budget
- [ ] Verify `plan_registry.record_outcome()` is called after each
      generation so outcomes are persisted

### Frontend — "Start plan" button

- [ ] Add a "Start" button to the plan detail view in
      `training-plans.html`
- [ ] On click: call `POST /api/training/start` with `plan_id` plus
      any required fields not in the plan (n_generations, n_epochs,
      train/test dates)
- [ ] Navigate to training monitor after successful start
- [ ] Disable button if plan has validation errors (already have
      `is_launchable` check)

### Tests

- [ ] Test `POST /api/training/start` with plan_id loads and passes plan
- [ ] Test plan outcomes are recorded after generation completes
- [ ] Existing training start tests still pass (plan_id is optional)

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
- [ ] Manual: save a plan, start it, verify orchestrator receives plan

---

## Session 2: Plan status + progress

Give visibility into which plan is running and how far through it is.

### Backend — plan status tracking

- [ ] Add `status` field to `TrainingPlan`: "draft" | "running" |
      "completed" | "failed" | "paused"
- [ ] Add `current_generation: int | None` and
      `started_at: str | None` fields
- [ ] When training starts with a plan: set status="running",
      started_at=now
- [ ] When a generation completes: update current_generation via
      plan_registry
- [ ] When run finishes: set status="completed"
- [ ] When run crashes: set status="failed"
- [ ] Add `GET /api/training-plans/{plan_id}/status` endpoint or
      include status in existing detail endpoint

### Frontend — plan status display

- [ ] Show status badge on plan cards in list view (draft/running/
      completed/failed)
- [ ] Show progress indicator in detail view: "Generation 2/5"
- [ ] Show outcomes table populated in real time as generations finish
- [ ] Link from plan detail to training monitor when status=running
- [ ] Link from training monitor back to active plan

### Tests

- [ ] Test status transitions: draft → running → completed
- [ ] Test status on crash: running → failed
- [ ] Test current_generation updates with each outcome

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean

---

## Session 3: Session splitting + auto-continue

Break large plans into manageable training sessions with checkpoints.

### Backend — session splitting model

- [ ] Add `sessions` concept to TrainingPlan: a plan can define
      `generations_per_session` (default: all in one session)
- [ ] Compute session boundaries: e.g. 10 gens with 3 per session →
      sessions [0-2, 3-5, 6-8, 9]
- [ ] Add `current_session: int` to plan status tracking
- [ ] When a session finishes, mark it complete and record which
      generation it reached

### Backend — auto-continue

- [ ] Add `auto_continue: bool` to TrainingPlan (default: false)
- [ ] When a training run completes and auto_continue=true: check if
      there are remaining sessions, auto-launch the next one
- [ ] The next session inherits the population from the previous
      session's final generation (survivors carry forward)
- [ ] If auto_continue=false: set status="paused", wait for manual
      "Continue" click

### Backend — resume

- [ ] Add `POST /api/training/resume` or extend start with
      `resume_from_session: int`
- [ ] Resume loads the plan, skips to the target session, loads the
      surviving population from the last completed generation
- [ ] Handle edge case: crashed mid-session → resume replays the
      crashed session from scratch (generation-level, not tick-level)

### Frontend — session UI

- [ ] Show session breakdown in plan detail: "Session 1: Gen 0-2 ✓,
      Session 2: Gen 3-5 (running), Session 3: Gen 6-8 (pending)"
- [ ] Add `generations_per_session` field to plan editor
- [ ] Add `auto_continue` checkbox to plan editor
- [ ] Add "Continue" button when status="paused"
- [ ] Show per-session timing in outcomes table

### Tests

- [ ] Test session boundary calculation
- [ ] Test auto-continue triggers next session
- [ ] Test resume from specific session
- [ ] Test crashed session replays correctly

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
