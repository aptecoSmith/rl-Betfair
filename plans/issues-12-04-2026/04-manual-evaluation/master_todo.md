# Master TODO — Manual Evaluation

## Session 1: Backend + worker plumbing

### Worker — CMD_EVALUATE command

- [ ] Add `CMD_EVALUATE` constant to `training/worker.py`
- [ ] Handler receives: `{ model_ids: list[str], test_dates: list[str] | null }`
- [ ] If `test_dates` is null, use all available processed days
- [ ] For each model_id: load weights from registry, instantiate policy,
      call `Evaluator.evaluate()`
- [ ] Publish progress via existing WebSocket — reuse evaluator's
      progress tracker with phase="evaluating"
- [ ] Handle errors per-model (skip failed, report, continue with next)
- [ ] Worker state during evaluation: "evaluating" (distinct from
      "training") so training monitor knows what's happening

### API — POST /api/evaluate endpoint

- [ ] Add `EvaluateRequest` schema: `{ model_ids: list[str],
      test_dates: list[str] | None }` to `api/schemas.py`
- [ ] Add `EvaluateResponse` schema: `{ accepted: bool, model_count: int,
      day_count: int }`
- [ ] Add `POST /api/evaluate` to `api/routers/training.py` (or a new
      `evaluation.py` router)
- [ ] Validate: all model_ids exist in registry, all test_dates are
      available in processed data
- [ ] Reject if worker is already busy (training or evaluating)
- [ ] Send CMD_EVALUATE to worker via IPC
- [ ] Return accepted response

### API — GET /api/evaluate/status

- [ ] Return current evaluation progress (or "idle" if not running)
- [ ] Include: model_ids being evaluated, current model, progress pct,
      ETA, completed count

### API — available days for eval

- [ ] Ensure `GET /api/admin/days` (or a new lightweight endpoint)
      returns all available processed days with enough info to pick
      them in the UI

### Tests

- [ ] Test CMD_EVALUATE runs evaluator for given model IDs
- [ ] Test CMD_EVALUATE with null test_dates uses all available days
- [ ] Test POST /api/evaluate rejects when worker is busy
- [ ] Test POST /api/evaluate validates model_ids exist
- [ ] Test progress events are published during standalone evaluation

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] Manual: call POST /api/evaluate with a known model_id, verify
      evaluation runs and results appear in registry

---

## Session 2: Frontend evaluation UI

### Evaluation page

- [ ] Create `frontend/src/app/evaluation/` component
- [ ] Add route `/evaluation` and nav link
- [ ] Model picker: searchable list of all models from scoreboard data,
      with checkboxes. Show model ID (short), architecture, composite
      score, garage status
- [ ] Date picker: list of available processed days with checkboxes.
      Show date, race count. Provide "select all", "select range",
      "last N days" shortcuts
- [ ] "Evaluate" button — calls POST /api/evaluate with selected
      model_ids and test_dates
- [ ] Disable button if worker is busy (poll status or check via WS)
- [ ] Progress section: reuse the existing progress bar components
      from training monitor (or extract shared component). Show
      current model, day progress, overall progress, ETA
- [ ] Results summary: after evaluation completes, show a table of
      results — model ID, total PnL, win rate, composite score per
      model. Link each to model detail page.

### Model detail — re-evaluate button

- [ ] Add "Re-evaluate" button to model detail page header
- [ ] On click: open a small dialog with date picker (pre-populated
      with the dates from the model's last evaluation run)
- [ ] "Evaluate" in dialog calls POST /api/evaluate with just this
      model + selected dates
- [ ] Show inline progress or navigate to evaluation page

### Scoreboard — bulk select

- [ ] Add checkbox column to scoreboard table
- [ ] Add "Select all" checkbox in header
- [ ] Add toolbar that appears when selection > 0: "Evaluate selected
      (N)" button
- [ ] On click: navigate to evaluation page with models pre-selected,
      or open date picker dialog then trigger directly

### Shared progress component (optional)

- [ ] If the progress bar markup is duplicated between training monitor
      and evaluation page, extract to a shared
      `progress-bars` component. Otherwise just duplicate — don't
      over-abstract for two uses.

### Tests

- [ ] Evaluation page renders with model and date pickers
- [ ] Selecting models and dates enables the evaluate button
- [ ] Re-evaluate button appears on model detail page

### Verify

- [ ] `cd frontend && ng build` — clean
- [ ] Manual: pick 2 models and 3 days, evaluate, verify results
      appear in scoreboard and model detail
