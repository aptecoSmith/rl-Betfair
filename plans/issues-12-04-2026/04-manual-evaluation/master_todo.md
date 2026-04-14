# Master TODO — Manual Evaluation

## Session 1: Backend + worker plumbing

### Worker — CMD_EVALUATE command

- [x] Add `CMD_EVALUATE` constant to `training/ipc.py`
- [x] Handler receives: `{ model_ids: list[str], test_dates: list[str] | null }`
- [x] If `test_dates` is null, use all available processed days
- [x] For each model_id: load weights from registry, instantiate policy,
      call `Evaluator.evaluate()`
- [x] Publish progress via existing WebSocket — reuse evaluator's
      progress tracker with phase="evaluating"
- [x] Handle errors per-model (skip failed, report, continue with next)
- [x] Worker state during evaluation: shared `running` flag (one job
      at a time); summary carries `manual_evaluation: true` flag

### API — POST /api/evaluate endpoint

- [x] Add `EvaluateRequest` schema
- [x] Add `EvaluateResponse` schema
- [x] Add `POST /evaluate` (new `api/routers/evaluation.py`)
- [x] Validate: all model_ids exist in registry, all test_dates are
      available in processed data
- [x] Reject if worker is already busy
- [x] Send CMD_EVALUATE to worker via IPC
- [x] Return accepted response

### API — GET /api/evaluate/status

- [x] Returns current evaluation progress (or "idle" if not running)
- [x] Includes: phase, detail, process snapshot, item snapshot,
      manual_evaluation flag

### API — available days for eval

- [x] Existing `GET /admin/days` already returns processed days with
      tick/race counts — no new endpoint needed.

### Tests

- [x] POST /evaluate: rejects when worker busy
- [x] POST /evaluate: validates model_ids exist
- [x] POST /evaluate: validates test_dates exist
- [x] POST /evaluate: default-all-dates expansion + explicit dates
- [x] GET /evaluate/status: idle + running variants

### Verify

- [x] `python -m pytest tests/ --timeout=120 -q` — 1811 passed
- [ ] Manual: call POST /api/evaluate with a known model_id, verify
      evaluation runs and results appear in registry (defer to session 2)

---

## Session 2: Frontend evaluation UI

### Evaluation page

- [x] Create `frontend/src/app/evaluation/` component
- [x] Add route `/evaluation` and nav link
- [x] Model picker: searchable list of all models from scoreboard data,
      with checkboxes. Show model ID (short), architecture, composite
      score, garage status
- [x] Date picker: list of available processed days with checkboxes.
      Show date, race count. Provide "select all", "last N days" shortcuts
- [x] "Evaluate" button — calls POST /api/evaluate with selected
      model_ids and test_dates
- [x] Disable button if worker is busy (uses TrainingService.isRunning)
- [x] Progress section reuses the training-monitor bar markup (overall,
      phase, current) fed from the shared TrainingService signal
- [x] Results summary table populated when an evaluating phase completes;
      rows link to model detail

### Model detail — re-evaluate button

- [x] Add "Re-evaluate" button to model detail page header
- [x] On click: open a small dialog with date picker (pre-populated
      with the dates from the model's last evaluation run)
- [x] "Evaluate" in dialog calls POST /api/evaluate with just this
      model + selected dates
- [x] "Open Evaluation page" shortcut also provided in the dialog

### Scoreboard — bulk select

- [x] Add checkbox column to scoreboard table
- [x] Add "Select all" checkbox in header
- [x] Add toolbar that appears when selection > 0: "Evaluate selected
      (N)" button
- [x] On click: navigate to evaluation page with models pre-selected
      via SelectionStateService.evaluationPreselected

### Shared progress component (optional)

- [x] Skipped — markup duplicated for two uses, no abstraction needed yet

### Tests

- [x] Evaluation page renders with model and date pickers
- [x] Selecting models and dates enables the evaluate button
- [x] Re-evaluate button appears on model detail page + opens dialog
      with metric_history dates pre-selected

### Verify

- [x] `cd frontend && ng build` — clean
- [x] Manual smoke via preview server: nav link works, page renders,
      bulk-select toolbar appears + routes to /evaluation with the
      model pre-selected, re-evaluate dialog opens with prior dates
      ticked
