# Progress — Manual Evaluation

## Session 1 — Backend + worker plumbing (✅ done)

- `CMD_EVALUATE` constant + `make_evaluate_cmd` / `make_evaluate_started_msg`
  helpers in `training/ipc.py`.
- `TrainingWorker._start_evaluation()` in `training/worker.py`:
  validates model_ids + test_dates, loads test days, reconstructs each
  policy from registry metadata + saved weights, calls
  `Evaluator.evaluate()` per model. Per-model errors are caught,
  logged, and reported via the progress queue — the batch continues.
  Emits `phase_start` / `phase_complete` so the existing WebSocket
  consumer flips `running` correctly.
- New `api/routers/evaluation.py` with:
  - `POST /evaluate` — validates models exist + have weights, validates
    dates against `paths.processed_data`, rejects when worker busy,
    sends CMD_EVALUATE.
  - `GET /evaluate/status` — projects the shared `training_state` into
    a lightweight evaluation snapshot.
- `EvaluateRequest` / `EvaluateResponse` / `EvaluateStatus` schemas in
  `api/schemas.py`. Router wired up in `api/main.py`.
- Unit tests `tests/test_api_evaluate.py` (9 tests) — covers reject
  while busy, invalid model_ids, invalid test_dates, default-all-dates
  expansion, explicit dates, idle status, running status.
- `python -m pytest tests/ --timeout=120 -q` → 1811 passed.

## Session 2 — Frontend evaluation UI (✅ done)

- New `frontend/src/app/evaluation/` page (`/evaluation` route + nav link):
  - Searchable model picker with checkboxes, sorted by composite score.
  - Day picker with select-all / last-7 / last-14 shortcuts.
  - "Evaluate" button → POST `/api/evaluate`.
  - Live progress card reuses the training-monitor's overall/process/item
    bar markup (status fed from the shared `TrainingService` signal).
  - Results section reloads scoreboard rows for the submitted models when
    an `evaluating` phase completes.
- Re-evaluate button + dialog on model-detail: pre-populates with the
  dates from the model's `metrics_history`, lets the user adjust, posts
  to `/api/evaluate`. Disabled while the worker is busy.
- Scoreboard: checkbox column + select-all header + bulk-action toolbar
  ("Evaluate selected (N)") that pre-fills the evaluation page via
  `SelectionStateService.evaluationPreselected`.
- Tests: new `evaluation.spec.ts` (8 tests) + new re-evaluate cases in
  `model-detail.spec.ts`. `ng build` clean; manual smoke through the
  preview server confirmed nav link, page render, toolbar, and dialog.
