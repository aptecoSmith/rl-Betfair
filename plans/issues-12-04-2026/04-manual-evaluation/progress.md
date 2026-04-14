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
