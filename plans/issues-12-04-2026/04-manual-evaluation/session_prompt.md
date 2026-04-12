# Session 1: Backend + worker plumbing for manual evaluation

Read `CLAUDE.md` and `plans/issues-12-04-2026/04-manual-evaluation/`
before starting. Follow session 1 of `master_todo.md`. Mark items done
as you go and update `progress.md` at the end.

## Context

The `Evaluator` class (`training/evaluator.py`) is fully functional but
only called from within `TrainingOrchestrator._run_generation()` during
a training run. There's no standalone path.

The evaluator needs:
- A `config` dict (betting constraints, env settings)
- A `ModelStore` (to create evaluation runs and write results)
- A loaded policy network (weights from registry)
- A list of `Day` objects (loaded from processed parquet files)
- Optional: `progress_queue` for progress events, `market_type_filter`

The worker (`training/worker.py`) currently supports:
- `CMD_START` — full training run
- `CMD_STOP` / `CMD_FINISH` — halt training
- `CMD_STATUS` — poll state

## What to do

### 1. Add CMD_EVALUATE to worker

In `training/worker.py`, add a new command handler:

```python
CMD_EVALUATE = "evaluate"
```

The handler should:
1. Parse `model_ids` and `test_dates` from the command payload
2. Load config, create ModelStore + Evaluator (same setup as training)
3. Load test days from processed data (filter by test_dates if provided,
   otherwise use all available)
4. For each model_id:
   a. Load weights: `store.load_weights(model_id)` → state_dict
   b. Get model metadata: `store.get_model(model_id)` → architecture,
      hyperparameters
   c. Instantiate policy from architecture + hyperparameters
   d. Load state_dict into policy
   e. Call `evaluator.evaluate(model_id, policy, test_days, ...)`
   f. Publish progress events via the WebSocket queue
5. Handle per-model errors gracefully — log, skip, continue

Reference the existing evaluation flow in `run_training.py:572-649`
for how to set up the evaluator and call it. The key difference is
that training creates policies during training — here we need to
reconstruct them from registry metadata + saved weights.

Look at how `PopulationManager` or the training loop instantiates
policies from architecture names and hyperparameters. You'll need
the same factory logic.

### 2. Add POST /api/evaluate endpoint

In `api/routers/training.py` (or a new `api/routers/evaluation.py`):

```python
@router.post("/api/evaluate")
async def start_evaluation(request: Request, body: EvaluateRequest):
    # Validate model_ids exist
    # Validate test_dates are available (if provided)
    # Reject if worker is busy
    # Send CMD_EVALUATE to worker
    # Return accepted response
```

The endpoint should mirror how `POST /api/training/start` sends
commands to the worker — check the IPC mechanism in
`api/routers/training.py`.

### 3. Add GET /api/evaluate/status

Return the evaluation progress state — whether idle or running,
which models are being evaluated, current progress.

## Key files

| File | What to change |
|------|----------------|
| `training/worker.py` | Add CMD_EVALUATE handler |
| `training/evaluator.py` | No changes — use as-is |
| `api/routers/training.py` | Add evaluate endpoints (or new router) |
| `api/schemas.py` | Add EvaluateRequest, EvaluateResponse |
| `registry/model_store.py` | Read-only — load weights + metadata |
| `agents/architectures.py` | Policy factory — instantiate from name |

## Constraints

- Don't modify the Evaluator class — it's already fit for purpose.
- Worker must reject CMD_EVALUATE if already training (and vice versa).
- Progress events should use the same WebSocket and schema as training
  progress so the frontend can reuse progress bar components.
- Per-model errors must not crash the entire evaluation — skip and
  report.
- `python -m pytest tests/ --timeout=120 -q` must pass.

## Commit

Single commit: `feat: standalone evaluation — worker command + API endpoint`
Push: `git push all`
