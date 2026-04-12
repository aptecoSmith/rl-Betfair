# Hard Constraints

- Do not modify the Evaluator class — it works correctly as-is.
- Worker must reject CMD_EVALUATE if already training, and reject
  CMD_START if already evaluating. Only one job at a time.
- Per-model evaluation errors must not crash the batch — log the
  error, skip the model, continue with the next.
- Progress events must use the same WebSocket channel and schema as
  training progress events so the frontend can reuse components.
- Evaluation results must be stored identically to training-triggered
  evaluations — same DB tables, same Parquet layout. There should be
  no way to distinguish manual vs training-triggered eval in the data.
- All model_ids and test_dates must be validated before starting —
  reject the request upfront, don't fail mid-evaluation on a bad ID.
- All tests pass: `python -m pytest tests/ --timeout=120 -q`.
- Frontend builds clean: `ng build`.
