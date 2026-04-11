# Lessons Learnt — Training Stop Options

Append-only. Date each entry.

---

## 2026-04-11 — Initial analysis

The infrastructure for graceful shutdown is already solid:
- Two `threading.Event` objects (`stop_event`, `finish_event`)
  passed from worker to orchestrator.
- WebSocket streaming of progress events to the frontend.
- "Finish Up" and "Stop Training" buttons with disabled-while-pending
  state.

The current "Finish Up" already does most of what "evaluate all
generated models" needs — it completes the current generation's
training, then evaluates.  The gap is that it waits for training to
complete first.  The new `skip_training_event` will let us jump
straight to evaluation.

The "finish current evaluation only" option is new — it requires
checking the new event inside the per-agent evaluation loop, which
currently only checks `stop_event` (between agents, not mid-eval).
The check point is after each agent's eval completes, so granularity
is per-model, not per-test-day.  This is acceptable — a single
model's eval takes ~1-2 minutes.
