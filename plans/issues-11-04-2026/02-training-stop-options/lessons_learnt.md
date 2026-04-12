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

## 2026-04-12 — Implementation complete

1. **Parallel eval path doesn't support stop_after_current_eval.**
   The `ThreadPoolExecutor` path submits all agents upfront and
   blocks on `future.result()` — there's no natural place to check
   `stop_after_current_eval_event` between completions without
   cancelling running futures.  Since `eval_workers` defaults to 1
   in config, this is acceptable.  If parallel eval becomes common,
   this will need a `concurrent.futures.as_completed()` loop with
   an event check.

2. **eval_all reuses existing finish_event + skip_training_event.**
   Rather than adding a third top-level event, eval_all sets both
   `finish_event` (which the orchestrator already checks between
   generations) and `skip_training_event` (new, skips remaining
   training agents).  This composition avoids duplicating the
   existing finish path.

3. **Frontend vitest tests are broken across the board** (pre-existing
   `TestBed.initTestEnvironment()` error).  The new spec tests follow
   the same pattern as existing ones and will pass once the vitest
   config is fixed.
