# Session: Training End Summary Modal

Read `CLAUDE.md` and `plans/issues-12-04-2026/12-training-end-summary/`
before starting. Follow `master_todo.md`. Mark items done as you go and
update `progress.md` at the end.

## Context

When training finishes, the UI silently reverts to the wizard idle state.
The only indication is a raw JSON dump in a `<pre>` tag.

**Current completion flow:**

1. Orchestrator emits `run_complete` event (`run_training.py:403-407`)
   with minimal data: `{ run_id, generations_completed, final_rankings count }`
2. Worker forwards to WebSocket (`worker.py:469-475`)
3. Frontend sets `running=false`, stores event as `latestEvent`
   (`training.service.ts:176-199`)
4. HTML shows `lastRunSummary() | json` in a `<pre>` block
   (`training-monitor.html:189`)

**What to change:**

1. Enrich the `run_complete` event with a proper summary
2. Show a formatted modal instead of raw JSON
3. Replace the idle-state raw JSON with a compact card

## Key files

| File | What to change |
|------|----------------|
| `training/run_training.py` | Enrich run_complete event with summary |
| `frontend/src/app/training-monitor/training-monitor.html` | Modal + compact card |
| `frontend/src/app/training-monitor/training-monitor.ts` | Modal state, summary parsing |
| `frontend/src/app/training-monitor/training-monitor.scss` | Modal + card styles |
| `frontend/src/app/services/training.service.ts` | Store enriched summary |
| `frontend/src/app/models/training.model.ts` | RunSummary type |

## Existing dialog pattern

The admin page (`frontend/src/app/admin/admin.html:54-66`) and the
stop-training dialog (`training-monitor.html:102-178`) both use the
same pattern:

```html
<div class="dialog-overlay">
  <div class="dialog">
    <h3>Title</h3>
    <!-- content -->
    <div class="dialog-actions">
      <button>Action</button>
    </div>
  </div>
</div>
```

Reuse this pattern. The modal should be wider than the stop dialog
to accommodate the top-5 table.

## Constraints

- Modal must auto-open when run_complete arrives — don't make the
  user click to find out training is done.
- Modal must be dismissable without navigating away.
- The compact idle-state card replaces the raw JSON — don't show both.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: training end summary modal with best model, top 5, and action buttons`
Push: `git push all`
