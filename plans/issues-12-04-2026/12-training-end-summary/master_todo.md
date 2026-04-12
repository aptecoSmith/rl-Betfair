# Master TODO — Training End Summary Modal

## Session 1: Backend summary + modal UI

### Backend — richer run_complete event

- [ ] In `run_training.py`, enrich the `run_complete` event with a
      proper summary dict:
      ```python
      {
          "run_id": str,
          "status": "completed" | "stopped" | "error",
          "generations_completed": int,
          "generations_requested": int,
          "total_agents_trained": int,
          "total_agents_evaluated": int,
          "wall_time_seconds": float,
          "best_model": {
              "model_id": str,
              "composite_score": float,
              "total_pnl": float,
              "win_rate": float,
              "architecture": str,
          },
          "population_summary": {
              "survived": int,
              "discarded": int,
              "garaged": int,
          },
          "top_5": [
              {"model_id": str, "composite_score": float, "pnl": float},
              ...
          ],
          "error_message": str | null,  # If status == "error"
      }
      ```
- [ ] Compute wall_time from run start to completion
- [ ] Pull best_model and top_5 from final_rankings
- [ ] Include stop reason if stopped early (immediate/eval_current/eval_all)

### Frontend — summary modal

- [ ] Create a modal component or reuse the existing dialog pattern
      from admin/stop-training dialogs
- [ ] Show when `run_complete` event arrives (auto-open)
- [ ] Modal content:
      - Header: "Training Complete" (green) / "Training Stopped"
        (amber) / "Training Failed" (red)
      - Duration: "3h 12m" (formatted wall time)
      - Generations: "3/3 completed" or "2/3 (stopped early)"
      - Best model card: model ID (short), score, PnL, win rate,
        architecture — clickable link to model detail
      - Top 5 table: rank, model ID, score, PnL, win rate
      - Population stats: N survived, N discarded, N garaged
      - If error: error message in a red box
- [ ] Action buttons at bottom:
      - "View Scoreboard" → navigate to scoreboard
      - "View Best Model" → navigate to model detail for best
      - "Start New Run" → dismiss modal, wizard is ready
      - "Dismiss" → close modal, stay on monitor page
- [ ] Modal auto-dismisses if user starts a new run

### Frontend — replace raw JSON display

- [ ] Remove or replace the `<pre>{{ lastRunSummary() | json }}</pre>`
      block (lines 189 in training-monitor.html)
- [ ] In idle state, show a compact "Last run" card instead of raw JSON:
      "Last run completed 5m ago — Best: abc123 (score 0.42, +£12.30)"
      with a "View summary" button that re-opens the modal

### Styling

- [ ] Modal should match existing dialog styles (`.dialog-overlay`,
      `.dialog` classes from admin page)
- [ ] Best model card: highlight colour, larger text for score
- [ ] Top 5 table: compact, sortable isn't needed — it's always by
      score
- [ ] Status-dependent header colour: green/amber/red

### Tests

- [ ] Test: run_complete event includes enriched summary data
- [ ] Test: wall_time_seconds is reasonable (> 0, < 24h)
- [ ] Test: best_model matches highest composite_score in final_rankings
- [ ] Test: error status includes error_message

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
- [ ] Manual: complete a short training run, verify modal appears with
      correct data
