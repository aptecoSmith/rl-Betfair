# 01 — Re-run lay-EV probe

See `session_prompts/00_autonomous_full_run.md` Phase 1 for the
full driver. This file is a terse pointer.

## What

Run `tools/probe_lay_outcome_distribution.py` on the held-out
window 2026-04-28/29/30 with current pwin lay threshold 0.40 to
confirm the calibration profile hasn't shifted. Use the per-
bucket EV table to set the new defaults for Phase 3.

## How

```
python -m tools.probe_lay_outcome_distribution \
    --days 2026-04-28 2026-04-29 2026-04-30 \
    --race-confidence-threshold 0.50 \
    --lay-threshold 0.40 \
    --device cuda
```

## Decision rules

- `predictor_p_win_lay_threshold` = lowest pwin bucket where
  EV/£ ≥ 0 across at least n=100 admitted runners. Expect 0.20.
- `lay_price_max` = highest lay-price bucket where EV/£ ≥ −£0.05.
  Expect 20.

## Stop conditions

- No bucket has positive EV → STOP, this is a new plan.
- Calibration hole has moved → STOP, surface the new
  observations and rethink.

## Output

Append the probe output table to `autonomous_run_log.md`. Surface
the chosen thresholds explicitly before locking them.
