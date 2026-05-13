# Session 03 — Launch cohort

Mirror the pwin-gate cohort launch + `--race-confidence-threshold
0.30`. Arm watcher that auto-fires held-out reeval at completion.
Loop into 1h heartbeat mode until cohort completes.

## Pre-checks

1. Session 01 committed (look for
   `feat(scalping-race-confidence-gate)` in git log)
2. Session 02 smoke PASSED (commit message contains "VERDICT:
   PASS")
3. GPU available (`nvidia-smi`)
4. All three predictor manifests exist

Any failing → STOP, write diagnostic.

## Launch command

```bash
TAG="_predictor_SCALPING_raceconf_$(date +%s)"
LOG="registry/${TAG}.log"
nohup python -m training_v2.cohort.runner \
  --n-agents 12 --generations 8 --days 6 \
  --data-dir data/processed --device cuda --seed 42 \
  --output-dir "registry/${TAG}" \
  --mutation-rate 0.2 \
  --strategy-mode arb \
  --predictor-bundle-manifests \
    ../betfair-predictors/production/race-outcome/manifest.json \
    ../betfair-predictors/production/race-outcome-ranker/manifest.json \
    ../betfair-predictors/production/direction-predictor/manifest.json \
  --use-race-outcome-predictor --use-direction-predictor --predictor-lean-obs \
  --predictor-p-win-back-threshold 0.20 \
  --predictor-p-win-lay-threshold 0.40 \
  --race-confidence-threshold 0.30 \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "$LOG" 2>&1 &
disown
```

Save the tag — used by the watcher and status updater.

## Verification

```bash
until grep -qE "Cohort:|Traceback|usage:|error:" registry/${TAG}.log 2>/dev/null; do
    sleep 4
done
head -10 registry/${TAG}.log | grep -vE "warnings.warn|valid feature names"
```

Must see:
- `Phase 5 enabled genes: [...]`
- `predictor bundle loaded: champion=... ranker=... direction=...`
- `Cohort: 12 agents × 8 generations ...`
- Generation 1 starting

Traceback → STOP. 10-min silent → STOP.

## Watcher

Create `/tmp/auto_reeval_raceconf_cohort.sh` from the pwin-gate
template:

```bash
#!/usr/bin/env bash
set -u
COHORT_DIR="registry/${TAG}"
SCOREBOARD="${COHORT_DIR}/scoreboard.jsonl"
TARGET_ROWS=96
REEVAL_LOG="${COHORT_DIR}/auto_reeval_2026-04-28_30.log"
REEVAL_OUTPUT="reeval_held_out_2026-04-28_30.jsonl"

echo "[$(date)] watcher started; waiting for ${TARGET_ROWS} rows"
while true; do
    if [ -f "$SCOREBOARD" ]; then
        ROWS=$(wc -l < "$SCOREBOARD" 2>/dev/null | tr -d ' ')
        if [ "$ROWS" -ge "$TARGET_ROWS" ]; then
            echo "[$(date)] cohort complete: ${ROWS} rows. firing reeval."
            break
        fi
        echo "[$(date)] rows=${ROWS}/${TARGET_ROWS} -- still running"
    fi
    sleep 300
done

python -m tools.reevaluate_cohort \
    --cohort-dir "${COHORT_DIR}" \
    --eval-days 2026-04-28 2026-04-29 2026-04-30 \
    --data-dir data/processed --device cuda --top-n 5 \
    --predictor-bundle-manifests \
        ../betfair-predictors/production/race-outcome/manifest.json \
        ../betfair-predictors/production/race-outcome-ranker/manifest.json \
        ../betfair-predictors/production/direction-predictor/manifest.json \
    --predictor-lean-obs --use-race-outcome-predictor --use-direction-predictor \
    --strategy-mode arb \
    --predictor-p-win-back-threshold 0.20 \
    --predictor-p-win-lay-threshold 0.40 \
    --race-confidence-threshold 0.30 \
    --output "${REEVAL_OUTPUT}" \
    > "${REEVAL_LOG}" 2>&1

echo "[$(date)] reeval done."
```

Substitute the actual tag value when writing the script. Launch
in background.

## Status updater

```bash
nohup python -m tools.show_cohort_status \
    registry/${TAG} --watch 60 \
    > /tmp/raceconf_status_watcher.log 2>&1 &
disown
```

## Heartbeat loop

Wake hourly. Each iteration:

1. Read `registry/${TAG}/status.txt`.
2. If progress unchanged 2+ consecutive iterations: tail the
   cohort log; STOP if Traceback.
3. If 96/96: skip to Session 04.
4. Otherwise: schedule next 1h wakeup, append single-line log
   entry "heartbeat N/96 rows".

## Acceptance

- Cohort PID running, log growing
- Generation 1 visible
- Watcher PID visible
- Status file populated within 5 min

## No commit during the run

Nothing to commit yet. Just log to `autonomous_run_log.md`.
Session 04 commits the verdict.
