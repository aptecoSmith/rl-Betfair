# Session 03 — Launch cohort

Mirror the pwin-gate cohort's launch shape exactly, plus the
`--direction-gate-enabled` flag. Arm a watcher that auto-fires
the held-out reeval at completion. Then loop into 1h heartbeat
mode until the cohort completes.

## Pre-checks (run once at the top of Session 03)

1. **Session 01 committed?** `git log --oneline -5 | grep
   "feat(scalping-direction-gate)"` should show the gate commit.
2. **Session 02 smoke PASSED?** Read the smoke output committed
   in the previous session — VERDICT line must read PASS.
3. **No other cohort process running?** Check GPU is available:
   `nvidia-smi` or peek at process list. The cohort will run on
   CUDA and should not contend with another training job.
4. **All three predictor manifests exist?**
   `../betfair-predictors/production/race-outcome/manifest.json`
   `../betfair-predictors/production/race-outcome-ranker/manifest.json`
   `../betfair-predictors/production/direction-predictor/manifest.json`

Any of these failing → stop loop, write diagnostic.

## Launch command

```bash
TAG="_predictor_SCALPING_dirgate_$(date +%s)"
LOG="registry/${TAG}.log"
echo "TAG=${TAG}"
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
  --direction-gate-enabled \
  --enable-gene stop_loss_pnl_threshold \
  --enable-gene open_cost \
  --enable-gene matured_arb_bonus_weight \
  --enable-gene naked_loss_scale \
  --enable-gene mature_prob_loss_weight \
  --enable-gene fill_prob_loss_weight \
  > "$LOG" 2>&1 &
disown
echo "PID=$!"
echo "$TAG" > /tmp/dirgate_tag.txt
```

Save the tag — used by the watcher and status updater.

## Verification

Within 5 minutes of launch:

```bash
# Wait for first log line beyond imports
until grep -qE "Cohort:|Traceback|usage:|error:" registry/${TAG}.log 2>/dev/null; do
    sleep 4
done
head -10 registry/${TAG}.log | grep -vE "warnings.warn|valid feature names"
```

Must see:
- `Phase 5 enabled genes: ['fill_prob_loss_weight', ...]`
- `predictor bundle loaded: champion=... ranker=... direction=...`
- `Cohort: 12 agents × 8 generations on 3 training days ...`
- Generation 1 lines starting.

If a Traceback appears: STOP loop, write diagnostic.
If the log is silent for >10 min: STOP loop, write diagnostic.

## Watcher

Create `/tmp/auto_reeval_dirgate_cohort.sh` modelled on
`/tmp/auto_reeval_pwingate_cohort.sh`:

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
    --direction-gate-enabled \
    --output "${REEVAL_OUTPUT}" \
    > "${REEVAL_LOG}" 2>&1

echo "[$(date)] reeval done."
```

Substitute the actual tag value for `${TAG}` when writing the
script. Launch in background:

```bash
chmod +x /tmp/auto_reeval_dirgate_cohort.sh
nohup bash /tmp/auto_reeval_dirgate_cohort.sh > /tmp/auto_reeval_dirgate_watcher.log 2>&1 &
disown
```

## Status updater

Launch the human-readable status file refresher:

```bash
nohup python -m tools.show_cohort_status \
    registry/${TAG} --watch 60 \
    > /tmp/dirgate_status_watcher.log 2>&1 &
disown
```

This writes `registry/${TAG}/status.txt` every 60s.

## Heartbeat loop

After launch, the loop wakes hourly. Each iteration:

1. Read `registry/${TAG}/status.txt` — note progress.
2. If progress unchanged across 2+ consecutive iterations
   (cohort stuck): tail the cohort log; if traceback, STOP.
3. If progress hit 96/96: skip to Session 04.
4. Otherwise: schedule next 1h wakeup.

Don't log a full log entry per heartbeat — append a single line
per iteration: `## YYYY-MM-DD HH:MM — heartbeat N/96 rows`.

## Acceptance

- Cohort process running (PID exists, log growing).
- Generation 1 visible in log.
- Watcher pid visible in process list.
- Status file populated within 5 min.

When complete: 96 rows in scoreboard.jsonl, watcher fires reeval,
reeval produces `<cohort_dir>/reeval_held_out_2026-04-28_30.jsonl`.

## Commit (after successful launch)

No commit during the run — nothing to commit. Just log to
autonomous_run_log.md. The next commit happens in Session 04.
