# pbt-breeding Step 5 -- A/B: PBT promotion ladder vs the gene-only GA.
#
# Paired: SAME --seed + SAME non-sealed day pool + SAME agent count / gens /
# selection metric. The ONLY difference is --breeding (ga vs pbt). Judged
# ONLY on the SEALED May 20-29 days, which are EXCLUDED from training +
# selection in BOTH arms (--exclude-days). The two arms train SEQUENTIALLY
# (not in parallel -- 2x16 workers would oversubscribe + contend), and the
# held-out re-eval runs AFTER all training (a "free box": never alongside
# training, which trips the RAM safety net -- operator note).
#
# NB Windows PowerShell 5.1 wraps a native exe's stderr as ErrorRecords, so
# do NOT set $ErrorActionPreference='Stop' (Python logs to stderr -- it would
# halt on the first log line). We check $LASTEXITCODE after each step instead.
# ASCII-only: PS 5.1 reads a BOM-less .ps1 as ANSI, so non-ASCII breaks it.
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"

$SEED    = 1234
$AGENTS  = 16
$GENS    = 6
$DAYS    = 12           # non-sealed pool: 3 rotations x 4 (2 train / 2 eval).
$PAR     = 16           # multiprocess workers
$SEALED  = @(
  "2026-05-20","2026-05-21","2026-05-22","2026-05-23","2026-05-24",
  "2026-05-25","2026-05-26","2026-05-27","2026-05-28","2026-05-29"
)
$OUT     = "registry\ab_pbt_$SEED"
$GA_DIR  = "$OUT\ga"
$PBT_DIR = "$OUT\pbt"
New-Item -ItemType Directory -Force -Path $OUT | Out-Null

function Log($m) { "$([DateTime]::Now.ToString('HH:mm:ss')) $m" | Tee-Object -Append "$OUT\runbook.log" }
function Die($m) { Log "FAILED: $m (exit $LASTEXITCODE)"; exit 1 }

# Arm 1: PBT promotion ladder (FIRST so warm-start is exercised early).
Log "ARM PBT: training $AGENTS agents x $GENS gens, seed $SEED, pool $DAYS days"
& $py -m training_v2.cohort.runner `
  --breeding pbt --n-agents $AGENTS --generations $GENS --days $DAYS `
  --exclude-days $SEALED --seed $SEED --parallel-agents $PAR --device cpu `
  --composite-score-mode locked_weighted `
  --pbt-rotations 3 --pbt-train-per-rotation 2 --pbt-eval-per-rotation 2 `
  --pbt-r2-size 6 --pbt-r3-size 4 --pbt-promote-from-r1 3 `
  --pbt-promote-from-r2 2 --pbt-freeze-top-r3 2 `
  --output-dir $PBT_DIR *> "$OUT\pbt_train.log"
if ($LASTEXITCODE -ne 0) { Die "PBT training" }
Log "ARM PBT: training done"

# Arm 2: gene-only GA (control; LSTM-only by construction).
Log "ARM GA: training $AGENTS agents x $GENS gens, seed $SEED, pool $DAYS days"
& $py -m training_v2.cohort.runner `
  --breeding ga --n-agents $AGENTS --generations $GENS --days $DAYS `
  --exclude-days $SEALED --seed $SEED --parallel-agents $PAR --device cpu `
  --n-eval-days 4 --composite-score-mode locked_weighted `
  --output-dir $GA_DIR *> "$OUT\ga_train.log"
if ($LASTEXITCODE -ne 0) { Die "GA training" }
Log "ARM GA: training done"

# Held-out re-eval on the SEALED days (AFTER all training -- free box).
Log "REEVAL PBT on sealed days"
& $py -m tools.reevaluate_cohort --cohort-dir $PBT_DIR --eval-days $SEALED `
  --device cpu --argmax-eval *> "$OUT\pbt_reeval.log"
if ($LASTEXITCODE -ne 0) { Die "PBT reeval" }
Log "REEVAL GA on sealed days"
& $py -m tools.reevaluate_cohort --cohort-dir $GA_DIR --eval-days $SEALED `
  --device cpu --argmax-eval *> "$OUT\ga_reeval.log"
if ($LASTEXITCODE -ne 0) { Die "GA reeval" }

# Analysis: heritability / selection-noise / diversity / arch leaderboard.
Log "ANALYSE pbt_lineage vs ga scoreboard"
& $py -m tools.analyze_pbt "$PBT_DIR\pbt_lineage.jsonl" `
  --ga "$GA_DIR\scoreboard.jsonl" *> "$OUT\analysis.txt"
Log "A/B COMPLETE -- see $OUT\analysis.txt + the *_reeval.log held-out rows"
