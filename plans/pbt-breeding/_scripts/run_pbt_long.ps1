# pbt-breeding -- LONG autonomous PBT campaign (operator left for ~18-20h).
#
# Runs the PBT promotion ladder continuously, accumulating R3 hall-of-fame
# champions, until a ~20h deadline. Each campaign run is capped at a modest
# generation count and the wrapper RELAUNCHES on exit (crash OR done) with a
# new seed: (1) a fresh process pool resets per-worker memory each run --
# guarding the warm-pool-growth risk; (2) a new seed explores fresh-blood
# configs the previous run didn't. All runs share ONE --output-dir, so
# pbt_hall_of_fame.jsonl + pbt_lineage.jsonl APPEND and the runner's per-gen
# leaderboard.txt + model_register.csv accumulate across the whole campaign.
#
# Viewable artifacts (regenerated every generation):
#   registry\pbt_long\leaderboard.txt      <- R3 champions, usual columns + frozen_at
#   registry\pbt_long\model_register.csv   <- every model: full settings + metrics
#
# ASCII-only + no $ErrorActionPreference='Stop' (PS 5.1 wraps native stderr;
# Python logs there). Check $LASTEXITCODE.
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$DIR = "registry\pbt_long"
$SEALED = @(
  "2026-05-20","2026-05-21","2026-05-22","2026-05-23","2026-05-24",
  "2026-05-25","2026-05-26","2026-05-27","2026-05-28","2026-05-29"
)
$DEADLINE = (Get-Date).AddHours(20)
New-Item -ItemType Directory -Force -Path $DIR | Out-Null
function Log($m) {
  "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" |
    Tee-Object -Append "$DIR\wrapper.log"
}

Log "PBT LONG campaign START -- deadline $($DEADLINE.ToString('yyyy-MM-dd HH:mm')); output $DIR"
$run = 0
while ((Get-Date) -lt $DEADLINE) {
  $run++
  $seed = 770 + $run
  Log "campaign run $run (seed $seed) starting -- 16 agents, 25 gens, 3x4 rotation"
  & $py -m training_v2.cohort.runner `
    --breeding pbt --n-agents 16 --generations 25 --days 12 `
    --exclude-days $SEALED --seed $seed --parallel-agents 16 --device cpu `
    --composite-score-mode locked_weighted `
    --pbt-rotations 3 --pbt-train-per-rotation 2 --pbt-eval-per-rotation 2 `
    --pbt-r2-size 6 --pbt-r3-size 4 --pbt-promote-from-r1 3 `
    --pbt-promote-from-r2 2 --pbt-freeze-top-r3 2 `
    --output-dir $DIR *>> "$DIR\train.log"
  $code = $LASTEXITCODE
  $champs = 0
  if (Test-Path "$DIR\pbt_hall_of_fame.jsonl") {
    $champs = (Get-Content "$DIR\pbt_hall_of_fame.jsonl" | Measure-Object -Line).Lines
  }
  Log "campaign run $run EXIT (code $code); R3 champions so far: $champs"
  Start-Sleep -Seconds 15
}
Log "PBT LONG campaign deadline reached -- stopping after $run run(s)."
