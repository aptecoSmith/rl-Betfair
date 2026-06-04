# pbt-breeding -- autonomous MULTI-ERA PBT campaign with a HARD wall-clock stop.
#
# Operator (2026-06-04): "kick off the multi era training run. You have until
# about 5pm tomorrow to keep training." So the campaign runs the PBT promotion
# ladder continuously until 2026-06-05 17:00, accumulating R3 hall-of-fame
# champions, then STOPS -- even mid-run (a 5-gen run at this 3x(6/4) rotation
# depth is ~10h, so the deadline can land mid-generation; we tree-kill the
# training process at the deadline rather than wait for the run to finish).
#
# Each run is capped at a modest generation count and the wrapper RELAUNCHES
# on exit (crash OR clean finish) with a NEW seed, as long as >2h remain:
#   (1) a fresh process pool resets per-worker memory each run (warm-pool guard);
#   (2) a new seed = a fresh-blood ERA exploring configs the prior run didn't.
# All runs share ONE --output-dir, so pbt_hall_of_fame.jsonl + pbt_lineage.jsonl
# APPEND and the per-gen leaderboard.txt + model_register.csv accumulate across
# the whole multi-era campaign. A mid-run deadline kill keeps every champion
# frozen up to the last completed generation (the hall-of-fame appends per gen).
#
# This era ALSO carries the pbt-gpu-forward genes: big-ctx transformers train on
# the GPU lane (--gpu-policy-lane, cap 2), and fresh blood draws the new
# transformer_ffn_mult {2,4} + transformer_pos_encoding {learned,rope} + widened
# d512/depth-6 sizes.
#
# Viewable artifacts (regenerated every generation):
#   registry\pbt_long\leaderboard.txt      <- R3 champions, usual columns + frozen_at
#   registry\pbt_long\model_register.csv   <- every model: full settings + metrics
#
# ASCII-only + no $ErrorActionPreference='Stop' (PS 5.1 wraps native stderr;
# Python logs there). Training runs as a BACKGROUND process per run so the
# wrapper can poll the deadline and tree-kill it on time.
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$DIR = "registry\pbt_long"
$SEALED = @(
  "2026-05-20","2026-05-21","2026-05-22","2026-05-23","2026-05-24",
  "2026-05-25","2026-05-26","2026-05-27","2026-05-28","2026-05-29"
)
$PRED = "..\betfair-predictors\production"
$MANIFESTS = @(
  "$PRED\race-outcome\manifest.json",
  "$PRED\race-outcome-ranker\manifest.json",
  "$PRED\direction-predictor\manifest.json"
)
# HARD stop: operator's "about 5pm tomorrow". Absolute target so it is exact
# regardless of when the wrapper is (re)launched.
$DEADLINE = Get-Date "2026-06-05 17:00:00"
$GENS = 5   # per-run cap: ~10h at 3x(6/4) rotation -> a full era completes and
            # relaunches a 2nd era inside the window; the deadline caps the last.
New-Item -ItemType Directory -Force -Path $DIR | Out-Null
function Log($m) {
  "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" |
    Tee-Object -Append "$DIR\wrapper.log"
}

Log "PBT MULTI-ERA campaign START -- HARD deadline $($DEADLINE.ToString('yyyy-MM-dd HH:mm')); output $DIR"
$run = 0
$champs = 0
while ((Get-Date) -lt $DEADLINE) {
  $remaining = ($DEADLINE - (Get-Date)).TotalHours
  if ($remaining -lt 2.0) {
    Log "less than 2h to deadline ($([math]::Round($remaining,2))h) -- not starting another era."
    break
  }
  $run++
  # Persistent seed counter so relaunches (new era, crash/reboot recovery, or
  # manual) NEVER repeat fresh-blood draws across the campaign.
  $seedFile = "$DIR\seed_counter.txt"
  if (Test-Path $seedFile) { $seed = [int]((Get-Content $seedFile -Raw).Trim()) } else { $seed = 771 }
  ($seed + 1) | Out-File -Encoding ascii -NoNewline $seedFile
  Log "era $run (seed $seed) starting -- 16 agents, $GENS gens, 3x(6/4) rotation, predictors-ON, BC-on, GPU lane (cap 2); ${remaining}h to deadline"

  # Argument list (array so --exclude-days / --predictor-bundle-manifests expand
  # cleanly). Training runs as a BACKGROUND process so we can deadline-kill it.
  $argList = @(
    '-m','training_v2.cohort.runner',
    '--breeding','pbt','--n-agents','16','--generations',"$GENS",'--days','30',
    '--exclude-days') + $SEALED + @(
    '--seed',"$seed",'--parallel-agents','16','--device','cpu',
    '--composite-score-mode','locked_weighted','--big-model-threads','1',
    '--gpu-policy-lane','--gpu-lane-max-concurrent','2',
    '--use-race-outcome-predictor','--bc-pretrain-steps','500',
    '--predictor-bundle-manifests',$MANIFESTS[0],$MANIFESTS[1],$MANIFESTS[2],
    '--pbt-rotations','3','--pbt-train-per-rotation','6','--pbt-eval-per-rotation','4',
    '--pbt-r2-size','6','--pbt-r3-size','4','--pbt-promote-from-r1','3',
    '--pbt-promote-from-r2','2','--pbt-freeze-top-r3','2',
    '--output-dir',$DIR
  )
  $outLog = "$DIR\train_era$run.out.log"
  $errLog = "$DIR\train_era$run.err.log"
  $proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog

  # Poll until the era finishes OR the deadline lands -- then tree-kill (/T kills
  # the 16 worker children + their CUDA contexts; confirmed safe).
  $killed = $false
  while (-not $proc.HasExited) {
    if ((Get-Date) -ge $DEADLINE) {
      Log "DEADLINE reached mid-era $run -- tree-killing PID $($proc.Id) (champions up to the last completed gen are saved)"
      & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
      $killed = $true
      Start-Sleep -Seconds 15
      break
    }
    Start-Sleep -Seconds 120
  }
  if (-not $proc.HasExited) { try { $proc.Kill() } catch {} }
  $code = if ($proc.HasExited) { $proc.ExitCode } else { "killed" }

  $champs = 0
  if (Test-Path "$DIR\pbt_hall_of_fame.jsonl") {
    $champs = (Get-Content "$DIR\pbt_hall_of_fame.jsonl" | Measure-Object -Line).Lines
  }
  Log "era $run EXIT (code $code); R3 champions so far: $champs"
  if ($killed) { break }
  Start-Sleep -Seconds 15
}
Log "PBT MULTI-ERA campaign STOPPED at deadline -- $run era(s), $champs champion(s)."
