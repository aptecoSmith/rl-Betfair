# pbt-gpu-forward -- FULL-GENE-SET champion campaign. Runs until interrupted.
#
# Operator (2026-06-06): "kick off a proper training run that generates champions.
# You don't need to limit by time now - keep going until I interrupt it, which
# won't be for at least 8 hours."
#
# Multi-ERA loop, NO deadline. Each era is a full 5-gen PBT ladder seeded with
# fresh blood (new seed), sharing ONE output dir so pbt_hall_of_fame.jsonl
# ACCUMULATES champions across eras. The ladder freezes top-R3 EVERY generation,
# so champions accumulate per-gen even within an unfinished era -- an interrupt
# never loses a completed generation. The wrapper relaunches the next era on any
# exit (crash OR clean finish) with the next seed. A 14h per-era wedge-guard
# tree-kills a hung era (the ctx<=128/d<=256 caps make a wedge unlikely; belt+
# braces). All 30 non-sealed days are the same set every era (only the rotation
# SPLIT changes with the seed), so the one pre-scan covers every era.
#
# Gene config: --enable-all-genes => ALL PHASE5 genes sampled
# (reward/aux/exit/direction-head/BC knobs); the 3 direction-LABEL-definition
# knobs pin 60/5/60 (one pre-scanned triple); BC pinned ON (500); predictors-ON;
# GPU lane cap 2.
#
# Kill to stop: taskkill /IM python.exe /F   (then kill this wrapper's powershell)
param([int]$Eras = 0)   # how many eras to run; 0 = loop until stopped (stop_pbt_loop.bat)
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$DIR = "registry\pbt_genes_v2"   # NEW-REWARD campaign (kept separate from the old spray-and-bail pbt_genes_full)
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
$GENS = 5
New-Item -ItemType Directory -Force -Path $DIR | Out-Null
function Log($m) {
  "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" |
    Tee-Object -Append "$DIR\wrapper.log"
}

Log "GENES-FULL campaign START -- $(if ($Eras -le 0) { 'loop until stopped' } else { [string]$Eras + ' era(s) then stop' }); --enable-all-genes (all PHASE5 sampled, label triple pinned 60/5/60); direction-predictor PINNED ON (obs is dir-dependent: +141 live features, so one dir-ON oracle covers all agents + BC is correct -- the direction GATE stays the per-agent gene); BC-on(500), predictors-ON, GPU lane cap 2, caps ctx<=128/d<=256; output $DIR"
$run = 0
while ($Eras -le 0 -or $run -lt $Eras) {
  $run++
  $seedFile = "$DIR\seed_counter.txt"
  if (Test-Path $seedFile) { $seed = [int]((Get-Content $seedFile -Raw).Trim()) } else { $seed = 900 }
  ($seed + 1) | Out-File -Encoding ascii -NoNewline $seedFile
  Log "era $run (seed $seed) starting -- 16 agents, $GENS gens, 3x(6/4) rotation, --enable-all-genes"
  $argList = @(
    '-m','training_v2.cohort.runner',
    '--breeding','pbt','--n-agents','16','--generations',"$GENS",'--days','30',
    '--exclude-days') + $SEALED + @(
    '--seed',"$seed",'--parallel-agents','16','--device','cpu',
    '--composite-score-mode','locked_weighted','--force-close-rate-penalty-weight','20',
    '--big-model-threads','1',
    '--gpu-policy-lane','--gpu-lane-max-concurrent','2',
    '--use-race-outcome-predictor','--use-direction-predictor','--bc-pretrain-steps','500',
    '--predictor-bundle-manifests',$MANIFESTS[0],$MANIFESTS[1],$MANIFESTS[2],
    '--enable-all-genes',
    '--pbt-rotations','3','--pbt-train-per-rotation','6','--pbt-eval-per-rotation','4',
    '--pbt-r2-size','6','--pbt-r3-size','4','--pbt-promote-from-r1','3',
    '--pbt-promote-from-r2','2','--pbt-freeze-top-r3','2',
    '--output-dir',$DIR
  )
  $outLog = "$DIR\train_seed${seed}.out.log"
  $errLog = "$DIR\train_seed${seed}.err.log"
  $proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog
  $proc.Id | Out-File -Encoding ascii -NoNewline "$DIR\train.pid"
  Log "era $run launched -- PID $($proc.Id); logs train_era$run.{out,err}.log"

  $deadline = (Get-Date).AddHours(14)
  while (-not $proc.HasExited) {
    if ((Get-Date) -ge $deadline) {
      Log "era $run WEDGE-GUARD 14h -- tree-killing PID $($proc.Id)"
      & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
      Start-Sleep -Seconds 15; break
    }
    Start-Sleep -Seconds 120
  }
  $code = if ($proc.HasExited) { $proc.ExitCode } else { "killed" }
  $champs = 0
  if (Test-Path "$DIR\pbt_hall_of_fame.jsonl") {
    $champs = (Get-Content "$DIR\pbt_hall_of_fame.jsonl" | Measure-Object -Line).Lines
  }
  Log "era $run EXIT (code $code); R3 champions accumulated: $champs"
  Start-Sleep -Seconds 15
}
