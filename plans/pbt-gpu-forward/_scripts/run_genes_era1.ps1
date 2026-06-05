# pbt-gpu-forward -- SINGLE-ERA validation run with the newly-promoted genes.
#
# Operator (2026-06-05): "Get the genes in to the mix, then kick off a 1 era
# training run. Likely that will finish in a few hours. We'll have a chat about
# reward later." + "if there is a pretrained bc, we should use it" + "same reward".
#
# This is NOT the multi-era campaign wrapper -- it's ONE 5-gen era, fresh output
# dir, that validates the promoted bail/exit machinery is sampled + helps:
#   * use_direction_predictor {on,off}  (gate's predictor; obs UNCHANGED -> BC ok)
#   * direction_gate_enabled  {on,off}  (COUPLED on -> predictor on; the open-blocker)
#   * force_close_before_off_seconds {0,60,120}  (auto-bail naked pairs at T-N)
#   * close_walk_ticks {0,5,10}          (let the close leg walk to complete the hedge)
#   * bc_pretrain_steps {0,500} gene EXISTS, but --bc-pretrain-steps 500 below pins
#     it ON for all (operator: "use the BC we did the work for").
# These auto-sample in fresh blood (ARCHITECTURE_GENE_NAMES) -- no --enable-gene.
# Reward shaping stays OFF (PHASE5 enabled_set empty) -- "chat about reward later".
#
# Transformer size caps (ctx<=128, d<=256) are in the fresh-blood sampler, so no
# d512/ctx256 straggler can wedge a generation (the gen-5 force-kill cause last era).
$env:PYTHONWARNINGS = "ignore"
$py = ".\.venv\Scripts\python.exe"
$DIR = "registry\pbt_genes_era1"
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
$SEED = 900
# 3 gens (not the wrapper's 5-gen "era") to fit the operator's "a few hours":
# gen 0 fresh-blood SAMPLES the new bail/exit genes; gens 1-2 let the PBT ladder
# PROMOTE the agents that use them well -> validates the genes propagate. A fuller
# multi-era campaign follows the reward chat. ~3-5h with the ctx<=128/d<=256 caps.
$GENS = 3
New-Item -ItemType Directory -Force -Path $DIR | Out-Null
function Log($m) {
  "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" |
    Tee-Object -Append "$DIR\wrapper.log"
}

Log "GENES-ERA1 START (seed $SEED) -- 16 agents, $GENS gens, 3x(6/4) rotation, predictors-ON, BC-on (pinned 500), GPU lane (cap 2); --enable-all-genes = ALL PHASE5 sampled (reward/aux/exit/direction-head/BC knobs) + the 5 bail/exit structural genes; the 3 direction-LABEL-definition knobs pinned 60/5/60 (one pre-scanned triple); caps ctx<=128/d<=256; output $DIR"

$argList = @(
  '-m','training_v2.cohort.runner',
  '--breeding','pbt','--n-agents','16','--generations',"$GENS",'--days','30',
  '--exclude-days') + $SEALED + @(
  '--seed',"$SEED",'--parallel-agents','16','--device','cpu',
  '--composite-score-mode','locked_weighted','--big-model-threads','1',
  '--gpu-policy-lane','--gpu-lane-max-concurrent','2',
  '--use-race-outcome-predictor','--bc-pretrain-steps','500',
  '--predictor-bundle-manifests',$MANIFESTS[0],$MANIFESTS[1],$MANIFESTS[2],
  '--enable-all-genes',
  '--pbt-rotations','3','--pbt-train-per-rotation','6','--pbt-eval-per-rotation','4',
  '--pbt-r2-size','6','--pbt-r3-size','4','--pbt-promote-from-r1','3',
  '--pbt-promote-from-r2','2','--pbt-freeze-top-r3','2',
  '--output-dir',$DIR
)
$outLog = "$DIR\train.out.log"
$errLog = "$DIR\train.err.log"
$proc = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
  -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id | Out-File -Encoding ascii -NoNewline "$DIR\train.pid"
Log "GENES-ERA1 launched -- PID $($proc.Id); logs $outLog / $errLog"

# Poll until the era finishes OR a 6h wedge-guard deadline lands (the ctx<=128/
# d<=256 caps make a straggler-wedge unlikely, but cap the tail in case a gen
# hangs -- per-gen champions are frozen to pbt_hall_of_fame.jsonl every gen, so a
# kill never loses a completed generation).
$DEADLINE = (Get-Date).AddHours(6)
while (-not $proc.HasExited) {
  if ((Get-Date) -ge $DEADLINE) {
    Log "WEDGE-GUARD 6h reached -- tree-killing PID $($proc.Id) (per-gen champions saved)"
    & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
    Start-Sleep -Seconds 15
    break
  }
  Start-Sleep -Seconds 120
}
$code = if ($proc.HasExited) { $proc.ExitCode } else { "killed" }
$champs = 0
if (Test-Path "$DIR\pbt_hall_of_fame.jsonl") {
  $champs = (Get-Content "$DIR\pbt_hall_of_fame.jsonl" | Measure-Object -Line).Lines
}
Log "GENES-ERA1 EXIT (code $code); R3 champions: $champs"
