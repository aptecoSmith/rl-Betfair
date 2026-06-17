@echo off
REM ============================================================================
REM  start_tick.bat — kick off a full-width TICK (explore) on the LOCKSTEP cohort.
REM
REM  Lockstep replaces the old tier-ladder PBT: ONE population marches through
REM  chronological ~10-day tranches (6 train / 4 eval), one generation per
REM  tranche; top --survivor-fraction survive as weight-clones, the rest are
REM  from-scratch catch-up mutants. A TICK is full-width (--enable-all-genes,
REM  NO --seed-gene) — it explores the gene space. (A TOCK seeds promising
REM  configs at gen 0 with --seed-gene; see start_tock.bat.)
REM
REM  Predictors + BC are ALWAYS ON (the real config). Selection uses the
REM  maturation-aware composite (locked_maturation) by default.
REM
REM  Held-out SELECTION (maturation-raising/holdout-selection.md): each tranche
REM  selects survivors on a FIXED validation set (fc=0), NOT the rotating
REM  tranche-eval days, so winners are chosen for GENERALISATION (cross-era
REM  finding 2026-06-11: in-sample/rotation doesn't predict held-out). For a
REM  TICK (explore) the validation flavor is 'sampled' — V days spread EVENLY
REM  across the timeline (regime-robust) rather than one recent window, so the
REM  explore isn't narrowed onto the latest regime. (A Tock uses 'contiguous' =
REM  deploy-recent.) The sealed final-test (--holdout-recent 7) stays inviolate.
REM  V=10 here is tunable; 0 reverts to the old per-tranche-eval selection.
REM
REM  Usage:  start_tick.bat [era_id]      (default era_id = tt_tick_002)
REM  Plan:   plans/lockstep-cohort/build_plan.md
REM ============================================================================
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PRED=..\betfair-predictors\production"
set "ERA=%~1"
if "%ERA%"=="" set "ERA=tt_tick_002"
set "DIR=registry\%ERA%"

if exist "%DIR%" (
  echo [start_tick] WARNING: %DIR% exists — wiping it for a fresh run.
  rmdir /s /q "%DIR%"
)
mkdir "%DIR%"

echo [start_tick] launching TICK %ERA%  ^(gauntlet, predictors+BC ON^)  -^>  %DIR%
"%PY%" -m training_v2.cohort.runner ^
  --breeding gauntlet --generations 5 --n-agents 16 --parallel-agents 16 --device cpu --seed 3002 ^
  --survivor-fraction 0.5 ^
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 ^
  --holdout-recent 7 ^
  --validation-holdout-recent 10 --validation-holdout-mode sampled ^
  --composite-score-mode locked_weighted ^
  --force-close-rate-penalty-weight 20 ^
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 ^
  --use-race-outcome-predictor --use-direction-predictor --bc-pretrain-steps 500 ^
  --predictor-bundle-manifests ^
      "%PRED%\race-outcome\manifest.json" ^
      "%PRED%\race-outcome-ranker\manifest.json" ^
      "%PRED%\direction-predictor\manifest.json" ^
  --enable-all-genes ^
  --era-type tick --era-id "%ERA%" ^
  --output-dir "%DIR%" > "%DIR%\tick.console.log" 2>&1

echo [start_tick] training done — building held-out leaderboard (re-eval on sealed-7)...
call "%~dp0score_holdout.bat" %ERA% >> "%DIR%\tick.console.log" 2>&1

echo [start_tick] TICK %ERA% done.  in-sample: %DIR%\model_register.csv   held-out (fc=120): %DIR%\holdout_board_fc120.txt
endlocal
