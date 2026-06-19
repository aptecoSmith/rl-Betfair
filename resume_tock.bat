@echo off
REM ============================================================================
REM  resume_tock.bat [era_id] — RESUME a stopped lockstep TOCK from its on-disk
REM  checkpoint (registry\<era>\_resume_state.json). Re-runs the in-progress
REM  generation; all earlier generations' weights + scoreboard rows are kept.
REM
REM  Use this after killing start_tock.bat mid-run. It does NOT wipe the dir and
REM  takes NO --seed-gene (the cohort is loaded from the checkpoint — survivors
REM  warm-start from their saved weights, the rest are from-scratch catch-up
REM  mutants, reconstructed in runner.py's resume branch).
REM
REM  CRITICAL: the day/tranche/predictor flags below MUST match start_tock.bat
REM  exactly (--pbt-train-per-rotation / --pbt-eval-per-rotation / --holdout-recent
REM  + the same predictor bundle + the same data dir) so the tranche schedule and
REM  n_generations rebuild identically. If you changed start_tock.bat, change this
REM  too. --seed / rng are overridden by the checkpoint's restored rng_state.
REM
REM  Usage:  resume_tock.bat [era_id]      (default tt_tock_003)
REM  Plan:   plans/lockstep-cohort/build_plan.md
REM ============================================================================
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PRED=..\betfair-predictors\production"
set "ERA=%~1"
if "%ERA%"=="" set "ERA=tt_tock_003"
set "DIR=registry\%ERA%"

if not exist "%DIR%\_resume_state.json" (
  echo [resume_tock] ERROR: no checkpoint at %DIR%\_resume_state.json — nothing to resume.
  echo [resume_tock] ^(Did the run reach the end of gen 0? Checkpoints are written at each gen start.^)
  exit /b 2
)

echo [resume_tock] RESUMING TOCK %ERA% from checkpoint  -^>  %DIR%
"%PY%" -m training_v2.cohort.runner ^
  --breeding lockstep --n-agents 16 --parallel-agents 16 --device cpu --seed 4003 ^
  --survivor-fraction 0.5 ^
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 ^
  --holdout-recent 7 ^
  --composite-score-mode locked_per_std ^
  --force-close-rate-penalty-weight 20 ^
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 ^
  --use-race-outcome-predictor --use-direction-predictor ^
  --predictor-bundle-manifests ^
      "%PRED%\race-outcome\manifest.json" ^
      "%PRED%\race-outcome-ranker\manifest.json" ^
      "%PRED%\direction-predictor\manifest.json" ^
  --enable-all-genes ^
  --era-type tock --hypothesis-id hypothesis_003 --era-id "%ERA%" ^
  --resume-from "%DIR%" ^
  --output-dir "%DIR%" >> "%DIR%\tock.console.log" 2>&1

echo [resume_tock] training done — building held-out leaderboard (re-eval on sealed-7)...
call "%~dp0score_holdout.bat" %ERA% >> "%DIR%\tock.console.log" 2>&1

echo [resume_tock] TOCK %ERA% done.  in-sample: %DIR%\model_register.csv   held-out (fc=120): %DIR%\holdout_board_fc120.txt
endlocal
