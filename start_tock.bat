@echo off
REM ============================================================================
REM  start_tock.bat — kick off a LOCKSTEP TOCK (exploit).
REM
REM  CURRENT ERA: tt_tock_004 / hypothesis_004 — the P1 MATURATION-GATE era
REM  (plans/maturation-raising/). It = the tock_003 held-out-winner recipe
REM  (unchanged seed block below) PLUS the one change under test: the Path-C
REM  mature_prob open-gate, band-seeded
REM    --seed-gene mature_prob_loss_weight=2.0:4.0     (train the mature head)
REM    --seed-gene mature_prob_open_threshold=0.15:0.35 (mask opens the head
REM                                                       predicts won't mature)
REM  Hypothesis: masking low-maturation opens at the decision site makes agents
REM  open FEWER, will-mature pairs -> opens down >=30%, naked down, maturation up.
REM  BASELINE TO BEAT: tt_tock_003 on the sealed-7, fc=120 NET day_pnl.
REM
REM  The 003 recipe block (below) — seed AUTO-DERIVED by:
REM    python -m tools.derive_seed --board registry/cross_era_holdout_board.jsonl --top-n 6
REM  which scans ALL 50 genes by the population-relative spread metric (winners'
REM  range / what the whole population explored). The 6 held-out winners are ALL
REM  LSTMs with the H001 direction-gate + BC machinery and CONVERGE on: low
REM  learning_rate + entropy_coeff, profit-seeker arb_spread ~0.045, reward_clip
REM  ~8, direction_gate_threshold ~0.31, naked_loss_scale ~0.65,
REM  direction_prob_loss_weight ~1.3. 12 seeds; the other ~38 genes did NOT
REM  converge so they drift full-width via --enable-all-genes. (Re-derive after
REM  any new era and paste the fresh seed — never hand-pick a subset.)
REM
REM  Selection = locked_weighted (the selector that PRODUCED those winners). NOT
REM  the maturation composite: the cross-era board showed maturation RATE does
REM  NOT generalise (the 46% in-sample collapsed to ~8% held-out) but LOCKED does.
REM
REM  Held-out SELECTION (maturation-raising/holdout-selection.md): each tranche
REM  selects survivors on a FIXED validation set (fc=0), NOT the rotating
REM  tranche-eval days, so winners are chosen for GENERALISATION (cross-era
REM  finding 2026-06-11: in-sample/rotation doesn't predict held-out). A TOCK is
REM  EXPLOIT, so the flavor is 'contiguous' = the V days immediately before the
REM  sealed final-test (deploy-recent: select for the regime you deploy into).
REM  (A Tick uses 'sampled' = spread across time; see start_tick.bat.) The sealed
REM  final-test (--holdout-recent 7) stays inviolate. V=7 here is tunable; 0
REM  reverts to the old per-tranche-eval selection.
REM
REM  Predictors + BC always ON (BC carried by the seed, no --bc-pretrain-steps).
REM  Usage:  start_tock.bat [era_id]      (default tt_tock_004)
REM  Plan:   plans/maturation-raising/ (P1) ; plans/lockstep-cohort/build_plan.md
REM ============================================================================
setlocal
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "PRED=..\betfair-predictors\production"
set "ERA=%~1"
if "%ERA%"=="" set "ERA=tt_tock_004"
set "DIR=registry\%ERA%"

if exist "%DIR%" (
  echo [start_tock] WARNING: %DIR% exists — wiping it for a fresh run.
  rmdir /s /q "%DIR%"
)
mkdir "%DIR%"

echo [start_tock] launching TOCK %ERA%  ^(gauntlet, seeded from held-out winners^)  -^>  %DIR%
"%PY%" -m training_v2.cohort.runner ^
  --breeding gauntlet --generations 5 --n-agents 16 --parallel-agents 16 --device cpu --seed 4004 ^
  --survivor-fraction 0.5 ^
  --pbt-train-per-rotation 6 --pbt-eval-per-rotation 4 ^
  --holdout-recent 7 ^
  --validation-holdout-recent 7 --validation-holdout-mode contiguous ^
  --composite-score-mode locked_weighted ^
  --force-close-rate-penalty-weight 20 ^
  --big-model-threads 1 --gpu-policy-lane --gpu-lane-max-concurrent 2 ^
  --use-race-outcome-predictor --use-direction-predictor ^
  --predictor-bundle-manifests ^
      "%PRED%\race-outcome\manifest.json" ^
      "%PRED%\race-outcome-ranker\manifest.json" ^
      "%PRED%\direction-predictor\manifest.json" ^
  --enable-all-genes ^
  --era-type tock --hypothesis-id hypothesis_004 --era-id "%ERA%" ^
  --seed-gene architecture=lstm ^
  --seed-gene use_direction_predictor=true ^
  --seed-gene direction_gate_enabled=true ^
  --seed-gene predictor_lean_obs=false ^
  --seed-gene bc_pretrain_steps=500 ^
  --seed-gene learning_rate=3e-5:9e-5 ^
  --seed-gene entropy_coeff=0.00014:0.0069 ^
  --seed-gene arb_spread_target_lock_pct=0.042:0.048 ^
  --seed-gene reward_clip=7.0:9.4 ^
  --seed-gene direction_gate_threshold=0.27:0.36 ^
  --seed-gene naked_loss_scale=0.48:0.83 ^
  --seed-gene direction_prob_loss_weight=0.97:1.72 ^
  --seed-gene mature_prob_loss_weight=2.0:4.0 ^
  --seed-gene mature_prob_open_threshold=0.15:0.35 ^
  --output-dir "%DIR%" > "%DIR%\tock.console.log" 2>&1

echo [start_tock] training done — building held-out leaderboard (re-eval on sealed-7)...
call "%~dp0score_holdout.bat" %ERA% >> "%DIR%\tock.console.log" 2>&1

echo [start_tock] TOCK %ERA% done.  in-sample: %DIR%\model_register.csv   held-out (fc=120): %DIR%\holdout_board_fc120.txt
endlocal
