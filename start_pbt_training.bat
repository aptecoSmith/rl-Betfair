@echo off
REM ====================================================================
REM  START the PBT full-gene champion campaign.
REM
REM  Set ERAS below = how many eras to run. Each era is a full 5-generation
REM  PBT ladder with fresh blood (~7h); champions accumulate across eras in
REM  registry\pbt_genes_full\pbt_hall_of_fame.jsonl.
REM    ERAS=0  -> loop forever until you run stop_pbt_loop.bat
REM    ERAS=N  -> run N eras, then stop on its own
REM ====================================================================
set ERAS=5

cd /d "%~dp0"
echo Launching PBT campaign for %ERAS% era(s)  (0 = until stopped)...
start "pbt-campaign" /min powershell -NoProfile -ExecutionPolicy Bypass -File "plans\pbt-gpu-forward\_scripts\run_genes_campaign.ps1" -Eras %ERAS%
echo.
echo Launched in a minimized "pbt-campaign" PowerShell window.
echo   leaderboard : registry\pbt_genes_full\leaderboard.txt
echo   champions   : registry\pbt_genes_full\pbt_hall_of_fame.jsonl
echo   stop it     : run stop_pbt_loop.bat
echo.
echo You can close this window now.
pause
