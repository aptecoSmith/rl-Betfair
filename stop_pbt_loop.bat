@echo off
REM ====================================================================
REM  STOP the PBT campaign.
REM  Kills the wrapper loop FIRST (so it stops relaunching eras), then the
REM  training python. Any champions already frozen are preserved on disk.
REM  (This kills ALL python.exe -- only run it when the PBT campaign is the
REM   python you're running.)
REM ====================================================================
echo Stopping the PBT campaign (wrapper loop + training python)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$me=$PID; Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'powershell.exe' -and $_.ProcessId -ne $me -and $_.CommandLine -and ($_.CommandLine -match 'run_genes_campaign') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }; Start-Sleep 2; Get-Process python -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue"
echo.
echo Stopped. Champions preserved in registry\pbt_genes_full\
pause
