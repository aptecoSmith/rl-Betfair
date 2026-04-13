@echo off
REM Stop all rl-betfair services.
REM Kills the supervisor (which kills its children on exit) + sweeps for orphans.
REM Safety: verifies each PID is python/node/npm before killing.
REM         Never uses /T (tree kill) to avoid crashing system/GPU processes.

echo Stopping rl-betfair services...
echo.

REM --- Kill by port, but only if the process is python/node/npm ---

for %%p in (9000 8002 8001 4202) do (
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%%p.*LISTENING"') do (
        call :safe_kill %%a %%p
    )
)

REM --- Kill by command line pattern (catches any stragglers) ---

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%training.worker%%' OR CommandLine like '%%api.main%%' OR CommandLine like '%%uvicorn%%api%%' OR CommandLine like '%%supervisor.py%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (matched rl-betfair command line)
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All rl-betfair services stopped.
exit /b 0


:safe_kill
REM %1 = PID, %2 = port
REM Check that the process image name starts with python, node, or npm
for /f "tokens=1 delims=," %%n in ('tasklist /FI "PID eq %1" /FO CSV /NH 2^>nul ^| findstr /i "python node npm"') do (
    echo   Killing %%~n on port %2 (PID %1)
    taskkill /F /PID %1 >nul 2>&1
    exit /b 0
)
echo   Skipping unknown process on port %2 (PID %1)
exit /b 0
