@echo off
REM Stop all rl-betfair services.
REM Kills the supervisor (which kills its children on exit) + sweeps for orphans.

echo Stopping rl-betfair services...
echo.

REM --- Kill supervisor by port ---

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":9000.*LISTENING"') do (
    echo   Killing supervisor PID %%a on port 9000
    taskkill /F /T /PID %%a >nul 2>&1
)

REM --- Kill any remaining child processes by port ---

for %%p in (8002 8001 4202) do (
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%%p.*LISTENING"') do (
        echo   Killing orphan PID %%a on port %%p
        taskkill /F /T /PID %%a >nul 2>&1
    )
)

REM --- Kill by command line pattern (catches any stragglers) ---

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%training.worker%%' OR CommandLine like '%%api.main%%' OR CommandLine like '%%uvicorn%%api%%' OR CommandLine like '%%supervisor.py%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (matched rl-betfair command line)
    taskkill /F /T /PID %%a >nul 2>&1
)

echo.
echo All rl-betfair services stopped.
