@echo off
REM Stop all rl-betfair services.
REM Kills by port AND by command line to catch orphaned child processes.

echo Stopping rl-betfair services...
echo.

REM --- Kill by port (catches the main listener) ---

for %%p in (8002 8001 4202) do (
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%%p.*LISTENING"') do (
        echo   Killing PID %%a on port %%p
        taskkill /F /PID %%a >nul 2>&1
        REM Also kill entire process tree rooted at this PID
        taskkill /F /T /PID %%a >nul 2>&1
    )
)

REM --- Kill any python processes running our modules ---
REM   Catches orphaned children (e.g. multiprocessing forks, training threads)

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%training.worker%%' OR CommandLine like '%%api.main%%' OR CommandLine like '%%uvicorn%%api%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (matched rl-betfair command line)
    taskkill /F /T /PID %%a >nul 2>&1
)

REM --- Kill Angular dev server (node) ---

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%ng serve%%' AND CommandLine like '%%4202%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (Angular dev server)
    taskkill /F /T /PID %%a >nul 2>&1
)

echo.
echo All rl-betfair services stopped.
