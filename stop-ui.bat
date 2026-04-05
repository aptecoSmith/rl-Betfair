@echo off
REM Stop the rl-betfair API backend and Angular frontend.
REM Kills by port AND by command line to catch orphaned child processes.

echo Stopping rl-betfair UI services...
echo.

REM --- Kill by port ---

for %%p in (8001 4202) do (
    for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%%p.*LISTENING"') do (
        echo   Killing PID %%a on port %%p
        taskkill /F /T /PID %%a >nul 2>&1
    )
)

REM --- Kill any python processes running the API ---

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%api.main%%' OR CommandLine like '%%uvicorn%%api%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (API process)
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
echo UI services stopped.
