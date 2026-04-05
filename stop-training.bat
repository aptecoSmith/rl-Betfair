@echo off
REM Stop the rl-betfair training worker.
REM Kills by port AND by command line to catch orphaned child processes.

echo Stopping training worker...
echo.

REM --- Kill by port ---

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8002.*LISTENING"') do (
    echo   Killing PID %%a on port 8002
    taskkill /F /T /PID %%a >nul 2>&1
)

REM --- Kill any python process running training.worker ---

for /f "tokens=2 delims=," %%a in (
    'wmic process where "CommandLine like '%%training.worker%%'" get ProcessId /format:csv 2^>nul ^| findstr /r "[0-9]"'
) do (
    echo   Killing orphan PID %%a (training.worker)
    taskkill /F /T /PID %%a >nul 2>&1
)

echo.
echo Training worker stopped.
