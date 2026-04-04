@echo off
REM Stop the rl-betfair API backend and Angular frontend.
REM Kills processes listening on ports 8001 and 4202.

echo Stopping rl-betfair services...

echo Killing API backend (port 8001)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo Killing Angular frontend (port 4202)...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":4202.*LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo All rl-betfair services stopped.
