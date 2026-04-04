@echo off
REM Stop the rl-betfair API backend and Angular frontend.
REM Kills processes by their window titles set in start-ui.bat.

echo Stopping rl-betfair services...

echo Killing API backend (port 8001)...
taskkill /FI "WINDOWTITLE eq rl-betfair API*" /T /F >nul 2>&1

echo Killing Angular frontend (port 4202)...
taskkill /FI "WINDOWTITLE eq rl-betfair UI*" /T /F >nul 2>&1

echo.
echo All rl-betfair services stopped.
