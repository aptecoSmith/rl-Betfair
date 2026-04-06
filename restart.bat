@echo off
REM Restart all rl-betfair services — stops everything, waits, starts fresh.

call "%~dp0stop-all.bat"

echo.
echo Waiting 2 seconds for ports to release...
timeout /t 2 /nobreak >nul

call "%~dp0start-all.bat"
