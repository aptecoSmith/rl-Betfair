@echo off
REM Start all rl-betfair services via the supervisor.
REM The supervisor manages the training worker, API, and frontend.
REM Run from the repo root.

echo Starting rl-betfair supervisor on http://localhost:9000 ...
start "rl-betfair Supervisor" cmd /k ".venv\Scripts\activate && python supervisor.py"

echo.
echo Supervisor launching in a separate window.
echo   Supervisor:  http://localhost:9000
echo   Worker:      ws://localhost:8002
echo   API:         http://localhost:8001
echo   UI:          http://localhost:4202
