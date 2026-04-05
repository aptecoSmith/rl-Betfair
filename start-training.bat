@echo off
REM Start the rl-betfair training worker.
REM Runs independently from the API — survives API restarts.
REM Run from the repo root.

echo Starting training worker on ws://localhost:8002 ...
start "rl-betfair Training" cmd /k ".venv\Scripts\activate && python -m training.worker"

echo.
echo Training worker launching in a separate window.
echo   Worker:  ws://localhost:8002
