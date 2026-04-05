@echo off
REM Start all rl-betfair services: training worker, API backend, and Angular frontend.
REM Run from the repo root.  Press Ctrl+C in any window to stop that service.

echo Starting training worker on ws://localhost:8002 ...
start "rl-betfair Training" cmd /k ".venv\Scripts\activate && python -m training.worker"

echo Starting API backend on http://localhost:8001 ...
start "rl-betfair API" cmd /k ".venv\Scripts\activate && uvicorn api.main:app --reload --reload-exclude .claude --reload-exclude *.log --port 8001"

echo Starting Angular frontend on http://localhost:4202 ...
start "rl-betfair UI" cmd /k "cd frontend && npm start"

echo.
echo All services launching in separate windows.
echo   Training worker:  ws://localhost:8002
echo   API:              http://localhost:8001
echo   UI:               http://localhost:4202
