@echo off
REM Start the rl-betfair API backend and Angular frontend.
REM Run from the repo root.  Press Ctrl+C in either window to stop.

echo Starting API backend on http://localhost:8001 ...
start "rl-betfair API" cmd /k ".venv\Scripts\activate && uvicorn api.main:app --reload --reload-exclude .claude --reload-exclude *.log --port 8001"

echo Starting Angular frontend on http://localhost:4202 ...
start "rl-betfair UI" cmd /k "cd frontend && npm start"

echo.
echo Both servers launching in separate windows.
echo   API:  http://localhost:8001
echo   UI:   http://localhost:4202
