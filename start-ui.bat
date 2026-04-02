@echo off
REM Start the rl-betfair API backend and Angular frontend.
REM Run from the repo root.  Press Ctrl+C in either window to stop.

echo Starting API backend on http://localhost:8000 ...
start "rl-betfair API" cmd /k ".venv\Scripts\activate && uvicorn api.main:app --reload --port 8000"

echo Starting Angular frontend on http://localhost:4200 ...
start "rl-betfair UI" cmd /k "cd frontend && npm start"

echo.
echo Both servers launching in separate windows.
echo   API:  http://localhost:8000
echo   UI:   http://localhost:4200
