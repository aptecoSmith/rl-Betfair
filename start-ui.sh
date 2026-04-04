#!/usr/bin/env bash
# Start the rl-betfair API backend and Angular frontend.
# Run from the repo root.  Press Ctrl+C to stop both.

set -e
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

cleanup() {
    echo "Stopping servers..."
    kill $API_PID $UI_PID 2>/dev/null
    wait $API_PID $UI_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "Starting API backend on http://localhost:8001 ..."
source "$REPO_DIR/.venv/bin/activate"
uvicorn api.main:app --reload --port 8001 &
API_PID=$!

echo "Starting Angular frontend on http://localhost:4202 ..."
cd "$REPO_DIR/frontend"
npm start &
UI_PID=$!

echo ""
echo "Both servers running."
echo "  API:  http://localhost:8001"
echo "  UI:   http://localhost:4202"
echo "Press Ctrl+C to stop both."

wait
