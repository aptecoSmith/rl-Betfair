#!/usr/bin/env bash
# Start the rl-betfair training worker.
# Runs independently from the API — survives API restarts.
# Run from the repo root.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting training worker on ws://localhost:8002 ..."

# Activate venv
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    source .venv/Scripts/activate
fi

python -m training.worker "$@"
