#!/usr/bin/env bash
# Live leaderboard watcher for the tt_tick_002 gauntlet run. Refreshes every
# 120s, writing all per-tranche boards to registry/tt_tick_002/leaderboards.txt
# (atomic write, read-only on the run's files). Launch DETACHED via
# tools/launch_detached.py so it outlives the launching shell (same job-object
# reason the training run is detached).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/Scripts/python.exe -m tools.gauntlet_leaderboard \
  --dir registry/tt_tick_002 --watch 120 --top 0
