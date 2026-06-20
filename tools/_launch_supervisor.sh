#!/usr/bin/env bash
# Detached launch wrapper for the rl-betfair supervisor (worker + api + frontend).
cd /c/Users/jsmit/source/repos/rl-betfair || exit 1
exec .venv/Scripts/python.exe supervisor.py > logs/supervisor_console.log 2>&1
