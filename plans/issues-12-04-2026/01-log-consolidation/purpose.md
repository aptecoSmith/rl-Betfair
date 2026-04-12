# 01 — Log Consolidation & Log Paths UI

## What

1. Move `registry/bet_logs/` under `logs/bet_logs/` so all log output
   lives under a single `logs/` tree.
2. Add `paths.bet_logs` to `config.yaml` (default: `logs/bet_logs`).
3. Update every production call site that constructs `bet_logs_dir` to
   read from config instead of hardcoding `registry/bet_logs`.
4. Add a `GET /api/admin/log-paths` endpoint that returns the resolved
   logs root path and subdirectory stats (file count per subfolder).
5. Add a "Log Files" card to the admin page showing the path, a copy
   button, and per-subfolder file counts.

## Why

- Logs are currently split across `logs/` and `registry/bet_logs/`.
  Consolidating them makes it easy to find, back up, or clean up all
  log output from one place.
- There's no way in the UI to see where logs live on disk or quickly
  open the folder. A copy-path button removes the guesswork.
