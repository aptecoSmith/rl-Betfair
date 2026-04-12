# Session: Log Consolidation & Log Paths UI

Read `CLAUDE.md` and `plans/issues-12-04-2026/01-log-consolidation/` before
starting. Follow `master_todo.md` in order. Mark items done as you go and
update `progress.md` at the end.

## Context

All log output should live under `logs/`. Currently `registry/bet_logs/`
contains evaluation bet log parquets — these need to move to `logs/bet_logs/`.

Three production sites hardcode `bet_logs_dir` relative to the registry DB:
- `api/main.py:54` — `str(Path(db_path).parent / "bet_logs")`
- `training/worker.py:122` — same pattern
- `scripts/session_9_shakeout.py:287` — `f"registry/{args.session_tag}_bet_logs"`

`ModelStore.__init__` (`registry/model_store.py:173`) defaults to
`db_path.parent / "bet_logs"` when no `bet_logs_dir` is passed.

## What to do

1. Add `paths.bet_logs: logs/bet_logs` to `config.yaml`.
2. Update the three production call sites to read from
   `config["paths"]["bet_logs"]` instead of deriving from `db_path`.
3. Update `ModelStore.__init__` default to `logs/bet_logs`.
4. Move existing data: `registry/bet_logs/` → `logs/bet_logs/`,
   `registry/session_9_bet_logs/` → `logs/session_9_bet_logs/`.
5. Add `GET /api/admin/log-paths` endpoint in `api/routers/admin.py`.
   Returns `{ logs_root: str, subdirs: [{ name: str, file_count: int }] }`.
   Scans the resolved `paths.logs` directory for subdirectories.
6. Add a "Log Files" card to `frontend/src/app/admin/admin.html` below
   the services section. Shows the logs root path, a copy button, and
   per-subfolder file counts. Add the model, service method, and styles.

## Constraints

- Test fixtures use `tmp_path` — they should be unaffected by the default
  change, but verify all tests pass.
- Don't break existing parquet read paths — the `ModelStore.bet_logs_dir`
  attribute is what matters, not the default.
- `python -m pytest tests/ --timeout=120 -q` must pass.
- `cd frontend && ng build` must be clean.

## Commit

Single commit: `feat: consolidate logs under logs/ + add log paths UI`
Push: `git push all`
