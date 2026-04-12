# Master TODO — Log Consolidation

## Session 1: Move bet_logs + config + API + UI

### Backend — move bet_logs under logs/

- [ ] Add `paths.bet_logs: logs/bet_logs` to `config.yaml`
- [ ] Update `api/main.py` to read `config["paths"]["bet_logs"]` instead
      of deriving from `db_path.parent / "bet_logs"`
- [ ] Update `training/worker.py` — same change
- [ ] Update `scripts/session_9_shakeout.py` — use config path with
      session tag suffix
- [ ] Update `ModelStore.__init__` default fallback to `logs/bet_logs`
      instead of `db_path.parent / "bet_logs"`
- [ ] Move existing `registry/bet_logs/` directory to `logs/bet_logs/`
- [ ] Move any `registry/session_9_bet_logs/` to `logs/session_9_bet_logs/`
- [ ] Update `.gitignore` if needed (bet_logs path reference)

### Backend — log-paths API endpoint

- [ ] Add `GET /api/admin/log-paths` to `api/routers/admin.py`
- [ ] Returns: `{ logs_root: str, subdirs: [{ name, file_count }] }`
- [ ] Reads the resolved `paths.logs` from config, scans subdirectories

### Frontend — Log Files card

- [ ] Add `LogPathsResponse` model to `admin.model.ts`
- [ ] Add `getLogPaths()` method to `api.service.ts`
- [ ] Add "Log Files" card to admin page below the services section
- [ ] Show resolved path with a "Copy path" button
- [ ] Show subdirectory list with file counts
- [ ] Style consistent with existing admin cards

### Tests

- [ ] Test `GET /api/admin/log-paths` returns correct structure
- [ ] Test bet_logs are written to `logs/bet_logs/` not `registry/bet_logs/`
- [ ] Existing bet-log tests still pass (test fixtures use tmp_path, unaffected)

### Verify

- [ ] `python -m pytest tests/ --timeout=120 -q` — all green
- [ ] `cd frontend && ng build` — clean
- [ ] Manual check: admin page shows log paths card
