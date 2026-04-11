# Fix Failing Tests — Session 01

## Before you start

Run the full test suite to see current state:
```bash
python -m pytest tests/ --timeout=60 -q 2>&1 | tail -20
```

Read `plans/issues-11-04-2026/05-fix-failing-tests/purpose.md` for
the full triage of known failures.

## Known failures (as of 2026-04-11)

### A. E2E: `test_e2e_training.py::test_full_training_flow`

WebSocket not ready within 30s. The worker subprocess is slow to
start (torch import + CUDA init). Fix the timeout or startup.

- Read `tests/test_e2e_training.py` — find `_wait_for_ws` and the
  30s timeout at line ~244.
- Increase to 60-90s.
- Consider whether a `@pytest.mark.slow` marker is appropriate so
  quick test runs can skip it.

### B. Timeouts: `test_integration_session_4_6.py`, `test_integration_session_4_7.py`

Policy forward pass on real data exceeds 30s global timeout.

- Add `@pytest.mark.timeout(120)` to the specific test functions.
- Or reduce model size in the test fixture.
- Verify they actually pass with more time.

### C. Data: `test_integration_session_2_7b.py` (3 failures)

`timeform_comment` and `past_races_json` columns are empty in the
database. Tests assert non-null counts > 0.

- Check if the MySQL table actually has data:
  ```sql
  SELECT COUNT(*) FROM RaceCardRunners
  WHERE timeform_comment IS NOT NULL AND timeform_comment != '';
  ```
- If empty: add `skipif` conditions on those specific tests for
  when the data columns are empty, with a reason string like
  "RaceCardRunners.timeform_comment not populated in test DB".
- If populated: the extraction query might be wrong — investigate.

## Exit criteria

- `python -m pytest tests/ --timeout=120 -q` → 0 failures, 0 errors.
- All fixes are minimal and justified.
- `progress.md` updated. Commit.
