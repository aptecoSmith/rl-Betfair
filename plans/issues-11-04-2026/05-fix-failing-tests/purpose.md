# 05 — Fix Failing Tests

## Problem

Several tests are failing or timing out, getting routinely skipped
with "it's pre-existing" during development. This erodes trust in
the test suite — real regressions can hide behind known failures.

## Current failures (as of 2026-04-11)

### 1. E2E training test — WebSocket timeout

**Test:** `test_e2e_training.py::test_full_training_flow`
**Symptom:** `Worker WebSocket not ready on port 18002 within 30s.`
**Category:** Infrastructure / startup timing.

The test spawns a worker subprocess and waits 30s for its WebSocket
to accept connections. On Windows the worker takes too long to start
(loading torch, initialising CUDA, etc.).

**Likely fixes:**
- Increase the timeout (30s may be too tight on a cold start).
- Add a retry with backoff to the `_wait_for_ws` polling loop.
- Pre-warm the worker by importing heavy modules before the test.
- Or mark the test as slow/optional and ensure it runs in CI with
  appropriate timeouts.

### 2. Integration test timeouts (session 4.6 + 4.7)

**Tests:**
- `test_integration_session_4_6.py::test_ppo_update_no_nans_on_real_data`
- `test_integration_session_4_7.py::test_opportunity_windows_computed`

**Symptom:** Both timeout at 30s during policy forward pass on real
data. These run actual model inference on full-sized observations.

**Likely fixes:**
- Increase per-test timeout for these specific tests (they're
  integration tests, not unit tests).
- Use `@pytest.mark.timeout(120)` on these tests.
- Or use smaller model architectures (tiny LSTM) for these tests.

### 3. Session 2.7b data tests — missing data

**Tests:**
- `test_integration_session_2_7b.py::TestExtraction::test_timeform_comment_populated`
- `test_integration_session_2_7b.py::TestEpisodeBuilder::test_past_races_populated`
- `test_integration_session_2_7b.py::TestEpisodeBuilder::test_timeform_comment_loaded`

**Symptom:** `timeform_comment` and `past_races_json` columns exist
but have zero non-null values in the extracted data.

**Category:** Data issue. These tests require the MySQL
`RaceCardRunners` table to have `timeform_comment` and past race
data populated. The table exists and has rows (test isn't skipped),
but the specific columns are empty.

**Likely fixes:**
- If the data source no longer provides these fields: remove or
  skip the tests with a clear reason.
- If the data should be there: investigate why `RaceCardRunners`
  has empty `timeform_comment` / `past_races_json` columns.
- Make the tests conditional: skip if the specific columns are
  empty (not just if the table exists).

## Files touched

| File | Change |
|---|---|
| `tests/test_e2e_training.py` | Increase WS timeout or fix startup |
| `tests/test_integration_session_4_6.py` | Increase per-test timeout |
| `tests/test_integration_session_4_7.py` | Increase per-test timeout |
| `tests/test_integration_session_2_7b.py` | Fix skip conditions or data |
