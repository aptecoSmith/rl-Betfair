# Master TODO — Fix Failing Tests

Single session — bug fix / maintenance.

---

- [ ] **Session 01 — Triage and fix all known test failures**

  Three categories:

  ### A. E2E WebSocket timeout

  - Investigate `_wait_for_ws` in `test_e2e_training.py` — is 30s
    enough on a cold start?
  - Check if torch import + CUDA init in the subprocess accounts
    for the delay.
  - Fix: increase timeout to 60-90s, or add exponential backoff,
    or lazy-import torch in the worker entry point.

  ### B. Integration test timeouts (4.6, 4.7)

  - These run real model inference and hit the 30s global timeout.
  - Fix: add `@pytest.mark.timeout(120)` to these specific tests.
  - Or: use a minimal model config (lstm_hidden_size=32) in the
    test fixture to speed up inference.
  - Verify they pass with the increased timeout.

  ### C. Session 2.7b data tests

  - Check the MySQL `RaceCardRunners` table: are `timeform_comment`
    and past race columns populated for any date?
  - If the data source stopped providing these: make the tests skip
    gracefully when the columns are empty (not just when the table
    is missing).
  - If the data should be there: fix the extraction query.

  **Exit criteria:**
  - `pytest tests/ -q` → 0 failures, 0 errors.
  - Slow tests either pass with adequate timeouts or are marked
    `@pytest.mark.slow` for optional execution.
  - `progress.md` updated.
