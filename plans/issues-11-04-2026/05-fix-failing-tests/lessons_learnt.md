# Lessons Learnt — Fix Failing Tests

Append-only. Date each entry.

---

## 2026-04-11 — Current state

Full suite: **1694 passed, 3 failed, 3 skipped, 1 error, 1 xfailed**
(excluding e2e and the two timeout tests).

The 3 failures are all in `test_integration_session_2_7b.py` and
relate to `timeform_comment` / `past_races_json` being empty in the
MySQL database. These tests depend on live data that may not be
populated in the dev environment.

The e2e test (`test_e2e_training.py`) errors at setup — the worker
subprocess doesn't start its WebSocket within 30s. This has been
reported as pre-existing in multiple session progress entries,
suggesting it's a persistent Windows-specific timing issue.

The integration tests (`test_integration_session_4_6.py`,
`test_integration_session_4_7.py`) timeout at 30s during real model
inference — they're doing actual forward passes on full-sized
observations with real architectures. The 30s global timeout is too
tight for these.
