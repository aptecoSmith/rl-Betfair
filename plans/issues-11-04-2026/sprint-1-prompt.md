# Sprint 1 — Quick Wins (3 sessions)

Work through these three issues sequentially. Each is a single session.
Commit after each. Run the full test suite after the last one.

## Before you start

Read `plans/issues-11-04-2026/order.md` for context on why these are first.

## Issue 05 — Fix Failing Tests

Read the full plan folder: `plans/issues-11-04-2026/05-fix-failing-tests/`

Start with `purpose.md` for the triage, then follow `session_prompt.md`.

Three categories to fix:
- E2E WebSocket timeout (increase 30s → 60-90s)
- Integration test timeouts 4.6/4.7 (add per-test `@pytest.mark.timeout(120)`)
- Session 2.7b data tests (add skip conditions for empty data columns)

**Exit:** `python -m pytest tests/ --timeout=120 -q` → 0 failures, 0 errors.

---

## Issue 03 — Training Log Autoscroll

Read the full plan folder: `plans/issues-11-04-2026/03-training-log-autoscroll/`

Start with `purpose.md`, then follow `session_prompt.md`.

Single feature: checkbox next to the activity log toggle that auto-scrolls
to bottom on new entries. Default on. Pauses when user scrolls up.

Files: `frontend/src/app/training-monitor/` (`.html`, `.ts`, `.scss`).

**Exit:** Auto-scroll works, manual scroll pauses it, all tests pass.

---

## Issue 07 — Anti-Passivity (Inactivity Penalty)

Read the full plan folder: `plans/issues-11-04-2026/07-anti-passivity/`

Start with `purpose.md` (important — read why force-bet was rejected),
then follow `session_prompt.md`.

Add `inactivity_penalty` gene (float, 0.0–2.0) to config.yaml. When a
model places zero bets in a race, apply `-inactivity_penalty` as shaped
reward. This prevents models learning that "never bet = safe".

Key files: `config.yaml`, `env/betfair_env.py` (`_REWARD_OVERRIDE_KEYS`,
`__init__`, `_settle_current_race`).

**Exit:** Tests pass including: zero-bet race with penalty, one-bet race
without penalty, backward compat with penalty=0, raw+shaped invariant.

---

## Sprint complete

After all three sessions:
1. Run `python -m pytest tests/ --timeout=120 -q` — should be fully green.
2. Commit each session separately.
3. Push: `git push origin master`.
4. Consider running a training session to validate the inactivity penalty.
