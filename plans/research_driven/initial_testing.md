# Initial Testing — Research-Driven

Fast CPU-only tests run **during every session** in this folder.
The contract: a session is not done until everything below passes
locally on a laptop in under a minute, with no GPU.

Anything slow, GPU-only, or requiring a full training run lives in
`integration_testing.md` instead. Anything that requires a human in
the loop lives in `manual_testing_plan.md`.

---

## Always-on (every session)

These are the existing repo-wide invariants. They predate this
folder; the rule is just "don't break them".

- `pytest tests/test_exchange_matcher.py` — single-price rule,
  junk-filter, max-price cap. P3/P4 add a new code path; the
  *aggressive* path tests must continue to pass unchanged.
- `pytest tests/test_bet_manager.py` — bet accounting, raw vs
  shaped accumulator invariant.
- `pytest tests/test_betfair_env.py` — `info["day_pnl"]` is the
  authoritative day P&L; `info["realised_pnl"]` is last-race-only.
- `pytest tests/test_reward_invariants.py` — `raw + shaped ≈ total`
  to floating-point tolerance.

If any of these are red at the *start* of a session, fix them
before doing the session work. Don't layer new code on a red bar.

---

## Per-proposal additions

Tests below are added by the session that ships the proposal. They
become part of the always-on set after that.

### P1 — money-pressure features

- Each new feature appears in `info["debug_features"]` for at least
  one race in a smoke-test fixture, with a sensible value (not NaN,
  not zero, not constant across ticks).
- `obi_topN` is in `[-1, 1]` for every emitted value across the
  smoke fixture.
- `weighted_microprice` lies between best back and best lay for
  every emitted value.
- `traded_delta_T` and `mid_drift_T` are zero on the first tick of
  a race (no prior window) and non-zero on at least one mid-race
  tick where traded volume is known to be non-trivial.
- Obs-vector schema version is bumped; loading a pre-P1 checkpoint
  with the new env raises a clear error, not a silent zero-pad.

### P2 — spread-cost shaped reward

- `info["spread_cost"]` is present per race and accumulates into
  the existing `shaped_bonus` accumulator, not into `raw`.
- `raw + shaped ≈ total` invariant test still passes.
- A purely-aggressive policy on a fixture race produces strictly
  positive `spread_cost`. A no-bet policy produces zero.
- Random-bet expected value of `spread_cost` is strictly positive
  (the term is a *cost*, not a zero-mean shaping like the existing
  early-pick bonus). This is documented in `lessons_learnt.md` so
  no future session "fixes" it back to zero-mean by mistake.

### P3 + P4 — passive orders, queue, cancel

- Passive bet placed at tick *N* with £200 ahead in queue, then
  £150 traded → still unfilled at tick *N+k*. Then £60 more
  traded → filled.
- Passive bet that never gets enough traded volume cancels cleanly
  at race-off with zero P&L and no exception.
- Passive bet explicitly cancelled mid-race releases its budget
  reservation and disappears from the open-orders set.
- Aggressive code path regression suite (existing matcher tests)
  continues to pass unchanged.
- Action-vector schema version is bumped; loading a pre-P3
  checkpoint with the new env raises a clear error.

### P5 — UI fill-side annotation

- Snapshot test of one replay-UI row showing the new annotation.
  No new pytest needed if the existing snapshot suite covers it.

---

## What stays out of this file

- Anything that needs the GPU → `integration_testing.md`.
- Anything that needs a full PPO loop → `integration_testing.md`.
- Anything that needs a human watching the UI →
  `manual_testing_plan.md`.

The fast feedback set must stay fast. If a test routinely runs over
five seconds, push it down to integration testing and replace it
with a unit test that exercises the same code path on a tiny
fixture.
