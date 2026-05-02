# Session prompt — Phase 4 Session 05: make attribution invariant assert opt-in / sampled

Use this prompt to open a new session in a fresh context. Self-
contained.

---

## The task

`_attribute_step_reward` runs an `np.isclose` invariant check
every tick (now after Session 01, on the modified incremental
attribution path):

```python
# (rollout.py lines 412-424)
total = float(per_runner.sum())
if not np.isclose(
    total, step_reward,
    rtol=0.0, atol=_ATTRIBUTION_TOLERANCE,
):
    raise AssertionError(...)
```

`np.isclose` is cheap, but at 12 k ticks/episode it compounds
and is one of the smaller-but-real per-tick costs. More
importantly, **the assert is doing development-time work that
production runs don't need** — it's catching a bug class
("attribution algebra silently breaks") that's worth checking
when code is being modified, but pointless to re-verify on
every tick of every production run after Session 01's changes
are bedded in.

**Replace per-tick check with a one-in-N sample by default,
plus a debug flag for per-tick.** The production default fires
once per ~100 ticks; a `PHASE4_STRICT_ATTRIBUTION=1` env var
restores per-tick checking for development / regression runs.
Optionally also fire on every settle-step tick (where
attribution is most likely to drift) regardless of the sample
rate.

End-of-session bar:

1. **CPU bit-identity preserved on attribution outputs.** The
   `per_runner_reward` arrays are unchanged; only the assertion
   frequency changes. Strict equality on the per-tick output
   regardless of which check path ran.
2. **Strict mode (PHASE4_STRICT_ATTRIBUTION=1) reproduces today's
   per-tick assert behaviour.** A tick that would have failed
   pre-change still fails post-change under strict mode.
3. **Sampled mode catches deliberately-injected drift** within
   the sample window. Synthesise a drift on a settle-step tick
   and verify it raises (settle-step always-checked, even in
   sampled mode).
4. **All pre-existing v2 tests pass on CPU.** The existing
   tests should be running in strict mode (set via test fixture
   or pytest conftest) so they remain regression guards.
5. **ms/tick measurement** vs Session 04's baseline.
6. **Verdict** GREEN / PARTIAL / FAIL.

## What you need to read first

1. `plans/rewrite/phase-4-restore-speed/purpose.md` and
   `findings.md`.
2. `training_v2/discrete_ppo/rollout.py` lines 405–426 — the
   current invariant assert.
3. CLAUDE.md §"Bet accounting: matched orders, not netted
   positions" — context for why the attribution invariant
   matters at all.
4. Search `tests/` for tests that rely on the per-tick assert
   firing — there may be tests that deliberately drift the
   attribution and expect a raise. Those tests must run in
   strict mode after this session.
5. `env/betfair_env.py::_settle_current_race` — understand which
   tick is "settle-step" so you can wire the always-check
   carve-out.

## Implementation sketch

```python
# Module-level
import os
_STRICT_ATTRIBUTION = os.environ.get(
    "PHASE4_STRICT_ATTRIBUTION", "0"
) == "1"
_SAMPLED_ATTRIBUTION_EVERY_N = 100
```

In `_attribute_step_reward`, replace the unconditional check
with:

```python
should_check = (
    _STRICT_ATTRIBUTION
    or self._steps_since_last_check >= _SAMPLED_ATTRIBUTION_EVERY_N
    or _is_settle_step(env)  # always-check on settle ticks
)
if should_check:
    total = float(per_runner.sum())
    if not np.isclose(total, step_reward, rtol=0.0, atol=_ATTRIBUTION_TOLERANCE):
        raise AssertionError(...)
    self._steps_since_last_check = 0
else:
    self._steps_since_last_check += 1
```

The `_is_settle_step` helper inspects the env's per-step state
to decide. One observable signal: `env._just_settled` (or
similar — read the env's state machine to find the right
flag); if no such flag exists, the closest proxy is "this tick
saw a non-zero entry in `env.all_settled_bets`" (i.e.
`len(env.all_settled_bets)` increased since last tick).

The `_steps_since_last_check` counter lives on `self`
(initialised in `__init__` and reset in `_collect`).

Add the strict-mode env-var to the test fixture so all existing
tests run in strict mode by default; opt them out only if a
specific test wants to verify sampled-mode behaviour.

## Tests to add

In `tests/test_v2_rollout_invariant_assert.py` (new file):

1. `test_strict_mode_fires_per_tick` — with
   `PHASE4_STRICT_ATTRIBUTION=1`, run a 1-day rollout and patch
   `np.isclose` to count calls. Assert the call count equals
   `n_steps`.

2. `test_sampled_mode_fires_at_most_once_per_n_plus_settle_ticks`
   — with strict mode OFF, run a 1-day rollout and assert the
   `np.isclose` call count is `≈ n_steps / N + n_settle_ticks`.

3. `test_strict_mode_raises_on_injected_drift` — synthesise a
   tick where `per_runner.sum() != step_reward` (monkeypatch the
   attribution loop to write a deliberate offset) and assert
   strict-mode raises.

4. `test_sampled_mode_raises_on_settle_step_drift` — same
   injection as above, but on a settle-step tick (the always-
   check carve-out). Assert sampled-mode raises.

5. `test_sampled_mode_misses_drift_on_non_sample_non_settle_tick`
   — same injection on a regular non-settle tick that falls
   between sample windows. Assert sampled-mode does NOT raise
   (this is the explicit trade-off; documenting it as a test
   makes the trade visible).

6. `test_attribution_outputs_unchanged_across_modes` — run the
   same fixed-seed rollout in strict and sampled modes; assert
   `per_runner_reward` arrays are byte-equal across all ticks.
   Catches a future regression where the assert path mutates
   state.

## Hard constraints

1. **Default ON in tests.** The test fixture / conftest must
   enable strict mode for every existing test that touches the
   rollout. Production code defaults to sampled mode; tests
   default to strict mode. The two defaults must not drift.
2. **Settle-step always-check is non-negotiable.** Settle is
   the highest-mutation tick of any episode; if anything is
   going to break attribution algebra, it'll break here. Skip
   the sample rate, always check.
3. **Don't change the attribution algebra.** This session is
   purely about *when* the check fires. The
   `per_runner_reward` computation is identical pre/post.
4. **Don't introduce new env-side coupling.** If
   `_is_settle_step` needs a new env flag, that's a Phase-4b
   candidate (env edits are out of scope per `purpose.md`
   §"Hard constraints" §1). Use only fields the env already
   exposes.

## Deliverables

- `training_v2/discrete_ppo/rollout.py` — sampled / strict
  invariant check.
- `tests/conftest.py` (or equivalent) — strict-mode env-var
  set for the v2 rollout test directory.
- `tests/test_v2_rollout_invariant_assert.py` (new) with the
  six tests above.
- `findings.md` updated.
- Commit: `feat(rewrite): phase-4 S05 (GREEN|PARTIAL) - sampled
  attribution-invariant check + strict mode for tests`.

## Estimate

~1.5 h. If past 2 h, stop — most likely the
`_is_settle_step` heuristic is harder to write without env
edits than expected; document and either ship without the
settle carve-out (sample-rate only) or escalate to Phase-4b.
