# Session 2 — Entropy floor & per-head logging

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — this is Phase 1, Session 2.
- `plans/arb-improvements/testing.md` — **no GPU, fast tests**.
- `plans/arb-improvements/hard_constraints.md` — the entropy floor
  scales the *coefficient*, never the distribution directly.
- `plans/arb-improvements/progress.md` — read the Session 1 entry.
  Session 1 delivered reward clipping; this session builds on top of
  it and expects the same default-off pattern.
- `plans/arb-improvements/lessons_learnt.md` — read the "PPO's KL
  clip is a trap" note. Entropy floor exists because of that lesson.

## Goal

When the policy's entropy collapses (the "don't bet" corner failure
mode), detect it and raise the entropy bonus until it recovers.
Per-head entropy is visible in the training monitor so the operator
can see the collapse in real time, not after the fact.

## Scope

**In scope:**

- `training.entropy_floor` (float, default `0.0` = off).
- `training.entropy_floor_window` (int, default `10` batches) —
  rolling window over which mean entropy is compared to the floor.
- `training.entropy_boost_max` (float, default `10.0`) — upper bound
  on the multiplier applied to `entropy_coefficient`.
- Controller: when rolling mean entropy < floor, scale
  `entropy_coefficient` to `floor / rolling_mean`, capped at
  `entropy_boost_max`. Restore to baseline once recovered.
- Per-head entropy (signal, stake, aggression, cancel, arb_spread)
  logged to the training monitor `action_stats` progress event.
  Add `action_stats` sub-dict if it doesn't already exist.
- Add an "entropy collapse" warning flag in the progress event:
  `true` when any head's entropy has been below floor for
  > N batches (N configurable, sensible default 5).

**Out of scope:**

- Action-bias warmup (Session 3).
- UI sparkline / panel — append to `ui_additions.md`, don't
  implement (Session 8 consolidates).
- Anything that rewrites the policy's action distribution directly
  (this is a coefficient controller, not a distribution override).

## Exact code path

1. `agents/ppo_trainer.py:220` — `self.entropy_coeff = hp.get(...)`.
   Capture the baseline here as `self._entropy_coeff_base`. Also
   read `entropy_floor`, `entropy_floor_window`,
   `entropy_boost_max` from `hp`.
2. PPO update loop — compute per-head entropy (the action
   distribution objects already exist; extract `.entropy()` per
   head). Accumulate a rolling window of mean entropies.
3. Before the next update's entropy-bonus term, check the rolling
   mean. If < floor, set
   `self.entropy_coeff = min(entropy_boost_max, floor/mean) *
   self._entropy_coeff_base`. If >= floor, restore to
   `self._entropy_coeff_base`.
4. `agents/ppo_trainer.py:645–648` progress event — add
   `action_stats = {"mean_entropy_signal": ..., "mean_entropy_stake":
   ..., ..., "entropy_collapse": bool, "entropy_coeff_active": float}`.

The controller is a small method — keep it inside `PPOTrainer` for
now. No new module needed.

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_entropy_floor.py`:

1. **Floor triggers scaling.** Feed a synthetic sequence of
   entropies declining below the floor. Assert
   `entropy_coeff_active` rises according to the ratio formula,
   bounded by `entropy_boost_max`.

2. **Recovery restores baseline.** After a sequence of low
   entropies, feed high entropies. Assert coefficient returns to
   the baseline.

3. **Floor off by default.** With `entropy_floor=0`, coefficient
   never changes from baseline regardless of entropy sequence.

4. **Per-head entropy in progress event.** Unit test the progress
   event construction — `action_stats` dict contains all five head
   entropies as floats.

5. **`entropy_collapse` flag.** With a head's entropy below floor
   for > 5 consecutive batches, flag is `true`. When recovered,
   flag flips back to `false`.

6. **`entropy_boost_max` caps the multiplier.** Extreme low entropy
   doesn't blow the coefficient past the cap.

7. **Existing `raw + shaped` invariant still holds** with entropy
   floor active. Floor touches `entropy_coefficient`, which is part
   of the PPO *loss*, not the environment reward. Raw/shaped
   accumulators are unaffected.

## Session exit criteria

- All 7 tests pass: `pytest tests/arb_improvements/ -x`.
- Existing tests still pass: `pytest tests/ -m "not gpu and not slow"`.
- `progress.md` Session 2 entry written.
- `ui_additions.md` Session 2 UI tasks confirmed present.
- `lessons_learnt.md` updated if anything surprising came up.
- Commit: `feat(train): entropy floor + per-head entropy diagnostics (default off)`.
- `git push all`.

## Do not

- Do not override the policy's action distribution directly. The
  fix is on the *bonus term*, not the distribution.
- Do not blend entropy across heads into a single scalar silently.
  Each head gets its own entropy in the progress event so an
  operator can see which head is collapsing.
- Do not add GPU tests.
- Do not touch the UI — Session 8 does that.
