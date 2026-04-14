# Session 1 — Reward & advantage clipping

## Before you start — read these

- `plans/arb-improvements/purpose.md` — why this plan exists.
- `plans/arb-improvements/master_todo.md` — where this session sits.
- `plans/arb-improvements/testing.md` — **no GPU, no full training
  loops. Add tests as you go.**
- `plans/arb-improvements/hard_constraints.md` — the clipping-is-
  training-signal-only rule is non-negotiable.
- `plans/arb-improvements/progress.md` — the 90fcb25f baseline is the
  failure this work exists to fix.
- Repo root `CLAUDE.md` — the `raw + shaped ≈ total_reward` invariant
  and the `info["day_pnl"]` authority rule.

## Goal

Prevent one outlier race in epoch 1 from collapsing the policy into
"don't bet". Add three independent clip knobs — reward clip (at the
training-signal layer), advantage clip (at the PPO update), and
value-loss clip (at the critic loss). All default off.

## Scope

**In scope:**

- `reward.reward_clip` (float, default `0.0` = off). When > 0, per-step
  reward fed into the rollout advantage/return computation is
  `np.clip(step_reward, -reward_clip, +reward_clip)`. Unclipped
  reward still flows into `EpisodeStats`, `info["day_pnl"]`, the
  log line, and the monitor progress events.
- `training.advantage_clip` (float, default `0.0` = off). Clamps
  per-transition advantage magnitude to
  `[-advantage_clip, +advantage_clip]` before the PPO ratio
  multiplies it.
- `training.value_loss_clip` (float, default `0.0` = off). Caps the
  per-sample value-loss contribution at `value_loss_clip**2` before
  the batch mean.
- Whitelist `reward_clip` in `BetfairEnv._REWARD_OVERRIDE_KEYS` so
  it can evolve per-agent.
- `clipped_reward_total` logged per episode alongside `total_reward`.
- `conftest.py` scaffolding: ensure `gpu` / `slow` pytest markers
  exist and skip by default (cross-reference with `arch-exploration`
  plan — reuse if already present, add if not).

**Out of scope:**

- Entropy floor (that's Session 2).
- Action-bias warmup (Session 3).
- Arb features (Phase 2).
- UI work for the new knobs — append tasks to
  `ui_additions.md`, don't implement. Session 8 consolidates.

## Exact code path

Trace from the design review:

1. `agents/ppo_trainer.py:220–223` — trainer pulls PPO knobs from
   `hp`. Add `advantage_clip` and `value_loss_clip` here.
2. `agents/ppo_trainer.py` rollout loop (where per-step rewards are
   pushed into the advantage/return buffer) — apply `reward_clip`
   to the buffer value, keep the unclipped value in a parallel
   buffer for telemetry.
3. `agents/ppo_trainer.py` PPO update (look for `advantages = ...`
   and the value-loss term `F.mse_loss(values, returns)`) — apply
   `advantage_clip` before the ratio, apply `value_loss_clip` as a
   `torch.clamp` on the per-sample squared error before
   `.mean()`.
4. `agents/ppo_trainer.py:419–427` `EpisodeStats` construction —
   add `clipped_reward_total` alongside `raw_pnl_reward` /
   `shaped_bonus`.
5. `agents/ppo_trainer.py:645–648` progress event — include
   `clipped_reward_total` so the monitor can flag it.
6. `env/betfair_env.py::_REWARD_OVERRIDE_KEYS` — add `reward_clip`.
   Do *not* change `_settle_current_race`; clipping happens
   downstream of the env's reward output.

Do not route clipping through `env._settle_current_race` — it must
be visible at the training-signal layer only, not injected into
the raw/shaped accumulators.

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_reward_clipping.py`:

1. **Reward clip — training signal.** Feed a synthetic sequence of
   per-step rewards including a ±100 outlier, with `reward_clip=5`.
   Assert the buffer used for advantage computation is clipped; the
   buffer used for episode logging is not.

2. **Reward clip — off by default.** With `reward_clip=0` (or
   missing from config), the training signal equals the raw reward
   exactly.

3. **Advantage clip.** Build a synthetic advantage tensor including
   outliers, apply `advantage_clip=2.0`, assert all values in
   `[-2.0, +2.0]` and the mean is still non-zero.

4. **Value-loss clip.** Synthetic value predictions + returns with
   one huge residual, compute loss with `value_loss_clip=1.0`,
   assert the total loss is within `value_loss_clip**2` of where it
   would be without the outlier.

5. **Trainer passes `reward_clip` into env overrides.** Mock
   `BetfairEnv`; assert `PPOTrainer` constructs it with
   `reward_overrides` containing `reward_clip` derived from
   `self.hyperparams`.

6. **`raw + shaped ≈ total_reward` invariant still holds** with
   `reward_clip=5` on a synthetic scalping race. Clipping touches
   the advantage path, not the reward accumulators.

7. **`clipped_reward_total` appears in EpisodeStats and progress
   event** when `reward_clip > 0`.

8. **Byte-identical rollout when all three knobs are off.** Compare
   a short synthetic rollout against a pre-session expected output
   fingerprint (check in a small deterministic fixture).

## Session exit criteria

- All 8 tests above pass: `pytest tests/arb_improvements/ -x`.
- Existing tests still pass: `pytest tests/ -m "not gpu and not slow"`.
- `progress.md` Session 1 entry written with commit hash and files
  changed.
- `ui_additions.md` Session 1 checkbox items added (or confirmed
  already present).
- `lessons_learnt.md` updated with anything surprising. Absence of
  news is also news — note that.
- Commit: `feat(train): reward / advantage / value-loss clipping (default off)`.
- `git push all`.

## Do not

- Do not change any reward formula. The fix is purely clipping on
  top of the existing reward stream.
- Do not remove or weaken `max_grad_norm`. The three new clips sit
  in front of it, not in place of it.
- Do not add any GPU-touching tests.
- Do not widen scope — entropy, warmup, features all have their
  own sessions.
