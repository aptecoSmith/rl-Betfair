# Session 3 — Signal-bias warmup & bet-rate diagnostics

## Before you start — read these

- `plans/arb-improvements/purpose.md`
- `plans/arb-improvements/master_todo.md` — Phase 1, Session 3.
- `plans/arb-improvements/testing.md`
- `plans/arb-improvements/hard_constraints.md` — warmup is a *soft
  prior* that decays to zero, never a hard override.
- `plans/arb-improvements/progress.md` — read Session 1 and 2
  entries.
- `plans/arb-improvements/lessons_learnt.md` — the "can't escape the
  abstention corner" lesson is why this session exists.

## Goal

Bias new agents toward placing bets during their first few epochs
so they never reach the "don't bet" corner before they've had a
chance to learn. The bias is a linearly-decaying additive constant
on the `signal` action mean; after the warmup window it's zero and
the policy is unaffected. Combined with Sessions 1 and 2, this
closes out Phase 1.

## Scope

**In scope:**

- `training.signal_bias_warmup` (int epochs, default `0` = off).
- `training.signal_bias_magnitude` (float, default `0.0`). Positive
  values bias toward "back"; negative toward "lay".
- Apply the bias inside the policy forward pass *only during
  training*, and only for the first `signal_bias_warmup` epochs.
  Linear decay: at epoch `e`, bias magnitude is
  `signal_bias_magnitude * max(0, 1 - e/signal_bias_warmup)`.
- Monitor: extend the `action_stats` progress event with
  `bet_rate` (fraction of steps where `signal` crossed ±0.33
  threshold), `arb_rate` (fraction of aggressive fills that paired),
  and `bias_active` (bool).
- UI tasks appended to `ui_additions.md`: bet-rate / arb-rate
  sparklines in the training monitor.

**Out of scope:**

- Biasing any head other than `signal`. Stake, aggression, cancel,
  and arb_spread stay at their default init.
- Any bias that persists past the warmup window.
- Direct UI implementation (Session 8 consolidates).

## Exact code path

1. Policy forward pass — in `agents/ppo_lstm.py`,
   `agents/ppo_time_lstm.py`, and `agents/ppo_transformer.py`,
   locate where the `signal` head's mean is computed. Add an
   optional `signal_bias: float` parameter to the forward signature
   that gets added to the mean before the distribution is
   constructed. Default `0.0` → no change.
2. `agents/ppo_trainer.py` — compute
   `current_bias = signal_bias_magnitude * max(0, 1 -
   current_epoch / signal_bias_warmup)` before each rollout step.
   Pass it into the policy's forward. After `current_epoch >=
   signal_bias_warmup`, pass `0.0`.
3. `agents/ppo_trainer.py:645–648` — progress event includes
   `bet_rate`, `arb_rate`, `bias_active` in `action_stats`.
4. No env changes. This is entirely a policy / trainer concern.

## Tests to add (all CPU-only, fast)

Create `tests/arb_improvements/test_signal_bias_warmup.py`:

1. **Bias applied at epoch 0.** Instantiate a policy, call
   `forward(obs, signal_bias=0.5)`. Assert the `signal` mean is
   0.5 higher than `forward(obs, signal_bias=0.0)`.

2. **Bias linearly decays.** Given `signal_bias_magnitude=1.0`
   and `signal_bias_warmup=10`, at epoch 5 the effective bias is
   0.5; at epoch 10 it is 0.0.

3. **No effect after warmup.** At epoch `>= signal_bias_warmup`,
   forward output is bit-identical to `signal_bias=0.0`.

4. **Bias off by default.** With `signal_bias_magnitude=0` or
   `signal_bias_warmup=0`, forward output is unchanged across all
   epochs.

5. **Bias only affects signal head.** Stake, aggression, cancel,
   and arb_spread head outputs are identical with vs without the
   bias.

6. **All three architectures honour the bias.** Parameterised test
   across `ppo_lstm_v1`, `ppo_time_lstm_v1`, `ppo_transformer_v1`.

7. **`bet_rate` / `arb_rate` in progress event.** Unit-test the
   event construction with a synthetic rollout; assert both
   fractions in `[0, 1]` and `bias_active` matches the expected
   boolean.

## Session exit criteria

- All 7 tests pass.
- Existing tests still pass.
- `progress.md` Session 3 entry written. If this completes Phase 1,
  add a "Phase 1 summary" subsection: did a short scalping run with
  all three knobs on (`reward_clip=5`, `entropy_floor=0.5`,
  `signal_bias_warmup=3`, `signal_bias_magnitude=0.3`) avoid the
  90fcb25f failure mode?
- `ui_additions.md` Session 3 UI tasks confirmed present.
- `lessons_learnt.md` updated with any surprises from the Phase 1
  smoke test.
- Commit: `feat(train): signal-bias warmup + bet-rate diagnostics`.
- `git push all`.

## Do not

- Do not bias any head other than `signal`.
- Do not leave the bias in past the warmup window. Decay to zero
  and stay there.
- Do not add GPU tests.
- Do not skip the short Phase-1 smoke test at the end of the
  session — it's the only non-GPU way to know Phase 1 works before
  the big verification run in Session 10.
