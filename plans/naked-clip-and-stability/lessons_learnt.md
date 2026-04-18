# Lessons learnt — Naked-Windfall Clip & Training Stability

Append-only. One bullet per surprise / gotcha / non-obvious
decision made during implementation.

---

## 2026-04-18 — Session 03 reward centering: units mismatch bug

**Symptom.** After the Session 05 launch, the smoke-test probe
reported `value_loss = 6.76e+08` on ep2 of a fresh transformer
probe (and `8.54e+08` on the LSTM). The full population that
leaked through before my days=/`policy_cls(env=env)` orchestrator
fix showed the same pattern across 111 agents — every ep2+ had
a value-head blow-up.

**Root cause.** Session 03's reward centering had a unit
mismatch between where the EMA was populated and where it was
applied:

- `agents/ppo_trainer.py::_ppo_update` populated
  `_reward_ema` with `sum(tr.training_reward for tr in
  transitions)` — the episode **sum**.
- `agents/ppo_trainer.py::_compute_advantages` subtracted
  `_reward_ema` from each `tr.training_reward` — a **per-step**
  quantity.

For a probe rollout with ~4742 steps and total reward −1551
(per-step mean ≈ −0.33), the EMA landed at −1551 after ep1.
On ep2, every per-step reward was shifted by +1551 before
GAE. The GAE accumulator converges to roughly
`shifted_reward / (1 − γλ) = 1551 / 0.0595 ≈ 26,000`, so
`returns` on ep2 sat around +26,000 while the value head
(trained on ep1 returns ~−30) still predicted near zero.
`value_loss = (value − return)^2.mean() ≈ 26,000² = 6.8e+08` —
matched the observed 6.76e+08 within 0.6%.

**Why the tests didn't catch it.**
`test_centering_fixes_uniformly_negative_rewards` assigns
`trainer._reward_ema = float(np.mean(rewards))` directly —
i.e. uses the **correct** per-step-mean unit. Production code
populated `_reward_ema` with the episode **sum**. The test
verified the correct-units code path; production used a
wrong-units code path; they never met.

**Fix.**
`agents/ppo_trainer.py` now computes
`per_step_mean_reward = sum(...) / max(1, len(transitions))`
before the `_update_reward_baseline(...)` call. The
`_update_reward_baseline` docstring now carries an explicit
UNITS CONTRACT stating callers must pass per-step mean, not
episode sum.

**Test-design takeaway.** When a value is computed in
production and asserted in tests, the two must share a
production-faithful fixture or the test becomes a checklist
item that proves nothing. The three new unit tests (which
mirror the aggregation via a helper) still wouldn't have
caught a caller-only drift. The one that MUST stay is
`test_real_ppo_update_feeds_per_step_mean_to_baseline` — it
spies on `_update_reward_baseline` during a real
`_ppo_update(rollout)` call and asserts the passed value
matches `sum / n_steps`. When reverted manually, this is the
only test that catches the bug.

General principle: for shared-mutable-state fixes, at least
one test must drive the full production code path rather than
an isolated helper. Unit tests that pre-compute the "correct"
value and then inject it into shared state are
self-confirming — they prove the consumer works given the
right input, not that the producer generates it.

---

## 2026-04-18 — Session 02 log-ratio clamp at ±20 is numerical, not a loss cap (NOT FIXED — watch-list)

**Observation.** The probe's ep1 `policy_loss = 7.23e+07` on
the transformer (and `3.64e+07` on the LSTM) was a 10-orders-
of-magnitude improvement over the untreated transformer
`0a8cacd3` baseline (`1.04e+17`) but still far above the
`EP1_POLICY_LOSS_MAX = 100` gate threshold.

**Why the existing defences don't bite hard enough on ep1.**
The clamp `torch.clamp(log_ratio, -20, +20)` caps
`ratio = exp(log_ratio)` at ~5e+08 — it prevents NaN overflow
but doesn't bound policy loss. On a bad in-epoch mini-batch
with negative advantages, PPO's `min(surr1, surr2)` picks the
unclamped surrogate (because negative × large-ratio is more
negative, and `min` selects the more-negative term for the
"-min" loss). Contribution: `exp(20) × |adv| ≈ 5e+08`. Average
it with normal mini-batches and you get the observed 1e+07
scale. The KL early-stop at epoch granularity cannot prevent
in-epoch spikes — by the time it fires, the damage is done
inside epoch 1.

**Why we're NOT fixing it yet.** The reward-centering fix
above changes the reward scale going into advantages.
Advantages get per-mini-batch normalised to O(1) anyway. Once
ep2+ value_loss isn't blowing up through the shared trunk,
ep1 policy_loss may well fall under 100 on its own. We revisit
this only if, after the centering fix, fresh runs still show
ep1 `policy_loss > 100`.

**If revisiting becomes necessary, the candidates are (in
order of increasing invasiveness):**
1. Tighten the clamp to `±5` (ratio in `[0.007, 148]`). Keeps
   the numerical backstop; adds a meaningful loss cap.
2. Move KL early-stop from epoch granularity to mini-batch
   granularity so a single runaway mini-batch aborts further
   updates in the same rollout.
3. Global-norm gradient clipping (`torch.nn.utils.
   clip_grad_norm_`) before `optimizer.step()`. Caps the
   actual weight delta regardless of loss magnitude.

**Watch-list signal.** After the centering fix ships and a
fresh run produces a few generations, pull the ep1
`policy_loss` distribution. If median > 100 OR max > 10,000,
open a follow-up plan and pick option 1 first.
