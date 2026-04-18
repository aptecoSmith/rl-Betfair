# Hard constraints — Policy Startup Stability

Non-negotiable rules. Anything that violates one gets rejected
in review before destabilising production.

## Scope

**§1** This plan changes ONE thing: how the PPO update consumes
its advantage tensor. Specifically, advantage normalisation per
mini-batch. Every code change must trace back to that single
intent. Anything else — reward function, action schema,
observation schema, env mechanics, sizing math, naked-penalty
shaping, locked-pnl floor, commission handling, matcher,
budget-reservation policy — STAYS as it is.

**§2** No new shaped-reward terms. The asymmetric raw reward
(`scalping_locked_pnl + sum(min(0, per_pair_naked_pnl))`, post
plan #13) stays. The reward magnitudes feeding the advantage
calculation are unchanged; only the gradient consumption of those
magnitudes is normalised.

**§3** No schema bumps. `OBS_SCHEMA_VERSION`,
`ACTION_SCHEMA_VERSION`, `SCALPING_ACTIONS_PER_RUNNER` stay.
Pre-existing checkpoints continue to load without migration.
The 3 garaged models remain valid and loadable.

**§4** No env changes. `env/betfair_env.py`, `env/bet_manager.py`,
`env/scalping_math.py`, `env/exchange_matcher.py`,
`env/tick_ladder.py` are all UNTOUCHED by this plan. The fix
lives entirely inside `agents/ppo_trainer.py`.

## Math

**§5** Advantage normalisation MUST be **per-mini-batch**, not
per-rollout-buffer. The standard PPO recipe is:

```python
adv_mean = batch_advantages.mean()
adv_std  = batch_advantages.std() + eps
batch_advantages = (batch_advantages - adv_mean) / adv_std
```

with `eps = 1e-8` (numerical stability — prevents division by
zero on the degenerate case where every advantage in the batch
is identical).

Per-batch normalisation is what the literature converged on.
Per-rollout normalisation (using rollout-wide mean/std for every
mini-batch) is a less common variant and is NOT what we want
here.

**§6** Normalisation happens on the advantage tensor used in
the **policy loss only**. The value loss continues to use the
un-normalised return targets. Mixing these gets you wrong value
estimates that bleed into future updates.

**§7** The normalisation MUST be applied AFTER GAE computation
(or whatever advantage estimator the trainer uses). GAE produces
the raw advantage signal; we normalise the result before feeding
it to the surrogate-loss calculation.

## Implementation

**§8** The change lives in **one function** in
`agents/ppo_trainer.py` — wherever the per-mini-batch surrogate
loss is computed. Find the line(s) computing `ratio * advantages`
or equivalent; the normalisation lands immediately above. Single
commit, single function.

**§9** The optional first-update LR warmup (§13 of `purpose.md`)
ships ONLY if the diagnostic test in Session 01 shows that
advantage normalisation alone leaves a residual spike. Default
position: ship just normalisation. The Session 01 prompt
explicitly walks the operator through the decision.

**§10** No changes to `max_grad_norm` (currently 0.5). The
existing gradient clip continues to backstop. Tightening it
further is out of scope; if the literature-standard
normalisation doesn't fix this, gradient clip alone won't either.

**§11** No changes to learning rate, PPO clip epsilon, n_epochs,
batch size, or any other hyperparameter that already has a
config knob or a gene. Those are tuning surfaces; this plan
only addresses the structural bug.

## Reward / metric protocol

**§12** This is NOT a reward-scale change in the
operator-visible sense. Values written to
`logs/training/episodes.jsonl` (`total_reward`,
`raw_pnl_reward`, `shaped_bonus`, `total_pnl`, etc.) are
**identical** before and after this plan because the reward
function itself is untouched. Scoreboard rows from before the
fix are directly comparable to scoreboard rows from after.

What changes is which agents the GA SELECTS, because the
gradient signal driving training is now stable enough to develop
all the action heads instead of saturating one or more on the
first update.

**§13** The Session 01 commit message MUST still mention the
change loudly — it's a training-loop change that affects every
subsequent training run — but it does NOT need the full
"reward-scale change" worked example template. A clear
description of what changed and why is enough.

**§14** Garaged models (`46187c46`, `ef453cd9`, `ab460eb9`) are
not migrated. They were trained without normalisation and
remain valid post-fix references. New training runs benefit
from the fix.

## Testing

**§15** Pre-existing
`test_invariant_raw_plus_shaped_equals_total_reward` (in
`tests/test_forced_arbitrage.py`) MUST stay green. The reward
accumulator is unchanged.

**§16** New test in `tests/test_ppo_trainer.py` (or a new file
`tests/test_ppo_advantage_normalisation.py` if the trainer test
file is too big to extend cleanly): a synthetic test that
demonstrates the spike-prevention behaviour:

1. Construct a freshly-initialised policy + a fake rollout
   buffer with deliberately-large reward magnitudes (±£500 to
   ±£2000 to mimic real scalping rollouts).
2. Run one PPO update WITHOUT normalisation; capture
   policy_loss. Assert it exceeds a threshold (the actual
   number depends on the synthesised advantages — pick a value
   that reliably triggers without normalisation but stays
   below the test's tolerance after the fix).
3. Reset the policy + buffer; run one PPO update WITH
   normalisation; assert policy_loss stays bounded under a
   small constant (e.g. 5).
4. Compare the action_head's output mean shift between the
   two cases — the normalised case must shift the head's mean
   by materially less than the un-normalised case.

This test is the load-bearing check that the fix actually
prevents the collapse pattern observed in production.

**§17** Smoke test (manual, run once at the end of Session 01,
recorded in `progress.md`): launch a tiny training run (1
agent, 5 episodes) with the fix in place; check
`logs/training/episodes.jsonl` for the agent's
policy_loss series. NO entry should exceed 100 on episode 1
(let alone 10⁴+). If the smoke run shows a spike, the fix isn't
sufficient and Session 01's stretch goal (LR warmup) lands.

**§18** Full `pytest tests/ -q` green on every commit in this
plan.

## Cross-session

**§19** Do NOT bundle the activation re-run into any
implementation commit. Session 02 resets the four activation
plans to draft as part of its scope; the operator launches the
re-run AFTER Session 02 lands.

**§20** Do NOT pre-emptively prune any models. Existing garaged
models stay; non-garaged orphans from the latest aborted run can
be pruned via `scripts/prune_non_garaged.py` whenever the
operator wants a clean state. That's separate from this plan.

**§21** Do NOT touch `plans/scalping-equal-profit-sizing/` or
any earlier plan folder. Reference them where useful in this
plan's prose; don't edit them.
