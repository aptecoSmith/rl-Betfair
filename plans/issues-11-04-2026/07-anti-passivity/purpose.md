# 07 — Anti-Passivity: Stop Models Learning to Never Bet

## Problem

A model can learn that not betting = no losses = safe. When shaped
reward terms (efficiency penalty, spread cost) punish betting, and
the precision bonus is zero when bet_count is 0, the safest policy
is to abstain entirely. This is a degenerate equilibrium: the model
achieves zero loss by doing nothing.

Current reward for a race with zero bets:
- `race_pnl = 0`
- `early_pick_bonus = 0`
- `precision_reward = 0` (guarded by `if race_bet_count > 0`)
- `efficiency_cost = 0`
- `drawdown_term = 0` (no P&L movement)
- `spread_cost_term = 0`
- **Total reward = 0**

Zero reward every race, every day. No gradient signal to bet. The
model's only incentive to bet comes from the terminal bonus
(`day_pnl / starting_budget`), which is also zero if it never bets.

## The tension

Forcing bets (e.g. "at least 1 per race") would make the model bet
when it has no edge — that's worse than not betting. The ideal model
abstains from bad races and bets on good ones. We want to encourage
*engagement* without mandating *bad bets*.

## Proposed approach: inactivity penalty as a shaped reward gene

Add a small shaped penalty for races where the model places zero
bets. This makes "do nothing" slightly negative instead of zero,
giving the optimizer a gradient to explore betting. But the penalty
is small enough that abstaining is still better than making a
clearly bad bet.

### Inactivity penalty

```python
if race_bet_count == 0:
    inactivity_penalty = -self._inactivity_penalty  # e.g. -0.5
else:
    inactivity_penalty = 0.0
```

Key properties:
- **Only fires on zero-bet races.** A model that bets once (even
  badly) doesn't pay it.
- **Small magnitude.** The penalty should be less than the expected
  loss from a random bad bet. If a random bet loses ~£5 on average,
  the inactivity penalty should be ~£0.50. This means the model
  is better off not betting when it has no edge (losing £0.50
  penalty < losing £5 on a bad bet), but worse off than a model
  that identifies a good bet.
- **Gene-controlled magnitude.** The penalty amount is a
  hyperparameter gene so the genetic algorithm can evolve the right
  balance. Some models may benefit from a stronger nudge; others
  may do better with a light touch.

### Gene definition

```yaml
inactivity_penalty:
  type: float
  min: 0.0
  max: 2.0
```

A value of 0.0 means no penalty (current behaviour). This lets
evolution discover whether inactivity punishment helps or hurts.

### Symmetry consideration

This penalty is **intentionally not zero-mean for random policies**.
A random policy that sometimes bets and sometimes doesn't would only
pay the penalty on no-bet races. This is similar to `spread_cost`
which is also intentionally asymmetric (see Session 23 comment in
`betfair_env.py:1066-1069`).

The justification: in live trading, watching a race go by without
betting has a real opportunity cost. The model held capital idle.
A small penalty for this is economically defensible, not just a
training trick.

## Alternative considered: force-at-least-one-bet

A training/inference option to force a minimum bet per race. Rejected
because:
- Forces bets with no edge — worse than abstaining.
- Changes the action semantics (model can no longer choose "no bet").
- Would need to pick *which* runner and *what* stake — either random
  (terrible) or the model's best signal (which it already chose not
  to act on).
- At inference time in ai-betfair, forcing bets with real money is
  actively harmful.

The inactivity penalty achieves the same goal (prevent passivity)
without forcing specific actions.

## Alternative considered: minimum bet rate over a day

Instead of per-race, require N bets per day. Softer than per-race
but:
- Doesn't help if the model front-loads all bets in early races
  and abstains for the rest.
- Harder to compute a per-race reward signal from a day-level
  constraint.
- The per-race inactivity penalty is simpler and more local.

Could be a future addition alongside the per-race penalty, but
start with the simpler mechanism.

## Files touched

| Layer | File | Change |
|---|---|---|
| Config | `config.yaml` | Add `inactivity_penalty` gene |
| Environment | `env/betfair_env.py` | Add penalty in `_settle_current_race()` |
| Environment | `env/betfair_env.py` | Add to `_REWARD_OVERRIDE_KEYS` |

## ai-betfair knock-on

None. The inactivity penalty is a training-only shaped reward term.
At inference time in ai-betfair, the policy runs its forward pass
as normal — the penalty doesn't affect the action output, only the
training gradient. No changes needed in ai-betfair.
