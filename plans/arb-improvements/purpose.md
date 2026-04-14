# Purpose — Arb Improvements

## Why this work exists

Forced-arbitrage (scalping) mode shipped in the 12-04-2026 sprint
(`plans/issues-12-04-2026/05-forced-arbitrage/`). The machinery is in
place — paired orders, arb_spread action, commission-aware locked
P&L, scalping-specific reward terms — but in practice **agents almost
never take arbs**. Run `90fcb25f` (2026-04-14) is the latest evidence:

```
Episode 1 [2026-04-06] | reward=-73.989  | P&L=+319.34 | loss=291555954607  arbs=6/124
Episode 2 [2026-04-07] | reward=-50.748  | P&L=-101.08 | loss=11.2271       arbs=0/18
Episode 3 [2026-04-08] | reward=-74.689  | P&L=-54.64  | loss=765.6595      arbs=3/16
Episode 4 [2026-04-06] | reward=-11.650  | P&L=+0.00   | loss=0.2772
...
Episode 18 [2026-04-08] | reward=-14.648 | P&L=+0.00  | loss=0.2849
```

The forced-arbitrage plan made arbs *possible*. This plan makes them
*happen*.

## What we discovered in design review

Three distinct problems are stacking on top of each other:

1. **Catastrophic value-function collapse.** Epoch 1 sees advantage
   targets in the ±£300 range. `value_loss_coeff=0.5` × MSE of those
   produces losses in the `10⁹–10¹²` range
   (`agents/ppo_trainer.py:221`). One update in that regime pushes
   the policy into the "don't bet" corner. Abstention has zero
   gradient signal, so PPO's KL clip cannot pull the policy back.
   Grad-norm clipping bounds magnitude but not direction — the
   direction learned on that batch is still "bet less". Every agent
   in `90fcb25f` shows this fingerprint.

2. **Arbs are invisible in the observation.** The policy currently
   sees raw `back_price_1`, `lay_price_1`, `spread`, `spread_pct` per
   runner (`env/betfair_env.py:132–189`). It must *derive* "there is
   a lockable arb here right now, worth £X after commission" from
   these inputs. That's a non-trivial transform for an LSTM/transformer
   to stabilise on within 3 generations — and doubly hard when layer
   1 kills exploration before the representation forms.

3. **No warm start for a sparse-reward skill.** Real arb moments are
   rare relative to tick count. Asking a randomly-initialised policy
   to discover "place aggressive + passive with N-tick offset" from
   random exploration is lottery-ticket learning. We already *know*
   where every arb moment in every training day was — the episode
   data contains the ground truth. Nothing in the pipeline uses it.

## What success looks like

- **No epoch-1 collapse.** Post-Phase-1 runs show `loss < 10⁷` after
  epoch 1, policy entropy stays above a configurable floor, and
  bet-rate stays positive across all 18 episodes of a 3-gen × 3-day
  run.
- **Arbs are first-class observation signal.** The policy sees
  `arb_lock_profit_pct`, `arb_spread_ticks`, `arb_fill_time_norm`
  per runner and `arb_opportunity_density_60s` globally. Feature
  functions live in `env/features.py` and are vendorable into
  `ai-betfair` unchanged.
- **BC warm start exists and is opt-in.** Each agent's policy is
  pretrained on an offline-generated oracle dataset of real
  training-day arb moments before PPO begins. Turned on from the
  wizard; default off so existing workflows are unaffected.
- **Head-to-head against baseline.** Re-running the `90fcb25f`
  configuration with all layers on shows measurably more arbs
  across episodes 2–18, higher mean locked P&L, and no runaway
  losses.
- **All new knobs default to off / no-op.** Non-scalping runs, and
  scalping runs that don't touch the new knobs, produce byte-identical
  training to the current code.

## Hard constraints (from CLAUDE.md, do not regress)

- **Reward shaping stays zero-mean for random policies.** No
  asymmetric positive-per-bet bonuses sneak in via the new features
  or BC loss.
- **`raw + shaped ≈ total_reward`** invariant holds for every
  scalping rollout. Reward clipping is a *training-signal* transform;
  `info["day_pnl"]`, `total_reward`, and the log line all stay on
  unclipped values.
- **`info["day_pnl"]` is authoritative.** `info["realised_pnl"]` is
  last-race-only and does not get reused for BC advantages or oracle
  filtering.
- **`ExchangeMatcher` single-price / LTP-filter / max-price rules
  are load-bearing.** Oracle action generation MUST respect the same
  filters — a BC target the env would reject is a phantom target.
- **Obs schema version bumps on any RUNNER_KEYS / MARKET_KEYS change.**
  Old checkpoints refuse to load; silent zero-padding is forbidden
  (`hard_constraints.md §13`).
- **One commission constant, imported.** Feature-layer
  post-commission arb profit and `BetManager.get_paired_positions`
  settlement must use the same number from the same place.
