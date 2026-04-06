# Purpose — Architecture & Hyperparameter Exploration

## Why this work exists

The rl-betfair simulator is now honest (phantom-profit bug fixed in commit
`e76ac98`). With honest rewards in place, we need to find a training
architecture and reward-shaping regime that produces **positive P&L on real
Betfair horse-racing markets**. We are currently nowhere near that.

The genetic population system is designed to search over reward shaping,
PPO hyperparameters, LSTM hyperparameters, and architecture choice. Gen 0
should give each agent in the population a meaningfully different starting
point so that three generations of evolution actually contain signal.

## What we discovered in design review

The genetic search is currently degenerate:

1. **Reward-shaping genes are dead code.** `reward_early_pick_bonus`,
   `reward_efficiency_penalty`, and `reward_precision_bonus` are sampled
   per-agent in `population_manager.py:220`, but the env never receives
   them — it reads straight from `config.yaml`. Every agent trains with
   identical reward shaping regardless of what its genome says.
2. **Only 3 PPO knobs vary.** `learning_rate`, `ppo_clip_epsilon`,
   `entropy_coefficient`. `gamma`, `gae_lambda`, `value_loss_coeff` are
   hardcoded even though the trainer would read them from `hp` if they
   were in the schema.
3. **LSTM structural params are hardcoded.** `num_layers=1`, no dropout,
   no layer norm. Both architectures (`ppo_lstm_v1`, `ppo_time_lstm_v1`)
   share the same encoder + head shapes.
4. **`observation_window_ticks` is sampled but never read** — dead gene.
5. **No third architecture exists.** `ppo_transformer_v1` is named in
   `PLAN.md` but has no scaffolding.
6. **No planning layer.** There is no record of what Gen 0
   configurations have been tried, no way to ensure fair coverage per
   architecture, and no way to phase the search.

## What success looks like

- Every gene in the mutation schema is actually plumbed through to the
  object that uses it. No silent ignores.
- Gen 0 can be deliberately planned: "run one full Gen 0 with an even
  mix of architectures, these reward ranges, these PPO ranges" — and we
  have a record of what's been tried.
- A third architecture (`ppo_transformer_v1`) exists and can be mixed
  into populations.
- Each phase of the rollout delivers a usable intermediate result; we
  are not blocked waiting for the whole plan to land before anything is
  testable.
- The UI exposes every new configurable knob. No genes that only a
  developer editing YAML can touch.

## Hard constraints (from CLAUDE.md, do not regress)

- Reward shaping stays zero-mean for random policies. No asymmetric
  positive-per-bet bonuses.
- `raw + shaped ≈ total_reward` is a test invariant. Every new reward
  term is classified as raw or shaped and bucketed correctly in
  `env/betfair_env.py::_settle_current_race`.
- `info["day_pnl"]` is authoritative; `info["realised_pnl"]` is
  last-race-only and exists only for backward compat.
- `ExchangeMatcher` single-price / LTP-filter / max-price rules are
  load-bearing. No new feature may depend on walking the ladder or
  peeking at unfiltered top-of-book.
