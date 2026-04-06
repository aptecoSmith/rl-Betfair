# Lessons Learnt — Architecture & Hyperparameter Exploration

Anything surprising, counter-intuitive, or that would have saved time
if we'd known it earlier. Append at the end of every session.

Project-wide conventions (e.g. CLAUDE.md) take precedence over anything
recorded here — this file is for learnings that are too narrow or too
provisional to belong there yet.

---

## 2026-04-06 — Design review findings

- **Sampled ≠ used.** The genetic algorithm can happily mutate values
  that no downstream code ever reads. `reward_precision_bonus` was in
  the schema for weeks, got mutated in every generation, and never
  changed any agent's reward. Rule: **every gene must have a test that
  asserts the env (or trainer) actually uses the sampled value**, not
  just that the value was sampled. A grep for a gene name should turn
  up at least one "read from hp" site AND one "passed to downstream
  consumer" site.

- **Architecture 2 is a tweak, not a replacement.** `ppo_time_lstm_v1`
  differs from `ppo_lstm_v1` only in a learnable `W_dt` parameter on
  the forget gate. Same encoders, same head, same pooling. So any
  perceived gap between the two architectures on Gen 0 is a signal
  about that one parameter, not about fundamentally different model
  families. Keep this in mind when reading results.

- **Gen 0 is already varied, not cloned.** Contrary to my initial
  suspicion, `population_manager.py:220` does call `sample_hyperparams`
  independently per agent. The problem wasn't "Gen 0 is uniform" —
  it was "several of the sampled values never reach the place that
  would use them".

- **Mutation is single-parameter, not all-at-once.** Good news —
  credit assignment across generations is possible.

- **Terminal bonus is raw.** `day_pnl / starting_budget` is added to
  `_cum_raw_reward`, not shaped. This is correct (it's real money) but
  was worth double-checking because the CLAUDE.md invariant
  `raw + shaped ≈ total_reward` breaks silently if you get this wrong.

## 2026-04-06 — Session 1 (reward plumbing)

- **The gene-name → config-key mapping is not 1:1.** The three reward
  genes sound like config keys but only two of them are: the existing
  `early_pick_bonus_min` / `_max` pair doesn't match the single scalar
  gene `reward_early_pick_bonus`. Session 1's fix maps the scalar to
  *both* ends of the interval (min == max → constant multiplier), which
  plumbs the gene through without changing the reward formula. Session 3
  will split it into proper min/max genes. This was the one non-obvious
  decision in an otherwise mechanical session; noting it so that
  reading `reward_overrides={"early_pick_bonus_min": 1.4,
  "early_pick_bonus_max": 1.4}` in a trainer log doesn't look like a bug.

- **Shared-config mutation is an easy trap.** My first draft of
  `BetfairEnv.__init__` merged overrides into `config["reward"]`
  directly. That would have been a horrible sleeper bug the moment two
  agents shared a config object — which is exactly what
  `PPOTrainer(... config=self.config ...)` does. Added an explicit
  regression test (`test_env_overrides_do_not_mutate_shared_config`) so
  nobody can accidentally reintroduce the aliasing.

- **Retiring `observation_window_ticks` was clean — no migration
  needed.** The sampler iterates over whatever specs exist in
  `search_ranges`, so removing the entry from `config.yaml` removes the
  gene from every checkpoint created after the change. Old checkpoints
  still contain the stale key but nothing reads it, so there's no
  crash path. The only code that *asserted* its presence was
  `tests/test_config.py:94` — updated to match.

- **Pre-existing failure `test_obs_dim_matches_env`.** Unrelated to
  this session: `tests/test_population_manager.py:408` hardcodes
  `obs_dim == 1630`, but current value is 1636. Confirmed stale on
  plain `master` via `git stash`. Left alone — fixing drive-by failures
  is exactly the "while I'm here" scope creep Session 1's plan warns
  against. Flag for a future cleanup pass.
