# Lessons Learnt — Arb Improvements

## From the run that triggered this plan (run 90fcb25f, 2026-04-14)

- **The scalping agents can see arbs — they stop looking for them.**
  Every agent in the run took arbs in episode 1 (6 / 5 / 6 / 5 / 6 / 4
  arbs completed across six agents on day 1), then flatlined at
  `P&L=+0.00` from episode 4 onwards. This is not a perception
  problem; it's a training-stability problem.

- **PPO's KL clip is a trap in the "don't bet" corner.** Once the
  policy has moved to abstention, there is no gradient signal to
  pull it back. The trust-region update has nothing to push against.
  Feature engineering alone cannot escape this corner — we need to
  prevent entering it in the first place (Phase 1) *and* seed the
  policy with arb-taking behaviour (Phase 3).

- **Gradient-norm clipping (`max_grad_norm=0.5`) is not sufficient.**
  It bounds gradient *magnitude* after the fact, but not the
  *direction* the outlier batch learned. The direction learned on a
  batch dominated by a ±£300 advantage target is "bet less", and
  that direction persists after clipping.

- **The missing-arb-metrics display turned out to be a diagnostic
  tell, not a bug.** The activity-log suffix in
  `agents/ppo_trainer.py:637` only prints when `arbs_completed or
  arbs_naked > 0`. Its absence means the agent placed zero bets —
  which is the failure mode. Keep the conditional (don't spam the
  log with `arbs=0/0`), but surface the information in the monitor's
  bet-rate sparkline instead (Session 3).

## From planning discussion (2026-04-14)

- The user's first framing was "why are some days missing arb
  metrics?" The answer was display-level (conditional suffix), but
  the real question hiding underneath was "why does the model stop
  looking for arbs?" Good planning meeting — follow the thread, not
  the symptom.

- The user instinct to "show the agent where the arbs were on
  training days" was correct, and it's exactly behaviour cloning on
  an oracle dataset. The training data already *is* the ground
  truth — we're not asking the policy to discover arbs, we're
  showing it.

- Ordering matters: stabilisation before features before BC. Without
  stabilisation, BC-pretrained weights get crushed by the first
  noisy PPO batch and we're back where we started. Without features,
  BC is regressing on a representation that can't generalise arb
  recognition to unseen days. Without BC, Phase-1 stability just
  means the agent reliably learns to not bet, faster.

## Prior art in this repo

- **`env/features.py` pattern.** Six pure, duck-typed, vendorable
  feature functions already live here (`compute_microprice`,
  `compute_obi`, `compute_traded_delta`, `compute_mid_drift`,
  `compute_book_churn`, `betfair_tick_size`). New arb features
  follow the same style exactly: stdlib-only, `PriceLevel`
  protocol, no env imports. Reference implementation for Session 4.

- **`OBS_SCHEMA_VERSION` + `validate_obs_schema()`** already refuse
  mismatched checkpoints. Session 5 increments the constant; the
  refusal path is proven and doesn't need new scaffolding.

- **Scalping mode end-to-end wiring already exists.** Forced-arb
  sessions 1–3 (`plans/issues-12-04-2026/05-forced-arbitrage/`)
  delivered: genes, wizard toggles, evaluator awareness, scalping
  reward terms, paired-order settlement, pair-tracking in
  `BetManager`, tick-ladder offset utilities. This plan layers on
  top — it does not re-implement any of it.

- **Commission constant**: `BetManager.get_paired_positions` already
  uses a commission rate to compute `locked_pnl`. Session 4 imports
  the same constant rather than redefining it — single source of
  truth for all post-commission profit calculations.

- **Reward overrides path**: `BetfairEnv._REWARD_OVERRIDE_KEYS` is
  the established whitelist for per-agent reward gene flow. Session 1
  adds `reward_clip` to it; the plumbing that makes it per-agent
  already exists.

## Anti-lessons (things NOT to do)

- **Do not silence reward spikes in the telemetry.** A real £300
  race produced a real number. Make that visible to the operator.
  The fix decouples *training signal* from *display*, not the other
  way around.

- **Do not make the arb features non-negative for "symmetry".** Same
  lesson as `CLAUDE.md` on reward shaping — negative values for a
  crossed post-commission book are the *signal*. That moment is
  exactly when the policy should be placing arbs. Clipping to zero
  discards the most valuable feature state.

- **Do not share BC-pretrained weights across a population.** Each
  agent pretrains from its own initialisation. Pretraining once and
  cloning collapses behavioural diversity on generation 0, defeating
  the point of the genetic search.

- **Do not run the oracle scan inside the training loop.** One-off,
  on-disk, keyed by date. A training run reads; it never writes.
  In-loop scanning is I/O + full-tick walks per agent per epoch —
  catastrophically slow for zero benefit.

- **Do not skip the verification session.** The failure this plan
  exists to fix is specific and measurable. Running the original
  `90fcb25f` config with the new machinery and comparing against the
  baseline table in `progress.md` is the only way to know the work
  shipped.

## Session 1 — Reward & advantage clipping (2026-04-14)

- **Absence of news:** nothing surprising. The scope was "three optional
  clip knobs, defaults off, don't touch telemetry". It landed in the
  shape the plan described — no refactor temptation, no new coupling.
  Noting explicitly so the next session doesn't assume there was a
  hidden wrinkle.

- **Design note that paid off:** splitting `Transition` into `reward`
  (telemetry-truth) and `training_reward` (fed into GAE) makes the
  `raw + shaped ≈ total_reward` invariant trivially safe to test —
  the clip operates on a dedicated field, so any reward-accumulator
  touch would show up as a clear diff. Worth keeping this pattern if
  Session 2 (entropy floor) or Session 3 (signal-bias warmup) introduce
  any more "training-signal-only" transforms.

- **Stub policy gotcha for testers:** a policy whose outputs don't
  depend on the input (e.g. `action_mean = out * 0`) trains no
  gradient even when `_ppo_update` runs end-to-end. The "policy
  parameters changed" assertion caught this immediately — tests that
  rely on the update actually doing work should be checking parameter
  drift, not just that no exception was raised.

## Session 2 — Entropy floor & per-head logging (2026-04-14)

- **Per-head entropy is trivially sliceable from a single flat Normal.**
  The policy packs its action space as
  `[signal × N | stake × N | aggression × N | cancel × N | arb_spread × N]`
  with N = `max_runners`. `dist.entropy()` is already per-dim, so a simple
  index slice gives per-head values — no need to change the policy
  network or introduce per-head Normals. The only ugliness is that
  directional-mode policies have 4 heads and scalping policies have 5;
  the slicer just iterates `min(per_runner_apd, len(_HEAD_NAMES))` and
  the unused arb_spread window stays empty on directional runs.

- **Patience is `streak > N`, not `streak >= N`.** The spec said
  "below floor for > N batches" — test 5 pins this explicitly so the
  boundary can't silently shift. With patience=5, the 6th consecutive
  collapsed batch is what trips the flag, matching the plain-English
  reading of the prompt.

- **Don't rebuild the controller when the floor is off.** The gate
  `self.entropy_floor > 0.0` guards the coefficient update, but the
  rolling per-head windows still accumulate. Operators get the
  diagnostic sparkline data for free, and flipping the floor on later
  in the session doesn't require a warm-up period to populate the
  window. This matches Session 1's philosophy: telemetry always on,
  training-signal transforms opt-in.

- **Stub policies need `max_runners` and `_per_runner_action_dim`.** The
  Session 1 stub got away with a flat 2-dim action space because the
  clipping tests don't care about heads. Session 2 tests need those
  attributes — `_compute_per_head_entropy` falls back to a single
  "signal" head when they're missing, which is friendly to code paths
  that hit the controller without a real multi-head policy, but the
  tests themselves explicitly set `max_runners=2, per_runner_action_dim=5`
  so the 5-head schema is actually exercised.

## Session 3 — Signal-bias warmup & bet-rate diagnostics (2026-04-14)

- **Optional kwarg, conditional pass.** Threading `signal_bias` as an
  optional `forward()` kwarg is the natural API, but the Session 1/2
  test stubs don't accept unknown kwargs and calling with the kwarg
  unconditionally broke them. Fix: in `_collect_rollout`, call
  `self.policy(obs, hidden, signal_bias=b)` only when `b != 0.0` and
  fall back to the original two-arg signature otherwise. Keeps the
  default-off path byte-identical AND preserves compatibility with
  pre-session-3 stub policies. Worth remembering for future sessions
  that extend the forward signature.

- **`_apply_signal_bias` is duplication-free by being module-level.**
  All three architectures build their `action_mean` the same way
  (`actor_head` → `(batch, max_runners, per_runner_action_dim)` → cat
  per head). A single helper that takes `actor_out` and adds to
  index 0 means one place to test and one place to reason about the
  "soft prior on head 0" contract. No inheritance games required.

- **Signal is head index 0 everywhere — lean on it.** `_HEAD_NAMES[0]
  == "signal"` from Session 2, the action layout is
  `[signal × N | stake × N | …]`, and the env's back/lay decision
  reads from the same slice. Session 3 added `bet_rate` on the same
  index assumption. Any future change to head ordering will need to
  touch all three (layout, bias, bet-rate). Low risk — the contract
  is load-bearing enough that no-one's touched it since Session 2.

- **`bet_rate` threshold matches the env's internal ±0.33 back/lay
  gate.** Picking the same value means "step where bet_rate tripped"
  and "step where the env placed a bet" don't diverge — which makes
  the diagnostic a faithful proxy for the sparkline the operator
  will read in the monitor.

## Open questions — decide as sessions land

- **Oracle coverage on low-liquidity days.** Unknown how dense arb
  moments are across the training data. If a day has < 10 oracle
  samples, BC on it is cheap-to-run-but-useless. Session 6 surfaces
  the density per day; Session 7 decides the minimum threshold below
  which BC is skipped for a day.

- **BC learning rate vs PPO learning rate.** BC might want a higher
  LR than the PPO schedule starts with. Session 7 exposes
  `bc_learning_rate` as a separate gene; default matches the PPO LR
  until we have evidence either way.

- **Aux head vs BC-only.** If Phase 3 alone fixes the collapse and
  arb-rate stays high through verification, Session 9 may be
  explicitly deferred or dropped. Document the decision in
  `progress.md` Session 10 entry.
