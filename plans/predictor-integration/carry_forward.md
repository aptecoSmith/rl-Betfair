# Carry-forward inventory

What survives from v2 / `plans/rewrite/` into the predictor-
integration plan, and why.

This is the audit the planning prompt asked for. It covers the
state of v2 as of 2026-05-10.

## Survives unchanged

| Component | Why |
|---|---|
| `env/betfair_env.py` (orchestration, day/race/tick loop, `_settle_current_race`) | Load-bearing; only adds a `strategy_mode` kwarg + 6 new RUNNER_KEYS, no mechanics changes. |
| `env/exchange_matcher.py` | Single-price matching, junk filter, hard caps, force-close relaxed path. CLAUDE.md §"Order matching". |
| `env/bet_manager.py` | Budget, liability, P&L tracking. Force-close overdraft. CLAUDE.md §"Force-close at T−N". |
| `env/scalping_math.py` | Equal-profit pair sizing. CLAUDE.md §"Equal-profit pair sizing". |
| `env/order_book.py`, `env/tick_ladder.py`, `env/features.py`, `env/feature_engineer.py` (mechanics) | Microstructure; orthogonal to predictors. |
| `data/extractor.py` | Stable parquet extraction. EW races are ALREADY captured (the win-market parquet carries `each_way_divisor` and `number_of_each_way_places` per race — `plans/ew-metadata-pipeline/`). No extractor work needed. |
| `data/episode_builder.py` | Same — EW metadata already loaded into `Race`. |
| `env/bet_manager.py::settle_race` (EW path) | Complete per `plans/ew-settlement/` — handles doubled stake + place-fraction at settle. The new `value_each_way` mode flips `bet.is_each_way = True` and reuses this path verbatim. |
| `agents_v2/discrete_policy.py` | NO shape change. New RUNNER_KEYS land inside the existing per-runner obs slice; `actor_input` concat from `plans/fill-prob-in-actor` / `plans/per-runner-credit` is preserved. |
| `agents_v2/action_space.py`, `agents_v2/env_shim.py` | Untouched. |
| `training_v2/discrete_ppo/{rollout,gae,trainer}.py` | Trainer reads `strategy_mode` for registry tagging only; no algorithmic change. Reward-centering, advantage-normalisation, KL early-stop, entropy controller, recurrent-state protocol — all unchanged. |
| `training_v2/cohort/{runner,worker,batched_worker}.py` | The `_build_trainer_hp` Path A merge from §"v2 stack consumes aux-head loss weights" is the same pattern this plan uses for the new genes. |
| `registry/model_store.py` (architecture-hash check) | Refuses incompatible checkpoints — correct behaviour when RUNNER_DIM changes. Extended in Session 03 to also refuse on predictor `experiment_id` mismatch. |
| Frontend (Vite/React, websocket events) | Schema-driven; new RUNNER_KEYS appear automatically. No frontend work in this plan. |
| Test infrastructure for env (`tests/test_*` for env/matcher/bet_manager) | Stays valid. |
| GA infrastructure (worker pool, breeding, mutation, gene schema) | Untouched mechanics; new genes added (`predictor_feature_gain`, `value_edge_threshold`, `value_kelly_fraction`, `each_way_edge_threshold`, `each_way_kelly_fraction`). |

## Survives but expected to be re-tuned (not in this plan)

| Component | What changes |
|---|---|
| Aux-head gene weights (`fill_prob_loss_weight`, `mature_prob_loss_weight`, `risk_loss_weight`) | Likely default to 0 in cohorts where champion `p_win` is the primary discrimination signal; the heads stay wired but earn their keep on a per-cohort basis. The follow-on Phase 7 tuning run from `plans/rewrite/` may reach a different conclusion when re-run with predictors enabled. |
| Reward shaping genes (`open_cost`, `mature_arb_bonus_weight`, `naked_loss_scale`, `force_close_before_off_seconds`, `mark_to_market_weight`, etc.) | Untouched in arb mode. Set to 0 / disabled in value modes (no scalping shaping for single-shot bets). |
| Entropy controller's `target_entropy` (current 150) | Unchanged in arb mode. May need re-tuning for value modes (action surface is 4-dim per runner, not 7-dim; natural entropy is lower). Decide in Session 05's smoke run. |
| BC pretrain (`bc_pretrain_steps`, `bc_learning_rate`) | Unchanged in arb. May or may not be useful for value modes; decide post-Session-05. |

## Modified (this plan touches)

| Component | What changes |
|---|---|
| `env/betfair_env.py:86` `OBS_SCHEMA_VERSION` | 7 → 8 |
| `env/betfair_env.py:297` `RUNNER_KEYS` | +6 race-level + 12 per-tick predictor keys |
| `env/betfair_env.py:__init__` | +`strategy_mode` kwarg, +`predictor_bundle` kwarg |
| `data/feature_engineer.py::engineer_tick` | adds predictor injection block |
| `config.yaml` | adds `observations.use_*`, `training.strategy_mode`, `predictors.*` |
| `training_v2/cohort/genes.py` (the `CohortGenes` dataclass) | adds 5 new genes |
| `training_v2/discrete_ppo/trainer.py` (registry record) | tags cohort with predictor `experiment_id`s + `strategy_mode` |
| `tools/reevaluate_cohort.py` | reads predictor `experiment_id`s from cohort row |
| `registry/model_store.py` | extends purge to refuse on predictor mismatch |
| `env/betfair_env.py` (Session 04) | adds `each_way` action dim in `value_each_way` mode; routes the flag to `bm.place_*` |
| `env/bet_manager.py` (Session 04) | `place_back` / `place_lay` accept `each_way: bool = False` kwarg and set `bet.is_each_way` accordingly |
| `agents_v2/action_space.py` (Session 04) | exposes the EW action signal in `value_each_way` mode |

## NEW (this plan introduces)

| Component | What |
|---|---|
| `predictors/__init__.py` | New top-level package |
| `predictors/loader.py` | `PredictorBundle` class; loads three manifests + segment_performance.json |
| `predictors/segment_router.py` | reads `segment_performance.json`, exposes `lookup(market_features) → consumer_hint` |
| `tests/test_predictor_loader.py` | unit tests for the loader |
| `tests/test_predictor_integration.py` | the byte-identical regression guard + per-mode smoke tests |
| `plans/predictor-integration/*` | This plan's docs |

## Out of scope

| Component | Where it's tracked |
|---|---|
| Live inference (ai-betfair) | Separate repo; cross-repo dependency, not bundled. |
| Each-way settlement adjustments | `plans/ew-settlement/`. Already separate. |
| StreamRecorder1 backup ingestion | Separate effort; cross-repo. |
| Frontend predictor visualisation | Future plan if Session 07 verdict is positive. |
| Action-space rewrite (continuous → discrete) | `plans/rewrite/`. Composes with this plan. |
| Internal supervised scorer (`plans/rewrite/phase-0`) | Stays as-is in `plans/rewrite/`. May be retired in a future plan once Session 07 verdict is positive. |

## Cross-references

- `plans/rewrite/README.md` — what's in the rewrite, why it's
  not the same plan as this.
- `plans/rewrite/phase-7-port-aux-heads/findings.md` — the
  empirical evidence that motivates the predictor integration
  hypothesis.
- `plans/fill-prob-in-actor/`, `plans/per-runner-credit/` — the
  actor_input concat pattern this plan re-uses.
- `betfair-predictors/docs/intended_consumer.md` — the documented
  RL-side integration shape.
