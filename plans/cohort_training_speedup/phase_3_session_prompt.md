# Session prompt: profile and meaningfully optimise rl-betfair cohort training

Self-contained brief for a fresh agent session. The user will paste a link to this
file (or copy its contents) as the opening message of a new conversation.

---

## Context

You're picking up a performance investigation on **rl-betfair**, a v2 cohort
GA-PPO trainer for Betfair scalping policies. The latest cohort runs are taking
~28 minutes per agent, and we want **~14 minutes per agent** (half that) so a
30-agent cohort completes in ~7 hours instead of ~14.

**Two prior optimisation attempts shipped without proper profiling and
under-delivered:**

1. **Phase 1 — V3 TVL feature vectorisation** in
   `data/predictor_features.py::_fill_tvl_features`. Replaced a per-level
   Python loop using `ticks_between()` (~1750 ms / race build) with a numpy
   `searchsorted` against a precomputed `BETFAIR_TICK_LADDER` (~43 ms /
   race). **This one actually delivered** ~10 min/agent saved because the
   inner loop was genuinely catastrophic (160× per-race speedup,
   regression-tested feature-for-feature against the predictor training-time
   builder).
2. **Phase 2 — `emit_debug_features=False` for cohort training** in
   `training_v2/cohort/worker.py::_build_env_for_day`. Disabled per-runner
   OBI/microprice/traded_delta/mid_drift/book_churn in `_get_info()` because
   nothing during training reads them. **Projected 12 min/agent saved.
   Actually saved <1 minute/agent** (6.53 ms → 6.29 ms per step). The
   micro-benchmark showed 48% reduction of `_get_info()` cost but that
   function was a small slice of total step cost.

**The pattern:** isolated function-level micro-benchmarks lied about impact
because the surrounding ~6 ms/step of OTHER work dominates. We never profiled
an actual full cohort training step.

## Your task

**Do real profiling first. No code changes until evidence is in.**

Specifically:

1. Profile a representative cohort training step (policy forward + env.step +
   bet manager work + PPO collector + occasional gradient update). Use either:
   - `cProfile` wrapped around a 2000-step rollout via
     `training_v2.discrete_ppo.rollout.RolloutCollector.collect_episode`, OR
   - `py-spy record --pid <runner_pid>` against a running cohort (already
     running — see "do not touch" below).
2. Produce a sorted top-30 by cumulative time. Identify functions accounting
   for ≥5% of cumulative time.
3. Build a **ranked candidate list** with: function, % of step time,
   hypothesised root cause, proposed fix, projected savings (based on measured
   cost, not a guess), implementation effort, and risk class.
4. **Do NOT implement anything** until the profile + ranked plan are reviewed.

## Hard targets

- Per-agent training time: **28 min → 14 min** (50 % reduction)
- Per-step time: **6.3 ms → ~3.2 ms** on the actual cohort training path
- Total cohort wall (30 agents × 5 gens): **~14h → ~7h**

If after profiling, 50 % savings looks infeasible without a major refactor,
say so. Don't dress up a small saving as a big one. The user has explicitly
said they prefer "honest small wins" over "optimistic projections that don't
land."

## Hard constraints — DO NOT TOUCH

- **The currently running cohort:**
  `registry/_predictor_SCALPING_postfix_e3_cohort_1779530050` (verify name on
  arrival — it may have been renamed or completed). Runner PID found via:
  `Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*training_v2.cohort.runner*' }`.
  Don't stop it.
- **Monitor eval must remain on every generation** (no reducing monitor
  frequency).
- **`emit_debug_features=False`** is now the cohort default — keep it.
- **Matcher fix (PassiveOrder.crossed gate)** in
  `env/bet_manager.py::PassiveOrderBook.on_tick` is correctness-critical.
  Don't remove it.
- **V4 predictor compatibility:** the env builds V4-variant tick windows; the
  predictor expects 39 dims. Manifests + feature pipeline are wired.
- **Regression tests must still pass:**
  - `tests/test_direction_features_v3_v4.py` — V3/V4 feature byte-equality
    against predictor training-time builder
  - `tests/test_emit_debug_features_off.py` — `emit_debug_features=False`
    invariants
  - `tests/test_passive_order_book_dual_mode.py` — matcher fix crossing gate
  - `tests/test_betfair_env.py`, `tests/test_bet_manager.py`,
    `tests/test_forced_arbitrage.py`, `tests/test_v2_select_days.py`

## Key files to read first

- `env/betfair_env.py` — `BetfairEnv`. Look at `step()` (~line 2913),
  `_get_obs()` (2153), `_get_agent_state()` (2160), `_get_position_vector()`
  (2189), `_get_info()` (2491).
- `env/bet_manager.py` — `BetManager` + `PassiveOrderBook`.
  `get_paired_positions` (1502), `get_naked_exposure` (1632),
  `get_positions` (1669), `passive_book.on_tick` (672).
- `agents_v2/discrete_policy.py` — `DiscreteLSTMPolicy` and base class.
  `forward()` is the hot inference path.
- `training_v2/discrete_ppo/rollout.py` — `RolloutCollector.collect_episode`.
- `training_v2/cohort/worker.py` — `train_one_agent`, `_build_env_for_day`,
  `_eval_rollout_stats`. Note `scalping_train_config()` produces the default
  cfg used by all builds.
- `data/predictor_features.py` — V2/V3/V4 feature builders (already optimised
  in phase 1; understand the contract).
- `plans/cohort_training_speedup/plan.md` and
  `plans/cohort_training_speedup/phase_2.md` — what's been tried; learn from
  the over-projection errors.

## Profiling pointers

The currently running cohort uses `--device cuda --strategy-mode arb` and an
LSTM-family policy. The per-day rollout is ~10k env steps. Production timing
per the cohort log: rollout takes ~65-68 seconds per day per agent, so
6.3-6.5 ms per step.

To replicate the production rollout in a profiler:

```python
# Mimic training_v2.cohort.worker.train_one_agent's per-day rollout
from training_v2.cohort.worker import _build_env_for_day, scalping_train_config
from training_v2.discrete_ppo.rollout import RolloutCollector
from agents_v2.discrete_policy import DiscreteLSTMPolicy
from predictors.loader import PredictorBundle
from pathlib import Path
import cProfile, pstats

bundle = PredictorBundle.from_manifests(
    champion_manifest=Path('../betfair-predictors/production/race-outcome/manifest.json'),
    ranker_manifest=Path('../betfair-predictors/production/race-outcome-ranker/manifest.json'),
    direction_manifest=Path('../betfair-predictors/production/direction-predictor/manifest.json'),
)
cfg = scalping_train_config()
env, shim = _build_env_for_day(
    day_str='2026-05-07',
    data_dir=Path('data/processed'),
    cfg=cfg,
    scorer_dir=Path('models/scorer'),  # confirm path
    reward_overrides={'force_close_before_off_seconds': 120.0,
                      'close_feasibility_max_spread_pct': 0.05},
    predictor_bundle=bundle,
    use_race_outcome_predictor=True,
    use_direction_predictor=True,
    predictor_lean_obs=True,
    predictor_p_win_back_threshold=0.20,
    predictor_p_win_lay_threshold=0.40,
    race_confidence_threshold=0.50,
)
policy = DiscreteLSTMPolicy(
    obs_dim=shim.obs_dim, action_space=shim.action_space, hidden_size=128,
)
policy.eval()  # or train mode; profile both
collector = RolloutCollector(shim=shim, policy=policy, device='cuda')

pr = cProfile.Profile()
pr.enable()
batch = collector.collect_episode(deterministic=False)
pr.disable()

pstats.Stats(pr).sort_stats('cumulative').print_stats(40)
```

This MUST be the path you profile — anything else will mislead you (as it
misled the previous two attempts).

## Specific candidates flagged by the previous attempts (verify before believing)

These are guesses, not measurements. The profile is the truth.

1. `bm.get_paired_positions(market_id=..., commission=...)` called per-step
   from `_get_agent_state` — iterates `self.bets` from scratch every step,
   with `max()`/`min()` Python comprehensions. As pairs accumulate (50-100 by
   end of race), this grows quadratically with race length. **Hypothesised
   fix:** maintain incremental aggregates on bet append + settle.
2. `bm.get_naked_exposure` and `bm.get_positions` — similar
   walk-self.bets-per-step shape.
3. Policy `forward()` — CPU LSTM at hidden_size=128/256 may be 1-2 ms / step.
   **Hypothesised fix:** none easy; maybe `torch.compile`.
4. `PassiveOrderBook.on_tick()` — walks all resting orders per tick, computes
   crossing gate. With 50+ resting orders, this is a real cost.
5. PPO gradient update overhead amortised across steps.

## Deliverables

1. **A profile dump** — committed under
   `plans/cohort_training_speedup/phase_3_profile.txt` or similar.
   Top-30 by cumulative time + top-30 by total time.
2. **`plans/cohort_training_speedup/phase_3.md`** with:
   - Profile summary (what's actually slow, % of total)
   - Ranked options with **measured-time-grounded** savings projections
   - For each option: implementation sketch + effort + risk + acceptance
     criteria
3. **NO code changes** until the user reviews the plan.

The over-projection rule: if your projected savings are >2× the actual
function's measured cost, your projection is wrong. State savings as
`measured_function_time × expected_reduction_fraction × calls_per_agent`
with all three numbers shown.

## Style notes

- Be skeptical of your own projections. Verify by re-profiling after each
  change.
- "Halve the agent time" is the user's STATED target but if the profile says
  it's impossible without a major refactor, say so honestly. They've
  explicitly said they prefer honesty over wishful thinking.
- Don't restart the running cohort. Don't change the recipe. Don't reduce
  monitoring. The user is sensitive to these specific things.

## When you're done

Present:

1. Top-10 hot functions with their measured % of step time.
2. 3-5 ranked candidate optimisations with savings projections grounded in the
   measured numbers above (not micro-benchmarks).
3. A clear go/no-go recommendation for the next cohort: "candidate X is worth
   implementing because it'll save Y minutes per agent based on measured
   Z ms/step in the profile" — OR — "no single optimisation cracks 5% of step
   time; the 50% target requires architectural change (option list)."

Then wait for the operator to choose which to implement.
