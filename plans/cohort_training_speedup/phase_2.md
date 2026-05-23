# Cohort speedup phase 2 — second pass

## Measured baseline

From the live cohort log (`_predictor_SCALPING_postfix_e3_cohort_1779478669`,
first agent on day 2026-04-16, post phase-1 vectorisation):

| Component | Wall (per day) | Share |
|---|---:|---:|
| `load_day` (parquet → Day) | 1.55 s | 2 % |
| `engineer_day` (static features) | 9.61 s | 12 % |
| Rollout (env.step × n_ticks) | **68 s** | **85 %** |
| **Per-day total** | **~80 s** | |

Per agent on 16 train + 7 eval = 23 day-rollouts: ~30 min wall.
Rollout is **6.5 ms / env.step**. The rollout is where the half-the-time
target has to come from — env build is only 12 % of the cost.

## What's spending the 6.5 ms / step?

Inspection of `env.step()` + `_get_info()` + `_get_agent_state()` +
`_get_obs()` reveals four candidate hot paths:

### 1. `emit_debug_features=True` does heavy per-runner feature work on every step that NOTHING reads during training

`env/betfair_env.py:2519-2565` runs per env.step() when
`_emit_debug_features=True` (the env default). For every runner on
the current tick it computes:

- `compute_obi(atb, atl, top_n=3)` — loops over price levels
- `compute_microprice(atb, atl, top_n=3, ltp)` — loops over price levels
- `compute_traded_delta(hist, ref_mp, window, now)` — walks deque
- `compute_mid_drift(hist, window, now, tick_size)` — walks deque
- `compute_book_churn(prev_atb, prev_atl, atb, atl, top_n)` — list diff

Plus `_update_runtime_windowed(tick)` (line 2944) maintains the
deques every step solely to feed those computations.

**The kicker:** these go into `info["debug_features"]`. Grep across
the training stack finds **zero consumers** of `info["debug_features"]`
during training rollouts. It's used by:
- Tests (`tests/research_driven/test_p1a_obi.py` etc.) — only when
  tests run
- Replay UI — not running during cohort training

`training/evaluator.py:255` already sets `emit_debug_features=False`.
The cohort path (`training_v2/cohort/worker.py::_build_env_for_day`)
does NOT — it falls through to the env default of True.

**Estimated savings:** 25-40 % of per-step time. Per agent: 8-12 min saved.
Per cohort (60 agents): 8-12 h saved.

**Risk:** zero. The tests use a separate env instance with the flag
explicitly set; no production code path reads these fields during
training.

**Effort:** one line in `training_v2/cohort/worker.py::_build_env_for_day`.

### 2. `_get_info()` always builds passive_orders/fills/cancels lists

`env/betfair_env.py:2583-2585`:

```python
"passive_orders": [o.to_dict() for o in bm.passive_book.orders],
"passive_fills": bm.passive_book.last_fills,
"passive_cancels": bm.passive_book.last_cancels,
```

`to_dict()` is called on every resting passive order every step.
Late in a race that's 50+ orders × ~10 fields = 500 dict ops per step.

Used by replay UI; not consumed during training. Same fix pattern:
gate on a flag.

**Estimated savings:** 5-10 % of per-step time. Per agent: 2-4 min saved.
Per cohort: 2-4 h saved.

**Risk:** low. Same audit as (1) — find any production consumer; if
none, gate it off during training.

**Effort:** ~10 lines (add flag, gate the three list builds, plumb
through cohort worker).

### 3. Per-step `get_paired_positions` + `get_naked_exposure` + `get_positions` re-walk all bets

`_get_agent_state()` (line 2176, 2180) and `_get_position_vector()`
(line 2251) each iterate `bm.bets` from scratch on every step. Bets
grow to ~50-100 per race; each walk allocates a fresh dict / list.

Per step: 3 walks × ~50 bets × ~5 ops each ≈ 750 ops. Per agent:
230k steps × 750 ops = **172 M Python ops**.

**The fix:** maintain incremental aggregates. Update on bet insert
(`place_back` / `place_lay`), bet match (passive book fill), bet
settle (settlement). Read in O(1).

**Estimated savings:** 10-20 % of per-step time. Per agent: 3-6 min saved.
Per cohort: 3-6 h saved.

**Risk:** medium. Invariant tracking across `place_back` /
`place_lay` / `passive_book.on_tick` / `_settle_current_race`. Tests
would catch most regressions but the failure mode (subtly wrong
exposure numbers) is annoying to debug.

**Effort:** 4-6 hours.

### 4. Engineer features + predictor inference are computed 12× per day per gen (once per agent)

The current cohort runner builds env per (agent, day) — so the same
day's `engineer_day` (9.6 s) + predictor inference (precompute) is
repeated 12× per generation × 23 days × 5 gens = 1380 redundant
builds × ~10 s = **3.8 hours per cohort**.

These computations are PURE FUNCTIONS of the day data + the
predictor bundle. They could be shared across agents within a
generation via a small `PrecomputedDay` cache object that lives at
`run_cohort` scope.

**Estimated savings:** ~3-4 h per cohort. NOT per-agent
(it's a per-cohort-gen amortisation), but the wall reduction is real.

**Risk:** medium. The cohort runner currently treats each (agent,
day) as independent; adding shared state means careful lifecycle
management. Bet manager state stays per-agent; only the static
observation tensor + predictor outputs are shared.

**Effort:** 4-6 hours.

## Recommended sequence

Stack-rank by impact / risk:

1. **Option 1: `emit_debug_features=False` during training.** Quick + safe + biggest single win. ~10 min/agent saved.
2. **Option 2: gate `passive_orders` / `passive_fills` / `passive_cancels` list-builds.** Same fix pattern; cheap. ~3 min/agent saved.

Combined options 1+2 expected savings: **~13 min/agent**, taking per-agent
from 28 min → ~15 min. **That's roughly the half-time target.**

3. **Option 4: per-day shared precompute cache.** Bigger refactor;
worth doing for future cohorts. Saves another ~30 min off cohort
wall after 1+2 land.

4. **Option 3: incremental bet aggregates.** Defer until 1+2 confirmed.
The per-step bet-walk cost matters less when the heavy debug features
are gone — re-profile before committing the engineering time.

## Implementation order

For the current cohort: nothing — let it run, it's already going.

For the NEXT cohort:
- Land options 1+2 (one PR, both share the same gating pattern)
- Smoke-test a 2-agent × 1-gen mini cohort
- Verify the existing `test_betfair_env.py` tests still pass (most assert
  `info["debug_features"]` content when emit_debug_features=True; they
  shouldn't run with False).
- Re-profile per-agent wall — expect ~15 min/agent
- If under target, ship. If still over, consider option 4 next.

## Caveat about the "ALWAYS WANT MONITORING" requirement

The user has explicitly said monitoring should not be reduced. To be
clear:

- **Options 1+2 do NOT touch monitor-eval.** They only disable
  `info["debug_features"]` (per-runner OBI/microprice/etc.) and
  passive-order list-building. The monitor eval is a SEPARATE
  post-generation step (`_evaluate_agents_on_monitor_days`) that
  runs an entire rollout pass on the held-out days; it's unaffected.
- **The "debug_features" name is misleading.** These are per-tick
  diagnostic fields for the replay UI, not the training monitor or
  the overfit tripwire. They were never visible to the agent and are
  never consumed during training rollout.

So options 1+2 are safe even under the "always monitor" constraint.

## Test plan for options 1+2

1. Run existing `tests/test_betfair_env.py` with the new flag set to
   False; expect failures only in tests that assert specific debug
   field content (which is the intended new behaviour).
2. Add a regression test:
   `tests/test_emit_debug_features_off.py::test_info_no_debug_features_when_off`
   — assert that `info` does not contain (or contains empty)
   `debug_features` / `passive_orders` when the new flag is False.
3. Smoke run: 2-agent × 1-gen × 2-day mini cohort, verify zero crashes.
4. Profile re-run: confirm per-step wall drops to ~3.5-4.5 ms (from 6.5).
5. Full 5-gen × 12-agent cohort: confirm avg per-agent wall lands
   ~15 min (from 28).
