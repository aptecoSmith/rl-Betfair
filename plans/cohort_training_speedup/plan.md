# Cohort training speedup plan

## TL;DR

Per-agent training cost **doubled** from ~17 min (old V2 cohort) to
~35 min (new V4+matcher-fix cohort). Hard profiling confirms the
**single largest cause is the V3 TVL feature builder calling
`ticks_between` ~33k times per race** (97 % of feature-build CPU).
Fixing just that one hot path should restore most of the lost
performance.

A ranked list of options follows. Each entry has measured impact,
implementation cost, risk class, and a "do now / do next / do later"
recommendation.

## How we got here

- **Old V2 cohort:** features = 26 dims, simple ladder + LTP windows.
  Per-race feature build: **~11 ms**. Per-agent training: ~17 min.
- **New V4 cohort:** features = 39 dims, adds V3 TVL aggregates (8)
  + V4 cross-runner stats (5). Per-race feature build: **~1750 ms**
  (160× slower). Per-agent training: ~35 min.

`cProfile` on a single race × variant V3:

| Function | Calls | Cumulative time |
|---|---:|---:|
| `_fill_tvl_features` | 1,926 | 10.96 s |
| `ticks_between` | 101,871 | 10.69 s |
| `tick_offset` | 4,426,794 | 9.62 s |
| `snap_to_tick` | 4,632,456 | 4.68 s |
| `builtins.round` | 13,652,760 | 2.94 s |

V4's additional cross-runner work adds only ~40 ms / race on top —
basically free relative to V3. So **V3 is the entire speedup target**.

## Cohort-level cost math

Per cohort run with current recipe (12 agents × 5 gens × (16 train +
~10 eval days) × ~80 races/day):

- Feature builds: ~125,000 per cohort
- At 1.75 s / build (current V3+V4): **61 hours**
- At 0.05 s / build (target post-fix): **1.7 hours**
- Recovered: ~59 hours of CPU. Per-agent training drops back to
  ~17-20 min. Full cohort wall: ~12-15 h instead of 35 h.

That's the prize.

## Ranked options

### A. Vectorise `_fill_tvl_features` (`data/predictor_features.py`)

**The headline fix.** Replace the per-level Python loop with numpy
array ops over the TradedVolumeLadder.

Current shape (simplified):

```python
for lvl in tvl:                     # ~30 entries per ladder
    price, size = lvl.price, lvl.size
    td = ticks_between(ltp_snapped, price)   # ~10 μs each, 33k calls/race
    if price < ltp_snapped:
        if td <= 5: feat[t_idx, 28] += size
        ...
```

Vectorised:

```python
prices = np.array([lvl.price for lvl in tvl], dtype=np.float64)
sizes  = np.array([lvl.size  for lvl in tvl], dtype=np.float64)
tds    = _ticks_between_vectorised(ltp_snapped, prices)  # numpy
below = prices < ltp_snapped
above = ~below
feat[t_idx, 28] = sizes[below & (tds <= 5)].sum()
feat[t_idx, 29] = sizes[below & (tds <= 10)].sum()
feat[t_idx, 30] = sizes[above & (tds <= 5)].sum()
feat[t_idx, 31] = sizes[above & (tds <= 10)].sum()
feat[t_idx, 27] = sizes[prices == ltp_snapped].sum()
feat[t_idx, 26] = sizes.sum()
feat[t_idx, 32] = float(len(tvl))
feat[t_idx, 33] = 1.0
```

The remaining piece is `_ticks_between_vectorised`. Two ways:

1. **Lookup-table cache**: precompute the full Betfair tick ladder
   (~200 prices from 1.01 to 1000) ONCE at import time, build a
   `dict[price → tick_idx]` or sorted `np.array` for `np.searchsorted`.
   `ticks_between(a, b) = idx[a] − idx[b]` becomes one searchsorted
   call.
2. **Inline the formula**: `tick_offset` is a piecewise-linear
   function of price; can be expressed in pure numpy.

Lookup-table is simpler, safer, and provably exact-match. Go with it.

**Savings:** ~1700 ms / race → ~10 ms / race. **~96% reduction in V3
build cost.** Per-cohort recovery: ~55 hours.

**Cost to implement:** 2-3 hours. Most of that is testing.

**Risks:**
- *Correctness*: V3 features must STILL match the predictor's
  training-time output byte-identically. The existing regression
  test `test_v3_v4_features_match_training_time_builder` catches
  this; I'd extend it to cover the vectorised path with another
  comparison run.
- *Float precision*: `ltp_snapped` is computed via `snap_to_tick`
  which rounds. The vectorised lookup must use the SAME snapping
  function (call `snap_to_tick` once, then index into the table).
  Risk: low if implemented carefully.

**Recommendation:** **DO FIRST.** Implement before the next cohort
launch.

### B. Cache feature windows across agents in a generation

V4 windows are a pure function of race data. The current code
rebuilds them for EVERY agent (12 agents × same race = 11 wasted
builds). Cache the `(windows, indices)` tuple per (date, race) once
per generation and reuse.

**Savings (without option A):** ~33 hours per cohort (12× speedup of
the V3 portion only, but only on training days; eval days still
rotate per gen).

**Savings (after option A):** Marginal — V3 build is ~50ms by then,
so 12× savings is only ~600ms per race × 80 races × 16 days × 5 gens
= 64 min total. Worth doing if cheap, not worth doing if hard.

**Cost to implement:** 1-2 hours. The cache key is `(date,
market_id)`, invalidated when generation ends.

**Risks:**
- *Memory*: a day with 80 races × ~500 windows × 32 ticks × 39
  features × 4 bytes = ~200 MB. Manageable.
- *Cache invalidation*: rotating eval days may force a rebuild each
  gen. Need to be careful that the cache lives only as long as a
  generation, not across gens.

**Recommendation:** **DO if option A's savings don't bring per-agent
under 20 min**, otherwise SKIP. Marginal value once V3 is fast.

### C. Drop training days from 16 → 13

Saves 18 % of training time directly. We chose 16 deliberately to
get more regime diversity for the rotating eval. With rotating
eval already implemented (7-of-10 sampled per gen), the marginal
gain from 16 vs 13 training days is modest.

**Savings:** ~5-7 hours per cohort (proportional to training day
reduction). Stacks linearly with option A.

**Cost to implement:** 30 seconds — change one list in the wrapper
script.

**Risks:**
- *Less regime variety in training data.* But rotation already
  forces the policy to generalise; this is mostly redundant.
- *Worst-case agent regression*: maybe ~1-2 % lower mean. Hard to
  predict.

**Recommendation:** **DO if combined wall is still too long after
A**. Skip otherwise.

### D. Reduce monitor-eval frequency from every gen to gens 2 & 5

Currently the monitor eval fires after every generation. Top-3 × 14
monitor days × ~30 s / rollout = ~21 min / gen × 5 gens = 1.75 hours
total. If we run it ONLY after gen 2 and gen 5, that's ~42 min
saved.

**Savings:** ~1 hour per cohort.

**Cost:** 10 lines of code in `run_cohort` (add a "monitor every N
gens" flag).

**Risks:**
- *Less granular overfit signal.* But 2 data points (mid + end) is
  enough to detect the divergence pattern.
- *Monitor-early-stop fires later.* Minor — early-stop on monitor
  is the safety net, not the main check.

**Recommendation:** **DO** alongside option A. Cheap, makes the
monitor-eval cost manageable.

### E. Use the existing `--batched` cluster runner

The runner supports `train_cluster_batched` which trains all agents
of the SAME architecture (same `hidden_size`) in one batched env
pass. With diverse hidden_size genes (the cohort has 64, 128, 256),
agents naturally split into 3 clusters; agents within a cluster
share env compute.

**Savings:** Potentially huge (2-5×) but depends on cluster sizes.
In practice cohort has roughly equal numbers per hidden_size, so
~3× speedup on per-agent training.

**BUT:** known limitations documented in the runner:
- `per_transition_credit=True` is ignored under batched
- `bc_pretrain_steps > 0` is ignored under batched
- **Monitor eval may or may not work** — would need testing

**Savings (with all caveats):** ~15-20 h per cohort if it works.

**Cost to implement:** Test thoroughly first. Maybe 4-6 hours of
verification. Could discover blocking incompatibilities.

**Risks:**
- *Unknown interaction with rotating eval + monitor eval* —
  neither was tested under batched mode.
- *Determinism may differ* — batched updates can change PPO update
  ordering subtly.

**Recommendation:** **DEFER to next iteration.** High potential but
also highest uncertainty. After A+B+C+D land and produce a clean
cohort, evaluate whether to invest the test time.

### F. Reduce agents per gen (12 → 8)

Smaller cohort means fewer trainings per gen. Linear savings.

**Savings:** 33 % faster per gen → ~12 h saved on a 35h cohort.

**Cost:** 1 line in the script.

**Risks:**
- *Less GA diversity per gen.* The whole point of the GA is
  population-based search; 8 is on the low end. Bred offspring
  would be highly correlated.
- *Worse final result is the biggest risk.* We've not validated
  cohort runs at 8 agents.

**Recommendation:** **DEFER.** Don't trade quality for speed at
this stage. Revisit after the first honest cohort lands.

### G. Reduce generations 5 → 3

Saves 40 % of compute. The early-stop machinery already does this
adaptively when improvement stalls.

**Savings:** ~14 h if compute is cohort-bound.

**Cost:** 1 line.

**Risks:**
- *Insufficient generations to see GA convergence.* With 3 gens,
  we get gen 0 (init), gen 1 (first breeding), gen 2 (second
  breeding). That's enough to see the trajectory but not to
  converge.
- *Less data for the monitor metric* to detect overfit pattern.

**Recommendation:** **DEFER.** Let early-stop be the mechanism. If
overfitting fires at gen 4-5, it'll catch it. If not, we want the
extra data points.

## Recommended sequence

For the **next cohort launch** (after current 35h run completes):

1. **A. Vectorise `_fill_tvl_features`** — biggest single win, low
   risk, regression-tested.
2. **D. Monitor every 2 gens, not every gen** — small cheap win.
3. **C. Optional: 16 → 13 training days** — only if A + D didn't
   get under target.

That stack delivers ~30 of the lost 35 hours back. Target wall:
**12-15 hours** for a 5-gen × 12-agent cohort.

Stretch options for a later iteration:

4. **B. Per-race feature cache** — only if A's win isn't enough.
5. **E. `--batched` mode** — bigger upside but unknown
   compatibility cost.

## Do NOT do during the current run

- All options here assume the current 35h cohort is allowed to
  finish. Restarting it now to apply any speedup means losing the
  first honest matcher-fix data we'd see (~6 h of progress already
  in).
- Implementation should land BEFORE the next cohort launch, not
  during this one.

## Test plan

For option A specifically:

1. Extend `tests/test_direction_features_v3_v4.py` with a test
   that runs build_direction_windows_for_race with `variant="V3"`
   AFTER the vectorisation lands and asserts byte-identical output
   against the predictor's training shard (`betfair-predictors/data/predictor_dataset/2026-05-07.parquet`).
2. Spot-check 3 different (race, tick, sid) combos — same as the
   current test does.
3. Run a tiny end-to-end cohort smoke (2 agents × 1 gen × 2 train
   days × 2 eval days) and verify the cohort log shows no
   `predict_tick_batch failed`.
4. Profile a single race build to confirm V3 build drops to
   ~10-20 ms.

Total test wall: ~30 min.

## Acceptance criteria for option A

- All existing tests in `test_direction_features_v3_v4.py` still
  pass
- 117/117 features still match the predictor training-time builder
  across the spot-check (3 race-tick-sid combos)
- V3 build time per race drops from ~1750 ms to under 50 ms
  (35× speedup minimum)
- 2-agent × 1-gen smoke cohort runs to completion with zero
  predictor warnings

If all four hold, ship.
