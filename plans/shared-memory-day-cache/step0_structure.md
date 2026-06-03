# Step 0 — `engineer_day` output structure, shareability, in-place-write audit, mechanism

**Status: COMPLETE. Stop-and-report point (per `01_execute.md`).**
Measured on the real Windows box, predictors-ON, FULL obs, day `2026-04-15`
(73 races, 11,353 ticks) via
`plans/shared-memory-day-cache/_measure/measure_day_structure.py`.

---

## Headline finding (the plan's premise needs correcting)

The plan assumed `engineer_day` returns **large numpy arrays** (the
`static_obs (n_races, n_ticks, n_feat)` tensors) that we'd memmap directly.

**It does not.** `engineer_day` returns a **nested list of Python dicts**:

```
day_features = [                       # list over races
  [                                    # list over ticks in that race
    { "market":          {k: float, ...},      # 37-ish market features
      "market_velocity": {k: float, ...},      # 11-ish velocity features
      "runners":         {sid: {k: float, ...}}# per-runner ~125 features
    },
    ...
  ], ...
]
```

This dict tree is **~0 % numpy** — it is pure Python objects (dicts +
floats + interned key strings). It is what the per-day cache (`mp_day_{day}.pkl`,
`_WORKER_DAY_CACHE`) holds, and it is the ~1–1.4 GB/day the firefight blamed.

The **numpy arrays** the plan wanted (`env._static_obs`) are built
**downstream, per-env, in `BetfairEnv._precompute`** by `_features_to_array`,
and are **NOT cached** today. They are **~10–20× smaller** than the dicts and
**are** memmappable.

> **Consequence for the mechanism:** you cannot `np.save`/memmap the
> `engineer_day` dicts — `.npy` is for homogeneous arrays. The correct
> realisation of the plan's intent is to **cache the downstream
> `static_obs` arrays** (which already exist, just aren't cached), not the
> dicts. This is a *bigger* change than "pickle → memmap the same object,"
> but it is also a *bigger win* (it shrinks the per-day footprint ~10–20×
> **before** any sharing) and it is the only form that can be shared
> cross-process at all.

---

## (1) Byte breakdown — the cached object (`engineer_day` dicts)

| quantity | value (day 2026-04-15) |
|---|---|
| races / ticks | 73 / 11,353 |
| tick dict keys | `market`, `market_velocity`, `runners` |
| runner features / runner | ~125 (pre-predictor; +18 injected later) |
| **RSS delta on build** | **1,899 MB** (incl. the `Day` object) |
| tracemalloc peak | 1,283 MB |
| `deep_sizeof` of `day_features` | **954 MB** |
| on-disk `pickle` (HIGHEST) | 241 MB |

The cached **dict object itself** is ~**0.95–1.3 GB** for this day (the
1.9 GB RSS also holds the parquet-loaded `Day`). This reconciles with the
`multiproc_worker.py` note ("each cached day is ~1.4 GB").
**≈100 % of that is Python-object overhead, ≈0 % shareable arrays.**

## (2) Byte breakdown — the downstream `static_obs` arrays

| quantity | value (predictors-ON, FULL obs) |
|---|---|
| obs dim / tick | **2050** = 37 market + 11 velocity + 14 runners × 143 |
| dtype | `float32` |
| **total `static_obs` / day** | **93.1 MB** |
| on-disk `.npy` | 93.1 MB |

**dict / array size ratio ≈ 10× (`deep_sizeof`) … 14× (tracemalloc) …
20× (RSS).** The 18 predictor columns are part of `RUNNER_KEYS`, so they are
real columns in `static_obs` (and in the 23-dim lean variant too).

LEAN obs (23-dim/runner) would be 37+11+14×23 = 370 floats/tick ≈ **17 MB/day**
— even smaller. The OOM case (predictors-ON + FULL obs) is the 93 MB figure.

## (3) Shareability — cohort-fixed inputs only ✓ (HC#6)

- `engineer_day` takes only `(day, obi_top_n, microprice_top_n,
  traded_delta_window_s, mid_drift_window_s, book_churn_top_n)` — all from
  the cohort-fixed `scalping_train_config()`. **No per-agent / per-gene /
  reward-override input.**
- The predictor columns baked into `static_obs` come from
  `_compute_race_predictor_outputs` / `_compute_tick_predictor_outputs`,
  which depend only on `(race, predictor_bundle, as_of_date=day)` — all
  cohort-fixed (same manifests across the cohort).
- **Measured:** two independent predictors-ON env builds produce
  **bit-identical `static_obs`** (`np.array_equal` across every race/tick).

→ One shared physical copy across all workers is **sound**.

## (4) In-place-write audit ✓ (HC#2)

Consumer search across the repo: the only site that **shares** the per-day
cache is `BetfairEnv._precompute` (reached via
`_build_env_for_day(feature_cache=…)` on the multiproc worker path). Other
`engineer_day` callers (`training_v2/arb_oracle.py`, `training/arb_oracle.py`,
tools, tests) build fresh and never touch the shared cache.

Two mutations exist in `_precompute`, **both on the dict tree, not on
`static_obs`**:

```python
# env/betfair_env.py:2248,2253 — predictor injection INTO the cached dicts
runners_dict[sid].update(predictor_keys)   # 6 race-level keys
runners_dict[sid].update(keys)             # 12 per-tick direction keys
```

**Measured:** building one env mutates the input dict — 18 keys appear that
were absent before:
`champion_p_win/p_placed/segment_strong`, `ranker_*` (3),
`dir_q{10,50,90}_{1,3,7}m` (9), `dir_fire_{drift,shorten,no_signal}` (3).

→ **Sharing the DICTS read-only is unsound** (a worker's `.update()` would
hit a read-only object / corrupt siblings). The array mechanism **eliminates
this**: predictors are baked **once at cache-build**, and `static_obs` is
read-only thereafter.

`static_obs` itself has **no in-place writers**: it is touched only at
`_precompute` init/append (build time) and read (no mutation) by `_get_obs`
(`env/betfair_env.py:2496`). The `np.nan_to_num(static, copy=False)` in
`_features_to_array` runs on a freshly-concatenated array at **build** time,
never on the shared memmap. → New shared object is **clean for read-only
sharing**.

## (5) Read-latency probe — memmap vs pickle

| | value |
|---|---|
| `np.load(mmap_mode='r')` first-touch | **15.6 ms** |
| `pickle.load` (full dict tree) | **1,365 ms** |
| `.npy` on disk | 93 MB |
| `.pkl` on disk | 241 MB |

memmap first-touch is ~**87× faster** and (the point) never materialises the
whole day into private RAM — touched pages live in the **OS page cache**,
shared across every process mapping the file.

---

## Mechanism decision

**Chosen: `np.memmap` / `.npy` of the per-day `static_obs` arrays**
(predictors baked in at cache-build), workers `np.load(mmap_mode='r')`.

Why this over the alternatives:

- **vs memmapping the `engineer_day` dicts** — impossible (`.npy` needs
  homogeneous arrays; dicts aren't).
- **vs keeping dicts + a cross-process proxy** (`Manager`) — per-tick IPC,
  catastrophically slow. Dead end. Cross-process sharing *requires* arrays,
  so the array conversion is forced regardless of mechanism.
- **vs `multiprocessing.shared_memory`** — same array requirement, but adds
  explicit lifecycle/cleanup that leaks on Windows *spawn* if a worker is
  killed (HC#4/#7). The page-cache (memmap) path is leak-free by
  construction and the read probe shows it is plenty fast.

Sub-decisions:

- **Bake predictors at cache-build** (not a per-worker overlay): one
  fully-populated read-only array/day; no dict, no `.update()`, no
  copy-on-write. The master already holds the bundle (`runner.py:2816`), so
  the predictor inference moves from N workers → 1 master (a bonus).
- **Cache key must encode obs-variant + predictor flags** (lean-23 vs
  full-143, predictors on/off) since `static_obs` columns depend on
  `_active_runner_keys`. A sidecar manifest validated on load prevents
  stale-cache cross-contamination (and HC#5 "no silent feature drops").
- **Gate caches** (`_race_p_win_by_race`, `_tick_drift_fires_by_race`,
  `_race_is_confident_by_race`) are NOT part of `static_obs`. Initial cut:
  the worker recomputes them from the bundle (cheap, correctness-preserving).
  Optional later: add them to the sidecar to skip per-worker inference.

### Projected memory (to be MEASURED in Step 3, not trusted yet)

| | today (dicts, private copies) | with shared `static_obs` memmap |
|---|---|---|
| per-day footprint | ~1.0–1.4 GB | **~0.09 GB** (full obs) |
| master holds (32 days) | ~32–45 GB | **~3 GB on disk**, ~0 RAM after write |
| each worker holds (day) | LRU × ~1.4 GB | **~0** (page-cache-shared views) |
| N=8 plateau | 128 GB → **OOM** | ≈ 8×6 (fixed) + ~3 shared ≈ **~51 GB** |

i.e. the win is materially **better** than the plan's "~47 GB shared / N=8 ≈
95 GB" — because the shared object is the ~3 GB array set, not a ~47 GB dict
set. **This is a projection; Step 3 measures the real plateau through the
training-start spike with the auto-kill watch before any N is raised.**

### Scope implication (why this is a stop-point)

This is **more than "pickle → memmap the same object."** It requires:

1. `prebuild_feature_cache` to run **with the bundle + obs flags** and
   capture `env._static_obs` (instead of predictor-OFF dicts).
2. A **new per-day memmap writer** (`static_obs.npy` + per-race tick-offset
   sidecar + variant/flags manifest), evolving `save_shared_cache_per_day`.
3. A **new additive env consume-path** — an optional
   `static_obs_cache=` kwarg on `BetfairEnv`; when present for the day,
   `_precompute` uses the memmapped arrays and skips `engineer_day` +
   `_features_to_array`. Default `None` ⇒ **byte-identical** to today.
4. Worker read path (`_worker_load_day` → memmap views;
   `_train_agent_worker` → pass `static_obs_cache`), `_WORKER_DAY_CACHE`
   shrinks to cheap views.

All changes are additive + gated; the dict path remains the fallback (HC#3).
The sequential / `--batched` in-process dict cache is **untouched**
(in-process sharing is by-reference, no duplication problem there).

### Gate

Byte breakdown reconciles with the measured ~1–1.4 GB/day dicts ✓.
Zero unaudited in-place writers into the **new shared object** (`static_obs`)
remain ✓ (only build-time writes; runtime read-only).
