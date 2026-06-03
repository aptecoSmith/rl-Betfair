"""Step 0 measurement: characterise engineer_day output vs static_obs.

Answers the Step-0 questions on FACTS, not assumptions:
  1. Byte breakdown of the cached object (engineer_day dicts) — RSS delta,
     tracemalloc, and a recursive sizeof cross-check.
  2. Byte size of the downstream static_obs float32 arrays the env actually
     reads at runtime (the natural memmap target).
  3. Determinism / shareability: two independent predictors-ON env builds
     produce bit-identical static_obs (cohort-fixed ⇒ shareable).
  4. The in-place-write the env performs on the cached dicts during
     _precompute (predictor `.update()`), confirming HC#2 applies.
  5. A read-latency probe: np.load(mmap_mode='r') first-touch vs pickle.load.

Run:
  python -m plans.shared-memory-day-cache._measure.measure_day_structure DAY
  (DAY defaults to 2026-04-15)
"""
from __future__ import annotations

import gc
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_DIR = REPO / "data" / "processed"
PRED = REPO.parent / "betfair-predictors" / "production"
MANIFESTS = (
    str(PRED / "race-outcome" / "manifest.json"),
    str(PRED / "race-outcome-ranker" / "manifest.json"),
    str(PRED / "direction-predictor" / "manifest.json"),
)

_PROC = psutil.Process()


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6


def deep_sizeof(obj, _seen=None) -> int:
    """Recursive sys.getsizeof over dict/list/tuple/set + their contents.

    Dedupes by id so shared/interned objects (e.g. small ints, interned
    key strings) are counted once — mirrors real RAM, where interned
    strings are one physical object.
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    _seen.add(oid)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_sizeof(k, _seen) + deep_sizeof(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for it in obj:
            size += deep_sizeof(it, _seen)
    return size


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else "2026-04-15"
    print(f"=== Step 0 structure measurement — day {day} ===")
    print(f"repo={REPO}")
    print(f"baseline RSS={rss_mb():.0f} MB\n")

    from training_v2.cohort.multiproc_worker import prebuild_feature_cache

    # ── 1. engineer_day dicts (the cached object today) ──────────────────
    gc.collect()
    rss0 = rss_mb()
    tracemalloc.start()
    cache: dict = {}
    t0 = time.perf_counter()
    prebuild_feature_cache([day], data_dir=DATA_DIR, into=cache)
    build_s = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    gc.collect()
    rss1 = rss_mb()

    day_features = cache[day]
    n_races = len(day_features)
    n_ticks = sum(len(r) for r in day_features)
    # Per-tick dict shape on a representative tick.
    sample_race = max(day_features, key=len)
    sample_tick = sample_race[len(sample_race) // 2]
    n_runners_in_tick = len(sample_tick.get("runners", {}))
    runner_feat_count = (
        len(next(iter(sample_tick["runners"].values())))
        if sample_tick.get("runners") else 0
    )

    deep = deep_sizeof(day_features)

    print("── (1) engineer_day output = nested list[race][tick] of dicts ──")
    print(f"  build time           : {build_s:.1f}s")
    print(f"  n_races              : {n_races}")
    print(f"  n_ticks (total)      : {n_ticks}")
    print(f"  runners / sample tick: {n_runners_in_tick}")
    print(f"  feats  / sample runr : {runner_feat_count}")
    print(f"  tick dict keys       : {sorted(sample_tick.keys())}")
    print(f"  RSS delta (build)    : {rss1 - rss0:8.1f} MB")
    print(f"  tracemalloc peak     : {peak / 1e6:8.1f} MB")
    print(f"  tracemalloc current  : {cur / 1e6:8.1f} MB")
    print(f"  deep_sizeof          : {deep / 1e6:8.1f} MB")
    print(f"  → per-day dict cost  : ~{(rss1 - rss0) / 1e3:.2f} GB (RSS)\n")

    # ── 2. static_obs arrays (predictors-ON, full obs) ───────────────────
    from data.episode_builder import load_day
    from env.betfair_env import BetfairEnv
    from predictors import PredictorBundle
    from training_v2.cohort.worker import scalping_train_config

    bundle = PredictorBundle.from_manifests(
        champion_manifest=MANIFESTS[0],
        ranker_manifest=MANIFESTS[1],
        direction_manifest=MANIFESTS[2],
    )
    cfg = scalping_train_config()
    cfg.setdefault("observations", {})
    cfg["observations"]["use_race_outcome_predictor"] = True
    cfg["observations"]["use_direction_predictor"] = True

    def build_static_obs(feature_cache):
        loaded = load_day(day, data_dir=DATA_DIR)
        env = BetfairEnv(
            loaded, cfg,
            feature_cache=feature_cache,
            predictor_bundle=bundle,
            use_race_outcome_predictor=True,
            use_direction_predictor=True,
            predictor_lean_obs=False,      # FULL obs = the problem case
            emit_debug_features=False,
        )
        return env

    # Fresh dict copy so the env's in-place predictor injection doesn't
    # pollute the measurement of the pristine cache.
    import copy as _copy
    cache_for_env = {day: _copy.deepcopy(day_features)}
    env_a = build_static_obs(cache_for_env)
    so = env_a._static_obs
    obs_bytes = sum(arr.nbytes for race in so for arr in race)
    obs_dim = so[0][0].shape[0]
    print("── (2) env._static_obs = list[race][tick] of float32 arrays ──")
    print(f"  obs_dim / tick       : {obs_dim}  (37 mkt + 11 vel + 14*143)")
    print(f"  dtype                : {so[0][0].dtype}")
    print(f"  total static_obs     : {obs_bytes / 1e6:8.1f} MB")
    print(f"  → per-day array cost : ~{obs_bytes / 1e6:.0f} MB\n")

    ratio = (rss1 - rss0) * 1e6 / max(obs_bytes, 1)
    print(f"  *** dict / array size ratio: {ratio:.1f}x ***\n")

    # ── 3. determinism / shareability ────────────────────────────────────
    cache_b = {day: _copy.deepcopy(day_features)}
    env_b = build_static_obs(cache_b)
    so_b = env_b._static_obs
    identical = all(
        np.array_equal(a, b)
        for ra, rb in zip(so, so_b) for a, b in zip(ra, rb)
    )
    print("── (3) determinism (two predictors-ON builds) ──")
    print(f"  static_obs bit-identical across builds: {identical}")
    print("  (cohort-fixed ⇒ one shared copy is sound, HC#6)\n")

    # ── 4. in-place write confirmation (HC#2) ────────────────────────────
    # The env injects predictor keys into the cached dicts during
    # _precompute. Build an env on a fresh dict and check the dict was
    # mutated (predictor keys present after, absent before).
    probe = {day: _copy.deepcopy(day_features)}
    a_race = probe[day][0]
    a_tick = a_race[len(a_race) // 2]
    a_runner_sid = next(iter(a_tick["runners"]))
    keys_before = set(a_tick["runners"][a_runner_sid].keys())
    build_static_obs(probe)  # mutates probe in place
    keys_after = set(probe[day][0][len(a_race) // 2]["runners"][a_runner_sid].keys())
    injected = sorted(keys_after - keys_before)
    print("── (4) in-place write into cached dicts during _precompute ──")
    print(f"  predictor keys injected by env: {injected}")
    print(f"  (these mutate the cached dict ⇒ read-only sharing of the "
          f"DICTS is unsound; arrays bake them once)\n")

    # ── 5. memmap vs pickle read-latency probe ───────────────────────────
    import pickle
    tmp = Path(REPO) / "plans" / "shared-memory-day-cache" / "_measure" / "_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    # Pack static_obs into one 2D array (total_ticks, obs_dim) for memmap.
    flat = np.concatenate([arr[None, :] for race in so for arr in race], axis=0)
    npy_path = tmp / "static_obs.npy"
    np.save(npy_path, flat)
    pkl_path = tmp / "day_dicts.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(day_features, fh, protocol=pickle.HIGHEST_PROTOCOL)

    npy_sz = npy_path.stat().st_size / 1e6
    pkl_sz = pkl_path.stat().st_size / 1e6

    t0 = time.perf_counter()
    mm = np.load(npy_path, mmap_mode="r")
    _ = np.asarray(mm[:64]).sum()  # first-touch a few pages
    mmap_first = time.perf_counter() - t0

    t0 = time.perf_counter()
    with open(pkl_path, "rb") as fh:
        _ = pickle.load(fh)
    pkl_load = time.perf_counter() - t0

    print("── (5) read-latency probe ──")
    print(f"  static_obs.npy on disk : {npy_sz:8.1f} MB")
    print(f"  day_dicts.pkl on disk  : {pkl_sz:8.1f} MB")
    print(f"  np.load(mmap) first-touch: {mmap_first * 1e3:7.1f} ms")
    print(f"  pickle.load (full)       : {pkl_load * 1e3:7.1f} ms")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
