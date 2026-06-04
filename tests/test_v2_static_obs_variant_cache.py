"""Per-worker static_obs cache must not collide across obs variants.

pbt-breeding 2026-06-04: lean obs is a fresh-blood gene, so one warm worker
serves agents on DIFFERENT obs representations (lean vs full), whose static_obs
for the same day live in different cache dirs. The per-worker
``_WORKER_STATIC_OBS_CACHE`` was keyed by DAY alone, so the first-loaded
variant for a day was returned for BOTH -> a StaticObsCacheMismatch crash at
gen 0. The cache is now keyed by ``(day, npy_path)``. This is the regression
guard (another 'missing test' from the OOM/lean postmortem).
"""

from __future__ import annotations

from training_v2.cohort import multiproc_worker as mw
from training_v2.cohort import static_obs_cache


def _fake_load(npy_path, sidecar_path, mmap=True):
    # Return a marker that identifies the variant by its path.
    return ("DayStaticObs", npy_path)


def test_same_day_different_variant_does_not_collide(monkeypatch):
    mw._WORKER_STATIC_OBS_CACHE.clear()
    monkeypatch.setattr(static_obs_cache.DayStaticObs, "load", _fake_load)

    day = "2026-05-17"
    full = mw._worker_load_static_obs(
        day, "out/mp_static_obs_cache_full/static_obs_2026-05-17.npy",
        "out/mp_static_obs_cache_full/meta_2026-05-17.pkl")
    lean = mw._worker_load_static_obs(
        day, "out/mp_static_obs_cache_lean/static_obs_2026-05-17.npy",
        "out/mp_static_obs_cache_lean/meta_2026-05-17.pkl")

    # The two variants for the SAME day must be distinct (no day-only hit).
    assert full == ("DayStaticObs",
                    "out/mp_static_obs_cache_full/static_obs_2026-05-17.npy")
    assert lean == ("DayStaticObs",
                    "out/mp_static_obs_cache_lean/static_obs_2026-05-17.npy")
    assert full != lean

    # A second call for each variant is a cache HIT returning the SAME variant.
    full2 = mw._worker_load_static_obs(
        day, "out/mp_static_obs_cache_full/static_obs_2026-05-17.npy",
        "out/mp_static_obs_cache_full/meta_2026-05-17.pkl")
    assert full2 == full
    mw._WORKER_STATIC_OBS_CACHE.clear()
