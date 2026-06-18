"""Guard: BC oracle obs is loaded as a SHARED read-only memmap, not a private
per-worker copy (training_v2/arb_oracle.load_samples).

The shared-memory day cache memmaps static_obs so 16 cohort workers share one
physical copy; BC oracle obs was NOT in that sharing — each worker np.load'd the
full ~100 MB/day obs (+ a per-sample .astype copy) into private RAM, 16x
duplicated. This test pins the fix: the obs block is extracted to a raw
``oracle_obs_d{dim}.npy`` and loaded ``mmap_mode='r'`` (one shared copy), with
byte-identical values.
"""
from __future__ import annotations

import numpy as np
import pytest

from training_v2.arb_oracle import load_samples


def _make_cache(tmp_path, date: str, n: int = 64, dim: int = 2254):
    cd = tmp_path / "oracle_cache_v2" / date
    cd.mkdir(parents=True)
    obs = np.random.RandomState(0).rand(n, dim).astype(np.float32)
    np.savez(
        cd / "oracle_samples.npz",
        tick_index=np.arange(n, dtype=np.int32),
        runner_idx=(np.arange(n) % 14).astype(np.int32),
        obs=obs,
        arb_spread_ticks=np.ones(n, dtype=np.int8),
        expected_locked_pnl=np.ones(n, dtype=np.float32),
        obs_schema_version=np.int32(0),
        action_schema_version=np.int32(0),
        obs_dim_stored=np.int32(dim),
    )
    return obs, cd


def test_oracle_obs_memmap_shared_and_byte_identical(tmp_path):
    date = "2099-01-01"
    obs, cd = _make_cache(tmp_path, date)
    data_dir = tmp_path / "processed"  # load_samples reads data_dir.parent/oracle_cache_v2
    data_dir.mkdir()

    samples = load_samples(date, data_dir, strict=False)
    assert len(samples) == 64

    # The shared raw .npy is built, keyed by obs_dim.
    npy = cd / "oracle_obs_d2254.npy"
    assert npy.exists(), "shared oracle_obs .npy not created"

    # obs is a memmap-backed VIEW (shared), not a fresh private full copy.
    o0 = samples[0].obs
    assert isinstance(o0, np.memmap) or isinstance(getattr(o0, "base", None), np.memmap)

    # Byte-identical to the original .npz obs.
    for i in (0, 1, 31, 63):
        assert np.array_equal(np.asarray(samples[i].obs), obs[i])


def test_oracle_obs_npy_reused_not_rebuilt(tmp_path):
    date = "2099-01-02"
    _obs, cd = _make_cache(tmp_path, date)
    data_dir = tmp_path / "processed"
    data_dir.mkdir()

    load_samples(date, data_dir, strict=False)
    npy = cd / "oracle_obs_d2254.npy"
    mtime = npy.stat().st_mtime_ns

    # A second load (e.g. another worker) must REUSE the shared file, not rebuild.
    load_samples(date, data_dir, strict=False)
    assert npy.stat().st_mtime_ns == mtime


def test_oracle_obs_keyed_by_dim(tmp_path):
    # lean (574) and full (2254) caches produce distinct shared files — no collision.
    for date, dim in (("2099-02-01", 574), ("2099-02-02", 2254)):
        _obs, cd = _make_cache(tmp_path, date, dim=dim)
        data_dir = tmp_path / f"processed_{dim}"
        data_dir.mkdir()
        # data_dir.parent must hold oracle_cache_v2; place caches accordingly.
    # Rebuild under a single parent so both dims coexist.
    base = tmp_path / "shared"
    (base / "processed").mkdir(parents=True)
    for date, dim in (("d574", 574), ("d2254", 2254)):
        cd = base / "oracle_cache_v2" / date
        cd.mkdir(parents=True)
        obs = np.random.RandomState(dim).rand(20, dim).astype(np.float32)
        np.savez(cd / "oracle_samples.npz",
                 tick_index=np.arange(20, dtype=np.int32),
                 runner_idx=np.zeros(20, dtype=np.int32),
                 obs=obs, arb_spread_ticks=np.ones(20, dtype=np.int8),
                 expected_locked_pnl=np.ones(20, dtype=np.float32),
                 obs_schema_version=np.int32(0), action_schema_version=np.int32(0),
                 obs_dim_stored=np.int32(dim))
        load_samples(date, base / "processed", strict=False)
        assert (cd / f"oracle_obs_d{dim}.npy").exists()
