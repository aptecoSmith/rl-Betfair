"""Multiprocess day-cache memory-budget guard.

The bug this guards against wasted a whole overnight PBT run: predictors-OFF
caches each ~1.45GB engineered day in EVERY worker, so 16 workers x 12 days
plateaued at ~128GB -> MemoryError at generation 2 (before the gauntlet ever
reached R3 -> zero champions), with NO guard to catch the config first. These
tests pin the estimator + the refuse-guard so that exact config is rejected
BEFORE training, and confirm the predictors-ON static_obs (shared memmap)
path is correctly judged safe.
"""

from __future__ import annotations

import pytest

from training_v2.cohort.multiproc_worker import (
    assert_day_cache_fits,
    estimate_day_cache_peak_gb,
)

_GB = 1_000_000_000


class TestEstimate:
    def test_dict_path_multiplies_by_workers(self):
        # 12 days x 0.30GB pickle, 16 workers, per-worker dict path.
        est = estimate_day_cache_peak_gb(
            day_cache_bytes=[int(0.30 * _GB)] * 12, n_workers=16, shared=False)
        # n_workers * sum * 5 (unpickle) + n_workers * 2.5 overhead
        #  = 16 * 3.6 * 5 + 16 * 2.5 = 288 + 40
        assert est == pytest.approx(328.0, abs=1.0)

    def test_shared_path_holds_one_copy(self):
        # SAME days but shared static_obs memmap -> one copy + overhead.
        est = estimate_day_cache_peak_gb(
            day_cache_bytes=[int(0.30 * _GB)] * 12, n_workers=16, shared=True)
        # sum + n_workers * overhead = 3.6 + 40
        assert est == pytest.approx(43.6, abs=1.0)

    def test_shared_is_far_cheaper_than_dict(self):
        days = [int(0.10 * _GB)] * 30
        dict_est = estimate_day_cache_peak_gb(
            day_cache_bytes=days, n_workers=16, shared=False)
        shared_est = estimate_day_cache_peak_gb(
            day_cache_bytes=days, n_workers=16, shared=True)
        assert shared_est < dict_est / 5


class TestRefuseGuard:
    def test_refuses_the_overnight_oom_config(self):
        """16 workers x 12 days x ~0.30GB pickle on a 128GB box -> ~328GB
        projected -> MUST refuse (this is the exact bug)."""
        with pytest.raises(MemoryError, match="OOM"):
            assert_day_cache_fits(
                day_cache_bytes=[int(0.30 * _GB)] * 12,
                n_workers=16, shared=False, total_ram_gb=128.0)

    def test_predictors_on_static_obs_config_is_allowed(self):
        """The SAME 16 workers x 12 days but predictors-ON static_obs
        (~0.097GB/day, shared) -> ~41GB -> allowed (the fix)."""
        est = assert_day_cache_fits(
            day_cache_bytes=[int(0.097 * _GB)] * 12,
            n_workers=16, shared=True, total_ram_gb=128.0)
        assert est < 0.9 * 128

    def test_small_dict_config_is_allowed(self):
        # 4 workers x 6 days x 0.30GB -> 4*1.8*5 + 10 = 46GB -> fits.
        est = assert_day_cache_fits(
            day_cache_bytes=[int(0.30 * _GB)] * 6,
            n_workers=4, shared=False, total_ram_gb=128.0)
        assert est < 0.9 * 128

    def test_unknown_ram_does_not_hard_refuse(self):
        # total_ram_gb=0 (unknown) -> estimate only, never raises.
        est = assert_day_cache_fits(
            day_cache_bytes=[int(0.30 * _GB)] * 12,
            n_workers=16, shared=False, total_ram_gb=0.0)
        assert est > 0
