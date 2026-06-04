"""Cross-process GPU concurrency cap (training_v2/cohort/gpu_slot.py).

The load-bearing property is CROSS-PROCESS mutual exclusion: at most
``max_concurrent`` processes hold a slot at once. Tested via timing — with a
cap of 1, two ~0.6s holders SERIALIZE (wall >= ~1.2s); with a cap of 2 they
run in PARALLEL (wall < ~1.0s).
"""
from __future__ import annotations

import multiprocessing as mp
import time

import pytest

from training_v2.cohort.gpu_slot import gpu_slot


def _hold(slot_dir: str, cap: int, hold_s: float, q) -> None:
    """Acquire a slot, record (acquire, release) wall times, hold hold_s."""
    t_acq = time.monotonic()
    with gpu_slot(cap, slot_dir=slot_dir, poll_seconds=0.05, label="test"):
        got = time.monotonic()
        time.sleep(hold_s)
        rel = time.monotonic()
    q.put((t_acq, got, rel))


def _run(tmp_path, cap: int, n_procs: int, hold_s: float):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    sdir = str(tmp_path / f"slots_cap{cap}")
    procs = [
        ctx.Process(target=_hold, args=(sdir, cap, hold_s, q))
        for _ in range(n_procs)
    ]
    t0 = time.monotonic()
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
    wall = time.monotonic() - t0
    intervals = [q.get() for _ in range(n_procs)]
    return wall, intervals


def _max_overlap(intervals) -> int:
    """Max number of (got, rel) windows overlapping at any instant."""
    events = []
    for _acq, got, rel in intervals:
        events.append((got, 1))
        events.append((rel, -1))
    events.sort()
    cur = peak = 0
    for _t, d in events:
        cur += d
        peak = max(peak, cur)
    return peak


class TestGpuSlotNoOp:
    def test_zero_cap_is_uncapped_noop(self, tmp_path):
        # max_concurrent <= 0 yields immediately without touching the fs.
        with gpu_slot(0, slot_dir=str(tmp_path / "none")) as h:
            assert h is None
        assert not (tmp_path / "none").exists()

    def test_single_process_acquire_release(self, tmp_path):
        with gpu_slot(2, slot_dir=str(tmp_path / "s")) as h:
            assert h is not None
        # Re-acquire after release works (lock was freed).
        with gpu_slot(2, slot_dir=str(tmp_path / "s")) as h:
            assert h is not None


@pytest.mark.timeout(60)
class TestGpuSlotCrossProcess:
    def test_cap_of_one_serializes(self, tmp_path):
        wall, intervals = _run(tmp_path, cap=1, n_procs=2, hold_s=0.6)
        # Only one holder at a time -> the two 0.6s holds cannot overlap.
        assert _max_overlap(intervals) == 1
        assert wall >= 1.1, f"cap=1 did not serialize (wall={wall:.2f}s)"

    def test_cap_of_two_parallelizes(self, tmp_path):
        wall, intervals = _run(tmp_path, cap=2, n_procs=2, hold_s=0.6)
        # Two slots -> both holders overlap; wall ~ one hold + spawn overhead.
        assert _max_overlap(intervals) == 2
        assert wall < 1.1, f"cap=2 did not parallelize (wall={wall:.2f}s)"

    def test_three_procs_cap_two_never_exceeds_two(self, tmp_path):
        _wall, intervals = _run(tmp_path, cap=2, n_procs=3, hold_s=0.5)
        # The third must wait for a slot -> at most two overlap, ever.
        assert _max_overlap(intervals) == 2
