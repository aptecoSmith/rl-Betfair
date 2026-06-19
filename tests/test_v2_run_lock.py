"""Guard: a single cohort.runner output_dir is single-writer.

Root cause of the 2026-06-18 silent-worker-collapse + ``FileNotFoundError`` on
``meta_*.pkl.tmp``: two masters writing one ``output_dir`` raced the atomic
static_obs cache rename AND doubled the worker pool. ``_acquire_run_lock``
makes a duplicate refuse cleanly while reclaiming a STALE lock (so resume after
a crash/reboot still works).
"""
from __future__ import annotations

import os

import pytest

from training_v2.cohort.runner import _acquire_run_lock


def test_acquire_creates_lock_with_pid(tmp_path):
    lock = _acquire_run_lock(tmp_path)
    assert lock.exists()
    assert str(os.getpid()) == lock.read_text(encoding="utf-8").split("|", 1)[0]


def test_second_acquire_refuses_when_holder_alive(tmp_path):
    # First acquire writes OUR pid; we (the test process) ARE a live python.
    # Force the holder-alive check to see a live holder by writing this pid
    # and having the cmdline contain cohort.runner is not guaranteed in the
    # test runner, so monkeypatch liveness via a live-pid sentinel file.
    _acquire_run_lock(tmp_path)
    # Overwrite the lock with a definitely-alive pid (this process) AND make
    # the liveness probe treat it as a cohort.runner by patching psutil.
    lock = tmp_path / ".run.lock"
    lock.write_text(f"{os.getpid()}|x", encoding="utf-8")

    import psutil

    real_proc = psutil.Process

    class _FakeProc:
        def __init__(self, pid):
            self._pid = pid

        def cmdline(self):
            return ["python", "-m", "training_v2.cohort.runner"]

    psutil.Process = _FakeProc  # type: ignore[assignment]
    try:
        with pytest.raises(SystemExit, match="already running"):
            _acquire_run_lock(tmp_path)
    finally:
        psutil.Process = real_proc  # type: ignore[assignment]


def test_stale_lock_is_reclaimed(tmp_path):
    # A dead-holder lock (pid that doesn't exist) must be reclaimed, not fatal.
    lock = tmp_path / ".run.lock"
    lock.write_text("999999999|2020-01-01T00:00:00", encoding="utf-8")
    got = _acquire_run_lock(tmp_path)  # must NOT raise
    assert got == lock
    assert str(os.getpid()) == lock.read_text(encoding="utf-8").split("|", 1)[0]
