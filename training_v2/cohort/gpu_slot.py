"""Cross-process GPU concurrency cap for the PBT GPU policy lane.

Big-context transformers route their policy (forward + batched PPO update) to
CUDA via ``--gpu-policy-lane``. In a 16-worker multiprocess cohort, several
such agents can land concurrently — the GPU-lane validation measured a peak of
**18.7 GB / 24 GB for FOUR** ctx256 transformers, and the widened size genes
(d512 / depth6) make a single agent ~2x heavier, so >2 concurrent risks a hard
CUDA OOM that takes down the worker.

This module caps the number of agents *simultaneously holding the GPU* to a
small ``N`` (default 2) using ``N`` OS advisory file locks. Each GPU-lane
agent acquires one lock before it builds its CUDA policy and releases it after
eval; a (N+1)-th agent blocks until a slot frees. The locks are **OS-advisory
byte-range locks**, which the kernel AUTO-RELEASES when the holding process
exits — so a crashed or killed worker never permanently wedges a slot (the
failure mode a naive named-semaphore counter would have).

The slot directory is box-wide by default (one physical GPU ⇒ one box-wide
cap is the correct semantics, even across independent campaigns sharing the
card). CPU-lane agents (LSTM, small transformers) never call this — they don't
touch the GPU, so they run fully unthrottled.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform.startswith("win")
if _IS_WINDOWS:
    import msvcrt
else:
    import fcntl


def default_slot_dir() -> Path:
    """Box-wide slot directory under the system temp dir."""
    return Path(tempfile.gettempdir()) / "rl_betfair_gpu_slots"


def _try_lock(fd: int) -> bool:
    """Non-blocking exclusive lock on byte 0. False if already held."""
    try:
        if _IS_WINDOWS:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd: int) -> None:
    try:
        if _IS_WINDOWS:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass


@contextmanager
def gpu_slot(
    max_concurrent: int = 2,
    *,
    slot_dir: "str | os.PathLike | None" = None,
    poll_seconds: float = 2.0,
    label: str = "",
):
    """Block until one of ``max_concurrent`` GPU slots is free, hold it for the
    duration of the ``with`` body, then release it.

    ``max_concurrent <= 0`` is a no-op (uncapped) — yields immediately without
    touching the filesystem, so callers can pass a disabled cap unconditionally.
    """
    if max_concurrent is None or int(max_concurrent) <= 0:
        yield None
        return

    n = int(max_concurrent)
    sdir = Path(slot_dir) if slot_dir is not None else default_slot_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    paths = [sdir / f"gpu_slot_{i}" for i in range(n)]
    for p in paths:
        # Ensure byte 0 exists for the range lock (idempotent across workers).
        if not p.exists():
            try:
                with open(p, "wb") as fh:
                    fh.write(b"\0")
            except FileExistsError:
                pass

    held_fd: int | None = None
    held_path: Path | None = None
    waited = False
    t0 = time.monotonic()
    while held_fd is None:
        for p in paths:
            fd = os.open(str(p), os.O_RDWR)
            if _try_lock(fd):
                held_fd, held_path = fd, p
                break
            os.close(fd)
        if held_fd is None:
            if not waited:
                logger.info(
                    "GPU lane full (%d slots) — %s waiting for a slot",
                    n, label or "agent",
                )
                waited = True
            time.sleep(poll_seconds)
    if waited:
        logger.info(
            "GPU slot acquired by %s after %.1fs wait (%s)",
            label or "agent", time.monotonic() - t0, held_path.name,
        )
    try:
        yield held_path
    finally:
        _unlock(held_fd)
        os.close(held_fd)
