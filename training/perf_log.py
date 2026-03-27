"""
training/perf_log.py -- Lightweight performance logging utility.

Provides a context manager that logs wall-clock time and optional GPU memory
usage for code blocks on the critical path.

Usage::

    from training.perf_log import perf_log

    with perf_log(logger, "Feature engineering", log_gpu=True):
        engineer_day(day)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

import torch


@contextmanager
def perf_log(
    logger: logging.Logger,
    label: str,
    *,
    log_gpu: bool = False,
    level: int = logging.INFO,
):
    """Log elapsed time (and optional GPU memory delta) for a code block.

    Parameters
    ----------
    logger :
        Logger instance to write to.
    label :
        Human-readable label for the operation being timed.
    log_gpu :
        If True and CUDA is available, log GPU memory allocated before/after.
    level :
        Logging level (default INFO).
    """
    gpu_available = log_gpu and torch.cuda.is_available()
    gpu_before = torch.cuda.memory_allocated() if gpu_available else 0

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start

    if gpu_available:
        gpu_after = torch.cuda.memory_allocated()
        gpu_delta = gpu_after - gpu_before
        logger.log(
            level,
            "%s completed in %.2fs | GPU: %.0f MB allocated (%+.0f MB)",
            label,
            elapsed,
            gpu_after / 1024**2,
            gpu_delta / 1024**2,
        )
    else:
        logger.log(level, "%s completed in %.2fs", label, elapsed)


def gpu_memory_summary() -> str | None:
    """Return a one-line GPU memory summary, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
    return (
        f"GPU mem: {allocated:.0f} MB allocated, {reserved:.0f} MB reserved | "
        f"peak: {peak_allocated:.0f} MB allocated, {peak_reserved:.0f} MB reserved"
    )
