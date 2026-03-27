"""Unit tests for training/perf_log.py -- Performance logging utility."""

from __future__ import annotations

import logging

import pytest
import torch

from training.perf_log import gpu_memory_summary, perf_log


class TestPerfLog:
    """Test the perf_log context manager."""

    def test_logs_elapsed_time(self, caplog):
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test.perf")
            with perf_log(logger, "Test operation"):
                pass
        assert "Test operation completed in" in caplog.text
        assert "s" in caplog.text

    def test_logs_at_custom_level(self, caplog):
        with caplog.at_level(logging.DEBUG):
            logger = logging.getLogger("test.perf.debug")
            with perf_log(logger, "Debug op", level=logging.DEBUG):
                pass
        assert "Debug op completed in" in caplog.text

    def test_measures_nonzero_time(self, caplog):
        import time

        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test.perf.time")
            with perf_log(logger, "Slow op"):
                time.sleep(0.05)
        # Should show at least 0.04s
        assert "completed in 0.0" in caplog.text

    def test_no_gpu_when_disabled(self, caplog):
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test.perf.nogpu")
            with perf_log(logger, "Simple op", log_gpu=False):
                pass
        assert "MB allocated" not in caplog.text

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No GPU available",
    )
    def test_logs_gpu_memory_when_enabled(self, caplog):
        with caplog.at_level(logging.INFO):
            logger = logging.getLogger("test.perf.gpu")
            with perf_log(logger, "GPU op", log_gpu=True):
                _ = torch.zeros(1000, device="cuda")
        assert "GPU" in caplog.text
        assert "MB allocated" in caplog.text

    def test_exception_propagates(self):
        logger = logging.getLogger("test.perf.exc")
        with pytest.raises(ValueError, match="test error"):
            with perf_log(logger, "Failing op"):
                raise ValueError("test error")


class TestGPUMemorySummary:
    """Test the gpu_memory_summary helper."""

    def test_returns_none_without_cuda(self):
        if torch.cuda.is_available():
            pytest.skip("GPU is available")
        assert gpu_memory_summary() is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No GPU available",
    )
    def test_returns_string_with_cuda(self):
        summary = gpu_memory_summary()
        assert summary is not None
        assert "GPU mem:" in summary
        assert "peak:" in summary
        assert "MB allocated" in summary
