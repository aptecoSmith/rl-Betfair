"""Unit tests for api/routers/system.py — system metrics endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import system


# ── Helpers ──────────────────────────────────────────────────────────


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(system.router)
    return TestClient(app)


def _mock_psutil():
    """Return a mock psutil module with realistic values."""
    mock = MagicMock()

    # virtual_memory
    vm = MagicMock()
    vm.used = 16 * 1024 * 1024 * 1024  # 16 GB
    vm.total = 32 * 1024 * 1024 * 1024  # 32 GB
    vm.percent = 50.0
    mock.virtual_memory.return_value = vm

    # cpu_percent
    mock.cpu_percent.return_value = 42.5

    # disk_io_counters
    disk_io = MagicMock()
    disk_io.read_bytes = 500 * 1024 * 1024  # 500 MB
    disk_io.write_bytes = 200 * 1024 * 1024  # 200 MB
    mock.disk_io_counters.return_value = disk_io

    # disk_usage
    disk_usage = MagicMock()
    disk_usage.used = 500 * 1024 * 1024 * 1024  # 500 GB
    disk_usage.total = 1000 * 1024 * 1024 * 1024  # 1 TB
    mock.disk_usage.return_value = disk_usage

    return mock


def _mock_pynvml():
    """Return a mock pynvml module with realistic GPU values."""
    mock = MagicMock()
    mock.NVML_TEMPERATURE_GPU = 0

    handle = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.return_value = handle
    mock.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3090"

    util = MagicMock()
    util.gpu = 78
    mock.nvmlDeviceGetUtilizationRates.return_value = util

    mem = MagicMock()
    mem.used = 8 * 1024 * 1024 * 1024  # 8 GB
    mem.total = 24 * 1024 * 1024 * 1024  # 24 GB
    mock.nvmlDeviceGetMemoryInfo.return_value = mem

    mock.nvmlDeviceGetTemperature.return_value = 65

    return mock


# ── Tests ────────────────────────────────────────────────────────────


class TestSystemMetrics:
    """Tests for GET /system/metrics."""

    @patch("api.routers.system.psutil", new_callable=_mock_psutil, create=True)
    def test_returns_200(self, _mock_ps):
        """Endpoint returns 200 OK."""
        client = _make_app()
        with patch.dict("sys.modules", {"psutil": _mock_psutil()}):
            resp = client.get("/system/metrics")
        assert resp.status_code == 200

    def test_response_schema_has_required_fields(self):
        """Response contains all expected top-level fields."""
        client = _make_app()
        mock_ps = _mock_psutil()
        with patch("api.routers.system._get_gpu_metrics", return_value=None), \
             patch.dict("sys.modules", {"psutil": mock_ps}):
            resp = client.get("/system/metrics")
        data = resp.json()
        for field in ["cpu_pct", "ram_used_mb", "ram_total_mb", "ram_pct",
                       "disk_read_mb_s", "disk_write_mb_s",
                       "disk_used_gb", "disk_total_gb", "gpu"]:
            assert field in data, f"Missing field: {field}"

    def test_cpu_value_is_numeric(self):
        """cpu_pct should be a float."""
        client = _make_app()
        mock_ps = _mock_psutil()
        with patch("api.routers.system._get_gpu_metrics", return_value=None), \
             patch.dict("sys.modules", {"psutil": mock_ps}):
            resp = client.get("/system/metrics")
        data = resp.json()
        assert isinstance(data["cpu_pct"], (int, float))

    def test_ram_values_positive(self):
        """RAM used and total should be positive integers."""
        client = _make_app()
        mock_ps = _mock_psutil()
        with patch("api.routers.system._get_gpu_metrics", return_value=None), \
             patch.dict("sys.modules", {"psutil": mock_ps}):
            resp = client.get("/system/metrics")
        data = resp.json()
        assert data["ram_used_mb"] > 0
        assert data["ram_total_mb"] > 0
        assert data["ram_used_mb"] <= data["ram_total_mb"]

    def test_disk_values_present(self):
        """Disk usage fields should be non-negative."""
        client = _make_app()
        mock_ps = _mock_psutil()
        with patch("api.routers.system._get_gpu_metrics", return_value=None), \
             patch.dict("sys.modules", {"psutil": mock_ps}):
            resp = client.get("/system/metrics")
        data = resp.json()
        assert data["disk_used_gb"] >= 0
        assert data["disk_total_gb"] > 0

    def test_gpu_null_when_unavailable(self):
        """gpu field is null when pynvml not available."""
        client = _make_app()
        mock_ps = _mock_psutil()
        with patch("api.routers.system._get_gpu_metrics", return_value=None), \
             patch.dict("sys.modules", {"psutil": mock_ps}):
            resp = client.get("/system/metrics")
        data = resp.json()
        assert data["gpu"] is None


class TestGpuMetrics:
    """Tests for the _get_gpu_metrics helper."""

    def test_returns_gpu_metrics_when_available(self):
        """Should return GpuMetrics when pynvml works."""
        mock_nvml = _mock_pynvml()
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = system._get_gpu_metrics()
        assert result is not None
        assert result.name == "NVIDIA GeForce RTX 3090"
        assert result.utilisation_pct == 78
        assert result.memory_used_mb == 8192
        assert result.memory_total_mb == 24576
        assert result.temperature_c == 65

    def test_returns_none_on_import_error(self):
        """Should return None when pynvml is not installed."""
        with patch.dict("sys.modules", {"pynvml": None}):
            result = system._get_gpu_metrics()
        assert result is None

    def test_returns_none_on_nvml_error(self):
        """Should return None when nvmlInit fails."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlInit.side_effect = Exception("NVML not found")
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = system._get_gpu_metrics()
        assert result is None

    def test_temperature_none_on_error(self):
        """Temperature should be None if reading fails."""
        mock_nvml = _mock_pynvml()
        mock_nvml.nvmlDeviceGetTemperature.side_effect = Exception("unsupported")
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = system._get_gpu_metrics()
        assert result is not None
        assert result.temperature_c is None

    def test_handles_bytes_gpu_name(self):
        """GPU name returned as bytes should be decoded."""
        mock_nvml = _mock_pynvml()
        mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA RTX 3090"
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = system._get_gpu_metrics()
        assert result is not None
        assert result.name == "NVIDIA RTX 3090"


class TestSystemMetricsIntegration:
    """Integration test — runs against real psutil (no GPU mock)."""

    @pytest.mark.integration
    def test_real_system_metrics(self):
        """Verify /system/metrics returns real data from this machine."""
        client = _make_app()
        resp = client.get("/system/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cpu_pct"] >= 0
        assert data["ram_total_mb"] > 1000  # At least 1 GB
        assert data["ram_used_mb"] > 0
        assert data["disk_total_gb"] > 0
