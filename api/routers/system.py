"""System metrics endpoint — CPU, RAM, GPU, disk stats."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.schemas import SystemMetrics, GpuMetrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["system"])


def _get_gpu_metrics() -> GpuMetrics | None:
    """Read GPU utilisation and VRAM via pynvml.  Returns None if unavailable."""
    try:
        import pynvml  # noqa: F811 — wraps nvidia-ml-py

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        temp = None
        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        pynvml.nvmlShutdown()

        return GpuMetrics(
            name=name,
            utilisation_pct=util.gpu,
            memory_used_mb=round(mem.used / 1024 / 1024),
            memory_total_mb=round(mem.total / 1024 / 1024),
            temperature_c=temp,
        )
    except Exception as exc:
        logger.debug("GPU metrics unavailable: %s", exc)
        return None


@router.get("/system/metrics", response_model=SystemMetrics)
def get_system_metrics():
    """Return current CPU, RAM, GPU, and disk stats."""
    import psutil

    vm = psutil.virtual_memory()
    disk = psutil.disk_io_counters()
    disk_usage = psutil.disk_usage("/")

    return SystemMetrics(
        cpu_pct=psutil.cpu_percent(interval=0.1),
        ram_used_mb=round(vm.used / 1024 / 1024),
        ram_total_mb=round(vm.total / 1024 / 1024),
        ram_pct=vm.percent,
        disk_read_mb_s=round(disk.read_bytes / 1024 / 1024, 1) if disk else 0,
        disk_write_mb_s=round(disk.write_bytes / 1024 / 1024, 1) if disk else 0,
        disk_used_gb=round(disk_usage.used / 1024 / 1024 / 1024, 1),
        disk_total_gb=round(disk_usage.total / 1024 / 1024 / 1024, 1),
        gpu=_get_gpu_metrics(),
    )


@router.get("/system/health")
def health_check():
    """Simple health check."""
    return {"status": "ok"}
