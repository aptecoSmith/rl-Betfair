export interface GpuMetrics {
  name: string;
  utilisation_pct: number;
  memory_used_mb: number;
  memory_total_mb: number;
  temperature_c: number | null;
}

export interface SystemMetrics {
  cpu_pct: number;
  ram_used_mb: number;
  ram_total_mb: number;
  ram_pct: number;
  disk_read_mb_s: number;
  disk_write_mb_s: number;
  disk_used_gb: number;
  disk_total_gb: number;
  gpu: GpuMetrics | null;
}
