# Step 3 RAM watch + auto-kill (shared-memory-day-cache).
#
# PowerShell (NOT python) on purpose: the emergency stop is
# `taskkill /F /IM python.exe /T`, which would kill a python monitor along
# with the cohort. powershell.exe survives it and completes the kill.
#
# Metric choice: "\Memory\Available MBytes" — memory available for new
# allocation INCLUDING reclaimable standby/file-cache. This is the correct
# OOM metric for the memmap design: the shared static_obs pages live in the
# file cache (standby), which is reclaimable, so Win32 "FreePhysicalMemory"
# (which excludes cache) would falsely look starved. Commit charge + per-
# process PRIVATE bytes are the true per-worker footprint (file-backed mmap
# pages are NOT private), so they show the shared-memory win directly.
#
# Auto-kills when Available drops below -ThresholdGB (default 12) and logs a
# LOWRAM_KILL row. Sample every -IntervalSec to a CSV.
param(
  [double]$ThresholdGB = 12.0,
  [int]$IntervalSec = 5,
  [string]$LogFile = "ram_watch.csv",
  [int]$MaxSamples = 100000
)

$thresholdMB = $ThresholdGB * 1024.0
"timestamp,available_mb,commit_gb,py_count,py_ws_gb,py_priv_gb" |
  Out-File -FilePath $LogFile -Encoding utf8

# Total physical for context.
$totalGB = [math]::Round(
  (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)
Write-Host "RAM watch armed: total=$totalGB GB, kill if Available < $ThresholdGB GB, every ${IntervalSec}s -> $LogFile"

for ($i = 0; $i -lt $MaxSamples; $i++) {
  try {
    $availMB = [math]::Round(
      (Get-Counter '\Memory\Available MBytes' -EA Stop).CounterSamples[0].CookedValue, 0)
    $commitGB = [math]::Round(
      (Get-Counter '\Memory\Committed Bytes' -EA Stop).CounterSamples[0].CookedValue / 1GB, 1)
  } catch {
    Start-Sleep -Seconds $IntervalSec; continue
  }
  $py = Get-Process python -ErrorAction SilentlyContinue
  $pyCount = ($py | Measure-Object).Count
  if ($py) {
    $wsGB = [math]::Round((($py | Measure-Object WorkingSet64 -Sum).Sum) / 1GB, 1)
    $privGB = [math]::Round((($py | Measure-Object PrivateMemorySize64 -Sum).Sum) / 1GB, 1)
  } else { $wsGB = 0; $privGB = 0 }

  $ts = Get-Date -Format "HH:mm:ss"
  $line = "{0},{1},{2},{3},{4},{5}" -f $ts, $availMB, $commitGB, $pyCount, $wsGB, $privGB
  $line | Out-File -FilePath $LogFile -Append -Encoding utf8
  Write-Host $line

  if ($availMB -lt $thresholdMB) {
    $msg = "!!! LOW RAM: Available ${availMB}MB < ${thresholdMB}MB -- taskkill /F /IM python.exe /T"
    Write-Host $msg -ForegroundColor Red
    "LOWRAM_KILL,$availMB,$commitGB,$pyCount,$wsGB,$privGB" |
      Out-File -FilePath $LogFile -Append -Encoding utf8
    taskkill /F /IM python.exe /T
    break
  }
  Start-Sleep -Seconds $IntervalSec
}
Write-Host "RAM watch exited."
