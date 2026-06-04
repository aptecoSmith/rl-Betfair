# PBT campaign supervisor (2026-06-04). Keeps the multi-day PBT campaign alive
# WITHOUT ever running two cohorts at once. Registered as a Windows Scheduled
# Task firing every ~20 min (survives crashes AND PC reboots -- the operator
# has lost an overnight run to a reboot before).
#
# Logic each tick:
#   * past the campaign target end-date  -> remove our own task and exit.
#   * a python run OR the wrapper is already alive (incl. the 15s gap between
#     the wrapper's internal per-run relaunches, where the wrapper PowerShell
#     is still sleeping) -> do nothing.
#   * otherwise the campaign is fully down -> (re)launch the wrapper detached,
#     with the repo as CWD so its relative paths resolve.
#
# ASCII-only + pure cmdlets (PS 5.1 reads BOM-less .ps1 as ANSI; no $LASTEXITCODE
# games needed here). Idempotent: safe to run as often as the scheduler fires.
$repo    = 'C:\Users\jsmit\source\repos\rl-betfair'
$wrapper = "$repo\plans\pbt-breeding\_scripts\run_pbt_long.ps1"
$dir     = "$repo\registry\pbt_long"
$logf    = "$dir\supervisor.log"
$target  = Get-Date '2026-06-08 12:00'   # campaign end target (~"a few days")

function SLog($m) {
  try { "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" | Out-File -Append -Encoding utf8 $logf } catch {}
}

if ((Get-Date) -ge $target) {
  SLog "past target end ($($target.ToString('yyyy-MM-dd HH:mm'))) -- removing supervisor task; campaign winds down on the running wrapper's own deadline."
  schtasks /delete /tn PBT_Supervisor /f | Out-Null
  exit 0
}

$pyAlive   = [bool](Get-Process python -ErrorAction SilentlyContinue)
$wrapAlive = [bool](Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
                    Where-Object { $_.CommandLine -like '*run_pbt_long.ps1*' })

if ($pyAlive -or $wrapAlive) { exit 0 }   # campaign alive (training, or wrapper between runs)

SLog "campaign DOWN -- relaunching wrapper (target end $($target.ToString('yyyy-MM-dd HH:mm')))."
Start-Process powershell `
  -ArgumentList '-ExecutionPolicy','Bypass','-NoProfile','-File',$wrapper `
  -WindowStyle Hidden -WorkingDirectory $repo
