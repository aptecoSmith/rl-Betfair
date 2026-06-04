# PBT campaign STOPPER (2026-06-04). Operator: "stop the run after R3."
#
# The current cohort is ONE wrapper run of 25 generations: it climbs R1->R2->R3
# and freezes its champions to the hall-of-fame (leaderboard.txt). This task
# fires every ~5 min; as soon as the wrapper logs a COMPLETED run ("EXIT
# (code"), the cohort's full R3 is built, so we tear the campaign down:
#   1. remove the relaunch supervisor (so it can't start a new cohort),
#   2. kill the wrapper (so its own loop can't start the next run),
#   3. remove ourselves.
# The hall-of-fame + register are left on disk untouched. Until the run
# completes, the supervisor stays in place so a crash mid-cohort still recovers.
#
# ASCII-only + pure cmdlets (PS 5.1 reads BOM-less .ps1 as ANSI).
$repo = 'C:\Users\jsmit\source\repos\rl-betfair'
$wlog = "$repo\registry\pbt_long\wrapper.log"
$logf = "$repo\registry\pbt_long\stopper.log"

function SLog($m) {
  try { "$([DateTime]::Now.ToString('yyyy-MM-dd HH:mm:ss')) $m" | Out-File -Append -Encoding utf8 $logf } catch {}
}

if (-not (Test-Path $wlog)) { exit 0 }

# wrapper.log is UTF-16 (Tee-Object); Select-String decodes it via the BOM.
$completed = Select-String -Path $wlog -Pattern 'EXIT \(code' -ErrorAction SilentlyContinue
if (-not $completed) { exit 0 }   # current cohort still climbing -> nothing to do

SLog "a cohort run completed (R3 champions frozen) -- stopping campaign per 'stop after R3'."

# 1. Remove the relaunch supervisor FIRST (so killing the wrapper can't trigger
#    a supervisor relaunch in the gap).
schtasks /delete /tn PBT_Supervisor /f 2>$null | Out-Null

# 2. Kill the wrapper tree so its loop can't start the next cohort. A run that
#    just started is gen-0 rookies (no champions) -- nothing of value lost.
$w = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
     Where-Object { $_.CommandLine -like '*-File *run_pbt_long.ps1*' }
foreach ($p in @($w)) {
  if ($p) { & taskkill /F /T /PID $p.ProcessId 2>$null | Out-Null }
}

SLog "campaign stopped; hall-of-fame preserved in leaderboard.txt + model_register.csv."

# 3. Remove ourselves.
schtasks /delete /tn PBT_Stopper /f 2>$null | Out-Null
