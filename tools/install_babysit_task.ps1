# Register a Windows scheduled task that runs babysit_loop.py every hour.
#
# Run once interactively as the user (no admin needed):
#   pwsh -ExecutionPolicy Bypass -File tools/install_babysit_task.ps1
#
# To remove later:
#   Unregister-ScheduledTask -TaskName "rl-betfair-babysit" -Confirm:$false

$ErrorActionPreference = "Stop"

$repo = "C:\Users\jsmit\source\repos\rl-betfair"
$python = "C:\Python314\python.exe"
$script = Join-Path $repo "tools\babysit_loop.py"
$taskName = "rl-betfair-babysit"

if (-not (Test-Path $python)) {
    Write-Error "Python not found at $python — edit this script with the correct path."
}
if (-not (Test-Path $script)) {
    Write-Error "babysit_loop.py not found at $script"
}

# Action: invoke python on the babysit script, with cwd = repo root.
$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "`"$script`"" `
    -WorkingDirectory $repo

# Trigger: every hour, indefinitely. Start at the next clock hour.
$now = Get-Date
$start = (Get-Date -Hour ($now.Hour + 1) -Minute 0 -Second 0)
if ($start -lt $now) { $start = $start.AddHours(1) }
$trigger = New-ScheduledTaskTrigger `
    -Once -At $start `
    -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 7)

# Settings: run whether user is logged in or not (logged-in is fine);
# don't wake the machine; allow start on demand; minimum stop on idle.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Principal: run as the current user, with the user's environment.
$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive

# Unregister first if exists.
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing task $taskName"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Hourly babysit of rl-betfair experiment loop." | Out-Null

Write-Host "Registered $taskName, starts at $start, repeats every hour for 7 days."
Write-Host "Run manually to test: Start-ScheduledTask -TaskName $taskName"
Write-Host "Inspect logs: Get-Content $repo\plans\recipe-expansion-and-robustness\babysit_log.txt -Tail 30"
