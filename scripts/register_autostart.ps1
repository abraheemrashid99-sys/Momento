# Registers a task that starts the plan 2 minutes after any logon. The plan itself is single-instance,
# so extra logons (e.g. remote-desktop connections) cannot create duplicates.
Set-Location (Split-Path $PSScriptRoot -Parent)
Unregister-ScheduledTask -TaskName "MomentoPlan" -Confirm:$false -ErrorAction SilentlyContinue
$act = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -WindowStyle Minimized -File `"$PWD\scripts\launch.ps1`"" -WorkingDirectory "$PWD"
$trg = New-ScheduledTaskTrigger -AtLogOn
$trg.Delay = "PT2M"
Register-ScheduledTask -TaskName "MomentoPlan" -Action $act -Trigger $trg -RunLevel Highest -Force | Out-Null
"Registered: MomentoPlan (starts 2 min after logon; single-instance)"
