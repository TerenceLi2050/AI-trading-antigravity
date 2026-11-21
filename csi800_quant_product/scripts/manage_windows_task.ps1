<#
.SYNOPSIS
    Manage a Windows Scheduled Task for post-close CSI 800 daily screening.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/manage_windows_task.ps1 -Action install
    powershell -ExecutionPolicy Bypass -File scripts/manage_windows_task.ps1 -Action status
    powershell -ExecutionPolicy Bypass -File scripts/manage_windows_task.ps1 -Action uninstall
#>
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "uninstall", "status")]
    [string]$Action,

    [string]$TaskName = "CSI800_Quant_Daily_Screen",
    [string]$RunTime = "15:40",
    [string]$PythonExe = "python",
    [string]$ModelPath = ""
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ScriptPath = Join-Path $Root "run_daily_screen.py"

switch ($Action) {
    "install" {
        $ArgList = "`"$ScriptPath`" --root `"$Root`" --mode daily"
        if ($ModelPath) {
            $ArgList += " --model `"$ModelPath`""
        }
        $TaskAction = New-ScheduledTaskAction -Execute $PythonExe -Argument $ArgList -WorkingDirectory $Root
        $Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $RunTime
        $Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
        Register-ScheduledTask -TaskName $TaskName -Action $TaskAction -Trigger $Trigger -Settings $Settings -Force | Out-Null
        Write-Host "Installed scheduled task '$TaskName' (Mon-Fri $RunTime) in $Root"
    }
    "uninstall" {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Removed scheduled task '$TaskName'"
    }
    "status" {
        Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue | Format-List TaskName, State, Actions, Triggers
        Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue | Format-List LastRunTime, LastTaskResult, NextRunTime
    }
}
