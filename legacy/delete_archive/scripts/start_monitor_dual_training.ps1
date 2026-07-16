$scriptPath = Join-Path $PSScriptRoot "monitor_dual_training.ps1"

Start-Process powershell `
    -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$scriptPath`"" `
    -WindowStyle Normal

Write-Host "Started monitor window: $scriptPath"
