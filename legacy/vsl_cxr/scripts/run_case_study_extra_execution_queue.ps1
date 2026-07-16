param(
    [string]$Manifest = "outputs\final_tables\case_study_extra_execution_manifest.csv",
    [string]$Gpu = "0",
    [int]$MaxMemoryMiB = 2000,
    [int]$PollSeconds = 180
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\case_study_full"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$queueLog = Join-Path $logDir "extra_execution_gpu${Gpu}.log"

function Write-QueueLog {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Out-File -FilePath $queueLog -Append -Encoding utf8
}

function Get-GpuMemoryMiB {
    param([string]$GpuIndex)
    $rows = nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    foreach ($row in $rows) {
        $parts = $row -split ","
        if ($parts.Count -ge 2 -and $parts[0].Trim() -eq $GpuIndex) {
            return [int]($parts[1].Trim())
        }
    }
    return 999999
}

function Wait-ForGpuWindow {
    param([string]$StepId)
    while ((Get-GpuMemoryMiB -GpuIndex $Gpu) -gt $MaxMemoryMiB) {
        $mem = Get-GpuMemoryMiB -GpuIndex $Gpu
        Write-QueueLog "WAIT $StepId gpu_memory_high=${mem}MiB"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Invoke-Logged {
    param(
        [string]$StepId,
        [string]$Command,
        [string]$SuccessPath,
        [bool]$ForceRun = $false
    )
    if (-not $ForceRun -and $SuccessPath -and (Test-Path -LiteralPath $SuccessPath)) {
        Write-QueueLog "SKIP $StepId exists=$SuccessPath"
        return
    }
    Wait-ForGpuWindow -StepId $StepId
    $env:CUDA_VISIBLE_DEVICES = $Gpu
    $stepLog = Join-Path $logDir "$StepId.extra.gpu$Gpu.log"
    Write-QueueLog "RUN $StepId command=$Command"
    "START $(Get-Date -Format o) $StepId" | Out-File -FilePath $stepLog -Append -Encoding utf8
    try {
        $oldErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        powershell -NoProfile -ExecutionPolicy Bypass -Command $Command *>> $stepLog
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch {
        $_ | Out-File -FilePath $stepLog -Append -Encoding utf8
        $code = 1
    } finally {
        $ErrorActionPreference = $oldErrorPreference
    }
    "EXITCODE $code $(Get-Date -Format o) $StepId" | Out-File -FilePath $stepLog -Append -Encoding utf8
    if ($code -ne 0 -and $SuccessPath -and (Test-Path -LiteralPath $SuccessPath)) {
        Write-QueueLog "ACCEPT_WITH_ARTIFACT $StepId success=$SuccessPath"
        $code = 0
    }
    if ($code -ne 0) {
        Write-QueueLog "FAIL $StepId exit=$code"
        exit $code
    }
    Write-QueueLog "DONE $StepId"
}

$rows = Import-Csv -Path $Manifest
Write-QueueLog "QUEUE_START rows=$($rows.Count) gpu=$Gpu max_memory=$MaxMemoryMiB"
foreach ($row in $rows) {
    $forceRun = $false
    if ($row.PSObject.Properties.Name -contains "force_run") {
        $forceRun = [string]$row.force_run -in @("1", "true", "True", "yes", "YES")
    }
    Invoke-Logged -StepId $row.step_id -Command $row.command -SuccessPath $row.success_path -ForceRun $forceRun
}
Write-QueueLog "QUEUE_DONE"
exit 0
