param(
    [Parameter(Mandatory = $true)]
    [string[]]$Ids,
    [string]$Gpu = "0",
    [int]$MaxMemoryMiB = 16000,
    [int]$PollSeconds = 120,
    [string]$CondaEnv = "vivid",
    [string]$Manifest = "outputs\next_stage_manifests\config_manifest.json",
    [string]$LpManifest = "outputs\next_stage_manifests\lp_config_manifest.json",
    [string]$WatcherName = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl_next_stage"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (-not $WatcherName) {
    $WatcherName = "postprocess_gpu$Gpu"
}
$watchLog = Join-Path $logDir "$WatcherName.log"

function Write-WatchLog {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Out-File -FilePath $watchLog -Append -Encoding utf8
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

function Test-RunReady {
    param([string]$Id)
    $trainDir = Join-Path $RepoRoot "outputs\qwen3vl_instruction\next_stage\$Id"
    $metrics = Join-Path $trainDir "metrics_final.json"
    $checkpoint = Join-Path $trainDir "checkpoints\best.pt"
    return ((Test-Path -LiteralPath $metrics) -and (Test-Path -LiteralPath $checkpoint))
}

Write-WatchLog "WATCH_START ids=$($Ids -join ',') gpu=$Gpu max_memory_mib=$MaxMemoryMiB"

foreach ($id in $Ids) {
    $trainDir = Join-Path $RepoRoot "outputs\qwen3vl_instruction\next_stage\$id"
    $packageMarkers = @(
        (Join-Path $trainDir "lp_results.md"),
        (Join-Path $trainDir "visual_dependence_results.md"),
        (Join-Path $trainDir "counterfactual_results.md"),
        (Join-Path $trainDir "ab_swap_results.md"),
        (Join-Path $trainDir "paraphrase_results.md"),
        (Join-Path $trainDir "instruction_audit.md"),
        (Join-Path $trainDir "cost_table.md")
    )
    $packaged = $true
    foreach ($marker in $packageMarkers) {
        if (-not (Test-Path -LiteralPath $marker)) {
            $packaged = $false
            break
        }
    }
    if ($packaged) {
        Write-WatchLog "SKIP $id package_exists"
        continue
    }
    while (-not (Test-RunReady -Id $id)) {
        Write-WatchLog "WAIT $id training_not_ready"
        Start-Sleep -Seconds $PollSeconds
    }
    while ((Get-GpuMemoryMiB -GpuIndex $Gpu) -gt $MaxMemoryMiB) {
        $mem = Get-GpuMemoryMiB -GpuIndex $Gpu
        Write-WatchLog "WAIT $id gpu_memory_high=${mem}MiB"
        Start-Sleep -Seconds $PollSeconds
    }
    $stamp = Get-Date -Format "yyyyMMddTHHmmss"
    $workerId = "$WatcherName-$id-$stamp" -replace '[^A-Za-z0-9_.-]', '_'
    $previousIsolationId = $env:NEXT_STAGE_ISOLATED_WORKER_ID
    $env:NEXT_STAGE_ISOLATED_WORKER_ID = $workerId
    Write-WatchLog "RUN $id postprocess isolated_worker=$workerId"
    try {
        & "$PSScriptRoot\run_next_stage_postprocess_queue.ps1" -Ids $id -Gpu $Gpu -CondaEnv $CondaEnv -Manifest $Manifest -LpManifest $LpManifest
        $code = $LASTEXITCODE
    } finally {
        if ($previousIsolationId) {
            $env:NEXT_STAGE_ISOLATED_WORKER_ID = $previousIsolationId
        } else {
            Remove-Item Env:\NEXT_STAGE_ISOLATED_WORKER_ID -ErrorAction SilentlyContinue
        }
    }
    Write-WatchLog "EXITCODE $code $id postprocess"
    if ($code -ne 0) {
        exit $code
    }
}

Write-WatchLog "WATCH_DONE"
exit 0
