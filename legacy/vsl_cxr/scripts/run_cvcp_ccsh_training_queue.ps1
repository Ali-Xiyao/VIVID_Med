param(
    [string]$Manifest = "outputs\final_tables\cvcp_ccsh_training_manifest.csv",
    [string]$Gpu = "0",
    [int]$LaneIndex = 0,
    [int]$LaneCount = 1,
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\cvcp_ccsh"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$queueLog = Join-Path $logDir "training_gpu${Gpu}_lane${LaneIndex}_of${LaneCount}.log"

function Write-QueueLog {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Out-File -FilePath $queueLog -Append -Encoding utf8
}

function Quote-Arg {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Get-LatestCheckpoint {
    param([string]$OutputDir)
    $checkpointDir = Join-Path $OutputDir "checkpoints"
    if (-not (Test-Path -LiteralPath $checkpointDir)) {
        return ""
    }
    $stepCheckpoint = Get-ChildItem -LiteralPath $checkpointDir -Filter "step_*.pt" -ErrorAction SilentlyContinue |
        ForEach-Object {
            if ($_.BaseName -match "^step_(\d+)$") {
                [pscustomobject]@{ Path = $_.FullName; Step = [int]$Matches[1] }
            }
        } |
        Sort-Object Step -Descending |
        Select-Object -First 1
    if ($stepCheckpoint) {
        return $stepCheckpoint.Path
    }
    $final = Join-Path $checkpointDir "final.pt"
    if (Test-Path -LiteralPath $final) {
        return $final
    }
    $best = Join-Path $checkpointDir "best.pt"
    if (Test-Path -LiteralPath $best) {
        return $best
    }
    return ""
}

function Move-PartialOutputDir {
    param(
        [string]$RunId,
        [string]$OutputDir
    )
    if (-not (Test-Path -LiteralPath $OutputDir)) {
        return ""
    }
    $successPath = Join-Path $OutputDir "metrics_final.json"
    if (Test-Path -LiteralPath $successPath) {
        return ""
    }
    if (Get-LatestCheckpoint -OutputDir $OutputDir) {
        return ""
    }

    $source = (Resolve-Path -LiteralPath $OutputDir).ProviderPath
    $categoryDir = Split-Path -Parent $source
    $outputRoot = Split-Path -Parent $categoryDir
    $archiveRoot = Join-Path $outputRoot "interrupted_runs"
    New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null

    $stamp = Get-Date -Format "yyyyMMddTHHmmss"
    $destination = Join-Path $archiveRoot "${RunId}_${stamp}"
    $suffix = 1
    while (Test-Path -LiteralPath $destination) {
        $destination = Join-Path $archiveRoot "${RunId}_${stamp}_${suffix}"
        $suffix++
    }

    Move-Item -LiteralPath $source -Destination $destination
    Write-QueueLog "ARCHIVE_PARTIAL $RunId source=$source destination=$destination reason=no_metrics_final_no_checkpoint"
    return $destination
}

function Invoke-Logged {
    param(
        [string]$RunId,
        [string]$Command,
        [string]$SuccessPath
    )
    $runLog = Join-Path $logDir "$RunId.train.gpu$Gpu.log"
    Write-QueueLog "RUN $RunId command=$Command"
    "START $(Get-Date -Format o) $RunId" | Out-File -FilePath $runLog -Append -Encoding utf8
    $env:CUDA_VISIBLE_DEVICES = $Gpu
    try {
        $oldErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        powershell -NoProfile -ExecutionPolicy Bypass -Command $Command *>> $runLog
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch {
        $_ | Out-File -FilePath $runLog -Append -Encoding utf8
        $code = 1
    } finally {
        $ErrorActionPreference = $oldErrorPreference
    }
    "EXITCODE $code $(Get-Date -Format o) $RunId" | Out-File -FilePath $runLog -Append -Encoding utf8
    if ($code -ne 0 -and (Test-Path -LiteralPath $SuccessPath)) {
        Write-QueueLog "ACCEPT_WITH_ARTIFACT $RunId success=$SuccessPath"
        $code = 0
    }
    if ($code -ne 0) {
        Write-QueueLog "FAIL $RunId exit=$code"
        exit $code
    }
    Write-QueueLog "DONE $RunId"
}

$rows = Import-Csv -Path $Manifest
Write-QueueLog "QUEUE_START rows=$($rows.Count) gpu=$Gpu lane=$LaneIndex/$LaneCount"

for ($i = 0; $i -lt $rows.Count; $i++) {
    if (($i % $LaneCount) -ne $LaneIndex) { continue }
    $row = $rows[$i]
    if (Test-Path -LiteralPath $row.success_path) {
        Write-QueueLog "SKIP $($row.id) metrics_final_exists"
        continue
    }
    $configArg = Quote-Arg $row.train_config
    $cmd = "conda run -n $CondaEnv python scripts\train_qwen3vl_cvcp.py --config $configArg"
    $resumePath = Get-LatestCheckpoint -OutputDir $row.train_output_dir
    if (-not $resumePath) {
        Move-PartialOutputDir -RunId $row.id -OutputDir $row.train_output_dir | Out-Null
    }
    if ($resumePath) {
        $resumeArg = Quote-Arg $resumePath
        Write-QueueLog "RESUME $($row.id) checkpoint=$resumePath"
        $cmd = "$cmd --resume $resumeArg"
    }
    Invoke-Logged -RunId $row.id -Command $cmd -SuccessPath $row.success_path
}

Write-QueueLog "QUEUE_DONE"
exit 0
