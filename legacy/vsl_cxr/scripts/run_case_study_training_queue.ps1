param(
    [string]$Manifest = "outputs\final_tables\case_study_full_execution_manifest.csv",
    [string]$Gpu = "0",
    [int]$LaneIndex = 0,
    [int]$LaneCount = 1,
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\case_study_full"
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
    $best = Join-Path $checkpointDir "best.pt"
    if (Test-Path -LiteralPath $best) {
        return $best
    }
    return ""
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
    $successPath = Join-Path $row.train_output_dir "metrics_final.json"
    if (Test-Path -LiteralPath $successPath) {
        Write-QueueLog "SKIP $($row.id) metrics_final_exists"
        continue
    }
    $configArg = Quote-Arg $row.train_config
    $cmd = "conda run -n $CondaEnv python scripts\train_qwen3vl_clinical_instruction.py --config $configArg --seed $($row.seed)"
    $resumePath = Get-LatestCheckpoint -OutputDir $row.train_output_dir
    if ($resumePath) {
        $resumeArg = Quote-Arg $resumePath
        Write-QueueLog "RESUME $($row.id) checkpoint=$resumePath"
        $cmd = "$cmd --resume $resumeArg"
    }
    Invoke-Logged -RunId $row.id -Command $cmd -SuccessPath $successPath
}

Write-QueueLog "QUEUE_DONE"
exit 0
