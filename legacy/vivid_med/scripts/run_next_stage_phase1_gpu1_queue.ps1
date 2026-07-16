param(
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:CUDA_VISIBLE_DEVICES = "1"
$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl_next_stage"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Get-LatestCheckpoint {
    param([string]$OutputDir)
    $checkpointDir = Join-Path $RepoRoot (Join-Path $OutputDir "checkpoints")
    if (-not (Test-Path -LiteralPath $checkpointDir)) {
        return $null
    }
    $stepCheckpoint = Get-ChildItem -LiteralPath $checkpointDir -Filter "step_*.pt" -ErrorAction SilentlyContinue |
        Sort-Object { [int](($_.BaseName -replace '^step_', '0')) } -Descending |
        Select-Object -First 1
    if ($stepCheckpoint) {
        return $stepCheckpoint.FullName
    }
    $best = Join-Path $checkpointDir "best.pt"
    if (Test-Path -LiteralPath $best) {
        return $best
    }
    return $null
}

function Invoke-Training {
    param(
        [string]$Log,
        [string]$Metrics,
        [string]$Config,
        [string]$Resume = ""
    )
    try {
        if ($Resume) {
            conda run -n $CondaEnv python scripts\train_qwen3vl_clinical_instruction.py --config $Config --resume $Resume *>> $Log
        } else {
            conda run -n $CondaEnv python scripts\train_qwen3vl_clinical_instruction.py --config $Config *>> $Log
        }
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    } catch {
        $_ | Out-File -FilePath $Log -Append -Encoding utf8
        $exitCode = 1
    }
    if ($exitCode -ne 0 -and (Test-Path -LiteralPath $Metrics)) {
        "CONTINUE $(Get-Date -Format o) metrics_final_exists_after_nonzero_exit" | Out-File -FilePath $Log -Append -Encoding utf8
        $exitCode = 0
    }
    return $exitCode
}

$runs = @(
    @{
        Id = "P2-field-query"
        Config = "configs\qwen3vl_instruction\next_stage\p2_field_query.yaml"
        OutputDir = "outputs\qwen3vl_instruction\next_stage\p2_field_query"
    },
    @{
        Id = "SHUF-heavy-QA8"
        Config = "configs\qwen3vl_instruction\next_stage\shuf_heavy_qa8.yaml"
        OutputDir = "outputs\qwen3vl_instruction\next_stage\shuf_heavy_qa8"
    },
    @{
        Id = "SHUF-TW-visual"
        Config = "configs\qwen3vl_instruction\next_stage\shuf_tw_visual.yaml"
        OutputDir = "outputs\qwen3vl_instruction\next_stage\shuf_tw_visual"
    }
)

foreach ($run in $runs) {
    $metrics = Join-Path $RepoRoot (Join-Path $run.OutputDir "metrics_final.json")
    $log = Join-Path $logDir "$($run.Id)_gpu1.log"
    if (Test-Path $metrics) {
        "SKIP $(Get-Date -Format o) $($run.Id) metrics_final_exists" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }

    "START $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Encoding utf8
    $resume = Get-LatestCheckpoint -OutputDir $run.OutputDir
    if ($resume) {
        "RESUME $(Get-Date -Format o) $($run.Id) $resume" | Out-File -FilePath $log -Append -Encoding utf8
        $exitCode = Invoke-Training -Log $log -Metrics $metrics -Config $run.Config -Resume $resume
    } else {
        $exitCode = Invoke-Training -Log $log -Metrics $metrics -Config $run.Config
    }
    "EXITCODE $exitCode $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

exit 0
