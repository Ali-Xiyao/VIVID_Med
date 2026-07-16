param(
    [Parameter(Mandatory = $true)]
    [string[]]$Ids,
    [string]$Gpu = "0",
    [string]$CondaEnv = "vivid",
    [string]$Manifest = "outputs\next_stage_manifests\config_manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$env:CUDA_VISIBLE_DEVICES = $Gpu

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
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
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
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    if ($exitCode -ne 0 -and (Test-Path -LiteralPath $Metrics)) {
        "CONTINUE $(Get-Date -Format o) metrics_final_exists_after_nonzero_exit" | Out-File -FilePath $Log -Append -Encoding utf8
        $exitCode = 0
    }
    return $exitCode
}

$manifestPath = Join-Path $RepoRoot $Manifest
$manifestObj = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$runById = @{}
foreach ($run in $manifestObj.configs) {
    $runById[$run.id] = $run
}

foreach ($id in $Ids) {
    if (-not $runById.ContainsKey($id)) {
        throw "Run id not found in manifest: $id"
    }
    $run = $runById[$id]
    $outputDir = "outputs\qwen3vl_instruction\next_stage\$id"
    $metrics = Join-Path $RepoRoot (Join-Path $outputDir "metrics_final.json")
    $log = Join-Path $logDir "$($id)_gpu$Gpu.log"
    if (Test-Path -LiteralPath $metrics) {
        "SKIP $(Get-Date -Format o) $id metrics_final_exists" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }
    "START $(Get-Date -Format o) $id run_id=$($run.run_id)" | Out-File -FilePath $log -Encoding utf8
    $resume = Get-LatestCheckpoint -OutputDir $outputDir
    if ($resume) {
        "RESUME $(Get-Date -Format o) $id $resume" | Out-File -FilePath $log -Append -Encoding utf8
        $exitCode = Invoke-Training -Log $log -Metrics $metrics -Config $run.config -Resume $resume
    } else {
        $exitCode = Invoke-Training -Log $log -Metrics $metrics -Config $run.config
    }
    "EXITCODE $exitCode $(Get-Date -Format o) $id" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

exit 0
