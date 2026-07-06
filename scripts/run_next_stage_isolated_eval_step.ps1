param(
    [Parameter(Mandatory = $true)]
    [string]$Id,
    [Parameter(Mandatory = $true)]
    [ValidateSet("visual", "counterfactual", "paraphrase")]
    [string]$Kind,
    [Parameter(Mandatory = $true)]
    [string]$Config,
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Gpu = "0",
    [string]$WorkerId = "",
    [string]$CondaEnv = "vivid",
    [int]$MaxSamples = 1000,
    [int]$BatchSize = 1,
    [string[]]$Modes = @("normal", "question_only", "image_shuffle", "hard_shuffle")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$env:CUDA_VISIBLE_DEVICES = $Gpu

if (-not $WorkerId) {
    $WorkerId = "gpu$Gpu-$Id-$Kind"
}

$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl_next_stage"
$diagDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_diagnostics"
$isoDir = Join-Path $diagDir "isolated\$WorkerId\$Id"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $isoDir | Out-Null

$isolatedOutput = Join-Path $isoDir "$Kind.json"
$manifest = Join-Path $isoDir "$Kind.merge_manifest.json"
$log = Join-Path $logDir "$($Id)_$($Kind)_isolated_$($WorkerId).log"

function Run-NativeStep {
    param(
        [string]$Name,
        [scriptblock]$Command,
        [string]$SuccessPath
    )
    "START $(Get-Date -Format o) $Name" | Out-File -FilePath $log -Append -Encoding utf8
    try {
        & $Command *>> $log
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    } catch {
        $_ | Out-File -FilePath $log -Append -Encoding utf8
        $exitCode = 1
    }
    if ($exitCode -ne 0 -and (Test-Path -LiteralPath $SuccessPath)) {
        "ACCEPTED_WITH_ARTIFACT $SuccessPath $(Get-Date -Format o) $Name" | Out-File -FilePath $log -Append -Encoding utf8
        $exitCode = 0
    }
    "EXITCODE $exitCode $(Get-Date -Format o) $Name" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

if (Test-Path -LiteralPath $Output) {
    "SKIP $(Get-Date -Format o) canonical_exists $Output" | Out-File -FilePath $log -Append -Encoding utf8
    exit 0
}

if ($Kind -eq "visual") {
    Run-NativeStep "$Id $Kind isolated_eval" {
        conda run -n $CondaEnv python scripts\evaluate_qwen3vl_visual_dependence.py --config $Config --checkpoint $Checkpoint --output $isolatedOutput --modes $Modes --max-samples $MaxSamples --batch-size $BatchSize --device cuda:0
    } $isolatedOutput
} elseif ($Kind -eq "counterfactual") {
    Run-NativeStep "$Id $Kind isolated_eval" {
        conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $Config --checkpoint $Checkpoint --output $isolatedOutput --max-samples $MaxSamples --batch-size $BatchSize --device cuda:0
    } $isolatedOutput
} elseif ($Kind -eq "paraphrase") {
    Run-NativeStep "$Id $Kind isolated_eval" {
        conda run -n $CondaEnv python scripts\evaluate_qwen3vl_paraphrase_robustness.py --config $Config --checkpoint $Checkpoint --output $isolatedOutput --max-samples $MaxSamples --batch-size $BatchSize --device cuda:0
    } $isolatedOutput
}

Run-NativeStep "$Id $Kind publish_isolated" {
    conda run -n $CondaEnv python scripts\merge_next_stage_isolated_outputs.py --output $Output --inputs $isolatedOutput --allow-existing --manifest $manifest
} $manifest

exit 0
