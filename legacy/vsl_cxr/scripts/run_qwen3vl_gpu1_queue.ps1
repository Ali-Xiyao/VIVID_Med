param(
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runs = @(
    @{ Id = "p4_d3_report_grounded_counterfactual"; Config = "configs\qwen3vl_instruction\p4_qwen3vl_d3_report_grounded_counterfactual.yaml" },
    @{ Id = "p5_d4_counterfactual_weighted"; Config = "configs\qwen3vl_instruction\p5_qwen3vl_d4_counterfactual_weighted.yaml" }
)

foreach ($run in $runs) {
    $log = Join-Path $logDir "$($run.Id)_gpu1.log"
    "START $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Encoding utf8
    conda run -n $CondaEnv python scripts\train_qwen3vl_clinical_instruction.py --config $run.Config *>> $log
    $exitCode = $LASTEXITCODE
    "EXITCODE $exitCode $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

exit 0
