param(
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$runs = @(
    @{ Id = "p2_d0_fixed_json_schema"; Config = "configs\qwen3vl_instruction\p2_qwen3vl_d0_fixed_json_schema.yaml" },
    @{ Id = "p3_d2_report_grounded_qa"; Config = "configs\qwen3vl_instruction\p3_qwen3vl_d2_report_grounded_qa.yaml" }
)

foreach ($run in $runs) {
    $log = Join-Path $logDir "$($run.Id)_gpu0.log"
    "START $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Encoding utf8
    conda run -n $CondaEnv python scripts\train_qwen3vl_clinical_instruction.py --config $run.Config *>> $log
    $exitCode = $LASTEXITCODE
    "EXITCODE $exitCode $(Get-Date -Format o) $($run.Id)" | Out-File -FilePath $log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

exit 0
