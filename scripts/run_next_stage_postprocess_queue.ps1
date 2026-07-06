param(
    [Parameter(Mandatory = $true)]
    [string[]]$Ids,
    [string]$Gpu = "0",
    [string]$CondaEnv = "vivid",
    [string]$Manifest = "outputs\next_stage_manifests\config_manifest.json",
    [string]$LpManifest = "outputs\next_stage_manifests\lp_config_manifest.json"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot
$env:CUDA_VISIBLE_DEVICES = $Gpu

$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl_next_stage"
$diagDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_diagnostics"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $diagDir | Out-Null
$isolationId = $env:NEXT_STAGE_ISOLATED_WORKER_ID

function Run-Step {
    param(
        [string]$Log,
        [string]$Name,
        [scriptblock]$Command,
        [string]$SuccessPath = ""
    )
    "START $(Get-Date -Format o) $Name" | Out-File -FilePath $Log -Append -Encoding utf8
    try {
        & $Command *>> $Log
        $exitCode = $LASTEXITCODE
        if ($null -eq $exitCode) {
            $exitCode = 0
        }
    } catch {
        $_ | Out-File -FilePath $Log -Append -Encoding utf8
        $exitCode = 1
    }
    if ($exitCode -ne 0 -and $SuccessPath -and (Test-Path -LiteralPath $SuccessPath)) {
        "ACCEPTED_WITH_ARTIFACT $SuccessPath $(Get-Date -Format o) $Name" | Out-File -FilePath $Log -Append -Encoding utf8
        $exitCode = 0
    }
    "EXITCODE $exitCode $(Get-Date -Format o) $Name" | Out-File -FilePath $Log -Append -Encoding utf8
    if ($exitCode -ne 0) {
        exit $exitCode
    }
}

function Get-IsolatedJsonPath {
    param(
        [string]$Id,
        [string]$Step,
        [string]$CanonicalPath
    )
    if (-not $isolationId) {
        return $CanonicalPath
    }
    $safeStep = $Step -replace '[^A-Za-z0-9_.-]', '_'
    $workerDir = Join-Path $diagDir "isolated\$isolationId\$Id"
    New-Item -ItemType Directory -Force -Path $workerDir | Out-Null
    return (Join-Path $workerDir "$safeStep.json")
}

function Publish-IsolatedJson {
    param(
        [string]$Log,
        [string]$Name,
        [string]$IsolatedPath,
        [string]$CanonicalPath
    )
    if (-not $isolationId) {
        return
    }
    if ($IsolatedPath -eq $CanonicalPath) {
        return
    }
    $manifestPath = Join-Path $diagDir "isolated\$isolationId\$Name.merge_manifest.json"
    Run-Step $Log "$Name publish_isolated" {
        conda run -n $CondaEnv python scripts\merge_next_stage_isolated_outputs.py --output $CanonicalPath --inputs $IsolatedPath --allow-existing --manifest $manifestPath
    } -SuccessPath $manifestPath
}

function Has-HardNegative {
    param([string]$ValPath)
    $path = Join-Path $RepoRoot ($ValPath -replace '^\.\\?/?', '')
    if (-not (Test-Path -LiteralPath $path)) {
        return $false
    }
    return [bool](Select-String -LiteralPath $path -Pattern 'hard_negative_image_path' -Quiet)
}

function Find-AbSwapPath {
    param([string]$ValPath)
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($ValPath)
    $candidate = Join-Path $RepoRoot "outputs\instruction_data\next_stage\$($stem)_ab_swap.jsonl"
    if (Test-Path -LiteralPath $candidate) {
        return "outputs\instruction_data\next_stage\$($stem)_ab_swap.jsonl"
    }
    return ""
}

$manifestObj = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot $Manifest) | ConvertFrom-Json
$lpObj = Get-Content -Raw -LiteralPath (Join-Path $RepoRoot $LpManifest) | ConvertFrom-Json
$runById = @{}
foreach ($run in $manifestObj.configs) { $runById[$run.id] = $run }
$lpById = @{}
foreach ($lp in $lpObj.configs) { $lpById[$lp.id] = $lp }

foreach ($id in $Ids) {
    if (-not $runById.ContainsKey($id)) {
        throw "Run id not found in manifest: $id"
    }
    if (-not $lpById.ContainsKey($id)) {
        throw "LP config not found in manifest: $id"
    }
    $run = $runById[$id]
    $lp = $lpById[$id]
    $trainDir = Join-Path $RepoRoot "outputs\qwen3vl_instruction\next_stage\$id"
    $metrics = Join-Path $trainDir "metrics_final.json"
    $checkpoint = Join-Path $trainDir "checkpoints\best.pt"
    $log = Join-Path $logDir "$($id)_post_gpu$Gpu.log"
    if (-not (Test-Path -LiteralPath $metrics)) {
        "WAIT $(Get-Date -Format o) $id missing metrics_final.json" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        "WAIT $(Get-Date -Format o) $id missing best checkpoint" | Out-File -FilePath $log -Append -Encoding utf8
        continue
    }

    $exportManifest = Join-Path $trainDir "vision_export_manifest.json"
    if (-not (Test-Path -LiteralPath $exportManifest)) {
        $exportDir = Join-Path $trainDir "vision_export"
        Run-Step $log "$id vision_export" {
            conda run -n $CondaEnv python scripts\extract_qwen3vl_vision_backbone.py --checkpoint $checkpoint --output-dir $exportDir
        } -SuccessPath (Join-Path $exportDir "vision_export_manifest.json")
        Copy-Item -LiteralPath (Join-Path $exportDir "vision_export_manifest.json") -Destination $exportManifest -Force
    }

    $lpDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_lp_runs\$($id)_chexpert_1k"
    if (-not (Test-Path -LiteralPath (Join-Path $lpDir "metrics_final.json"))) {
        Run-Step $log "$id lp_chexpert_1k" {
            conda run -n $CondaEnv python scripts\train_qwen3vl_vision_lp.py --config $lp.config --device cuda:0
        } -SuccessPath (Join-Path $lpDir "metrics_final.json")
    }

    $transferDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_transfer\$($id)_nih_1k"
    if (-not (Test-Path -LiteralPath (Join-Path $transferDir "transfer_metrics.json"))) {
        Run-Step $log "$id nih_transfer_1k" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_lp_transfer.py --lp-config $lp.config --val-ums-path data\dataset\processed\nih_external_test_ums.jsonl --data-root "H:/Xiyao_Wang/000_Public Dataset" --output-dir $transferDir --max-samples 1000 --batch-size 2 --device cuda:0 --verify-images
        } -SuccessPath (Join-Path $transferDir "transfer_metrics.json")
    }

    $modes = @("normal", "question_only", "image_shuffle")
    if (Has-HardNegative -ValPath $run.val) { $modes += "hard_shuffle" }
    $visualOut = Join-Path $diagDir "$($id)_visual_dependence.json"
    if (-not (Test-Path -LiteralPath $visualOut)) {
        $visualStepOut = Get-IsolatedJsonPath $id "visual_dependence" $visualOut
        Run-Step $log "$id visual_dependence" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_visual_dependence.py --config $run.config --checkpoint $checkpoint --output $visualStepOut --modes $modes --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $visualStepOut
        Publish-IsolatedJson $log "$id visual_dependence" $visualStepOut $visualOut
    }

    $cfOut = Join-Path $diagDir "$($id)_counterfactual_diagnostics.json"
    if (-not (Test-Path -LiteralPath $cfOut)) {
        $cfStepOut = Get-IsolatedJsonPath $id "counterfactual_diagnostics" $cfOut
        Run-Step $log "$id counterfactual" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $run.config --checkpoint $checkpoint --output $cfStepOut --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $cfStepOut
        Publish-IsolatedJson $log "$id counterfactual_diagnostics" $cfStepOut $cfOut
    }

    $abSwapVal = Find-AbSwapPath -ValPath $run.val
    if ($abSwapVal) {
        $swapConfig = Join-Path $diagDir "$($id)_ab_swap_config.yaml"
        $swapOut = Join-Path $diagDir "$($id)_ab_swap_counterfactual_diagnostics.json"
        if (-not (Test-Path -LiteralPath $swapOut)) {
            Run-Step $log "$id ab_swap_config" {
                conda run -n $CondaEnv python scripts\write_eval_config_override.py --config $run.config --val-instruction-path $abSwapVal --output $swapConfig
            } -SuccessPath $swapConfig
            $swapStepOut = Get-IsolatedJsonPath $id "ab_swap_counterfactual_diagnostics" $swapOut
            Run-Step $log "$id ab_swap_counterfactual" {
                conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $swapConfig --checkpoint $checkpoint --output $swapStepOut --max-samples 1000 --batch-size 1 --device cuda:0
            } -SuccessPath $swapStepOut
            Publish-IsolatedJson $log "$id ab_swap_counterfactual_diagnostics" $swapStepOut $swapOut
        }
    }

    $paraOut = Join-Path $diagDir "$($id)_paraphrase_robustness.json"
    if (-not (Test-Path -LiteralPath $paraOut)) {
        $paraStepOut = Get-IsolatedJsonPath $id "paraphrase_robustness" $paraOut
        Run-Step $log "$id paraphrase" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_paraphrase_robustness.py --config $run.config --checkpoint $checkpoint --output $paraStepOut --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $paraStepOut
        Publish-IsolatedJson $log "$id paraphrase_robustness" $paraStepOut $paraOut
    }

    Run-Step $log "$id package" {
        conda run -n $CondaEnv python scripts\package_next_stage_run_outputs.py --run-id $id
    }
    Run-Step $log "$id summarize" {
        conda run -n $CondaEnv python scripts\summarize_next_stage_results.py
    }
}

exit 0
