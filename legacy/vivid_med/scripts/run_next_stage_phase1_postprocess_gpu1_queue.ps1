param(
    [string]$CondaEnv = "vivid"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:CUDA_VISIBLE_DEVICES = "1"
$logDir = Join-Path $RepoRoot "outputs\logs\qwen3vl_next_stage"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

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

function Process-Run {
    param(
        [string]$Id,
        [string]$Config,
        [string]$LpConfig,
        [string]$AbSwapVal = "",
        [switch]$HardShuffle
    )
    $trainDir = Join-Path $RepoRoot "outputs\qwen3vl_instruction\next_stage\$Id"
    $metrics = Join-Path $trainDir "metrics_final.json"
    $checkpoint = Join-Path $trainDir "checkpoints\best.pt"
    $log = Join-Path $logDir "$($Id)_post_gpu1.log"
    if (-not (Test-Path -LiteralPath $metrics)) {
        "WAIT $(Get-Date -Format o) $Id missing metrics_final.json" | Out-File -FilePath $log -Append -Encoding utf8
        return
    }
    if (-not (Test-Path -LiteralPath $checkpoint)) {
        "WAIT $(Get-Date -Format o) $Id missing best checkpoint" | Out-File -FilePath $log -Append -Encoding utf8
        return
    }

    $exportManifest = Join-Path $trainDir "vision_export_manifest.json"
    if (-not (Test-Path -LiteralPath $exportManifest)) {
        $exportDir = Join-Path $trainDir "vision_export"
        Run-Step $log "$Id vision_export" {
            conda run -n $CondaEnv python scripts\extract_qwen3vl_vision_backbone.py --checkpoint $checkpoint --output-dir $exportDir
        } -SuccessPath (Join-Path $exportDir "vision_export_manifest.json")
        Copy-Item -LiteralPath (Join-Path $exportDir "vision_export_manifest.json") -Destination $exportManifest -Force
    }

    $lpDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_lp_runs\$($Id)_chexpert_1k"
    if (-not (Test-Path -LiteralPath (Join-Path $lpDir "metrics_final.json"))) {
        Run-Step $log "$Id lp_chexpert_1k" {
            conda run -n $CondaEnv python scripts\train_qwen3vl_vision_lp.py --config $LpConfig --device cuda:0
        } -SuccessPath (Join-Path $lpDir "metrics_final.json")
    }

    $transferDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_transfer\$($Id)_nih_1k"
    if (-not (Test-Path -LiteralPath (Join-Path $transferDir "transfer_metrics.json"))) {
        Run-Step $log "$Id nih_transfer_1k" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_lp_transfer.py --lp-config $LpConfig --val-ums-path data\dataset\processed\nih_external_test_ums.jsonl --data-root "H:/Xiyao_Wang/000_Public Dataset" --output-dir $transferDir --max-samples 1000 --batch-size 2 --device cuda:0 --verify-images
        } -SuccessPath (Join-Path $transferDir "transfer_metrics.json")
    }

    $modes = @("normal", "question_only", "image_shuffle")
    if ($HardShuffle) {
        $modes += "hard_shuffle"
    }
    $diagDir = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_diagnostics"
    New-Item -ItemType Directory -Force -Path $diagDir | Out-Null
    $visualOut = Join-Path $diagDir "$($Id)_visual_dependence.json"
    if (-not (Test-Path -LiteralPath $visualOut)) {
        Run-Step $log "$Id visual_dependence" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_visual_dependence.py --config $Config --checkpoint $checkpoint --output $visualOut --modes $modes --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $visualOut
    }

    $cfOut = Join-Path $diagDir "$($Id)_counterfactual_diagnostics.json"
    if (-not (Test-Path -LiteralPath $cfOut)) {
        Run-Step $log "$Id counterfactual" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $Config --checkpoint $checkpoint --output $cfOut --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $cfOut
    }

    if ($AbSwapVal) {
        $swapConfig = Join-Path $RepoRoot "outputs\qwen3vl_next_stage_diagnostics\$($Id)_ab_swap_config.yaml"
        $swapOut = Join-Path $diagDir "$($Id)_ab_swap_counterfactual_diagnostics.json"
        if (-not (Test-Path -LiteralPath $swapOut)) {
            Run-Step $log "$Id ab_swap_config" {
                conda run -n $CondaEnv python scripts\write_eval_config_override.py --config $Config --val-instruction-path $AbSwapVal --output $swapConfig
            } -SuccessPath $swapConfig
            Run-Step $log "$Id ab_swap_counterfactual" {
                conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $swapConfig --checkpoint $checkpoint --output $swapOut --max-samples 1000 --batch-size 1 --device cuda:0
            } -SuccessPath $swapOut
        }
    }

    $paraOut = Join-Path $diagDir "$($Id)_paraphrase_robustness.json"
    if (-not (Test-Path -LiteralPath $paraOut)) {
        Run-Step $log "$Id paraphrase" {
            conda run -n $CondaEnv python scripts\evaluate_qwen3vl_paraphrase_robustness.py --config $Config --checkpoint $checkpoint --output $paraOut --max-samples 1000 --batch-size 1 --device cuda:0
        } -SuccessPath $paraOut
    }

    Run-Step $log "$Id package" {
        conda run -n $CondaEnv python scripts\package_next_stage_run_outputs.py --run-id $Id
    }
}

Process-Run "p2_field_query" "configs\qwen3vl_instruction\next_stage\p2_field_query.yaml" "configs\qwen3vl_instruction\next_stage_lp\lp_p2_field_query_chexpert_1k.yaml"
Process-Run "shuf_heavy_qa8" "configs\qwen3vl_instruction\next_stage\shuf_heavy_qa8.yaml" "configs\qwen3vl_instruction\next_stage_lp\lp_shuf_heavy_qa8_chexpert_1k.yaml" -AbSwapVal "outputs\instruction_data\next_stage\shuf_heavy_qa8_val_ab_swap.jsonl" -HardShuffle
Process-Run "shuf_tw_visual" "configs\qwen3vl_instruction\next_stage\shuf_tw_visual.yaml" "configs\qwen3vl_instruction\next_stage_lp\lp_shuf_tw_visual_chexpert_1k.yaml" -AbSwapVal "outputs\instruction_data\next_stage\d7_hard_shuffle_val200_ab_swap.jsonl" -HardShuffle

exit 0
