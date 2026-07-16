$ErrorActionPreference = "Stop"
Set-Location "H:\Xiyao_Wang\021_260129VIVID"

function Run-Transfer {
    param(
        [string]$Name,
        [string]$Config,
        [string]$OutputDir
    )
    $log = "outputs\logs\qwen3vl_transfer_$Name.log"
    $metrics = Join-Path $OutputDir "transfer_metrics.json"
    if (Test-Path -LiteralPath $metrics) {
        "SKIP $Name existing $metrics $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8 -Append
        return
    }
    "START $Name $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
    conda run -n vivid python scripts\evaluate_qwen3vl_lp_transfer.py `
        --lp-config $Config `
        --val-ums-path data\dataset\processed\nih_external_test_ums.jsonl `
        --data-root "H:/Xiyao_Wang/000_Public Dataset" `
        --output-dir $OutputDir `
        --max-samples 1000 `
        --batch-size 2 `
        --device cuda:0 `
        --verify-images *>> $log
    $code = $LASTEXITCODE
    "EXITCODE $code $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8 -Append
    if ($code -ne 0) { exit $code }
}

Run-Transfer "base_nih_1k" "configs\qwen3vl_instruction\lp_qwen3vl_base_chexpert_1k.yaml" "outputs\qwen3vl_transfer\base_nih_1k"
Run-Transfer "p2_nih_1k" "configs\qwen3vl_instruction\lp_qwen3vl_p2_chexpert_1k.yaml" "outputs\qwen3vl_transfer\p2_nih_1k"
Run-Transfer "p3_nih_1k" "configs\qwen3vl_instruction\lp_qwen3vl_p3_chexpert_1k.yaml" "outputs\qwen3vl_transfer\p3_nih_1k"
