$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Audit = Join-Path $Root "outputs/final_tables/vindr_cxr_formal_integrity_audit.json"
$Manifest = Join-Path $Root "local_runs/bives_cxr/vindr_expert_sc_intake/vindr_test_expert_sc.jsonl"
$ManifestLock = Join-Path $Root "local_runs/bives_cxr/vindr_expert_sc_intake/vindr_test_expert_sc_lock.json"
$Checkpoint = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_sc_b2_sparse_k16_seed17/best.pt"
$CacheLock = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_weak_sc_cache/cache_lock.json"
$Thresholds = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_sc_b2_sparse_k16_seed17/locked_thresholds.json"
$B0Dir = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_sc_b0_pooled_seed17"
$Model = "H:/Xiyao_Wang/001_models/Qwen3.5-2B"
$ExpertOut = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_vindr_expert_sc_seed17"
$InterventionOut = Join-Path $Root "local_runs/bives_cxr/qwen35_2b_vindr_interventions_seed17"

Write-Host "[$(Get-Date -Format o)] WAIT_VINDR_INTEGRITY"
while ($true) {
    if (Test-Path -LiteralPath $Audit) {
        try {
            $payload = Get-Content -LiteralPath $Audit -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($payload.status -eq "pass" -and $payload.sha256_checked -eq 18006 -and $payload.dicom_decode_checked -ge 3000) {
                break
            }
            if ($payload.status -eq "fail") {
                throw "VinDr integrity audit failed; expert gate will not start"
            }
        } catch [System.ArgumentException] {
            # The audit writer may be replacing the file; retry without bypassing the gate.
        }
    }
    Start-Sleep -Seconds 30
}

$env:CUDA_VISIBLE_DEVICES = "0"
if (-not (Test-Path -LiteralPath (Join-Path $ExpertOut "metrics_final.json"))) {
    Write-Host "[$(Get-Date -Format o)] START_EXPERT_SC"
    python scripts/evaluate_bives_vindr_sc.py `
        --manifest $Manifest `
        --manifest-lock $ManifestLock `
        --integrity-audit $Audit `
        --checkpoint $Checkpoint `
        --training-cache-lock $CacheLock `
        --locked-thresholds $Thresholds `
        --b0-dir $B0Dir `
        --model-path $Model `
        --output-dir $ExpertOut `
        --device cuda:0 `
        --dtype bf16 `
        --batch-size 32
    if ($LASTEXITCODE -ne 0) {
        throw "expert S/C evaluation failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $InterventionOut "metrics_final.json"))) {
    Write-Host "[$(Get-Date -Format o)] START_PIXEL_INTERVENTIONS"
    python scripts/evaluate_bives_vindr_interventions.py `
        --manifest $Manifest `
        --integrity-audit $Audit `
        --expert-sc-dir $ExpertOut `
        --checkpoint $Checkpoint `
        --training-cache-lock $CacheLock `
        --model-path $Model `
        --output-dir $InterventionOut `
        --device cuda:0 `
        --dtype bf16 `
        --dilations 0,0.1
    if ($LASTEXITCODE -ne 0) {
        throw "pixel intervention evaluation failed with exit code $LASTEXITCODE"
    }
}

Write-Host "[$(Get-Date -Format o)] EXPERT_GATE_DONE"
