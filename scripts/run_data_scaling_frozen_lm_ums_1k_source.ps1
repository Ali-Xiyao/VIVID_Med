$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "outputs/logs"
New-Item -ItemType Directory -Force $logDir | Out-Null

$logPath = Join-Path $logDir "data_scaling_frozen_lm_ums_1k_source.log"

$env:HF_ENDPOINT = "https://huggingface.co"
$env:HF_HUB_DISABLE_XET = "1"
$env:CUDA_VISIBLE_DEVICES = "0"

"START $(Get-Date -Format o)" | Set-Content -Path $logPath
"COMMAND python -u scripts/train_cxr.py --config configs/data_scaling/frozen_lm_ums_1k.yaml" |
    Add-Content -Path $logPath

& python -u scripts/train_cxr.py --config configs/data_scaling/frozen_lm_ums_1k.yaml *>&1 |
    Tee-Object -FilePath $logPath -Append

$code = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
"EXITCODE $code" | Add-Content -Path $logPath
exit $code
