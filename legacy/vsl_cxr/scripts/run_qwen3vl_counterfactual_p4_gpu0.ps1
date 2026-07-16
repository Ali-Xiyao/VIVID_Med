$ErrorActionPreference = "Stop"
Set-Location "H:\Xiyao_Wang\021_260129VIVID"
New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

$log = "outputs\logs\qwen3vl_counterfactual_p4.log"
"START p4_d3_report_grounded_counterfactual $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
cmd /c "conda run -n vivid python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config configs\qwen3vl_instruction\p4_qwen3vl_d3_report_grounded_counterfactual.yaml --checkpoint outputs\qwen3vl_instruction_runs\p4_d3_report_grounded_counterfactual\checkpoints\best.pt --output outputs\qwen3vl_diagnostics\p4_counterfactual_diagnostics.json --batch-size 2 --device cuda:0 >> $log 2>>&1"
$exitCode = $LASTEXITCODE
"EXITCODE $exitCode $(Get-Date -Format o)" | Add-Content -Path $log -Encoding utf8
exit $exitCode

