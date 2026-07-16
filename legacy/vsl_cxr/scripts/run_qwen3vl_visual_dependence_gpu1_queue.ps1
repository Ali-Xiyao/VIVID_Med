$ErrorActionPreference = "Stop"
Set-Location "H:\Xiyao_Wang\021_260129VIVID"
New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

$runs = @(
  @{
    Id = "p4_d3_report_grounded_counterfactual"
    Config = "configs\qwen3vl_instruction\p4_qwen3vl_d3_report_grounded_counterfactual.yaml"
    Checkpoint = "outputs\qwen3vl_instruction_runs\p4_d3_report_grounded_counterfactual\checkpoints\best.pt"
    Output = "outputs\qwen3vl_diagnostics\p4_visual_dependence.json"
    Device = "cuda:1"
  },
  @{
    Id = "p5_d4_counterfactual_weighted"
    Config = "configs\qwen3vl_instruction\p5_qwen3vl_d4_counterfactual_weighted.yaml"
    Checkpoint = "outputs\qwen3vl_instruction_runs\p5_d4_counterfactual_weighted\checkpoints\best.pt"
    Output = "outputs\qwen3vl_diagnostics\p5_visual_dependence.json"
    Device = "cuda:1"
  }
)

foreach ($run in $runs) {
  $log = "outputs\logs\qwen3vl_visual_$($run.Id).log"
  "START $($run.Id) $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
  cmd /c "conda run -n vivid python scripts\evaluate_qwen3vl_visual_dependence.py --config $($run.Config) --checkpoint $($run.Checkpoint) --output $($run.Output) --batch-size 1 --device $($run.Device) >> $log 2>>&1"
  $exitCode = $LASTEXITCODE
  "EXITCODE $exitCode $(Get-Date -Format o)" | Add-Content -Path $log -Encoding utf8
  if ($exitCode -ne 0) {
    exit $exitCode
  }
}

