$ErrorActionPreference = "Stop"
Set-Location "H:\Xiyao_Wang\021_260129VIVID"
New-Item -ItemType Directory -Force -Path "outputs\logs" | Out-Null

$runs = @(
  @{
    Id = "p2_d0_fixed_json_schema"
    Config = "configs\qwen3vl_instruction\p2_qwen3vl_d0_fixed_json_schema.yaml"
    Checkpoint = "outputs\qwen3vl_instruction_runs\p2_d0_fixed_json_schema\checkpoints\best.pt"
    Output = "outputs\qwen3vl_diagnostics\p2_paraphrase_robustness.json"
    Device = "cuda:0"
  },
  @{
    Id = "p3_d2_report_grounded_qa"
    Config = "configs\qwen3vl_instruction\p3_qwen3vl_d2_report_grounded_qa.yaml"
    Checkpoint = "outputs\qwen3vl_instruction_runs\p3_d2_report_grounded_qa\checkpoints\best.pt"
    Output = "outputs\qwen3vl_diagnostics\p3_paraphrase_robustness.json"
    Device = "cuda:0"
  }
)

foreach ($run in $runs) {
  $log = "outputs\logs\qwen3vl_paraphrase_$($run.Id).log"
  "START $($run.Id) $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
  cmd /c "conda run -n vivid python scripts\evaluate_qwen3vl_paraphrase_robustness.py --config $($run.Config) --checkpoint $($run.Checkpoint) --output $($run.Output) --batch-size 2 --device $($run.Device) >> $log 2>>&1"
  $exitCode = $LASTEXITCODE
  "EXITCODE $exitCode $(Get-Date -Format o)" | Add-Content -Path $log -Encoding utf8
  if ($exitCode -ne 0) {
    exit $exitCode
  }
}

