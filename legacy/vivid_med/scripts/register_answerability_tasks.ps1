$ErrorActionPreference = "Stop"

$Root = "H:\Xiyao_Wang\021_260129VIVID"
$RunnerTasks = @(
    @{ Name = "VIVID_ansmask_resume_gpu1"; Path = "scripts\run_ansmask_resume_gpu1.cmd" },
    @{ Name = "VIVID_lp_ansmask_gpu1"; Path = "scripts\run_lp_ansmask_gpu1.cmd" },
    @{ Name = "VIVID_null_as_negative_gpu1"; Path = "scripts\run_null_as_negative_gpu1.cmd" },
    @{ Name = "VIVID_lp_null_as_negative_gpu1"; Path = "scripts\run_lp_null_as_negative_gpu1.cmd" },
    @{ Name = "VIVID_cf_prefix_dependency_gpu1"; Path = "scripts\run_cf_prefix_dependency_gpu1.cmd" },
    @{ Name = "VIVID_random_lm_gpu1"; Path = "scripts\run_random_lm_gpu1.cmd" },
    @{ Name = "VIVID_lp_random_lm_gpu1"; Path = "scripts\run_lp_random_lm_gpu1.cmd" },
    @{ Name = "VIVID_field_paraphrase_gpu1"; Path = "scripts\run_field_paraphrase_gpu1.cmd" }
)

$PeriodicTasks = @(
    @{ Name = "VIVID_answerability_queue_once"; Script = "scripts\answerability_gpu1_queue_once.ps1"; Minutes = 5 },
    @{ Name = "VIVID_answerability_resource_guard_once"; Script = "scripts\answerability_resource_guard_once.ps1"; Minutes = 5 }
)

function Invoke-Schtasks {
    param([string[]]$SchtasksArgs)
    $output = & schtasks.exe @SchtasksArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "schtasks $($SchtasksArgs -join ' ') failed: $($output | Out-String)"
    }
    $output
}

foreach ($task in $RunnerTasks) {
    $runnerPath = Join-Path $Root $task.Path
    Invoke-Schtasks @(
        "/Create",
        "/F",
        "/TN", $task.Name,
        "/SC", "ONCE",
        "/ST", "23:59",
        "/SD", "2099/12/31",
        "/TR", "`"$runnerPath`""
    ) | Out-Host
}

foreach ($task in $PeriodicTasks) {
    $scriptPath = Join-Path $Root $task.Script
    $taskRun = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
    Invoke-Schtasks @(
        "/Create",
        "/F",
        "/TN", $task.Name,
        "/SC", "MINUTE",
        "/MO", "$($task.Minutes)",
        "/TR", $taskRun
    ) | Out-Host
}

Write-Host "registered VIVID answerability tasks"
