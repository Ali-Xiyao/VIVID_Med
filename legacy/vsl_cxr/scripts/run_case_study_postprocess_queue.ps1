param(
    [string]$Manifest = "outputs\final_tables\case_study_full_execution_manifest.csv",
    [string]$Gpu = "0",
    [int]$LaneIndex = 0,
    [int]$LaneCount = 1,
    [string]$CondaEnv = "vivid",
    [int]$MaxMemoryMiB = 18000,
    [int]$PollSeconds = 120,
    [string]$NihManifest = "data\dataset\processed\nih_external_test_ums.jsonl",
    [string]$DataRoot = "H:/Xiyao_Wang/000_Public Dataset"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\case_study_full"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$queueLog = Join-Path $logDir "postprocess_gpu${Gpu}_lane${LaneIndex}_of${LaneCount}.log"
# CUDA_VISIBLE_DEVICES below maps the selected physical GPU to logical cuda:0.
$device = "cuda:0"

function Write-QueueLog {
    param([string]$Message)
    "$(Get-Date -Format o) $Message" | Out-File -FilePath $queueLog -Append -Encoding utf8
}

function Get-GpuMemoryMiB {
    param([string]$GpuIndex)
    $rows = nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    foreach ($row in $rows) {
        $parts = $row -split ","
        if ($parts.Count -ge 2 -and $parts[0].Trim() -eq $GpuIndex) {
            return [int]($parts[1].Trim())
        }
    }
    return 999999
}

function Wait-ForGpuWindow {
    param([string]$RunId)
    while ((Get-GpuMemoryMiB -GpuIndex $Gpu) -gt $MaxMemoryMiB) {
        $mem = Get-GpuMemoryMiB -GpuIndex $Gpu
        Write-QueueLog "WAIT $RunId gpu_memory_high=${mem}MiB"
        Start-Sleep -Seconds $PollSeconds
    }
}

function Quote-Arg {
    param([string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Invoke-Logged {
    param(
        [string]$RunId,
        [string]$StepName,
        [string]$Command,
        [string]$SuccessPath,
        [bool]$Required = $true
    )
    if ($SuccessPath -and (Test-Path -LiteralPath $SuccessPath)) {
        Write-QueueLog "SKIP $RunId $StepName exists=$SuccessPath"
        return
    }
    Wait-ForGpuWindow -RunId "$RunId/$StepName"
    $runLog = Join-Path $logDir "$RunId.$StepName.gpu$Gpu.log"
    Write-QueueLog "RUN $RunId $StepName command=$Command"
    "START $(Get-Date -Format o) $RunId $StepName" | Out-File -FilePath $runLog -Append -Encoding utf8
    $env:CUDA_VISIBLE_DEVICES = $Gpu
    try {
        $oldErrorPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        powershell -NoProfile -ExecutionPolicy Bypass -Command $Command *>> $runLog
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch {
        $_ | Out-File -FilePath $runLog -Append -Encoding utf8
        $code = 1
    } finally {
        $ErrorActionPreference = $oldErrorPreference
    }
    "EXITCODE $code $(Get-Date -Format o) $RunId $StepName" | Out-File -FilePath $runLog -Append -Encoding utf8
    if ($code -ne 0 -and $SuccessPath -and (Test-Path -LiteralPath $SuccessPath)) {
        Write-QueueLog "ACCEPT_WITH_ARTIFACT $RunId $StepName success=$SuccessPath"
        $code = 0
    }
    if ($code -ne 0 -and $Required) {
        Write-QueueLog "FAIL $RunId $StepName exit=$code"
        exit $code
    }
    if ($code -ne 0) {
        Write-QueueLog "WARN $RunId $StepName exit=$code"
    } else {
        Write-QueueLog "DONE $RunId $StepName"
    }
}

$rows = Import-Csv -Path $Manifest
Write-QueueLog "QUEUE_START rows=$($rows.Count) gpu=$Gpu lane=$LaneIndex/$LaneCount max_memory=$MaxMemoryMiB"

for ($i = 0; $i -lt $rows.Count; $i++) {
    if (($i % $LaneCount) -ne $LaneIndex) { continue }
    $row = $rows[$i]
    $trainMetrics = Join-Path $row.train_output_dir "metrics_final.json"
    $checkpoint = Join-Path $row.train_output_dir "checkpoints\best.pt"
    while (-not ((Test-Path -LiteralPath $trainMetrics) -and (Test-Path -LiteralPath $checkpoint))) {
        Write-QueueLog "WAIT $($row.id) training_not_ready"
        Start-Sleep -Seconds $PollSeconds
    }

    $lpMetrics = Join-Path $row.lp_output_dir "metrics_final.json"
    $lpConfig = Quote-Arg $row.lp_config
    $trainConfig = Quote-Arg $row.train_config
    $checkpointArg = Quote-Arg $checkpoint
    $nihManifestArg = Quote-Arg $NihManifest
    $dataRootArg = Quote-Arg $DataRoot
    $lpOutputDir = Quote-Arg $row.lp_output_dir
    $nihOutputDir = Quote-Arg $row.nih_output_dir
    $visualOutput = Quote-Arg $row.visual_output
    $counterfactualOutput = Quote-Arg $row.counterfactual_output
    $paraphraseOutput = Quote-Arg $row.paraphrase_output

    Invoke-Logged -RunId $row.id -StepName "lp" -SuccessPath $lpMetrics -Command "conda run -n $CondaEnv python scripts\train_qwen3vl_vision_lp.py --config $lpConfig --device $device"

    $nihMetrics = Join-Path $row.nih_output_dir "transfer_metrics.json"
    Invoke-Logged -RunId $row.id -StepName "nih_available" -SuccessPath $nihMetrics -Command "conda run -n $CondaEnv python scripts\evaluate_qwen3vl_lp_transfer.py --lp-config $lpConfig --probe-dir $lpOutputDir --val-ums-path $nihManifestArg --data-root $dataRootArg --output-dir $nihOutputDir --max-samples 0 --batch-size 2 --device $device --verify-images"

    Invoke-Logged -RunId $row.id -StepName "visual" -SuccessPath $row.visual_output -Command "conda run -n $CondaEnv python scripts\evaluate_qwen3vl_visual_dependence.py --config $trainConfig --checkpoint $checkpointArg --output $visualOutput --modes normal question_only image_shuffle hard_shuffle --max-samples 1000 --batch-size 1 --device $device"

    Invoke-Logged -RunId $row.id -StepName "counterfactual" -SuccessPath $row.counterfactual_output -Command "conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $trainConfig --checkpoint $checkpointArg --output $counterfactualOutput --max-samples 1000 --batch-size 1 --device $device" -Required $false

    if ($row.PSObject.Properties.Name -contains "ab_swap_input" -and $row.ab_swap_input) {
        $valInstructionPath = Quote-Arg $row.val_instruction_path
        $abSwapInput = Quote-Arg $row.ab_swap_input
        $abSwapConfig = Quote-Arg $row.ab_swap_config
        $abSwapOutput = Quote-Arg $row.ab_swap_output
        Invoke-Logged -RunId $row.id -StepName "ab_swap_jsonl" -SuccessPath $row.ab_swap_input -Command "conda run -n $CondaEnv python scripts\generate_ab_swap_jsonl.py --input $valInstructionPath --output $abSwapInput --max-records 1000" -Required $false
        $abRows = 0
        if (Test-Path -LiteralPath $row.ab_swap_input) {
            $abRows = (Get-Content -LiteralPath $row.ab_swap_input | Measure-Object -Line).Lines
        }
        if ($abRows -gt 0) {
            Invoke-Logged -RunId $row.id -StepName "ab_swap_config" -SuccessPath $row.ab_swap_config -Command "conda run -n $CondaEnv python scripts\write_eval_config_override.py --config $trainConfig --val-instruction-path $abSwapInput --output $abSwapConfig"
            Invoke-Logged -RunId $row.id -StepName "ab_swap_counterfactual" -SuccessPath $row.ab_swap_output -Command "conda run -n $CondaEnv python scripts\evaluate_qwen3vl_counterfactual_diagnostics.py --config $abSwapConfig --checkpoint $checkpointArg --output $abSwapOutput --max-samples 1000 --batch-size 1 --device $device" -Required $false
        } else {
            Write-QueueLog "WARN $($row.id) ab_swap no_rows input=$($row.ab_swap_input)"
        }
    }

    Invoke-Logged -RunId $row.id -StepName "paraphrase" -SuccessPath $row.paraphrase_output -Command "conda run -n $CondaEnv python scripts\evaluate_qwen3vl_paraphrase_robustness.py --config $trainConfig --checkpoint $checkpointArg --output $paraphraseOutput --max-samples 1000 --batch-size 1 --device $device" -Required $false
}

Write-QueueLog "QUEUE_DONE"
exit 0
