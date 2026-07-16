param(
    [string]$Manifest = "outputs\final_tables\cvcp_ccsh_module_combo_manifest.csv",
    [string]$Gpu = "0",
    [int]$LaneIndex = 0,
    [int]$LaneCount = 1,
    [string]$CondaEnv = "vivid",
    [int]$MaxMemoryMiB = 17000,
    [int]$PollSeconds = 180
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$logDir = Join-Path $RepoRoot "outputs\logs\cvcp_ccsh_module_combos"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$queueLog = Join-Path $logDir "module_combo_gpu${Gpu}_lane${LaneIndex}_of${LaneCount}.log"
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
    while (-not (Test-Path -LiteralPath $trainMetrics)) {
        Write-QueueLog "WAIT $($row.combo_id) backbone_training_not_ready"
        Start-Sleep -Seconds $PollSeconds
    }
    if ($row.checkpoint_path) {
        while (-not (Test-Path -LiteralPath $row.checkpoint_path)) {
            Write-QueueLog "WAIT $($row.combo_id) checkpoint_not_ready"
            Start-Sleep -Seconds $PollSeconds
        }
    }

    $trainConfig = Quote-Arg $row.train_config
    $checkpointArg = ""
    if ($row.checkpoint_path) {
        $checkpointArg = "--checkpoint " + (Quote-Arg $row.checkpoint_path)
    }
    $trainNpz = Quote-Arg $row.train_npz
    $trainMeta = Quote-Arg $row.train_metadata
    $trainManifest = Quote-Arg $row.train_manifest
    $valNpz = Quote-Arg $row.val_npz
    $valMeta = Quote-Arg $row.val_metadata
    $valManifest = Quote-Arg $row.val_manifest
    $valInstruction = Quote-Arg $row.val_instruction_path
    $maxTrain = [int]$row.max_export_train
    $maxVal = [int]$row.max_export_val

    Invoke-Logged -RunId $row.combo_id -StepName "export_train_embeddings" -SuccessPath $row.train_npz -Command "conda run -n $CondaEnv python scripts\export_qwen3vl_instruction_embeddings.py --config $trainConfig $checkpointArg --output-npz $trainNpz --metadata-jsonl $trainMeta --manifest $trainManifest --max-samples $maxTrain --batch-size 2 --device $device"
    Invoke-Logged -RunId $row.combo_id -StepName "export_val_embeddings" -SuccessPath $row.val_npz -Command "conda run -n $CondaEnv python scripts\export_qwen3vl_instruction_embeddings.py --config $trainConfig $checkpointArg --instruction-jsonl $valInstruction --output-npz $valNpz --metadata-jsonl $valMeta --manifest $valManifest --max-samples $maxVal --batch-size 2 --device $device"

    foreach ($module in ($row.modules -split ';')) {
        if (-not $module) { continue }
        $moduleLower = $module.ToLowerInvariant()
        $moduleSuccess = Join-Path $row.output_root "$moduleLower\metrics_final.json"
        $outputRoot = Quote-Arg $row.output_root
        $steps = [int]$row.max_steps
        $batchSize = [int]$row.batch_size
        Invoke-Logged -RunId $row.combo_id -StepName "train_module_$moduleLower" -SuccessPath $moduleSuccess -Command "conda run -n $CondaEnv python scripts\cvcp_ccsh_driver.py train_module_stack --module $module --train-embeddings $trainNpz --train-metadata $trainMeta --val-embeddings $valNpz --val-metadata $valMeta --output-root $outputRoot --max-steps $steps --batch-size $batchSize --device $device"
    }
    Invoke-Logged -RunId $row.combo_id -StepName "summarize" -SuccessPath "" -Command "conda run -n $CondaEnv python scripts\summarize_cvcp_ccsh_module_combos.py"
}

Write-QueueLog "QUEUE_DONE"
exit 0
