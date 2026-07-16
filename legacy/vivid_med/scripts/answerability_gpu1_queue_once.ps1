$ErrorActionPreference = "Continue"
$Root = "H:\Xiyao_Wang\021_260129VIVID"
Set-Location $Root
$LogPath = "outputs/logs/answerability_queue_once.log"
$PauseFlag = "outputs/run_state/VIVID_MICCAI_PAUSED.flag"
$TargetGpu = 1
$LaunchMaxTotalPowerW = 220.0
$HardMaxTotalPowerW = 400.0
$HardMaxTempC = 83
$LaunchBusyMemoryMiB = 500
$StopAfterAnsmaskFinal = $false

function Write-QueueLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message" -Encoding UTF8
}

function Get-GpuStatus {
    param([int]$Index)
    try {
        $line = & nvidia-smi -i $Index --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-QueueLog "GPU${Index} query failed: $($line | Out-String)"
            return $null
        }
        $parts = ($line | Select-Object -First 1).ToString().Split(",")
        return [pscustomobject]@{
            Index = $Index
            Util = [int]$parts[0].Trim()
            Used = [int]$parts[1].Trim()
            Temp = [int]$parts[2].Trim()
            Power = [double]$parts[3].Trim()
        }
    } catch {
        Write-QueueLog "GPU${Index} query exception: $($_.Exception.Message)"
        return $null
    }
}

function Get-AllGpuStatus {
    $statuses = @()
    foreach ($idx in 0, 1) {
        $status = Get-GpuStatus $idx
        if ($null -eq $status) {
            return $null
        }
        $statuses += $status
    }
    return $statuses
}

function Test-GpuComputeRunning {
    param(
        [int]$Index,
        [int]$MinUsedMiB = 500
    )
    try {
        $apps = & nvidia-smi -i $Index --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-QueueLog "GPU${Index} compute-app query failed: $($apps | Out-String)"
            return $true
        }
        $active = @()
        foreach ($app in $apps) {
            $text = $app.ToString().Trim()
            if ($text.Length -eq 0) { continue }
            $parts = $text.Split(",")
            if ($parts.Count -lt 2) { continue }
            $usedText = $parts[1].Trim()
            $usedMiB = 0
            if ([int]::TryParse($usedText, [ref]$usedMiB) -and $usedMiB -ge $MinUsedMiB) {
                $active += $text
            }
        }
        return ($active.Count -gt 0)
    } catch {
        Write-QueueLog "GPU${Index} compute-app exception: $($_.Exception.Message)"
        return $true
    }
}

function Test-LaunchAllowed {
    $all = Get-AllGpuStatus
    if ($null -eq $all) {
        Write-QueueLog "launch blocked: GPU status unavailable"
        return $false
    }
    $target = $all | Where-Object { $_.Index -eq $TargetGpu } | Select-Object -First 1
    $totalPower = [Math]::Round((($all | Measure-Object -Property Power -Sum).Sum), 2)
    $maxTemp = (($all | Measure-Object -Property Temp -Maximum).Maximum)

    Write-QueueLog "power-guard policy active; target_gpu=${TargetGpu} util=$($target.Util)%, mem=$($target.Used)MiB, temp=$($target.Temp)C, power=$($target.Power)W; total_power=${totalPower}W; max_temp=${maxTemp}C"
    if ($maxTemp -ge $HardMaxTempC) {
        Write-QueueLog "launch blocked: max temp ${maxTemp}C >= ${HardMaxTempC}C"
        return $false
    }
    if ($totalPower -ge $LaunchMaxTotalPowerW) {
        Write-QueueLog "launch blocked: current total power ${totalPower}W >= launch threshold ${LaunchMaxTotalPowerW}W"
        return $false
    }
    if ($target.Used -ge $LaunchBusyMemoryMiB -or (Test-GpuComputeRunning $TargetGpu $LaunchBusyMemoryMiB)) {
        Write-QueueLog "launch blocked: target GPU${TargetGpu} is busy"
        return $false
    }
    if ($totalPower -ge $HardMaxTotalPowerW) {
        Write-QueueLog "launch blocked: total power ${totalPower}W >= hard cap ${HardMaxTotalPowerW}W"
        return $false
    }
    return $true
}

function Start-TaskIfAllowed {
    param(
        [string]$TaskName,
        [string]$Reason
    )
    if (Test-LaunchAllowed) {
        Write-QueueLog "launching ${TaskName}: ${Reason}"
        schtasks /Run /TN $TaskName | Out-String | Add-Content -Path $LogPath -Encoding UTF8
    } else {
        Write-QueueLog "not launching ${TaskName}: resource policy blocked"
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
if (Test-Path $PauseFlag) {
    Write-QueueLog "pause flag present at ${PauseFlag}; queue launch skipped"
    return
}
Write-QueueLog "----- queue once begin -----"

$ansTrainFinal = Test-Path "outputs/ablation_ums_ansmask_12label/checkpoints/final.pt"
$ansTrainBest = Test-Path "outputs/ablation_ums_ansmask_12label/checkpoints/best.pt"
$ansLpFinal = Test-Path "outputs/lp_ums_ansmask_12label/metrics_final.json"
$nullTrainFinal = Test-Path "outputs/ablation_ums_null_as_negative_12label/checkpoints/final.pt"
$nullLpFinal = Test-Path "outputs/lp_ums_null_as_negative_12label/metrics_final.json"
$cfPrefixFinal = Test-Path "outputs/counterfactual_prefix_dependency_A_ums_12label_128.json"
$fieldParaphraseFinal = Test-Path "outputs/field_paraphrase_robustness_A_ums_12label_128.json"
$randomLmTrainFinal = Test-Path "outputs/ablation_ums_random_lm_12label/checkpoints/final.pt"
$randomLmLpFinal = Test-Path "outputs/lp_ums_random_lm_12label/metrics_final.json"
$nullLpRunning = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "train_vit_baseline\.py" -and $_.CommandLine -match "lp_ums_null_as_negative_12label\.yaml" }
).Count -gt 0
$randomLmRunning = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "train_cxr\.py" -and $_.CommandLine -match "ablation_ums_random_lm_12label\.yaml" }
).Count -gt 0
$randomLmLpRunning = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "train_vit_baseline\.py" -and $_.CommandLine -match "lp_ums_random_lm_12label\.yaml" }
).Count -gt 0
$cfPrefixRunning = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "eval_counterfactual_prefix_dependency\.py" -and $_.CommandLine -match "counterfactual_prefix_dependency_A_ums_12label_128\.json" }
).Count -gt 0
$fieldParaphraseRunning = @(
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -match "eval_field_paraphrase_robustness\.py" -and $_.CommandLine -match "field_paraphrase_robustness_A_ums_12label_128\.json" }
).Count -gt 0

Write-QueueLog "status: ans_train_final=${ansTrainFinal}; ans_lp_final=${ansLpFinal}; null_train_final=${nullTrainFinal}; null_lp_final=${nullLpFinal}; null_lp_running=${nullLpRunning}; cf_prefix_final=${cfPrefixFinal}; cf_prefix_running=${cfPrefixRunning}; field_paraphrase_final=${fieldParaphraseFinal}; field_paraphrase_running=${fieldParaphraseRunning}; random_lm_train_final=${randomLmTrainFinal}; random_lm_running=${randomLmRunning}; random_lm_lp_final=${randomLmLpFinal}; random_lm_lp_running=${randomLmLpRunning}"

if ($StopAfterAnsmaskFinal -and $ansTrainFinal) {
    Write-QueueLog "stop-after-ansmask enabled: ansmask final exists; not launching LP or downstream experiments"
} elseif (-not $ansTrainFinal -and $ansTrainBest) {
    Start-TaskIfAllowed "VIVID_ansmask_resume_gpu1" "resume ansmask pretraining"
} elseif ($ansTrainFinal -and -not $ansLpFinal) {
    Start-TaskIfAllowed "VIVID_lp_ansmask_gpu1" "run ansmask linear probe"
} elseif ($ansLpFinal -and -not $nullTrainFinal) {
    Start-TaskIfAllowed "VIVID_null_as_negative_gpu1" "run null-as-negative pretraining"
} elseif ($nullTrainFinal -and -not $nullLpFinal) {
    if ($nullLpRunning) {
        Write-QueueLog "not launching VIVID_lp_null_as_negative_gpu1: null-as-negative LP already running"
    } else {
        Start-TaskIfAllowed "VIVID_lp_null_as_negative_gpu1" "run null-as-negative linear probe"
    }
} elseif ($nullLpFinal -and -not $randomLmTrainFinal) {
    if ($randomLmRunning) {
        Write-QueueLog "not launching VIVID_random_lm_gpu1: random-LM already running"
    } else {
        Start-TaskIfAllowed "VIVID_random_lm_gpu1" "run random-LM same-architecture UMS control"
    }
} elseif ($randomLmTrainFinal -and -not $randomLmLpFinal) {
    if ($randomLmLpRunning) {
        Write-QueueLog "not launching VIVID_lp_random_lm_gpu1: random-LM LP already running"
    } else {
        Start-TaskIfAllowed "VIVID_lp_random_lm_gpu1" "run random-LM linear probe"
    }
} elseif ($randomLmLpFinal -and -not $cfPrefixFinal) {
    if ($cfPrefixRunning) {
        Write-QueueLog "not launching VIVID_cf_prefix_dependency_gpu1: counterfactual prefix eval already running"
    } else {
        Start-TaskIfAllowed "VIVID_cf_prefix_dependency_gpu1" "run counterfactual prefix dependency eval"
    }
} elseif ($cfPrefixFinal -and -not $fieldParaphraseFinal) {
    if ($fieldParaphraseRunning) {
        Write-QueueLog "not launching VIVID_field_paraphrase_gpu1: field paraphrase eval already running"
    } else {
        Start-TaskIfAllowed "VIVID_field_paraphrase_gpu1" "run field paraphrase robustness eval"
    }
} else {
    Write-QueueLog "no launch needed"
}

Write-QueueLog "----- queue once end -----"
