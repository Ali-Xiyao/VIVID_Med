$ErrorActionPreference = "Continue"
$Root = "H:\Xiyao_Wang\021_260129VIVID"
Set-Location $Root
$LogPath = "outputs/logs/answerability_resource_guard_once.log"
$PauseFlag = "outputs/run_state/VIVID_MICCAI_PAUSED.flag"
$HardMaxTotalPowerW = 400.0
$WarnTotalPowerW = 350.0
$HardMaxTempC = 83
$VividTasks = @(
    "VIVID_ansmask_resume_gpu0",
    "VIVID_lp_ansmask_gpu0",
    "VIVID_null_as_negative_gpu0",
    "VIVID_lp_null_as_negative_gpu0",
    "VIVID_cf_prefix_dependency_gpu0",
    "VIVID_random_lm_gpu0",
    "VIVID_lp_random_lm_gpu0",
    "VIVID_field_paraphrase_gpu0",
    "VIVID_ansmask_resume_gpu1",
    "VIVID_lp_ansmask_gpu1",
    "VIVID_null_as_negative_gpu1",
    "VIVID_lp_null_as_negative_gpu1",
    "VIVID_cf_prefix_dependency_gpu1",
    "VIVID_random_lm_gpu1",
    "VIVID_lp_random_lm_gpu1",
    "VIVID_field_paraphrase_gpu1"
)

function Write-GuardLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message" -Encoding UTF8
}

function Get-GpuStatus {
    param([int]$Index)
    try {
        $line = & nvidia-smi -i $Index --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-GuardLog "GPU${Index} query failed: $($line | Out-String)"
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
        Write-GuardLog "GPU${Index} query exception: $($_.Exception.Message)"
        return $null
    }
}

function Stop-VividTasks {
    param([string]$Reason)
    Write-GuardLog "hard guard triggered: ${Reason}; ending VIVID scheduled tasks"
    foreach ($task in $VividTasks) {
        try {
            $result = schtasks /End /TN $task 2>&1
            Write-GuardLog "schtasks /End ${task}: $($result | Out-String)"
        } catch {
            Write-GuardLog "failed to end ${task}: $($_.Exception.Message)"
        }
    }
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
if (Test-Path $PauseFlag) {
    Write-GuardLog "pause flag present at ${PauseFlag}; resource guard skipped"
    return
}
Write-GuardLog "----- resource guard begin -----"

$statuses = @()
foreach ($idx in 0, 1) {
    $status = Get-GpuStatus $idx
    if ($null -ne $status) {
        $statuses += $status
    }
}

if ($statuses.Count -lt 2) {
    Write-GuardLog "power-guard policy active; incomplete GPU status; no stop action"
} else {
    $totalPower = [Math]::Round((($statuses | Measure-Object -Property Power -Sum).Sum), 2)
    $maxTemp = (($statuses | Measure-Object -Property Temp -Maximum).Maximum)
    foreach ($gpu in $statuses) {
        Write-GuardLog "GPU$($gpu.Index): util=$($gpu.Util)%, mem=$($gpu.Used)MiB, temp=$($gpu.Temp)C, power=$($gpu.Power)W"
    }
    Write-GuardLog "power summary: total_power=${totalPower}W, max_temp=${maxTemp}C, warn_power=${WarnTotalPowerW}W, hard_power=${HardMaxTotalPowerW}W, hard_temp=${HardMaxTempC}C"
    if ($totalPower -ge $HardMaxTotalPowerW) {
        Stop-VividTasks "total power ${totalPower}W >= ${HardMaxTotalPowerW}W"
    } elseif ($maxTemp -ge $HardMaxTempC) {
        Stop-VividTasks "max temp ${maxTemp}C >= ${HardMaxTempC}C"
    } elseif ($totalPower -ge $WarnTotalPowerW) {
        Write-GuardLog "warning: total power ${totalPower}W >= ${WarnTotalPowerW}W; no stop until hard cap"
    } else {
        Write-GuardLog "resource guard OK; no stop action"
    }
}

Write-GuardLog "----- resource guard end -----"
