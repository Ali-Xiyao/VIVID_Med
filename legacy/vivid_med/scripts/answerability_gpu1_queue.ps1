param(
    [int]$IntervalSeconds = 300,
    [string]$LogPath = "outputs/logs/answerability_queue.log"
)

$ErrorActionPreference = "Continue"

function Write-QueueLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message" -Encoding UTF8
}

function Get-MatchingProcesses {
    param([string]$Pattern)
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like $Pattern -and $_.Name -ne "powershell.exe" -and $_.Name -ne "pwsh.exe" } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

function Get-GpuUtilization {
    param([int]$Index)

    try {
        $query = & nvidia-smi -i $Index --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-QueueLog "GPU${Index} query failed: $($query | Out-String)"
            return $null
        }

        $parts = ($query | Select-Object -First 1).ToString().Split(",")
        return @{
            Util = [int]$parts[0].Trim()
            Used = [int]$parts[1].Trim()
        }
    } catch {
        Write-QueueLog "GPU${Index} query exception: $($_.Exception.Message)"
        return $null
    }
}

function Test-Gpu1LaunchAllowed {
    $gpu1 = Get-GpuUtilization 1

    if ($null -eq $gpu1) {
        Write-QueueLog "launch blocked: GPU1 status unavailable"
        return $false
    }

    Write-QueueLog "GPU1-only policy active; GPU1 util=$($gpu1.Util)%, mem=$($gpu1.Used)MiB"

    if ($gpu1.Used -ge 1000) {
        Write-QueueLog "launch blocked: GPU1 is not idle"
        return $false
    }

    return $true
}

function Start-Gpu1Job {
    param(
        [string]$Name,
        [string]$ScriptPath
    )

    Write-QueueLog "launching ${Name}: ${ScriptPath}"
    Start-Process -WindowStyle Hidden -WorkingDirectory (Get-Location) -FilePath $ScriptPath
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
Write-QueueLog "queue started; interval=${IntervalSeconds}s"

while ($true) {
    Write-QueueLog "----- queue check begin -----"

    $ansTrainRunning = Get-MatchingProcesses "*ablation_ums_ansmask_12label.yaml*"
    $ansLpRunning = Get-MatchingProcesses "*lp_ums_ansmask_12label.yaml*"
    $nullTrainRunning = Get-MatchingProcesses "*ablation_ums_null_as_negative_12label.yaml*"
    $nullLpRunning = Get-MatchingProcesses "*lp_ums_null_as_negative_12label.yaml*"

    $ansTrainFinal = Test-Path "outputs/ablation_ums_ansmask_12label/checkpoints/final.pt"
    $ansLpFinal = Test-Path "outputs/lp_ums_ansmask_12label/metrics_final.json"
    $nullTrainFinal = Test-Path "outputs/ablation_ums_null_as_negative_12label/checkpoints/final.pt"
    $nullLpFinal = Test-Path "outputs/lp_ums_null_as_negative_12label/metrics_final.json"

    Write-QueueLog "status: ans_train_final=${ansTrainFinal}; ans_lp_final=${ansLpFinal}; null_train_final=${nullTrainFinal}; null_lp_final=${nullLpFinal}"
    Write-QueueLog "running: ans_train=$($ansTrainRunning.Count); ans_lp=$($ansLpRunning.Count); null_train=$($nullTrainRunning.Count); null_lp=$($nullLpRunning.Count)"

    $anyRunning = $ansTrainRunning -or $ansLpRunning -or $nullTrainRunning -or $nullLpRunning

    if (-not $ansTrainFinal -and -not $ansTrainRunning -and (Test-Path "outputs/ablation_ums_ansmask_12label/checkpoints/best.pt")) {
        if (Test-Gpu1LaunchAllowed) {
            Start-Gpu1Job `
                -Name "ablation_ums_ansmask_12label_resume" `
                -ScriptPath "scripts\run_ansmask_resume_gpu1.cmd"
        } else {
            Write-QueueLog "ansmask resume ready but GPU policy blocks launch"
        }
    } elseif ($ansTrainFinal -and -not $ansLpFinal -and -not $anyRunning) {
        if (Test-Gpu1LaunchAllowed) {
            Start-Gpu1Job `
                -Name "lp_ums_ansmask_12label" `
                -ScriptPath "scripts\run_lp_ansmask_gpu1.cmd"
        } else {
            Write-QueueLog "ansmask LP ready but GPU policy blocks launch"
        }
    } elseif ($ansLpFinal -and -not $nullTrainFinal -and -not $anyRunning) {
        if (Test-Gpu1LaunchAllowed) {
            Start-Gpu1Job `
                -Name "ablation_ums_null_as_negative_12label" `
                -ScriptPath "scripts\run_null_as_negative_gpu1.cmd"
        } else {
            Write-QueueLog "null-as-negative training ready but GPU policy blocks launch"
        }
    } elseif ($nullTrainFinal -and -not $nullLpFinal -and -not $anyRunning) {
        if (Test-Gpu1LaunchAllowed) {
            Start-Gpu1Job `
                -Name "lp_ums_null_as_negative_12label" `
                -ScriptPath "scripts\run_lp_null_as_negative_gpu1.cmd"
        } else {
            Write-QueueLog "null-as-negative LP ready but GPU policy blocks launch"
        }
    } elseif (-not $ansTrainFinal -and -not $ansTrainRunning) {
        Write-QueueLog "ansmask pretraining is incomplete and not running; manual inspection required"
    } else {
        Write-QueueLog "no launch condition met"
    }

    Write-QueueLog "----- queue check end -----"
    Start-Sleep -Seconds $IntervalSeconds
}
