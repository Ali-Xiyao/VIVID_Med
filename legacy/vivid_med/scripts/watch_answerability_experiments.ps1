param(
    [int]$IntervalSeconds = 300,
    [string]$LogPath = "outputs/logs/answerability_watch.log"
)

$ErrorActionPreference = "Continue"

function Write-WatchLog {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $LogPath -Value "[$stamp] $Message" -Encoding UTF8
}

function Get-RunProcesses {
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.CommandLine -like "*ablation_ums_ansmask_12label*" -or
            $_.CommandLine -like "*ablation_ums_null_as_negative_12label*" -or
            $_.CommandLine -like "*lp_ums_ansmask_12label*" -or
            $_.CommandLine -like "*lp_ums_null_as_negative_12label*"
        } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}

New-Item -ItemType Directory -Force -Path (Split-Path $LogPath) | Out-Null
Write-WatchLog "watch started; interval=${IntervalSeconds}s"

while ($true) {
    Write-WatchLog "----- check begin -----"

    try {
        $gpuAll = & nvidia-smi 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-WatchLog "nvidia-smi all: OK"
            Add-Content -Path $LogPath -Value ($gpuAll | Out-String) -Encoding UTF8
        } else {
            Write-WatchLog "nvidia-smi all: FAILED"
            Add-Content -Path $LogPath -Value ($gpuAll | Out-String) -Encoding UTF8
        }
    } catch {
        Write-WatchLog "nvidia-smi all exception: $($_.Exception.Message)"
    }

    foreach ($idx in 0, 1) {
        try {
            $gpuOne = & nvidia-smi -i $idx --query-gpu=timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-WatchLog "nvidia-smi gpu${idx}: OK"
                Add-Content -Path $LogPath -Value ($gpuOne | Out-String) -Encoding UTF8
            } else {
                Write-WatchLog "nvidia-smi gpu${idx}: FAILED"
                Add-Content -Path $LogPath -Value ($gpuOne | Out-String) -Encoding UTF8
            }
        } catch {
            Write-WatchLog "nvidia-smi gpu${idx} exception: $($_.Exception.Message)"
        }
    }

    try {
        $procs = Get-RunProcesses
        if ($procs) {
            Write-WatchLog "matching processes:"
            Add-Content -Path $LogPath -Value ($procs | Format-Table -Wrap -AutoSize | Out-String) -Encoding UTF8
        } else {
            Write-WatchLog "matching processes: none"
        }
    } catch {
        Write-WatchLog "process check exception: $($_.Exception.Message)"
    }

    foreach ($dir in "outputs/ablation_ums_ansmask_12label", "outputs/ablation_ums_null_as_negative_12label", "outputs/lp_ums_ansmask_12label", "outputs/lp_ums_null_as_negative_12label") {
        try {
            Write-WatchLog "latest files under ${dir}:"
            if (Test-Path $dir) {
                $files = Get-ChildItem $dir -Recurse -Force -ErrorAction SilentlyContinue |
                    Sort-Object LastWriteTime -Descending |
                    Select-Object -First 12 FullName, Length, LastWriteTime
                if ($files) {
                    Add-Content -Path $LogPath -Value ($files | Format-Table -AutoSize | Out-String) -Encoding UTF8
                } else {
                    Write-WatchLog "  no files"
                }
            } else {
                Write-WatchLog "  directory missing"
            }
        } catch {
            Write-WatchLog "file check exception for ${dir}: $($_.Exception.Message)"
        }
    }

    try {
        $logs = Get-ChildItem outputs/logs/*ums*12label*train.log -ErrorAction SilentlyContinue |
            Select-Object Name, Length, LastWriteTime
        if ($logs) {
            Write-WatchLog "training log files:"
            Add-Content -Path $LogPath -Value ($logs | Format-Table -AutoSize | Out-String) -Encoding UTF8
        }
    } catch {
        Write-WatchLog "training log check exception: $($_.Exception.Message)"
    }

    Write-WatchLog "----- check end -----"
    Start-Sleep -Seconds $IntervalSeconds
}
