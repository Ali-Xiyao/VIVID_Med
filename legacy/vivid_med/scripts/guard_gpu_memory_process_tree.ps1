param(
    [Parameter(Mandatory = $true)]
    [int]$Gpu,

    [Parameter(Mandatory = $true)]
    [int]$ThresholdMiB,

    [Parameter(Mandatory = $true)]
    [int]$RootPid,

    [string]$LogPath = "outputs/logs/qwen3vl_next_stage/gpu_memory_guard.log",

    [int]$PollSeconds = 15
)

$ErrorActionPreference = "Stop"

function Write-GuardLog {
    param([string]$Message)
    $dir = Split-Path -Parent $LogPath
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $Message | Out-File -FilePath $LogPath -Append -Encoding utf8
}

function Convert-ProcessCreationTime {
    param($Process)

    $value = $Process.CreationDate
    if ($value -is [datetime]) {
        return $value
    }

    if ($null -ne $value) {
        $text = [string]$value
        if (-not [string]::IsNullOrWhiteSpace($text)) {
            try {
                return [System.Management.ManagementDateTimeConverter]::ToDateTime($text)
            } catch {
                # Fall through to Get-Process.StartTime; CIM providers on this
                # host sometimes surface CreationDate in a non-DMTF shape.
            }
        }
    }

    $liveProcess = Get-Process -Id ([int]$Process.ProcessId) -ErrorAction Stop
    return $liveProcess.StartTime
}

function Get-ProcessTreeIds {
    param([int]$Root)

    $all = @(Get-CimInstance Win32_Process)
    $rootProcess = $all | Where-Object { [int]$_.ProcessId -eq $Root } | Select-Object -First 1
    if (-not $rootProcess) {
        return @()
    }
    try {
        $rootCreated = Convert-ProcessCreationTime -Process $rootProcess
    } catch {
        Write-GuardLog ("ROOT_TIME_READ_FAILED {0} root_pid={1} error={2}" -f (Get-Date -Format o), $Root, $_.Exception.Message)
        return @($Root)
    }
    $queue = New-Object System.Collections.Generic.Queue[int]
    $seen = New-Object System.Collections.Generic.HashSet[int]

    $queue.Enqueue($Root)
    [void]$seen.Add($Root)

    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        foreach ($child in ($all | Where-Object { $_.ParentProcessId -eq $current })) {
            try {
                $childCreated = Convert-ProcessCreationTime -Process $child
            } catch {
                Write-GuardLog ("INCLUDE_CHILD_TIME_READ_FAILED {0} root_pid={1} parent_pid={2} child_pid={3} error={4}" -f (Get-Date -Format o), $Root, $current, $child.ProcessId, $_.Exception.Message)
                $childCreated = $rootCreated.AddSeconds(1)
            }
            if ($childCreated -lt $rootCreated) {
                Write-GuardLog ("SKIP_STALE_PARENT {0} root_pid={1} parent_pid={2} child_pid={3} child_created={4} root_created={5}" -f (Get-Date -Format o), $Root, $current, $child.ProcessId, $childCreated.ToString("o"), $rootCreated.ToString("o"))
                continue
            }
            $childId = [int]$child.ProcessId
            if ($seen.Add($childId)) {
                $queue.Enqueue($childId)
            }
        }
    }

    return @($seen)
}

Write-GuardLog ("GUARD_START {0} gpu={1} threshold_mib={2} root_pid={3} poll_seconds={4}" -f (Get-Date -Format o), $Gpu, $ThresholdMiB, $RootPid, $PollSeconds)

while ($true) {
    Start-Sleep -Seconds $PollSeconds

    $root = Get-Process -Id $RootPid -ErrorAction SilentlyContinue
    if (-not $root) {
        Write-GuardLog ("GUARD_EXIT root_missing {0} root_pid={1}" -f (Get-Date -Format o), $RootPid)
        exit 0
    }

    $line = & nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        Where-Object { $_ -match ("^{0}," -f $Gpu) } |
        Select-Object -First 1

    if (-not $line) {
        Write-GuardLog ("GUARD_WARN gpu_line_missing {0} gpu={1}" -f (Get-Date -Format o), $Gpu)
        continue
    }

    $usedMiB = [int](($line -split ",")[1].Trim())
    if ($usedMiB -lt $ThresholdMiB) {
        continue
    }

    Write-GuardLog ("GUARD_TRIGGER {0} gpu={1} used_mib={2} threshold_mib={3} root_pid={4}" -f (Get-Date -Format o), $Gpu, $usedMiB, $ThresholdMiB, $RootPid)
    try {
        $ids = @(Get-ProcessTreeIds -Root $RootPid | Sort-Object -Descending)
    } catch {
        Write-GuardLog ("TREE_BUILD_FAILED {0} root_pid={1} error={2}" -f (Get-Date -Format o), $RootPid, $_.Exception.Message)
        $ids = @($RootPid)
    }

    if ($ids.Count -eq 0) {
        Write-GuardLog ("TREE_EMPTY_FALLBACK_ROOT {0} root_pid={1}" -f (Get-Date -Format o), $RootPid)
        $ids = @($RootPid)
    }

    foreach ($targetPid in $ids) {
        try {
            Stop-Process -Id $targetPid -Force -ErrorAction Stop
            Write-GuardLog ("STOPPED {0} pid={1}" -f (Get-Date -Format o), $targetPid)
        } catch {
            Write-GuardLog ("STOP_FAILED {0} pid={1} error={2}" -f (Get-Date -Format o), $targetPid, $_.Exception.Message)
        }
    }

    exit 0
}
