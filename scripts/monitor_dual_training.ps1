param(
    [string]$Log10k = ".\outputs\cxr_chexpert_v4_2_dense10k\train.err.log",
    [string]$Log5k = ".\outputs\cxr_chexpert_v4_2_dense5k\train.err.log",
    [int]$TailLines = 12,
    [int]$IntervalSec = 5,
    [switch]$Once
)

function Get-LastProgressLine {
    param([string]$Path)

    if (!(Test-Path -LiteralPath $Path)) {
        return "log not found"
    }

    $match = Get-Content -LiteralPath $Path -Tail 400 |
        Select-String -Pattern "Training:\s+([0-9]+)%.*\|\s*([0-9]+)/([0-9]+)" |
        Select-Object -Last 1

    if ($null -eq $match) {
        return "training started, waiting for first tqdm line"
    }

    return $match.Line.Trim()
}

function Get-GpuSummary {
    try {
        return nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    }
    catch {
        return @("nvidia-smi unavailable")
    }
}

function Show-Block {
    param(
        [string]$Title,
        [string]$Path,
        [int]$TailLines
    )

    Write-Host "==== $Title ===="
    Write-Host "path: $Path"
    Write-Host "progress: $(Get-LastProgressLine -Path $Path)"

    if (Test-Path -LiteralPath $Path) {
        Write-Host "-- tail $TailLines lines --"
        Get-Content -LiteralPath $Path -Tail $TailLines
    }
    else {
        Write-Host "log not found"
    }
    Write-Host ""
}

do {
    Clear-Host
    Write-Host ("[{0}] dual training monitor" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Write-Host ""

    Write-Host "GPU summary (index, util%, usedMB, totalMB):"
    Get-GpuSummary | ForEach-Object { Write-Host $_ }
    Write-Host ""

    Show-Block -Title "10k run" -Path $Log10k -TailLines $TailLines
    Show-Block -Title "5k run" -Path $Log5k -TailLines $TailLines

    if ($Once) {
        break
    }

    Start-Sleep -Seconds $IntervalSec
}
while ($true)
