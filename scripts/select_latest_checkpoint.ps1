param(
    [Parameter(Mandatory = $true)]
    [string]$Dir,
    [string]$Default = ""
)

$checkpoint = $null
if (Test-Path -LiteralPath $Dir) {
    $checkpoint = Get-ChildItem -LiteralPath $Dir -File |
        Where-Object { $_.Name -eq "best.pt" -or $_.Name -match '^step_\d+\.pt$' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

if ($checkpoint) {
    Write-Output $checkpoint.FullName
} else {
    Write-Output $Default
}
