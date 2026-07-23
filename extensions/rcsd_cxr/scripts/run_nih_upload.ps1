param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$LocalRoot = 'H:\Xiyao_Wang\000_Public Dataset\NIH Chest X-rays',
    [string]$RemoteRoot = '/ipfs/inspurfileset/home/dqxy/dqxy11/projects/xiyaowang/02101_data/NIH_Chest_X-rays'
)

$ErrorActionPreference = 'Continue'
$runtimeRoot = Join-Path $ProjectRoot 'local_runs\nih_upload_20260722'
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
$pidPath = Join-Path $runtimeRoot 'uploader.pid'
$PID | Set-Content -LiteralPath $pidPath -Encoding ascii

$uploader = Join-Path $ProjectRoot 'scripts\upload_rawpty_sftp.py'
$key = Join-Path $env:USERPROFILE '.ssh\sues_hpc_dqxy11'
$knownHosts = Join-Path $env:USERPROFILE '.ssh\known_hosts'
$mapping = "${LocalRoot}::${RemoteRoot}"

try {
    $attempt = 0
    do {
        $attempt++
        Write-Output "NIH_UPLOAD_ATTEMPT=$attempt TIME=$(Get-Date -Format o)"
        & python $uploader `
            --key $key `
            --known-hosts $knownHosts `
            --tree $mapping
        $result = $LASTEXITCODE
        Write-Output "NIH_UPLOAD_EXIT=$result TIME=$(Get-Date -Format o)"
        if ($result -ne 0) {
            Start-Sleep -Seconds 20
        }
    } while ($result -ne 0)
}
finally {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
