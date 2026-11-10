# The benchmark on the real datasets, in Docker (read-only datasets, no network, no capabilities), from Windows PowerShell.
# Usage: .\tools\run_real.ps1 [-Unsw datasets\unsw] [-Cic datasets\cic] [-Out results] [-Jobs 4] [-Sample 0.2] [-Quick]
# -Sample reads only that fraction of each file (if the computer has little memory); the report says so.
# It first runs `idsbench check` on each folder, so a wrong file is reported in seconds and not after an hour.
param(
    [string]$Unsw = "datasets\unsw",
    [string]$Cic = "datasets\cic",
    [string]$Out = "results",
    [int]$Jobs = 4,
    [double]$Sample = 0,
    [switch]$Quick
)
# Windows PowerShell 5.1 turns anything a program writes on stderr into an error under "Stop": docker and idsbench
# print their progress there, so only the exit code is looked at.
$ErrorActionPreference = "Continue"

$mounts = @()
$folders = @()
foreach ($pair in @(@("unsw", $Unsw), @("cic", $Cic))) {
    if (Test-Path -LiteralPath $pair[1]) {
        $full = (Resolve-Path -LiteralPath $pair[1]).Path
        $mounts += @("-v", "${full}:/datasets/$($pair[0]):ro")
        $folders += "/datasets/$($pair[0])"
    }
}
if ($folders.Count -eq 0) { Write-Host "No dataset folder found. See datasets\README.md."; exit 2 }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$outFull = (Resolve-Path -LiteralPath $Out).Path

docker build -t idsbench .
if ($LASTEXITCODE -ne 0) { Write-Host "docker build failed"; exit 1 }

$secure = @("--rm", "--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network", "none")
Write-Host "== check =="
docker run @secure @mounts idsbench check @folders
if ($LASTEXITCODE -ne 0) { Write-Host "check failed: see the message above"; exit 2 }

$run = @("run", "--out", "/out", "--jobs", "$Jobs")
if (Test-Path -LiteralPath $Unsw) { $run += @("--unsw", "/datasets/unsw") }
if (Test-Path -LiteralPath $Cic) { $run += @("--cic", "/datasets/cic") }
if ($Sample -gt 0) { $run += @("--sample", "$Sample") }
if ($Quick) { $run += "--quick" }
Write-Host "== run =="
docker run @secure @mounts -v "${outFull}:/out" idsbench @run
if ($LASTEXITCODE -ne 0) { Write-Host "run failed (exit code $LASTEXITCODE)"; exit 1 }
Write-Host "Done. Send me $Out\report.json (and look at $Out\summary.png)."
