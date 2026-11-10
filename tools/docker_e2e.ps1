# End to end inside the image, from Windows PowerShell: synthetic copies of the two datasets are written to a volume,
# described, benchmarked, and the result files are checked (read-only root, no capabilities, no network).
# Needs Docker Desktop.
# Usage: .\tools\docker_e2e.ps1 [-Image idsbench]
param([string]$Image = "idsbench")
$ErrorActionPreference = "Stop"

$id = Get-Random
$vol = "idsbench-e2e-$id"
$secure = @("--read-only", "--tmpfs", "/tmp", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--network", "none", "-v", "${vol}:/data")

# Windows PowerShell 5.1 turns a program's message on stderr into an error under "Stop": the commands that print
# progress on stderr or are expected to fail run with errors not terminating, and only their exit code is looked at.
function Invoke-IdsCode([int[]]$Allowed, [string[]]$IdsArgs) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = docker run --rm @secure $Image @IdsArgs 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $old
    if ($Allowed -notcontains $code) { throw "idsbench $IdsArgs exited with $code : $out" }
    ($out | Out-String)
}
function Invoke-Ids([string[]]$IdsArgs) { Invoke-IdsCode @(0) $IdsArgs }
function Read-Volume($path) {
    $out = docker run --rm --entrypoint cat -v "${vol}:/data" $Image $path
    if ($LASTEXITCODE -ne 0) { throw "cannot read $path" }
    ($out | Out-String)
}
function Assert-Contains($text, $needle) { if ($text -notmatch [regex]::Escape($needle)) { throw "expected '$needle' in: $text" } }
function Assert-Missing($text, $needle) { if ($text -match [regex]::Escape($needle)) { throw "did not expect '$needle' in: $text" } }

try {
    docker volume create $vol | Out-Null
    docker run --rm --user 0 -v "${vol}:/data" --entrypoint chown $Image 10001 /data
    if ($LASTEXITCODE -ne 0) { throw "could not prepare the volume" }

    Invoke-Ids @("synth", "unsw", "/data/u", "--rows", "900") | Out-Null
    Invoke-Ids @("synth", "cic", "/data/c", "--rows", "400") | Out-Null

    $check = Invoke-Ids @("check", "/data/u", "/data/c")
    foreach ($want in "unsw", "cic", "dos hulk", "categorical: proto, service, state", "duplicates") { Assert-Contains $check $want }

    Assert-Contains (Invoke-Ids @("run", "--unsw", "/data/u", "--cic", "/data/c", "--out", "/data/out", "--models", "dummy,tree", "--quick")) "written: /data/out/report.json"
    $json = Read-Volume "/data/out/report.json"
    Assert-Contains $json '"synthetic": false'
    Assert-Contains $json '"id": "cic-by-day-no-port"'
    Assert-Contains $json '"id": "unsw-official-no-ttl"'
    Assert-Contains $json '"id": "transfer-unsw-to-cic"'
    $md = Read-Volume "/data/out/evaluation.md"
    Assert-Contains $md "# Evaluation"
    Assert-Missing $md "SYNTHETIC DATA"

    Invoke-Ids @("demo", "--rows", "500", "--models", "dummy,tree", "--quick", "--out", "/data/demo") | Out-Null
    Assert-Contains (Read-Volume "/data/demo/evaluation.md") "SYNTHETIC DATA"

    Invoke-Ids @("report", "/data/out/report.json", "-o", "/data/out/again.md") | Out-Null
    if ((Read-Volume "/data/out/again.md") -ne $md) { throw "the report rebuilt from the JSON is not identical" }

    Invoke-IdsCode @(2) @("check", "/data/nope") | Out-Null
    Write-Host "docker end-to-end: OK"
}
finally {
    $ErrorActionPreference = "Continue"
    docker volume rm -f $vol 2>&1 | Out-Null
}
