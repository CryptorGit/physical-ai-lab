param(
    [int]$BatchIndex = 0,
    [int]$SeedRoot = 20268021,
    [string]$PythonExe = "C:\isaacsim\python.bat",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$env:PYTHONPATH = (
    (Join-Path $RepoRoot "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src"),
    (Join-Path $RepoRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"),
    $RepoRoot,
    $env:PYTHONPATH
) -join ";"
Set-Location $RepoRoot

& $PythonExe `
    (Join-Path $PSScriptRoot "collect_stage2g_on_policy.py") `
    --batch-index $BatchIndex `
    --seed-root $SeedRoot `
    --headless `
    --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2G collection batch $BatchIndex failed with exit code $LASTEXITCODE"
}
