param(
    [string]$PythonExe = "C:\isaacsim\python.bat",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$Exp012Src = Join-Path $RepoRoot "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src"
$Exp005Src = Join-Path $RepoRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"

$env:PYTHONPATH = "$Exp012Src;$Exp005Src;$RepoRoot;$env:PYTHONPATH"
Set-Location $RepoRoot

& $PythonExe `
    (Join-Path $PSScriptRoot "diagnose_stage2d_reachability.py") `
    --headless `
    --device $Device

if ($LASTEXITCODE -ne 0) {
    throw "Stage 2D diagnosis failed with exit code $LASTEXITCODE"
}
