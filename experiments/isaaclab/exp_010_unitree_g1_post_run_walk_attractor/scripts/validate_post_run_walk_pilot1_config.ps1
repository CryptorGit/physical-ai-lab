$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $repo
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not (Test-Path $launcher)) {
    throw "Isaac Lab launcher not found: $launcher"
}
& $launcher -p ".\experiments\isaaclab\exp_010_unitree_g1_post_run_walk_attractor\scripts\validate_post_run_walk_pilot1.py"
