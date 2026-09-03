param(
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $repo

& ".\experiments\isaaclab\exp_010_unitree_g1_post_run_walk_attractor\scripts\validate_post_run_walk_pilot1_config.ps1"
if ($ValidateOnly) {
    Write-Host "Validation complete. PPO iterations: 0. Optimizer updates: 0."
    exit 0
}

$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not (Test-Path $launcher)) {
    throw "Isaac Lab launcher not found: $launcher"
}
& $launcher -p ".\experiments\isaaclab\exp_010_unitree_g1_post_run_walk_attractor\scripts\execute_post_run_walk_pilot1.py" --headless
