$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $repo
& ".\experiments\isaaclab\exp_010_unitree_g1_post_run_walk_attractor\scripts\train_post_run_walk_pilot1.ps1" -ValidateOnly
