[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.6, 0.8, 1.0, 1.2)][double]$Speed = 1.0,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
$standToWalk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\model_0.pt"
$walkToStand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100\model_0.pt"
foreach ($required in @($launcher, $stand, $walk, $standToWalk, $walkToStand)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$arguments = @(
    "-p", "$PSScriptRoot\play_walk_to_stand.py",
    "--stand-checkpoint", $stand,
    "--walk-checkpoint", $walk,
    "--stand-to-walk-checkpoint", $standToWalk,
    "--walk-to-stand-checkpoint", $walkToStand,
    "--speed", $Speed
)
if ($ValidateOnly) { $arguments += "--validate-only" }
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 4 GUI failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
