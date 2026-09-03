[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Parent,
    [ValidateSet(8, 1024)][int]$NumEnvs,
    [ValidateSet(2, 100, 150)][int]$Iterations,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$RunName,
    [ValidateSet(1.4, 1.6, 1.8)][double]$RampDuration = 1.6,
    [ValidateSet(-2.0, -3.0)][double]$ReverseWeight = -2.0
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
$standToWalk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\model_0.pt"
foreach ($required in @($launcher, $stand, $walk, $standToWalk, $Parent)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $launcher -p "$PSScriptRoot\train_walk_to_stand.py" --stand-checkpoint $stand --walk-checkpoint $walk --stand-to-walk-checkpoint $standToWalk --parent $Parent --num-envs $NumEnvs --iterations $Iterations --seed $Seed --run-name $RunName --ramp-duration $RampDuration --reverse-weight $ReverseWeight --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 4 training failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
