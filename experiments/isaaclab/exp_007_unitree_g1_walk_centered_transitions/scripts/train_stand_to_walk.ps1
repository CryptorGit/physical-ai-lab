[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Parent,
    [ValidateSet(8, 1024)][int]$NumEnvs,
    [ValidateSet(2, 100, 150)][int]$Iterations,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$RunName,
    [ValidateSet(-0.10, -0.20)][double]$TargetAlignmentWeight = -0.10,
    [ValidateSet(-0.10, -0.20)][double]$SourceAlignmentWeight = -0.10
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
foreach ($required in @($launcher, $stand, $walk, $Parent)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $launcher -p "$PSScriptRoot\train_stand_to_walk.py" --stand-checkpoint $stand --walk-checkpoint $walk --parent $Parent --num-envs $NumEnvs --iterations $Iterations --seed $Seed --run-name $RunName --target-alignment-weight $TargetAlignmentWeight --source-alignment-weight $SourceAlignmentWeight --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 training failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
