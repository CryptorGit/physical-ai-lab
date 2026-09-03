[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("baseline", "pilot", "formal", "smoke")][string]$Mode,
    [string]$TransitionCheckpoint = "",
    [Parameter(Mandatory = $true)][string]$Label,
    [Parameter(Mandatory = $true)][int]$Seed,
    [string]$Output = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage3_stand_to_walk"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
foreach ($required in @($launcher, $stand, $walk)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
if ($Mode -ne "baseline" -and -not (Test-Path -LiteralPath $TransitionCheckpoint -PathType Leaf)) {
    throw "Transition checkpoint missing: $TransitionCheckpoint"
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$arguments = @("-p", "$PSScriptRoot\evaluate_stand_to_walk.py", "--stand-checkpoint", $stand, "--walk-checkpoint", $walk, "--mode", $Mode, "--label", $Label, "--seed", $Seed, "--output", $Output, "--headless")
if ($TransitionCheckpoint) { $arguments += @("--transition-checkpoint", $TransitionCheckpoint) }
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 evaluation failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
