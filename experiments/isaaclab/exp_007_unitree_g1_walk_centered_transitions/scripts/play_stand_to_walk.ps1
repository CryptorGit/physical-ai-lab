[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.6, 0.8, 1.0, 1.2)][double]$Speed = 1.0,
    [int]$Seed = 20260818,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$gatePath = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\stage3_stand_to_walk\gate.json"
$gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
if ($gate.status -ne "PASS") { throw "Stage 3 is not PASS: $($gate.status)" }
$stand = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
$transition = Join-Path $repositoryRoot $gate.selected_checkpoint
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($stand, $walk, $transition, $launcher)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$arguments = @("-p", "$PSScriptRoot\play_stand_to_walk.py", "--stand-checkpoint", $stand, "--transition-checkpoint", $transition, "--walk-checkpoint", $walk, "--speed", $Speed, "--seed", $Seed, "--viz", "kit")
if ($ValidateOnly) { $arguments += "--validate-only" }
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 GUI failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
