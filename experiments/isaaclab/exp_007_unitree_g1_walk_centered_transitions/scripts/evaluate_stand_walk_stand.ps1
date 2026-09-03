[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("preflight", "smoke", "formal", "repeatability")][string]$Mode = "formal",
    [int]$Seed = 20260901,
    [int]$Cycles = 1,
    [string]$Output = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage5_stand_walk_stand_integration"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$manifest = Join-Path $experimentRoot "integration_manifest.json"
foreach ($required in @($launcher, $manifest)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required file missing: $required" }
}
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$arguments = @(
    "-p", "$PSScriptRoot\evaluate_stand_walk_stand.py",
    "--mode", $Mode,
    "--seed", $Seed,
    "--cycles", $Cycles,
    "--output", $Output,
    "--manifest", $manifest,
    "--headless"
)
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 5 evaluation failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
