[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.6, 0.8, 1.0, 1.2)][double]$Speed = 1.0,
    [ValidateSet(1, 3)][int]$Cycles = 1,
    [switch]$RequireValidStandContract,
    [switch]$ValidateOnly
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
    "-p", "$PSScriptRoot\play_stand_walk_stand.py",
    "--manifest", $manifest,
    "--speed", $Speed,
    "--cycles", $Cycles
)
if ($ValidateOnly) { $arguments += "--validate-only" }
if ($RequireValidStandContract) { $arguments += "--require-valid-stand-contract" }
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 5 GUI failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
