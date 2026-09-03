[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("reproduce", "stand", "forward", "reverse", "full")]
    [string]$Test,
    [ValidateSet(0.6, 0.8)][double]$Speed = 0.6,
    [Parameter(Mandatory = $true)][int]$Episodes,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Output = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage5d_integration_failure_diagnosis"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$manifest = Join-Path $experimentRoot "integration_manifest.json"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$arguments = @(
    "-p", "$PSScriptRoot\diagnose_stage5_failures.py",
    "--test", $Test, "--episodes", $Episodes, "--seed", $Seed,
    "--label", $Label, "--output", $Output, "--manifest", $manifest, "--headless"
)
if ($Test -notin @("reproduce", "stand")) { $arguments += @("--speed", $Speed) }
Push-Location $repositoryRoot
try {
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 5D diagnostic failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
