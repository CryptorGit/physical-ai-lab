[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("startup", "main", "zero_point_six")]
    [string]$Mode,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Output = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage5e_state_conditioned_confirmation"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$manifest = Join-Path $experimentRoot "integration_manifest.json"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $launcher -p "$PSScriptRoot\evaluate_state_conditioned.py" `
        --mode $Mode --seed $Seed --label $Label --output $Output `
        --manifest $manifest --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 5E evaluation failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
