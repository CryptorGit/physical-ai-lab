[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("preflight", "audit", "formal")]
    [string]$Mode,
    [Parameter(Mandatory = $true)][int]$Seed,
    [Parameter(Mandatory = $true)][string]$Label,
    [ValidateRange(1, 20)][int]$EpisodesPerSpeed = 4,
    [string]$Output = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage6_run_low_steady_state"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;$repositoryRoot\experiments\isaaclab\exp_006_unitree_g1_command_skills\src" + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try {
    & $launcher -p "$PSScriptRoot\evaluate_run_low.py" `
        --mode $Mode --seed $Seed --label $Label `
        --episodes-per-speed $EpisodesPerSpeed --output $Output --viz none
    if ($LASTEXITCODE -ne 0) { throw "Stage 6 evaluation failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $oldPythonPath
}
