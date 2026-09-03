[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet("preflight", "pilot", "formal")][string]$Mode,
    [Parameter(Mandatory = $true)][string]$Label,
    [string]$Output = "results\exp_007_unitree_g1_walk_centered_transitions\stage2w_independent_walk",
    [int]$Seed = 20260728
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try {
    & $launcher -p (Join-Path $PSScriptRoot "evaluate_walk_steady_state.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path --mode $Mode `
        --output $Output --label $Label --seed $Seed --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 2W evaluation failed: $LASTEXITCODE" }
}
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
