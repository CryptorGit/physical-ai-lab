[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateRange(2, 1024)][int]$NumEnvs = 1024,
    [ValidateRange(1, 300)][int]$Iterations = 150,
    [int]$Seed = 20260729,
    [string]$RunName = "stage2w_independent_walk_pilot",
    [ValidateSet("pilot1", "pilot2")][string]$RewardProfile = "pilot1"
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
    & $launcher -p (Join-Path $PSScriptRoot "train_walk_steady_state.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path --num-envs $NumEnvs `
        --iterations $Iterations --seed $Seed --run-name $RunName --reward-profile $RewardProfile --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 2W training failed: $LASTEXITCODE" }
}
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
