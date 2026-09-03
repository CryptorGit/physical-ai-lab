[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet(100, 150)][int]$Iterations = 100,
    [int]$Seed,
    [Parameter(Mandatory = $true)][string]$RunName,
    [ValidateSet(-0.02, -0.05)][double]$YawOscillationWeight = -0.02
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $launcher -p (Join-Path $PSScriptRoot "train_walk_stabilization.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path --num-envs 1024 `
        --iterations $Iterations --seed $Seed --run-name $RunName `
        --yaw-oscillation-weight $YawOscillationWeight --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 2W-B training failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
