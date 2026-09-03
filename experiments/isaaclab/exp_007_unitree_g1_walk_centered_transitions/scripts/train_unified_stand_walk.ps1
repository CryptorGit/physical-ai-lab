[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("R0", "R1", "R2", "R3", "R4")]
    [string]$Phase,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [ValidateRange(2, 1024)]
    [int]$NumEnvs = 1024,

    [ValidateRange(1, 200)]
    [int]$Iterations,

    [int]$Seed = 20260725,

    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path

if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Isaac Lab launcher not found: $launcher"
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $launcher -p (Join-Path $PSScriptRoot "train_unified_stand_walk.py") `
        --phase $Phase `
        --checkpoint $checkpointPath `
        --num-envs $NumEnvs `
        --iterations $Iterations `
        --seed $Seed `
        --run-name $RunName `
        --headless
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 2R $Phase training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
