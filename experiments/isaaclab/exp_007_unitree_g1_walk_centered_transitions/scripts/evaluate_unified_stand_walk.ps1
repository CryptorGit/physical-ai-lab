[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [ValidateSet("R1", "R2", "R3", "R4", "FORMAL")]
    [string]$Phase,

    [string]$Output = "results\exp_007_unitree_g1_walk_centered_transitions\stage2r_unified_stand_walk",

    [int]$Seed = 20260726,

    [ValidateRange(1, 100)]
    [int]$WalkEpisodesPerSpeed = 20,

    [ValidateRange(1, 100)]
    [int]$StandEpisodes = 50
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$($experimentRoot)\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Push-Location $repositoryRoot
try {
    & $launcher -p (Join-Path $PSScriptRoot "evaluate_unified_stand_walk.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path `
        --phase $Phase `
        --output $Output `
        --seed $Seed `
        --walk-episodes-per-speed $WalkEpisodesPerSpeed `
        --stand-episodes $StandEpisodes `
        --headless
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 2R evaluation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
