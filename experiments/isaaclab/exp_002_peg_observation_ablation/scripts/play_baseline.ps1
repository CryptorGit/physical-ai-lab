param(
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\..\..\.."
$IsaacLauncher = Join-Path $ProjectRoot "tools\run-isaaclab.ps1"

if (-not (Test-Path $IsaacLauncher)) {
    throw "Isaac Lab launcher not found: $IsaacLauncher"
}

if (-not (Test-Path $Checkpoint)) {
    throw "Checkpoint not found: $Checkpoint"
}

$Task = "Isaac-Factory-PegInsert-Direct-v0"

& $IsaacLauncher `
    "scripts/reinforcement_learning/rl_games/play.py" `
    "--task" $Task `
    "--checkpoint" (Resolve-Path $Checkpoint)

if ($LASTEXITCODE -ne 0) {
    throw "Baseline playback failed with exit code $LASTEXITCODE"
}