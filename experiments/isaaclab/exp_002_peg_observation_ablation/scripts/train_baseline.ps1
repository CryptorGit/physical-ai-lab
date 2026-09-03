$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path "$PSScriptRoot\..\..\..\.."
$IsaacLauncher = Join-Path $ProjectRoot "tools\run-isaaclab.ps1"

if (-not (Test-Path $IsaacLauncher)) {
    throw "Isaac Lab launcher not found: $IsaacLauncher"
}

$Task = "Isaac-Factory-PegInsert-Direct-v0"
$Seed = 42
$NumEnvs = 256

& $IsaacLauncher `
    "scripts/reinforcement_learning/rl_games/train.py" `
    "--task" $Task `
    "--seed" $Seed `
    "--num_envs" $NumEnvs `
    "--headless"

if ($LASTEXITCODE -ne 0) {
    throw "Baseline training failed with exit code $LASTEXITCODE"
}