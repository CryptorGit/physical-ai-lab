[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(2.4, 2.6, 2.8)][double]$Speed = 2.6,
    [int]$Seed = 20261031,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$checkpoint = Join-Path $repositoryRoot "logs\rsl_rl\physical_ai_g1_command_skills\2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0\model_0.pt"
$player = Join-Path $PSScriptRoot "play_run_low_steady_state.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpoint, $player, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
Write-Host "STATE=UNINITIALIZED_FOR_RUN/RUN_LOW MODEL=run_low_steady_state_expert_v1"
Write-Host "TARGET_SPEED=$Speed TURN=DISABLED supported=[2.4,2.6,2.8]"
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--speed", $Speed, "--turn-degrees", 0, "--checkpoint", $checkpoint, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "RUN_LOW GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location }
