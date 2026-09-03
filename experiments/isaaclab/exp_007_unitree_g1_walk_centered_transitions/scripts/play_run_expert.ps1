[CmdletBinding(PositionalBinding = $false)]
param(
    [double]$Speed = 2.6,
    [ValidateSet(0, 45, 90)][int]$TurnDegrees = 0,
    [ValidateSet("Left", "Right")][string]$Direction = "Left",
    [int]$Seed = 20260723,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpointPath = Join-Path $repositoryRoot $manifest.experts.RUN.checkpoint
$player = Join-Path $PSScriptRoot "play_run_expert.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpointPath, $player, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
Write-Host "exp_007 Stage 0 single RUN expert playback"
Write-Host "actor: exp006 G1CommandResidualActor candidate A"
Write-Host "checkpoint: $checkpointPath"
Write-Host "command: RUN speed=$Speed m/s turn=$Direction $TurnDegrees degrees"
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--speed", $Speed, "--turn-degrees", $TurnDegrees, "--direction", $Direction, "--checkpoint", $checkpointPath, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "RUN GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location }
