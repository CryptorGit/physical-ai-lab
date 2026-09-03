[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("STAND", "WALK")][string]$Mode = "STAND",
    [double]$Speed = 1.5,
    [int]$Seed = 20260723,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpointPath = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$player = Join-Path $PSScriptRoot "play_walk_expert.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpointPath, $player, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
if ($Mode -eq "STAND") { $Speed = 0.0 }
Write-Host "exp_007 Stage 0 single WALK/STAND expert playback"
Write-Host "actor: exp005 Stage 2 legacy MLP (123 -> 37)"
Write-Host "checkpoint: $checkpointPath"
Write-Host "command: $Mode speed=$Speed m/s"
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--mode", $Mode, "--speed", $Speed, "--checkpoint", $checkpointPath, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "WALK/STAND GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location }
