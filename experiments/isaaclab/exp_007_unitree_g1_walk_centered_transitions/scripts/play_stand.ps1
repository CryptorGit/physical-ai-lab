[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Seed = 20260723,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$player = Join-Path $PSScriptRoot "play_stand.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpoint, $player, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
$actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $manifest.experts.WALK_STAND.sha256) {
    throw "Checkpoint SHA-256 mismatch: expected=$($manifest.experts.WALK_STAND.sha256) actual=$actualHash"
}
Write-Host "exp_007 STAND home-state playback"
Write-Host "active state: STAND"
Write-Host "active expert: Stage 2 model_4246"
Write-Host "checkpoint: $checkpoint"
Write-Host "command: vx=0 vy=0 yaw_rate=0"
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--checkpoint", $checkpoint, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "STAND GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location }
