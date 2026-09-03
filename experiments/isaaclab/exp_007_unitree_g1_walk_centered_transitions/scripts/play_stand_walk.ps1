[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)][double]$Speed = 1.2,
    [ValidateSet("ZeroYaw", "FixedTarget")][string]$HeadingMode = "ZeroYaw",
    [int]$Seed = 20260724,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$player = Join-Path $PSScriptRoot "play_stand_walk.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpoint, $player, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
$actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $manifest.experts.WALK_STAND.sha256) {
    throw "Checkpoint SHA-256 mismatch: expected=$($manifest.experts.WALK_STAND.sha256) actual=$actualHash"
}
Write-Warning "Stage 2 formal status is FAIL. This command is diagnostic playback and does not claim a supported WALK transition."
Write-Host "active expert: Stage 2 model_4246 (RUN expert is not loaded)"
Write-Host "speed: $Speed m/s; ramp: fixed 2.0 s minimum-jerk; heading mode: $HeadingMode; camera: world-orientation-fixed"
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--checkpoint", $checkpoint, "--speed", $Speed, "--ramp-duration", "2.0", "--heading-mode", $HeadingMode, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $isaacLabBat @arguments
    if ($LASTEXITCODE -ne 0) { throw "STAND-WALK GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location }
