[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.6, 0.8, 1.0, 1.2)]
    [double]$Speed = 1.0,

    [int]$Seed = 20260803,

    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$gatePath = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\stage2wb_walk_stabilization\gate.json"
$gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $gate.selected_checkpoint
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$player = Join-Path $PSScriptRoot "play_walk_steady_state.py"

foreach ($required in @($gatePath, $checkpoint, $launcher, $player)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file missing: $required"
    }
}
$actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $gate.selected_checkpoint_sha256) {
    throw "Stage 2W-B checkpoint SHA mismatch: expected=$($gate.selected_checkpoint_sha256) actual=$actualHash"
}
Write-Host "STATE: WALK; CAPABILITY: FULL; MODEL: walk_steady_state_expert_v1; TRANSITION: NONE"
Write-Host "supported range: 0.6-1.2 m/s"
Write-Host "speed: $Speed m/s; camera: world-orientation-fixed; RUN and transition experts are not loaded"

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--checkpoint", $checkpoint, "--speed", $Speed, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 2W-B WALK GUI failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
