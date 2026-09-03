[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet(0.6, 0.8, 1.0, 1.2)]
    [double]$Speed = 1.0,

    [int]$Seed = 20260726,

    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$gatePath = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\stage2r_unified_stand_walk\gate.json"
$gate = Get-Content -LiteralPath $gatePath -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $gate.best_diagnostic_checkpoint
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$player = Join-Path $PSScriptRoot "play_unified_stand_walk.py"

foreach ($required in @($gatePath, $checkpoint, $launcher, $player)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file missing: $required"
    }
}
$actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $gate.best_diagnostic_checkpoint_sha256) {
    throw "Diagnostic checkpoint SHA mismatch: expected=$($gate.best_diagnostic_checkpoint_sha256) actual=$actualHash"
}
Write-Warning "Stage 2R is NO_GO_RETRAIN. This is diagnostic GUI playback, not a supported skill or formal transition."
Write-Host "active expert: unified Stage 2R diagnostic only; RUN expert and transition bridge are not loaded"
Write-Host "speed: $Speed m/s; camera: world-orientation-fixed; controller: smoothed fixed heading"

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    $arguments = @("-p", $player, "--checkpoint", $checkpoint, "--speed", $Speed, "--seed", $Seed, "--viz", "kit")
    if ($ValidateOnly) { $arguments += "--validate-only" }
    & $launcher @arguments
    if ($LASTEXITCODE -ne 0) { throw "Stage 2R diagnostic GUI failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
