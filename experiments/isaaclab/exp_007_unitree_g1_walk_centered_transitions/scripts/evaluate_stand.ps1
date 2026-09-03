[CmdletBinding(PositionalBinding = $false)]
param(
    [int]$Episodes = 50,
    [int]$Seed = 20260723,
    [double]$StandHoldSeconds = 8.0,
    [ValidateSet("formal", "smoke")][string]$RunLabel = "formal",
    [string]$Output = ""
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$evaluator = Join-Path $PSScriptRoot "evaluate_stand.py"
$audit = Join-Path $PSScriptRoot "audit_experts.ps1"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
foreach ($required in @($checkpoint, $evaluator, $audit, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
$actualHash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $manifest.experts.WALK_STAND.sha256) {
    throw "Checkpoint SHA-256 mismatch: expected=$($manifest.experts.WALK_STAND.sha256) actual=$actualHash"
}
if (-not $Output) {
    $leaf = if ($RunLabel -eq "formal") { "stage1_stand_formal" } else { "stage1_stand_smoke" }
    $Output = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\$leaf"
}
Write-Host "exp_007 Stage 1 STAND $RunLabel evaluation"
Write-Host "checkpoint: $checkpoint"
Write-Host "episodes: $Episodes seed: $Seed settle: <=2.0s/0.4s hold: $StandHoldSeconds s"
Write-Host "command: vx=0 vy=0 yaw_rate=0; RUN/bridge/scripted contributions=0"
Push-Location $repositoryRoot
try {
    & $audit
    if ($LASTEXITCODE -ne 0) { throw "Stage 0 reference audit failed: $LASTEXITCODE" }
    & $isaacLabBat -p $evaluator --checkpoint $checkpoint --episodes $Episodes --seed $Seed `
        --settle-timeout-s 2.0 --settle-hold-s 0.4 --stand-hold-s $StandHoldSeconds `
        --run-label $RunLabel --output $Output --viz none
    if ($LASTEXITCODE -ne 0) { throw "Stage 1 STAND evaluation failed: $LASTEXITCODE" }
}
finally { Pop-Location }
