[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("smoke", "baseline", "pilot", "formal", "all")][string]$Mode = "all",
    [int]$Seed = 20260724
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$evaluator = Join-Path $PSScriptRoot "evaluate_stand_walk.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\stage2_stand_walk"
foreach ($required in @($checkpoint, $evaluator, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
$hash = (Get-FileHash $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $manifest.experts.WALK_STAND.sha256) { throw "Checkpoint SHA mismatch: $hash" }
$modes = if ($Mode -eq "all") { @("smoke", "baseline", "pilot", "formal") } else { @($Mode) }
Push-Location $repositoryRoot
try {
    foreach ($current in $modes) {
        if ($current -eq "formal" -and -not (Test-Path (Join-Path $output "selected_controller.json"))) {
            throw "Formal controller is not frozen; run -Mode pilot first."
        }
        Write-Host "Stage 2 STAND-WALK mode=$current checkpoint=$checkpoint seed=$Seed"
        & $isaacLabBat -p $evaluator --checkpoint $checkpoint --mode $current --seed $Seed --output $output --viz none
        if ($LASTEXITCODE -ne 0) { throw "Stage 2 $current failed: $LASTEXITCODE" }
    }
    @'
# Abrupt diagnostic, pilot freeze, and formal evaluation
cd "$HOME\workspace\physical-ai-lab"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand_walk.ps1 -Mode all
'@ | Set-Content -LiteralPath (Join-Path $output "reproduction_commands.ps1") -Encoding utf8
}
finally { Pop-Location }
