[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("candidates", "sweep", "compare", "transition", "all")][string]$Mode = "all",
    [int]$Seed = 20260725
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$manifest = Get-Content (Join-Path $experimentRoot "expert_manifest.json") -Raw | ConvertFrom-Json
$checkpoint = Join-Path $repositoryRoot $manifest.experts.WALK_STAND.checkpoint
$evaluator = Join-Path $PSScriptRoot "audit_walk_operating_envelope.py"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repositoryRoot "results\exp_007_unitree_g1_walk_centered_transitions\stage2b_walk_operating_envelope"
foreach ($required in @($checkpoint, $evaluator, $isaacLabBat)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required file missing: $required" }
}
$hash = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $manifest.experts.WALK_STAND.sha256) { throw "Checkpoint SHA mismatch: $hash" }
$modes = if ($Mode -eq "all") { @("candidates", "sweep", "compare", "transition") } else { @($Mode) }
Push-Location $repositoryRoot
try {
    foreach ($current in $modes) {
        if ($current -ne "candidates" -and -not (Test-Path (Join-Path $output "selected_heading_controller.json"))) {
            throw "Heading controller is not frozen; run -Mode candidates first."
        }
        if ($current -eq "compare") {
            foreach ($headingMode in @("ZeroYaw", "FixedTarget")) {
                Write-Host "Stage 2B mode=compare heading=$headingMode seed=$Seed"
                & $isaacLabBat -p $evaluator --checkpoint $checkpoint --output $output --mode compare --heading-mode $headingMode --seed $Seed --viz none
                if ($LASTEXITCODE -ne 0) { throw "Stage 2B compare/$headingMode failed: $LASTEXITCODE" }
            }
        }
        else {
            Write-Host "Stage 2B mode=$current heading=FixedTarget seed=$Seed"
            & $isaacLabBat -p $evaluator --checkpoint $checkpoint --output $output --mode $current --heading-mode FixedTarget --seed $Seed --viz none
            if ($LASTEXITCODE -ne 0) { throw "Stage 2B $current failed: $LASTEXITCODE" }
        }
    }
    @'
cd "$HOME\workspace\physical-ai-lab"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_operating_envelope.ps1 -Mode all -Seed 20260725
'@ | Set-Content -LiteralPath (Join-Path $output "reproduction_commands.ps1") -Encoding utf8
}
finally { Pop-Location }
