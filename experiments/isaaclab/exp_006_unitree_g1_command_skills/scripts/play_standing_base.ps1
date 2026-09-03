[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [string]$Candidate = "standing_candidate",
    [int]$Seed = 42,
    [double]$StandHoldSeconds = 8.0
)

$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repositoryRoot "results\exp_006_unitree_g1_command_skills\standing_base_gui\$Candidate"

Push-Location $repositoryRoot
try {
    & $isaacLabBat -p (Join-Path $PSScriptRoot "evaluate_standing_base.py") `
        --candidate $Candidate --checkpoint $Checkpoint --episodes 1 --seed $Seed `
        --stand-hold-s $StandHoldSeconds --output $output --viz kit
    if ($LASTEXITCODE -ne 0) { throw "Standing-base GUI replay failed with exit code $LASTEXITCODE" }
}
finally { Pop-Location }
