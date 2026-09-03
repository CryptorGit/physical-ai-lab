[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet("0.08", "0.09", "0.10", "0.15")] [string]$Depth = "0.09",
    [string]$Checkpoint = "artifacts/exp_006_unitree_g1_command_skills/crouch_standing_option_stage2_model4246/model_0.pt"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = "results/exp_006_unitree_g1_command_skills/crouch_shallow_scripted_v1/gui_$Depth"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $isaacLabBat -p (Join-Path $PSScriptRoot "evaluate_crouch.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path `
        --task Isaac-Motion-Flat-G1-Command-CrouchShallow-Play-v0 --episodes 1 --num-envs 1 `
        --fixed-depth ([double]$Depth) --output $output --console-status --viz kit
    if ($LASTEXITCODE -ne 0) { throw "CROUCH_SHALLOW GUI playback failed: $LASTEXITCODE" }
}
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
