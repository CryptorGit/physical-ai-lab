[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Checkpoint = "artifacts/exp_006_unitree_g1_command_skills/crouch_standing_option_stage2_model4246/model_0.pt",
    [string]$Output = "results/exp_006_unitree_g1_command_skills/crouch_shallow_scripted_v1/formal_50",
    [int]$Seed = 20260722,
    [switch]$SkipUnsupported
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$srcRoot = Join-Path $experimentRoot "src"
$flatRunSrc = Join-Path $repositoryRoot "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$srcRoot;$flatRunSrc" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
Push-Location $repositoryRoot
try {
    & $isaacLabBat -p (Join-Path $PSScriptRoot "evaluate_crouch.py") --checkpoint $checkpointPath `
        --task Isaac-Motion-Flat-G1-Command-CrouchShallow-Eval-v0 --episodes 50 --num-envs 50 `
        --seed $Seed --output (Join-Path $Output "supported") --viz none
    if ($LASTEXITCODE -ne 0) { throw "Formal CROUCH_SHALLOW evaluation failed: $LASTEXITCODE" }
    if (-not $SkipUnsupported) {
        $depths = ((@("0.11") * 5) + (@("0.12") * 5) + (@("0.15") * 5)) -join ","
        & $isaacLabBat -p (Join-Path $PSScriptRoot "evaluate_crouch.py") --checkpoint $checkpointPath `
            --task Isaac-Motion-Flat-G1-Command-CrouchShallow-Eval-v0 --episodes 15 --num-envs 15 `
            --seed ($Seed + 110) --fixed-depths $depths --output (Join-Path $Output "unsupported") --viz none
        if ($LASTEXITCODE -ne 0) { throw "Unsupported-depth evaluation failed: $LASTEXITCODE" }
    }
}
finally { Pop-Location; $env:PYTHONPATH = $previousPythonPath }
