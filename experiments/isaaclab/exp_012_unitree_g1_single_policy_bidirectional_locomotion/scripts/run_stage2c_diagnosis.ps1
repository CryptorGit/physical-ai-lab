[CmdletBinding()]
param(
    [int]$WarmupSteps = 360,
    [switch]$Headless = $true
)
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$exp005 = Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;$exp005" + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
try {
    $arguments = @(
        "-p",
        (Join-Path $PSScriptRoot "diagnose_stage2c_gradients.py"),
        "--warmup-steps", "$WarmupSteps"
    )
    if ($Headless) { $arguments += "--headless" }
    & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" @arguments
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
