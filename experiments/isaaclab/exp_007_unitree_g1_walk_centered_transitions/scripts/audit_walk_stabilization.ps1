[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][string]$Checkpoint,
    [ValidateSet("timeline", "controller", "formal-full", "formal-low")][string]$Mode,
    [Parameter(Mandatory = $true)][string]$Label,
    [ValidateSet("ZeroYaw", "FixedTarget")][string]$HeadingMode,
    [double]$KHeading = 0.8,
    [double]$KYawRate = 0.10,
    [double]$YawRateLimit = 0.30,
    [double]$LowPassAlpha = 0.15,
    [double]$SlewLimit = 0.01,
    [int]$Seed,
    [string]$Output = "results\exp_007_unitree_g1_walk_centered_transitions\stage2wb_walk_stabilization"
)
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$experimentRoot\src;$repositoryRoot\experiments\isaaclab\exp_005_unitree_g1_flat_run\src" + $(if ($previousPythonPath) { ";$previousPythonPath" } else { "" })
Push-Location $repositoryRoot
try {
    & $launcher -p (Join-Path $PSScriptRoot "audit_walk_stabilization.py") `
        --checkpoint (Resolve-Path -LiteralPath $Checkpoint).Path --mode $Mode --label $Label `
        --heading-mode $HeadingMode --k-heading $KHeading --k-yaw-rate $KYawRate `
        --yaw-rate-limit $YawRateLimit --low-pass-alpha $LowPassAlpha --slew-limit $SlewLimit `
        --seed $Seed --output $Output --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 2W-B audit failed: $LASTEXITCODE" }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
