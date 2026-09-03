[CmdletBinding(PositionalBinding = $false)]
param()
$ErrorActionPreference = "Stop"
$experimentRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repositoryRoot = (Resolve-Path (Join-Path $experimentRoot "..\..\..")).Path
$isaacLabBat = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
if (-not (Test-Path $isaacLabBat)) { throw "Isaac Lab launcher missing: $isaacLabBat" }
Push-Location $repositoryRoot
try {
    Write-Host "Stage 0 expert audit; no simulation and no training"
    & $isaacLabBat -p (Join-Path $PSScriptRoot "audit_experts.py")
    if ($LASTEXITCODE -ne 0) { throw "Stage 0 expert audit failed: $LASTEXITCODE" }
}
finally { Pop-Location }
