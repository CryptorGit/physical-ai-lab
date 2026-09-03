[CmdletBinding(PositionalBinding=$false)]
param()
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
Push-Location $root
try {
  & $isaac -p (Join-Path $PSScriptRoot "validate_run_to_walk_pilot1_config.py")
  if ($LASTEXITCODE -ne 0) { throw "RUN_TO_WALK Pilot 1 frozen config validation failed" }
}
finally {
  Pop-Location
}
