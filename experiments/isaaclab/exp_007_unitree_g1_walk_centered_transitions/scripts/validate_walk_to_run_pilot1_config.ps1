[CmdletBinding(PositionalBinding=$false)]
param()
$ErrorActionPreference="Stop"
$root=(Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac=Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
Push-Location $root
try {
  & $isaac -p (Join-Path $PSScriptRoot "validate_walk_to_run_pilot1_config.py")
  if ($LASTEXITCODE -ne 0) { throw "Frozen Stage 7R Pilot 1 config validation failed" }
} finally { Pop-Location }
