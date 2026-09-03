[CmdletBinding(PositionalBinding=$false)]
param([switch]$ValidateOnly)
$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "validate_run_to_walk_pilot1_config.ps1")
if ($LASTEXITCODE -ne 0) { throw "Pilot launcher refused invalid frozen config" }
if ($ValidateOnly) { exit 0 }
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$completedGate = Join-Path $root "results\exp_007_unitree_g1_walk_centered_transitions\stage8c_run_to_walk_pilot1_execution\gate.json"
if (Test-Path $completedGate) {
  throw "STAGE8C_SINGLE_RUN_ALREADY_COMPLETED: execution authorization is re-locked."
}
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
Push-Location $root
try {
  & $isaac -p (Join-Path $PSScriptRoot "execute_stage8c_pilot1.py")
  if ($LASTEXITCODE -ne 0) { throw "Authorized Stage 8C RUN_TO_WALK Pilot 1 failed" }
}
finally {
  Pop-Location
}
