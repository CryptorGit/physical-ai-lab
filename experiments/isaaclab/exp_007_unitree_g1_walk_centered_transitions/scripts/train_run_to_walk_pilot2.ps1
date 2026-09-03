param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path (Split-Path $root -Parent) "IsaacLab\isaaclab.bat"
$gate = Join-Path $root "results\exp_007_unitree_g1_walk_centered_transitions\stage8d_run_to_walk_pilot2_walk_acquisition\gate.json"

if (Test-Path -LiteralPath $gate) {
  throw "Stage 8D single-run authorization is already consumed: $gate"
}

Push-Location $root
try {
  & $isaac -p (Join-Path $PSScriptRoot "execute_stage8d_pilot2.py")
  if ($LASTEXITCODE -ne 0) {
    throw "Stage 8D Pilot 2 failed with exit code $LASTEXITCODE"
  }
  & (Join-Path (Split-Path $isaac -Parent) "env_isaaclab\Scripts\python.exe") `
    (Join-Path $PSScriptRoot "finalize_stage8d.py")
  if ($LASTEXITCODE -ne 0) {
    throw "Stage 8D finalization failed with exit code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}
