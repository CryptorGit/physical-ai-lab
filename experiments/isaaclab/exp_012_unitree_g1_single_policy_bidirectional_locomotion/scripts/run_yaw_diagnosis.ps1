[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$exp005 = Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$checkpoint = Join-Path $repo "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$old = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;$exp005" + $(if ($old) { ";$old" } else { "" })
$env:PYTHONUTF8 = "1"
try {
  & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p (Join-Path $PSScriptRoot "diagnose_yaw_controllability.py") --checkpoint $checkpoint --headless
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { $env:PYTHONPATH = $old }
