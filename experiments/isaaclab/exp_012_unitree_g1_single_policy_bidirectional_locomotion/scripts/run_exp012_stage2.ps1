[CmdletBinding()]
param([switch]$WiringOnly)
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$exp005 = Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$checkpoint = Join-Path $repo "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$old = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;$exp005" + $(if ($old) { ";$old" } else { "" })
try {
  $mode = if ($WiringOnly) { "wiring" } else { "pilot" }
  $output = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2_pilot1_run"
  & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p (Join-Path $PSScriptRoot "train_stage2.py") --mode $mode --checkpoint $checkpoint --output $output --headless
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally { $env:PYTHONPATH = $old }
