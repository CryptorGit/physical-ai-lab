[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$exp005 = Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src"
$checkpoint = Join-Path $repo "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$collector = Join-Path $PSScriptRoot "collect_yaw_cancellation_preflight.py"
$old = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;$exp005" + $(if ($old) { ";$old" } else { "" })
$env:PYTHONUTF8 = "1"
try {
  foreach ($mode in @("steady", "transition")) {
    foreach ($controller in @("off", "on")) {
      & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p $collector `
        --mode $mode --controller $controller --seed 20261201 --checkpoint $checkpoint --headless
      if ($LASTEXITCODE) { exit $LASTEXITCODE }
    }
  }
  python (Join-Path $PSScriptRoot "analyze_yaw_cancellation_preflight.py")
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
  $env:PYTHONPATH = $old
}
