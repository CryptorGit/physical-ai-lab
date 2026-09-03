[CmdletBinding()]
param([string]$Device = "cuda:0")
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$checkpoint = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2q_final_sequence_integration\raw\dagger_round_2_student.pt"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;" + (Join-Path $repo "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src") + ";" + (Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src") + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
try {
  foreach ($case in @("vx", "vy", "yaw", "gait")) {
    & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p (Join-Path $PSScriptRoot "audit_command_case.py") --case $case --checkpoint $checkpoint --headless --device $Device
    if ($LASTEXITCODE) { throw "Command audit case '$case' failed with exit code $LASTEXITCODE" }
  }
  & python (Join-Path $PSScriptRoot "finalize_command_audit.py")
  if ($LASTEXITCODE) { throw "Command audit gate failed with exit code $LASTEXITCODE" }
} finally {
  $env:PYTHONPATH = $oldPythonPath
}
