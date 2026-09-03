[CmdletBinding()]
param(
  [string]$Device = "cuda:0",
  [int]$Seed = 20261399,
  [string]$Checkpoint = ""
)
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
if (-not $Checkpoint) {
  $Checkpoint = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2q_final_sequence_integration\raw\dagger_round_2_student.pt"
}
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;" + (Join-Path $repo "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src") + ";" + (Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src") + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
try {
  & "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat" -p (Join-Path $PSScriptRoot "play_directional_baseline.py") --checkpoint $Checkpoint --seed $Seed --device $Device --viz full
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
  $env:PYTHONPATH = $oldPythonPath
}
