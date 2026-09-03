[CmdletBinding()]
param([string]$Device = "cuda:0")
$ErrorActionPreference = "Stop"
$exp = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repo = (Resolve-Path (Join-Path $exp "..\..\..")).Path
$isaac = "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat"
$python = "$env:USERPROFILE\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$stage2q = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2q_final_sequence_integration\raw\dagger_round_2_student.pt"
$stage2n = Join-Path $repo "results\exp_012_unitree_g1_single_policy_bidirectional_locomotion\stage2n_gait_conditioned_ppo_retention_preflight\checkpoints\model_initial.pt"
$oldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$exp\src;" + (Join-Path $repo "experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src") + ";" + (Join-Path $repo "experiments\isaaclab\exp_005_unitree_g1_flat_run\src") + $(if ($oldPythonPath) { ";$oldPythonPath" } else { "" })
try {
  & (Join-Path $PSScriptRoot "run_command_audit.ps1") -Device $Device
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
  foreach ($candidate in @(@("stage2q", $stage2q), @("stage2n", $stage2n))) {
    & $isaac -p (Join-Path $PSScriptRoot "evaluate_stage0.py") --suite candidate --checkpoint $candidate[1] --tag $candidate[0] --device $Device --viz none
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
  }
  & python (Join-Path $PSScriptRoot "finalize_parent_selection.py")
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_stage0.py") --suite anchor --checkpoint $stage2q --device $Device --viz none
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
  foreach ($suite in @("translation_walk", "translation_run", "yaw", "translation_yaw", "independence", "transitions", "random")) {
    & $isaac -p (Join-Path $PSScriptRoot "evaluate_stage0.py") --suite $suite --checkpoint $stage2q --device $Device --viz none
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
  }
  & $python (Join-Path $PSScriptRoot "finalize_stage0.py")
  if ($LASTEXITCODE) { exit $LASTEXITCODE }
} finally {
  $env:PYTHONPATH = $oldPythonPath
}
