param([switch]$SkipCollection, [switch]$SkipTraining)
$ErrorActionPreference = "Stop"
$exp = Split-Path $PSScriptRoot -Parent
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = "$HOME\workspace\IsaacLab\isaaclab.bat"
if (-not $SkipCollection) {
  & $isaac -p (Join-Path $PSScriptRoot "collect_teacher_dataset.py") --headless
}
if (-not $SkipTraining) {
  & $isaac -p (Join-Path $PSScriptRoot "train_unified_student.py")
}
& $isaac -p (Join-Path $PSScriptRoot "evaluate_unified_student.py") --headless
# The frozen Stage-0 BC result fails retention, so the protocol permits exactly
# one DAgger round. RUN_TO_WALK states are excluded by the evaluator.
$result = Join-Path $repo "results\exp_009_unitree_g1_unified_walk_run_student\stage0_multiteacher_distillation"
$epoch1 = Join-Path $result "checkpoints\epoch_1.pt"
& $isaac -p (Join-Path $PSScriptRoot "evaluate_unified_student.py") --headless --checkpoint $epoch1 --collect-dagger
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" (Join-Path $PSScriptRoot "train_dagger_round1.py")
$dagger = Join-Path $result "checkpoints\dagger_round1.pt"
& $isaac -p (Join-Path $PSScriptRoot "evaluate_unified_student.py") --headless --checkpoint $dagger --append
& "$HOME\workspace\IsaacLab\env_isaaclab\Scripts\python.exe" (Join-Path $PSScriptRoot "finalize_stage0.py")
