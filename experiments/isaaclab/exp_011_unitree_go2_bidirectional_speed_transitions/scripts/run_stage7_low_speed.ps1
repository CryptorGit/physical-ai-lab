param(
  [switch]$AuditOnly,
  [switch]$TrainingOnly,
  [string]$Device = "cuda:0"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage7_low_speed_gait_stabilization"
$parent = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage4_resumed_optimizer_training\checkpoints\model_50.pt"
Push-Location $repo
try {
  & $isaac -p (Join-Path $PSScriptRoot "prepare_stage7.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 7 curriculum audit failed." }
  if ($AuditOnly) { return }
  & $isaac -p (Join-Path $PSScriptRoot "train_stage7_low_speed.py") `
    --checkpoint $parent --output $output --num-envs 2048 --iterations 200 --device $Device --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 7 training stopped fail-closed." }
  if ($TrainingOnly) { return }
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_stage7.py") `
    --mode validation --num-envs 50 --device $Device --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 7 validation failed." }
  $selection = Get-Content (Join-Path $output "selected_checkpoint.json") -Raw | ConvertFrom-Json
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_stage7.py") `
    --mode formal --num-envs 50 --checkpoint $selection.checkpoint --device $Device --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 7 formal evaluation failed." }
  & $isaac -p (Join-Path $PSScriptRoot "finalize_stage7.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 7 finalization failed." }
} finally {
  Pop-Location
}
