param([switch]$TrainingOnly, [string]$Device = "cuda:0")
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stage1 = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage1_single_policy_baseline\stage0_selected_baseline.json"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage4_resumed_optimizer_training"
$checkpoint = (Get-Content $stage1 -Raw | ConvertFrom-Json).selected.checkpoint_path
Push-Location $repo
try {
  & $isaac -p (Join-Path $PSScriptRoot "train_stage4_resumed_optimizer.py") `
    --checkpoint $checkpoint --output $output --device $Device --headless
  if ($LASTEXITCODE -ne 0) { throw "Stage 4 training stopped fail-closed." }
  if ($TrainingOnly) { return }
  Write-Host "Training complete. Run validation and formal evaluation with the selected checkpoint."
} finally {
  Pop-Location
}
