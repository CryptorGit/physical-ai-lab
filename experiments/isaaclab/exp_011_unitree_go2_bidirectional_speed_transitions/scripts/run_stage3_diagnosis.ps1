param([string]$Device = "cuda:0", [switch]$RecaptureBatch)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$stage1 = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage1_single_policy_baseline\stage0_selected_baseline.json"
$stage2 = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage2_continuous_0_to_2_training"
$checkpoint = (Get-Content $stage1 -Raw | ConvertFrom-Json).selected.checkpoint_path
$unstable = Join-Path $stage2 "checkpoints\model_1_unstable.pt"
$output = Join-Path $repo "results\exp_011_unitree_go2_bidirectional_speed_transitions\stage3_first_update_stability_diagnosis"
$batch = Join-Path $output "initial_rollout_batch.pt"
Push-Location $repo
try {
  if ($RecaptureBatch) {
    & $isaac -p (Join-Path $PSScriptRoot "diagnose_stage3_first_update.py") `
      --checkpoint $checkpoint --unstable-checkpoint $unstable --device $Device --headless
    if ($LASTEXITCODE -ne 0) { throw "Stage 3 batch recapture failed closed." }
  }
  if (-not (Test-Path $batch)) {
    throw "Preserved batch is absent. Re-run once with -RecaptureBatch."
  }
  $python = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
  & $python (Join-Path $PSScriptRoot "analyze_stage3_fixed_batch.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 3 offline analysis failed closed." }
  & $python (Join-Path $PSScriptRoot "finalize_stage3_diagnosis.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 3 finalization failed." }
} finally {
  Pop-Location
}
