[CmdletBinding(PositionalBinding=$false)]
param([switch]$ValidateOnly)
$ErrorActionPreference="Stop"
& (Join-Path $PSScriptRoot "validate_walk_to_run_pilot1_config.ps1")
if ($LASTEXITCODE -ne 0) { throw "Pilot launcher refused invalid frozen config" }
if ($ValidateOnly) { exit 0 }
$root=(Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac=Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$out="results/exp_007_unitree_g1_walk_centered_transitions/stage7r7_frozen_pilot1_execution"
$checkpointDir=Join-Path $root "$out/checkpoints"
$common=@(
  "--mode","baseline","--seed","20261122","--episodes-per-target","22",
  "--output","$out/evaluations",
  "--stand","logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
  "--stand-to-walk","logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
  "--walk","logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
  "--run","logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
)
Push-Location $root
try {
  if (-not (Test-Path (Join-Path $checkpointDir "initial.pt"))) {
    & $isaac -p (Join-Path $PSScriptRoot "execute_stage7r7_pilot1.py") --phase prepare
    if ($LASTEXITCODE -ne 0) { throw "Frozen Pilot 1 preparation failed" }
  }
  & $isaac -p (Join-Path $PSScriptRoot "evaluate_walk_to_run.py") @common --label initial --transition-checkpoint "$out/checkpoints/initial.pt"
  if ($LASTEXITCODE -ne 0) { throw "Initial baseline failed" }
  & $isaac -p (Join-Path $PSScriptRoot "execute_stage7r7_pilot1.py") --phase train
  if ($LASTEXITCODE -ne 0) { throw "Frozen Pilot 1 training failed" }
  $checkpoints=@(
    @("first_post_update","first_post_update.pt"),@("model_10","model_10.pt"),
    @("model_25","model_25.pt"),@("model_50","model_50.pt"),
    @("model_75","model_75.pt"),@("model_100","model_100.pt")
  )
  foreach($item in $checkpoints) {
    & $isaac -p (Join-Path $PSScriptRoot "evaluate_walk_to_run.py") @common --label $item[0] --transition-checkpoint "$out/checkpoints/$($item[1])"
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation failed: $($item[0])" }
  }
  & $isaac -p (Join-Path $PSScriptRoot "finalize_stage7r7.py")
  if ($LASTEXITCODE -ne 0) { throw "Stage 7R7 finalization failed" }
} finally { Pop-Location }
