param(
  [Parameter(Mandatory=$true)][int]$Seed,
  [Parameter(Mandatory=$true)][string]$Label,
  [int]$AttemptsPerSource = 30
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
Set-Location $repo
& $isaac -p `
  "$PSScriptRoot\evaluate_run_to_walk.py" `
  --headless `
  --seed $Seed `
  --attempts-per-source $AttemptsPerSource `
  --label $Label `
  --output "results/exp_007_unitree_g1_walk_centered_transitions/stage8a_run_to_walk_audit/raw" `
  --stand "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt" `
  --stand-to-walk "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt" `
  --walk "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt" `
  --run "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt" `
  --walk-to-run "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
