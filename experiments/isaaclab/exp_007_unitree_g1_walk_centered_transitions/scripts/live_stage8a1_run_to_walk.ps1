[CmdletBinding(PositionalBinding=$false)]
param(
  [Parameter(Mandatory=$true)][int]$NumEnvs,
  [Parameter(Mandatory=$true)][int]$CohortSize,
  [Parameter(Mandatory=$true)][int]$Cohorts,
  [Parameter(Mandatory=$true)][int]$Seed,
  [Parameter(Mandatory=$true)][string]$Label
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$arguments = @(
  "-p", (Join-Path $PSScriptRoot "live_stage8a1_run_to_walk.py"),
  "--headless",
  "--num-envs", $NumEnvs,
  "--cohort-size", $CohortSize,
  "--cohorts", $Cohorts,
  "--seed", $Seed,
  "--label", $Label,
  "--output", "results/exp_007_unitree_g1_walk_centered_transitions/stage8a1_run_to_walk_live_handoff/raw",
  "--stand", (Join-Path $root "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"),
  "--stand-to-walk", (Join-Path $root "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt"),
  "--walk", (Join-Path $root "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"),
  "--run", (Join-Path $root "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"),
  "--walk-to-run", (Join-Path $root "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt")
)
Push-Location $root
try {
  & $isaac @arguments
  if ($LASTEXITCODE -ne 0) { throw "Stage 8A1 live cohort failed" }
}
finally {
  Pop-Location
}
