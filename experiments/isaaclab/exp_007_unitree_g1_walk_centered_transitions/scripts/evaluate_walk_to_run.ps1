[CmdletBinding(PositionalBinding=$false)]
param([ValidateSet("preflight","baseline","formal")][string]$Mode,[int]$Seed,[int]$EpisodesPerTarget=10,[string]$Label,[string]$Output="results\exp_007_unitree_g1_walk_centered_transitions\stage7_walk_to_run")
$ErrorActionPreference="Stop";$root=(Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path;$isaac=Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$a=@("-p",(Join-Path $PSScriptRoot "evaluate_walk_to_run.py"),"--mode",$Mode,"--seed",$Seed,"--episodes-per-target",$EpisodesPerTarget,"--label",$Label,"--output",$Output,
"--stand",(Join-Path $root "logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"),
"--stand-to-walk",(Join-Path $root "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\model_0.pt"),
"--walk",(Join-Path $root "logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"),
"--run",(Join-Path $root "logs\rsl_rl\physical_ai_g1_command_skills\2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0\model_0.pt"),"--headless")
Push-Location $root;try{&$isaac @a;if($LASTEXITCODE-ne 0){throw "Stage7 evaluation failed"}}finally{Pop-Location}
