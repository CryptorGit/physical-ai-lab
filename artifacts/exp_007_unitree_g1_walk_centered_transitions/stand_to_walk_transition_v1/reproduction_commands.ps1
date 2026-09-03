# Exact Stage 3 reproduction commands
cd "$HOME\workspace\physical-ai-lab"
$stand = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$walk = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\model_100.pt"
$selected = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\model_0.pt"
$out = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage3_stand_to_walk"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand_to_walk.ps1 -Mode baseline -Label hard_switch_baseline -Seed 20260812 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_stand_to_walk.ps1 -Parent $walk -NumEnvs 1024 -Iterations 100 -Seed 20260813 -RunName stage3_stand_to_walk_pilot1_validrun_1024_100
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_stand_to_walk.ps1 -Parent $walk -NumEnvs 1024 -Iterations 100 -Seed 20260815 -RunName stage3_stand_to_walk_pilot2_source_align_1024_100 -SourceAlignmentWeight -0.20
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_stand_to_walk.ps1 -Mode formal -TransitionCheckpoint $selected -Label formal_final -Seed 20260817 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_stand_to_walk.ps1 -Speed 1.0
