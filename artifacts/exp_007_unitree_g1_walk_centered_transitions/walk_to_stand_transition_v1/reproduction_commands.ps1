# Exact Stage 4 reproduction commands
cd "$HOME\workspace\physical-ai-lab"
$parent = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$selected = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100\model_0.pt"
$out = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage4_walk_to_stand"
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_to_stand.ps1 -Mode baseline -Label direct_switch_baseline -Seed 20260822 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_to_stand.ps1 -Parent $parent -NumEnvs 1024 -Iterations 100 -Seed 20260824 -RunName stage4_walk_to_stand_pilot1_1024_100 -RampDuration 1.6 -ReverseWeight -2.0
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\evaluate_walk_to_stand.ps1 -Mode formal -TransitionCheckpoint $selected -Label formal_final -Seed 20260826 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_to_stand.ps1 -Speed 1.0
