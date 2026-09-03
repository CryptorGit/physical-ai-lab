# Exact Stage 2W-B reproduction commands
cd "$HOME\workspace\physical-ai-lab"
$parent = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_22-32-29_stage2w_independent_walk_pilot1_1024_150\model_150.pt"
$pilot1 = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-12-44_stage2wb_stabilization_pilot1_valid_1024_100"
$pilot2 = ".\logs\rsl_rl\physical_ai_g1_walk_centered\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100"
$out = ".\results\exp_007_unitree_g1_walk_centered_transitions\stage2wb_walk_stabilization"

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode timeline -Label stage2w_failure_replay -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260731 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_zero_yaw -HeadingMode ZeroYaw -Seed 20260804 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_current -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260804 -Output $out
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint $parent -Mode controller -Label controller_lower_bandwidth -HeadingMode FixedTarget -KHeading 0.5 -KYawRate 0.10 -YawRateLimit 0.25 -LowPassAlpha 0.08 -SlewLimit 0.005 -Seed 20260804 -Output $out

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_stabilization.ps1 -Checkpoint $parent -Iterations 100 -Seed 20260805 -RunName stage2wb_stabilization_pilot1_valid_1024_100 -YawOscillationWeight -0.02
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\train_walk_stabilization.ps1 -Checkpoint "$pilot1\model_100.pt" -Iterations 100 -Seed 20260807 -RunName stage2wb_stabilization_pilot2_1024_100 -YawOscillationWeight -0.05
.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\audit_walk_stabilization.ps1 -Checkpoint "$pilot2\model_100.pt" -Mode formal-full -Label full_range_formal_candidate -HeadingMode FixedTarget -KHeading 0.8 -KYawRate 0.10 -YawRateLimit 0.30 -LowPassAlpha 0.15 -SlewLimit 0.01 -Seed 20260809 -Output $out

.\experiments\isaaclab\exp_007_unitree_g1_walk_centered_transitions\scripts\play_walk_steady_state.ps1 -Speed 1.0
