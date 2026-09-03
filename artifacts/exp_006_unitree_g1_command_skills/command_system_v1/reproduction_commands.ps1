$s = ".\experiments\isaaclab\exp_006_unitree_g1_command_skills\scripts"
$stage2 = ".\logs\rsl_rl\physical_ai_g1_flat_run\2026-07-17_21-40-39_stage2_1024_750\model_4246.pt"
$model31 = ".\logs\rsl_rl\physical_ai_g1_command_skills\2026-07-20_14-34-35_pilot_stop_stage_a_braking\model_31.pt"
$root = ".\results\exp_006_unitree_g1_command_skills\command_system_v1"
$isaac = "$env:USERPROFILE\workspace\IsaacLab\isaaclab.bat"

& $isaac -p "$s\evaluate_standing_base.py" --candidate stage2_model4246 --checkpoint $stage2 --episodes 50 --seed 20260723 --stand-hold-s 8 --output "$root\stand_formal_50"
& $isaac -p "$s\evaluate_run_turn_run.py" --checkpoint $model31 --episodes 50 --seed 20260723 --run-duration-min 2.5 --run-duration-max 3.0 --recovery-duration-min 2.5 --recovery-duration-max 3.2 --output "$root\run_turn_run_formal_50"
& $isaac -p "$s\audit_command_router.py" --output "$root\unsupported_requests"
& python "$s\build_command_system_artifact.py"

# The STAND-CROUCH-STAND aggregate reuses the verified formal 50-episode
# scripted_shallow_v1 result at results/.../crouch_shallow_scripted_v1/formal_50.
# GUI: & "$s\play_command_system.ps1" -Demo RUN_TURN_RUN
