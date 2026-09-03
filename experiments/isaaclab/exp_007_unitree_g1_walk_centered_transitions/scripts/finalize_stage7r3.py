"""Write the Stage 7R3 fail-closed audit record."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r3_in_place_cohort"
R2 = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r2_transition_only_runner"
OUT.mkdir(parents=True, exist_ok=True)


def write(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load(name: str):
    return json.loads((R2 / name).read_text(encoding="utf-8"))


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
walk = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
run = REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

write("stage7r2_reference.json", {
    "path": str(R2.relative_to(REPO)).replace("\\", "/"),
    "status": "FAIL",
    "completed": ["transition-only storage", "GAE isolation", "gradient isolation"],
    "missing": ["live_ready_cohort_512", "physical_state_handoff_verified"],
    "unchanged": True,
})
write("cohort_design.json", {
    "method": "IN_PLACE_ENV_ID_COHORT", "physical_envs": 1024, "cohort_size": 512,
    "selection": "deterministic_seeded_random_sample", "state_copy": False,
    "one_segment_per_env_per_batch": True,
})
write("source_ready_contract.json", {
    "controller": "frozen WALK expert", "target_speed_mps": 1.2, "hold_s": 1.0,
    "speed_error_max_mps": 0.20, "heading_error_max_rad": 0.12,
    "required_false": ["fall", "dangerous_slip", "excessive_flight", "long_dwell_saturation"],
    "finite_observation_action": True, "revalidated_at_launch": True,
})
write("ready_env_management.json", {
    "fields": ["source_ready", "source_ready_since", "source_contract_duration",
               "ready_generation_id", "selected_for_cohort", "cohort_id",
               "transition_segment_id"],
    "waiting_controller": "frozen WALK expert @ 1.2 m/s",
    "freeze": False, "replacement_during_batch": False,
})
write("cohort_selection.json", {
    "seed": 20261110, "algorithm": "torch.randperm over sorted ready IDs",
    "launch_condition": "ready_count >= cohort_size", "first_ready_only": False,
})
write("in_place_handoff_contract.json", {
    "same_environment_id": True, "state_copy": False, "setter_calls": 0,
    "teleport_calls": 0, "switch_action": "last WALK action",
    "first_transition_observation_previous_action": "last applied WALK action",
    "next_control_step_transition_action": True,
})
write("in_place_handoff_audit.json", {
    "status": "NOT_VERIFIED_IN_ISAAC_SIM",
    "manager_tensor_mapping": "PASS",
    "same_env_id": "PASS_AT_MANAGER_LAYER",
    "previous_action_bitwise": "PASS_AT_MANAGER_LAYER",
    "root_joint_continuity": "NOT_MEASURED",
    "contact_sensor_continuity": "NOT_MEASURED",
    "sensor_timestamp_continuity": "NOT_MEASURED",
    "state_setter_calls": 0, "teleport_calls": 0,
})
write("source_occupancy_bias.json", {
    "reference": "Stage 7 direct-switch source occupancy",
    "ready_success": {"count": 29, "attempts": 30, "rate": 29 / 30},
    "phase_bias_status": "REFERENCE_ONLY_NOT_LIVE_R3",
})
write("source_phase_counts.json", {
    "reference_only": True, "left_support": 14, "right_support": 13,
    "double_support": 2, "flight": 0, "total_ready": 29,
})
write("ppo_storage_audit.json", {
    **load("ppo_storage_audit.json"),
    "regression_source": "Stage 7R2 deterministic audit",
    "live_r3_storage_executed": False,
})
write("gae_regression_test.json", {
    "manual_gae": load("gae_unit_test.json"),
    "prefix_contamination": load("prefix_reward_contamination_test.json"),
    "source_duration_invariance": load("source_duration_invariance_test.json"),
    "status": "PASS_REGRESSION_ONLY",
})
write("gradient_audit.json", {
    "source": "Stage 7R2", "transition_actor_gradient": True,
    "transition_critic_gradient": True, "live_r3_gradient_step": False,
})
write("frozen_parameter_audit.json", {
    "walk_sha256": sha(walk), "run_sha256": sha(run),
    "frozen_gradient_zero": True, "live_r3_hash_before_after": "UNCHANGED",
})
write("r0_interface_gate.json", {
    "status": "FAIL",
    "passes": [
        "in-place cohort manager 64/32", "in-place cohort manager 1024/512",
        "seeded physical-env ID mapping", "previous-action gather bitwise",
        "Stage 7R2 storage/GAE/gradient regression",
    ],
    "failures": [
        "64/32 test was not an Isaac Sim physics rollout",
        "1024/512 live ready cohort was not formed in Isaac Sim",
        "root/joint/contact/sensor continuity was not measured across a live switch",
        "contact history and sensor timestamp continuity remain unverified",
    ],
    "pilot_authorized": False,
})
write("training_config.json", {
    "pilot_authorized": False, "pilot_count": 0, "iterations": 0,
    "physical_envs": 1024, "cohort_size": 512,
    "reason": "R0 complete gate failed; no PPO pilot permitted",
})
write("pilot_results.json", {"executed": False, "pilot_count": 0, "reason": "R0 FAIL"})
write("checkpoint_sweep.json", {"executed": False, "selected_checkpoint": None})
write("formal_summary.json", {"executed": False, "classification": "FAIL", "reason": "R0 infrastructure gate failed"})
write("per_seed_results.json", {"executed": False, "results": []})
write("per_target_results.json", {"executed": False, "targets_mps": [2.4, 2.6, 2.8], "results": []})
for name, fields in {
    "episodes.csv": ["episode_id", "target_speed", "success"],
    "transition_timelines.csv": ["segment_id", "env_id", "step", "phase"],
}.items():
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
write("failure_counts.json", {
    "in_place_isaac_live_handoff_not_verified": 1,
    "live_ready_cohort_512_not_verified": 1,
    "pilot_not_started": 1,
})
write("gate.json", {
    "stage": "7R3", "status": "FAIL", "eligible_for_stage8": False,
    "r0_status": "FAIL", "pilot": False, "formal": False, "artifact": False,
    "capability_manifest_updated": False,
    "failures": ["live_ready_cohort_512 missing", "IN_PLACE_STATE_CONTINUITY_VERIFIED missing"],
    "git_revision_before_result_commit": head,
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p '
    '".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\audit_stage7r3_cohort.py"\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p '
    '".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage7r3.py"\n',
    encoding="utf-8",
)
print(json.dumps({"status": "FAIL", "pilot": False, "walk_sha": sha(walk), "run_sha": sha(run)}))
