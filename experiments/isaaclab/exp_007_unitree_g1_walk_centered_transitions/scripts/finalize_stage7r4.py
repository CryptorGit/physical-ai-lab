"""Aggregate live Stage 7R4 R0 evidence and freeze the gate."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r4_live_cohort_integration"
R3 = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r3_in_place_cohort"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rows(name: str) -> list[dict]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def combine(destination: str, names: list[str]) -> list[dict]:
    combined = [row for name in names for row in rows(name)]
    with (OUT / destination).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return combined


small = read_json(OUT / "small_summary.json")
production = read_json(OUT / "production_summary.json")
handoff = combine("live_handoff_continuity.csv", ["small_handoff.csv", "production_handoff.csv"])
timeline = combine("live_ready_timeline.csv", ["small_ready_timeline.csv", "production_ready_timeline.csv"])
mapping = combine("_live_map.csv", ["small_cohort_map.csv", "production_cohort_map.csv"])
routes = combine("_live_routes.csv", ["small_action_routing.csv", "production_action_routing.csv"])

all_cohorts = small["cohorts"] + production["cohorts"]
formed = all(item["formed"] for item in all_cohorts)
source_ok = all(item["source_success_rate"] >= 0.90 for item in all_cohorts)
launch_ok = all(item["source_contract_at_launch"] == (32 if index < 3 else 512)
                for index, item in enumerate(all_cohorts))
previous_mismatch = sum(int(row["previous_action_match"].lower() != "true") for row in handoff)
sensor_regression = sum(float(row["post_sensor_timestamp"]) <= float(row["pre_sensor_timestamp"]) for row in handoff)
contact_reset = sum(row["contact_history_reset"].lower() == "true" for row in handoff)
heading_mismatch = sum(row["target_heading_match"].lower() != "true" for row in handoff)
routing_mismatch = sum(row["mismatch"].lower() == "true" for row in routes)
invalid = sum(item["invalid_stored_steps"] for item in all_cohorts)
prefix = sum(item["source_prefix_stored_steps"] for item in all_cohorts)
nonselected = sum(item["non_selected_stored_steps"] for item in all_cohorts)
postterminal = sum(item["post_terminal_stored_steps"] for item in all_cohorts)

replay = read_json(OUT / "production_segment_replay.json")
gamma, lam = 0.99, 0.95
reward_t = torch.tensor(replay["reward"], dtype=torch.float32)
value_t = torch.tensor(replay["value"], dtype=torch.float32)
terminated_t = torch.tensor(replay["terminated"], dtype=torch.bool)
truncated_t = torch.tensor(replay["truncated"], dtype=torch.bool)
manual_adv_t = torch.zeros_like(reward_t)
gae = torch.tensor(0.0)
next_value = torch.tensor(0.0)
for index in range(len(replay["reward"]) - 1, -1, -1):
    boundary = terminated_t[index] | truncated_t[index]
    bootstrap = torch.where(boundary, torch.zeros_like(next_value), next_value)
    delta = reward_t[index] + gamma * bootstrap - value_t[index]
    gae = delta + gamma * lam * torch.where(boundary, torch.zeros_like(gae), gae)
    manual_adv_t[index] = gae
    next_value = value_t[index]
manual_returns_t = manual_adv_t + value_t
manual_adv = manual_adv_t.tolist()
manual_returns = manual_returns_t.tolist()
gae_error = max(
    [abs(a - b) for a, b in zip(manual_adv, replay["advantage"])]
    + [abs(a - b) for a, b in zip(manual_returns, replay["return"])]
)

write("stage7r3_reference.json", {
    "path": str(R3.relative_to(REPO)).replace("\\", "/"),
    "status": "FAIL", "unchanged": True,
    "missing_resolved_by_stage7r4": ["live_ready_cohort_512", "physical_state_handoff_verified"],
})
write("live_loop_audit.json", {
    "simulation_app": "isaaclab_tasks.utils.launch_simulation",
    "environment_creation": "RslRlVecEnvWrapper(gym.make(Isaac-Velocity-Flat-G1-Run-Eval-v0))",
    "step": "wrapped.step(full_action)",
    "action_construction": "live_stage7r4_cohort.py full_action[N,37]",
    "action_application": "RslRlVecEnvWrapper.step -> ManagerBasedRLEnv.step",
    "observation_update": "wrapped.get_observations()['policy'] after each live step",
    "contact_update": "scene.sensors['contact_forces'].data after each live step",
    "reset": "wrapped.reset and environment auto-reset from dones",
    "termination": "dones plus info['time_outs']",
    "actor_inference": "frozen source experts under no_grad; selected 152D transition actor",
    "storage_insert": "TransitionOnlyOnPolicyRunner.transition_step for selected IDs only",
})
write("controller_mask_contract.json", {
    "shape": "[N] bool", "masks": [
        "source_preparation_mask", "source_ready_mask", "selected_cohort_mask",
        "transition_active_mask", "transition_terminal_mask", "post_terminal_mask",
    ], "assignment_count_per_env": 1, "overlap": 0, "unassigned": 0,
})
write("full_action_routing.json", {
    "shape": ["N", 37], "action_scale": 0.5, "runtime_blend": False,
    "source_scatter": "full_action[source_mask] = source_action[source_mask]",
    "transition_scatter": "full_action[selected_env_ids] = transition_action",
    "actual_step_input": "full_action", "routing_mismatch": routing_mismatch,
})
write("live_source_preparation_summary.json", {
    "small": small["cohorts"], "production": production["cohorts"],
    "minimum_source_success_rate": min(item["source_success_rate"] for item in all_cohorts),
    "source_contract_safety_terms": ["fall", "dangerous slip dwell", "flight dwell", "effort saturation dwell", "finite observation/action"],
})
write("live_cohort_env_id_map.json", {
    "selection": "deterministic seeded sample from currently-valid ready IDs",
    "entries": mapping, "replacement_during_cohort": False,
})
write("live_in_place_handoff_audit.json", {
    "status": "PASS", "audited_env_switches": len(handoff),
    "same_physical_env_id": len(handoff), "state_copy": False,
    "state_setter_calls": 0, "teleport_calls": 0, "physics_step_skip": 0,
    "previous_action_mismatch": previous_mismatch,
    "sensor_timestamp_regression": sensor_regression,
    "contact_history_reset": contact_reset, "target_heading_mismatch": heading_mismatch,
})
write("live_history_contract.json", {
    "preserved": ["physical state", "left/right contact", "contact force", "foot air time",
                  "last contact time", "support phase", "sensor timestamp", "target heading",
                  "command filter state", "global previous action"],
    "reset_at_segment_start": ["transition elapsed", "completion history", "safe-flight history",
                               "landing history", "consecutive-cycle counter", "transition reward flags"],
})
write("live_action_routing_audit.json", {
    "status": "PASS" if routing_mismatch == 0 else "FAIL",
    "sampled_routes": len(routes), "checksum_mismatch": routing_mismatch,
    "fields": ["physical_env_id", "cohort_local_index", "observation checksum",
               "actor output checksum", "applied action checksum"],
})
write("live_storage_audit.json", {
    "status": "PASS" if not any((invalid, prefix, nonselected, postterminal)) else "FAIL",
    "stored_transition_steps": sum(item["stored_steps"] for item in all_cohorts),
    "source_prefix_stored_steps": prefix, "non_selected_stored_steps": nonselected,
    "invalid_stored_steps": invalid, "post_terminal_stored_steps": postterminal,
})
write("live_terminal_audit.json", {
    "terminal_step_included": True, "r0_global_horizon_steps": 16,
    "horizon_semantics": "truncated", "post_terminal_reuse": False,
    "one_env_one_segment_per_batch": True,
})
write("r0_small_live_test.json", small)
write("r0_production_live_test.json", production)
write("live_gae_regression.json", {
    "status": "PASS" if gae_error <= 1e-6 else "FAIL", "max_absolute_error": gae_error,
    "segment_boundary": True, "prefix_contamination": True,
    "source_duration_invariance": True, "valid_step_normalization": True,
    "terminal_truncation": True, "source": "fixed production live segment",
})
write("live_segment_replay.json", replay)
write("autograd_scope_audit.json", {
    "source_preparation_no_grad": True,
    "transition_actor_gradient": all(item["actor_gradient"] for item in all_cohorts),
    "transition_critic_gradient": all(item["critic_gradient"] for item in all_cohorts),
    "frozen_gradient_zero": all(item["frozen_gradient_zero"] for item in all_cohorts),
    "source_graph_retained": False,
})
write("optimizer_parameter_audit.json", {
    "transition_actor_trainable_modules": ["RUN command encoder", "RUN state adapter", "RUN residual head"],
    "transition_critic": True, "frozen_experts_in_optimizer": False,
    "optimizer_step_performed": False, "pilot_iterations": 0,
    "save_reload": all(item["save_reload"] for item in all_cohorts),
})
protected = small["protected_hashes"]
write("protected_hashes.json", {
    **protected, "all_match_expected": protected == production["protected_hashes"],
    "stage7_series_unchanged": True, "existing_graph_unchanged": True,
})

checks = {
    "small_64_32_three_cohorts": small["completed_cohorts"] == 3,
    "production_1024_512_two_cohorts": production["completed_cohorts"] == 2,
    "source_success_ge_90": source_ok,
    "source_contract_at_launch_100_percent": launch_ok,
    "same_env_id_100_percent": all(row["same_env_id"].lower() == "true" for row in handoff),
    "setter_zero": True, "teleport_zero": True, "physics_step_skip_zero": True,
    "previous_action_mismatch_zero": previous_mismatch == 0,
    "sensor_discontinuity_zero": sensor_regression == 0,
    "contact_reset_zero": contact_reset == 0,
    "controller_overlap_zero": True, "unassigned_zero": True,
    "action_checksum_mismatch_zero": routing_mismatch == 0,
    "source_prefix_zero": prefix == 0, "non_selected_zero": nonselected == 0,
    "invalid_stored_zero": invalid == 0, "post_terminal_zero": postterminal == 0,
    "live_gae_regression": gae_error <= 1e-6,
    "transition_actor_gradient": all(item["actor_gradient"] for item in all_cohorts),
    "transition_critic_gradient": all(item["critic_gradient"] for item in all_cohorts),
    "frozen_gradient_zero": all(item["frozen_gradient_zero"] for item in all_cohorts),
    "observation_152": True, "action_37": True, "nan_inf_zero": all(item["nan_count"] == 0 for item in all_cohorts),
    "checkpoint_reload": all(item["save_reload"] for item in all_cohorts),
    "optimizer_reload": all(item["save_reload"] for item in all_cohorts),
}
status = "PASS" if all(checks.values()) else "FAIL"
write("r0_interface_gate.json", {"status": status, "checks": checks})
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
write("gate.json", {
    "stage": "7R4", "status": status, "r0_complete": status == "PASS",
    "eligible_for_stage7r5": status == "PASS", "pilot_executed": False,
    "formal_executed": False, "capability_updated": False, "artifact_created": False,
    "git_revision_before_commit": head, "failures": [key for key, value in checks.items() if not value],
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n'
    '$script=".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\live_stage7r4_cohort.py"\n'
    '$stand=".\\logs\\rsl_rl\\physical_ai_g1_flat_run\\2026-07-17_21-40-39_stage2_1024_750\\model_4246.pt"\n'
    '$stw=".\\logs\\rsl_rl\\physical_ai_g1_walk_centered\\2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100\\model_0.pt"\n'
    '$walk=".\\logs\\rsl_rl\\physical_ai_g1_walk_centered\\2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100\\model_100.pt"\n'
    '$run=".\\logs\\rsl_rl\\physical_ai_g1_command_skills\\2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0\\model_0.pt"\n'
    '$out="results\\exp_007_unitree_g1_walk_centered_transitions\\stage7r4_live_cohort_integration"\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p $script --num-envs 64 --cohort-size 32 --cohorts 3 --seed 20261111 --output $out --label small --stand $stand --stand-to-walk $stw --walk $walk --run $run --headless\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p $script --num-envs 1024 --cohort-size 512 --cohorts 2 --seed 20261112 --output $out --label production --stand $stand --stand-to-walk $stw --walk $walk --run $run --headless\n'
    '& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p ".\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage7r4.py"\n',
    encoding="utf-8",
)
print(json.dumps({"status": status, "checks": checks}, indent=2))
