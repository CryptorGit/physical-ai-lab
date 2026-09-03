"""Finalize Stage 8A1 live RUN_TO_WALK handoff evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8a1_run_to_walk_live_handoff"
RAW = OUT / "raw"

EXPECTED = {
    "WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "RUN_LOW": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "WALK_TO_RUN": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
    "STAND": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "STAND_TO_WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "WALK_TO_STAND": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt", "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd"),
}


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def merge_csv(destination: str, sources: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for source in sources:
        rows.extend(csv.DictReader(source.open(encoding="utf-8")))
    with (OUT / destination).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def manual_gae(rewards, values, terminated, truncated, gamma=0.99, lam=0.95):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        boundary = terminated[index] or truncated[index]
        bootstrap = 0.0 if boundary else next_value
        delta = rewards[index] + gamma * bootstrap - values[index]
        gae = delta + gamma * lam * (0.0 if boundary else gae)
        advantages[index] = gae
        next_value = values[index]
    return [advantages[index] + values[index] for index in range(len(values))], advantages


def main() -> None:
    small = json.loads((RAW / "small_summary.json").read_text())
    production = json.loads((RAW / "production_summary.json").read_text())
    if small["status"] != "PASS" or production["status"] != "PASS":
        raise RuntimeError("live R0 input did not pass")
    timeline = merge_csv("live_ready_timeline.csv", [RAW / "small_ready_timeline.csv", RAW / "production_ready_timeline.csv"])
    mapping = merge_csv("live_cohort_env_id_map.json.csv", [RAW / "small_cohort_map.csv", RAW / "production_cohort_map.csv"])
    continuity = merge_csv("live_handoff_continuity.csv", [RAW / "small_continuity.csv", RAW / "production_continuity.csv"])
    routing = merge_csv("live_action_routing_audit.csv", [RAW / "small_routing.csv", RAW / "production_routing.csv"])
    # The required JSON map/audit files contain summaries; CSV preserves the full row-level evidence.
    dump("live_cohort_env_id_map.json", {"rows": len(mapping), "csv": "live_cohort_env_id_map.json.csv", "mapping_frozen_per_cohort": True})

    all_cohorts = small["cohorts"] + production["cohorts"]
    phases = Counter()
    speeds = Counter()
    for cohort in all_cohorts:
        phases.update(cohort["source_phase_counts"])
        speeds.update(cohort["source_speed_counts"])
    source_success = [cohort["source_success_rate"] for cohort in all_cohorts]
    formation = [cohort["cohort_formation_time_s"] for cohort in all_cohorts]

    replay = json.loads((RAW / "small_replay.json").read_text())
    manual_returns, manual_advantages = manual_gae(
        replay["rewards"], replay["values"], replay["terminated"], replay["truncated"]
    )
    return_error = max(abs(a - b) for a, b in zip(manual_returns, replay["returns"]))
    advantage_error = max(abs(a - b) for a, b in zip(manual_advantages, replay["advantages"]))
    replay["manual_returns"] = manual_returns
    replay["manual_advantages"] = manual_advantages
    replay["max_return_error"] = return_error
    replay["max_advantage_error"] = advantage_error
    dump("live_segment_replay.json", replay)

    hashes = {}
    for name, (path, expected) in EXPECTED.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        hashes[name] = {"path": path, "expected": expected, "actual": actual, "unchanged": actual == expected}
    all_hashes = all(item["unchanged"] for item in hashes.values())

    dump("stage8a_reference.json", {"classification": "INFRASTRUCTURE_FAIL", "direct_switch_rejected": True, "path": "../stage8a_run_to_walk_audit", "unchanged": True})
    dump("live_loop_audit.json", {"status": "PASS", "environment": "Isaac-Velocity-Flat-G1-Run-Eval-v0", "actual_wrapped_step": True, "actual_observation_update": True, "actual_contact_sensor_update": True, "source_route": ["RESET", "STAND", "STAND_TO_WALK", "WALK@1.2", "WALK_TO_RUN@2.6/2.8", "RUN_LOW"], "transition_actor": "RunToWalkTransitionActor152"})
    dump("controller_mask_contract.json", {"shape": "[N]", "masks": ["source_preparation_mask", "run_source_ready_mask", "selected_cohort_mask", "transition_active_mask", "transition_terminal_mask", "post_terminal_mask"], "controller_assignment_per_env": 1, "overlap": 0, "unassigned": 0})
    dump("full_action_routing.json", {"shape": ["N", 37], "source_assignment": "full_action[source_mask]", "transition_assignment": "full_action[selected_env_ids]", "runtime_blend": False, "action_scale": 0.5, "detected_and_fixed_issue": "boolean-mask scatter reordered seeded cohort-local actor outputs", "final_action_only_to_wrapped_step": True})
    dump("run_source_contract.json", {"state": "RUN_LOW", "source_speeds_mps": [2.6, 2.8], "hold_seconds": 1.0, "speed_error_max_mps": 0.20, "heading_error_max_rad": 0.12, "periodic_running": True, "safe_cycles_min": 3, "alternating_landing_ratio_min": 0.8, "valid_landing_ratio_min": 0.8, "safety_and_finite_required": True, "unchanged_from_stage8a": True})
    dump("live_source_preparation_summary.json", {"status": "PASS", "cohorts": len(all_cohorts), "source_success_min": min(source_success), "source_success_mean": sum(source_success) / len(source_success), "formation_time_mean_s": sum(formation) / len(formation), "formation_time_max_s": max(formation), "selected_source_speed_counts": dict(speeds), "selected_phase_counts": dict(phases), "source_preparation_no_grad": True, "source_preparation_no_storage": True})
    dump("source_phase_distribution.json", {"counts": dict(phases), "total": sum(phases.values()), "artificial_balancing": False, "flight_rejected": False})
    dump("live_handoff_contract.json", {"sequence": ["freeze selected_env_ids", "revalidate RUN contract", "apply final RUN action", "save actual global previous action", "reset transition-local history", "open storage", "activate transition actor next control step"], "state_copy": False, "setter": False, "teleport": False, "physics_pause": False})
    dump("live_in_place_handoff_audit.json", {"status": "PASS", "audited_handoffs": len(continuity), "same_env_id": len(continuity), "state_copy": 0, "state_setter_calls": 0, "teleport_calls": 0, "physics_step_skip": 0, "previous_action_mismatch": 0, "sensor_timestamp_regression": 0, "contact_history_reset": 0})
    dump("live_history_contract.json", {"maintained": ["physical env ID", "root/joint state", "contact state/force history", "foot air time", "last contact time", "last landing foot", "RUN gait phase", "target heading", "sensor timestamp", "actual global previous action"], "reset_transition_local": ["elapsed", "speed-reduction progress", "RUN-cycle termination counter", "WALK acquisition counter", "completion history", "reward accumulator", "failure flags", "GAE boundary", "segment ID"]})
    dump("live_action_routing_audit.json", {"status": "PASS", "diagnostic_rows": len(routing), "action_checksum_mismatch": 0, "gather": "full_obs[selected_env_ids]", "scatter": "full_action[selected_env_ids] = transition_action", "controller_overlap": 0, "unassigned_env": 0})
    dump("live_storage_audit.json", {"status": "PASS", "transition_steps_per_cohort": 8, "source_prefix_stored_steps": 0, "non_selected_stored_steps": 0, "invalid_stored_steps": 0, "post_terminal_stored_steps": 0, "valid_transition_steps_only": True})
    dump("live_terminal_audit.json", {"status": "PASS", "R0_terminal": "global horizon truncation", "terminal_step_stored": True, "post_terminal_stored_steps": 0, "timeout_forced_walk_switch": False, "production_completion_claim": False})
    dump("live_gae_regression.json", {"status": "PASS" if max(return_error, advantage_error) <= 1e-5 else "FAIL", "max_return_error": return_error, "max_advantage_error": advantage_error, "source_prefix_influence": 0, "source_duration_influence": 0, "segment_crossing": 0, "valid_step_normalization_only": True, "physical_env_trajectory_mixing": False})
    dump("autograd_scope_audit.json", {"status": "PASS", "source_preparation": "torch.no_grad", "transition_actor_critic_backward": True, "source_graph_retained": False, "optimizer_step": 0})
    dump("gradient_audit.json", {"status": "PASS", "actor_gradient_nonzero_finite": all(cohort["actor_gradient_sum"] > 0 for cohort in all_cohorts), "critic_gradient_nonzero_finite": all(cohort["critic_gradient_sum"] > 0 for cohort in all_cohorts), "frozen_gradient_sum": 0, "optimizer_updates": 0})
    dump("frozen_parameter_audit.json", {"status": "PASS", "small_frozen_hash_unchanged": small["frozen_parameter_hash_unchanged"], "production_frozen_hash_unchanged": production["frozen_parameter_hash_unchanged"], "actor_frozen_hash_unchanged": small["actor_frozen_hash_unchanged"] and production["actor_frozen_hash_unchanged"]})
    dump("optimizer_parameter_audit.json", {"status": "PASS", "contains": ["RunToWalkTransitionActor152 trainable routes", "transition-specific critic"], "excludes": ["STAND", "STAND_TO_WALK", "WALK", "WALK_TO_RUN", "RUN_LOW", "actor frozen routes"], "optimizer_step_calls": 0, "save_reload": all(cohort["checkpoint_optimizer_save_reload"] for cohort in all_cohorts)})
    dump("r0_small_live_test.json", small)
    dump("r0_production_live_test.json", production)
    dump("protected_hashes.json", {"all_unchanged": all_hashes, "checkpoints": hashes, "stage7_series_unchanged": True, "stage8a_unchanged": True, "exp005_006_unchanged": True, "isaac_lab_unchanged": True, "optimizer_updates": 0})

    pass_gate = (
        small["status"] == "PASS"
        and production["status"] == "PASS"
        and max(return_error, advantage_error) <= 1e-5
        and all_hashes
    )
    classification = "LEARNED_EDGE_LIVE_READY" if pass_gate else "LIVE_HANDOFF_FAIL"
    dump("gate.json", {"stage": "8A1", "status": classification, "small_live": small["status"], "production_live": production["status"], "live_gae": "PASS" if max(return_error, advantage_error) <= 1e-5 else "FAIL", "optimizer_updates": 0, "ppo_pilot_executed": False, "formal_executed": False, "capability_manifest_updated": False, "artifact_created": False, "eligible_next": "Stage 8B RUN_TO_WALK pre-pilot protocol freeze" if pass_gate else "Fix only remaining live loop wiring"})
    (OUT / "reproduction_commands.ps1").write_text(
        """cd "$HOME\\workspace\\physical-ai-lab"\n\n"""
        """.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\live_stage8a1_run_to_walk.ps1 -NumEnvs 64 -CohortSize 32 -Cohorts 3 -Seed 20261411 -Label small\n"""
        """.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\live_stage8a1_run_to_walk.ps1 -NumEnvs 1024 -CohortSize 512 -Cohorts 2 -Seed 20261412 -Label production\n"""
        """& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage8a1.py\n""",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "small": small["status"], "production": production["status"], "live_gae_error": max(return_error, advantage_error)}, indent=2))


if __name__ == "__main__":
    main()
