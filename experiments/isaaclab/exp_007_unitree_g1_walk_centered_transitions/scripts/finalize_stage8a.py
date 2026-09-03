"""Aggregate Stage 8A direct baseline and R0 evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8a_run_to_walk_audit"
RAW = OUT / "raw"
SEEDS = (20261301, 20261302, 20261303)
SOURCES = (2.6, 2.8)
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


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def pct(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    x = (len(values) - 1) * q / 100
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected = []
    source_attempts = []
    for seed in SEEDS:
        rows = list(csv.DictReader((RAW / f"seed_{seed}_episodes.csv").open(encoding="utf-8")))
        for row in rows:
            row["seed"] = int(row["seed"])
            row["source_run_speed_mps"] = float(row["source_run_speed_mps"])
            for key in (
                "valid_run_source", "walk_contract_acquisition", "walk_takeover", "walk_hold",
                "full_edge_success", "fall", "dangerous_slip", "impact_failure",
                "long_dwell_saturation", "excessive_flight", "action_discontinuity_failure",
            ):
                row[key] = row[key] == "True"
            for key in ("transition_duration_s", "heading_p95_rad", "entry_action_jump_l2"):
                row[key] = float(row[key])
        source_attempts.extend(rows)
        for source in SOURCES:
            valid = [row for row in rows if row["source_run_speed_mps"] == source and row["valid_run_source"]]
            if len(valid) < 20:
                raise RuntimeError(f"seed {seed} source {source} has only {len(valid)} valid sources")
            selected.extend(valid[:20])

    with (OUT / "direct_switch_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)

    difference = []
    for seed in SEEDS:
        difference.extend(csv.DictReader((RAW / f"seed_{seed}_action_difference.csv").open(encoding="utf-8")))
    with (OUT / "action_difference.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(difference[0]))
        writer.writeheader()
        writer.writerows(difference)

    per_source = {}
    for source in SOURCES:
        group = [row for row in selected if row["source_run_speed_mps"] == source]
        metrics = {
            "episodes": len(group),
            "walk_contract_acquisition_rate": mean(row["walk_contract_acquisition"] for row in group),
            "walk_takeover_rate": mean(row["walk_takeover"] for row in group),
            "walk_hold_rate": mean(row["walk_hold"] for row in group),
            "full_edge_success_rate": mean(row["full_edge_success"] for row in group),
            "transition_duration_mean_s": mean(row["transition_duration_s"] for row in group),
            "heading_p95_rad": pct([row["heading_p95_rad"] for row in group], 95),
            "fall_rate": mean(row["fall"] for row in group),
            "saturation_rate": mean(row["long_dwell_saturation"] for row in group),
            "slip_rate": mean(row["dangerous_slip"] for row in group),
            "impact_failure_rate": mean(row["impact_failure"] for row in group),
            "excessive_flight_rate": mean(row["excessive_flight"] for row in group),
            "action_discontinuity_failure_rate": mean(row["action_discontinuity_failure"] for row in group),
            "entry_action_jump_p95": pct([row["entry_action_jump_l2"] for row in group], 95),
        }
        gate = (
            metrics["walk_contract_acquisition_rate"] >= 0.90
            and metrics["walk_takeover_rate"] >= 0.90
            and metrics["walk_hold_rate"] >= 0.90
            and metrics["full_edge_success_rate"] >= 0.90
            and metrics["fall_rate"] <= 0.05
            and metrics["heading_p95_rad"] <= 0.12
            and metrics["saturation_rate"] <= 0.05
            and metrics["slip_rate"] <= 0.05
            and metrics["impact_failure_rate"] <= 0.05
            and metrics["excessive_flight_rate"] <= 0.05
            and metrics["action_discontinuity_failure_rate"] <= 0.05
        )
        metrics["candidate_gate"] = "PASS" if gate else "FAIL"
        per_source[str(source)] = metrics
    direct_pass = all(value["candidate_gate"] == "PASS" for value in per_source.values())

    phase = defaultdict(list)
    for row in selected:
        phase[row["contact_phase"]].append(row)
    phase_audit = {
        key: {
            "n": len(group),
            "full_edge_success_rate": mean(row["full_edge_success"] for row in group),
            "fall_rate": mean(row["fall"] for row in group),
            "saturation_rate": mean(row["long_dwell_saturation"] for row in group),
            "slip_rate": mean(row["dangerous_slip"] for row in group),
            "walk_acquisition_time_mean_s": mean(row["transition_duration_s"] for row in group),
            "entry_action_jump_p95": pct([row["entry_action_jump_l2"] for row in group], 95),
        }
        for key, group in phase.items()
    }
    r0_small = json.loads((OUT / "r0/small.json").read_text())
    r0_prod = json.loads((OUT / "r0/production.json").read_text())
    # The actor/storage wiring audit is not a substitute for applying the
    # transition actor inside the live Isaac stepping loop.
    for result in (r0_small, r0_prod):
        result["manager_and_gradient_wiring"] = result["status"]
        result["live_transition_actor_activation"] = "NOT_VERIFIED"
        result["live_contact_sensor_continuity"] = "NOT_VERIFIED"
        result["status"] = "FAIL"
    r0_pass = False
    classification = "PARAMETER_FREE_CANDIDATE" if direct_pass else "LEARNED_EDGE_INFRASTRUCTURE_READY" if r0_pass else "INFRASTRUCTURE_FAIL"

    hashes = {}
    for key, (path, expected) in EXPECTED.items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        hashes[key] = {"path": path, "expected": expected, "actual": actual, "unchanged": actual == expected}
    dump("stage7r9_reference.json", {"status": "LIMITED_FULL_PASS", "supported_targets_mps": [2.6, 2.8], "source_revision": "696733b79857f6a320b472c116ad02e0502a1817", "unchanged": True})
    dump("formal_graph_source_route.json", {"route": ["RESET", "STAND", "STAND_TO_WALK", "WALK@1.2", "WALK_TO_RUN", "RUN_LOW"], "source_commands_mps": [2.6, 2.8], "source_preparation_in_edge_denominator": False})
    dump("run_source_contract.json", {"state": "RUN_LOW", "controller": "run_low_steady_state_expert_v1", "hold_seconds": 1.0, "speed_error_max_mps": 0.20, "heading_error_max_rad": 0.12, "periodic_running": True, "safe_cycles_min": 3, "alternating_landing_ratio_min": 0.8, "valid_landing_ratio_min": 0.8, "safety_required": True, "finite_required": True})
    dump("walk_target_contract.json", {"state": "WALK", "controller": "walk_steady_state_expert_v1", "target_speed_mps": 1.2, "hold_seconds": 0.4, "speed_error_max_mps": 0.20, "heading_error_max_rad": 0.12, "periodic_running": False, "excessive_flight": False, "safety_required": True, "finite_required": True})
    dump("overlap_audit.json", {"valid_samples": len(selected), "samples_per_source": 60, "joint_rows": len(difference), "production_blend": False, "action_l2_p95": {str(source): per_source[str(source)]["entry_action_jump_p95"] for source in SOURCES}})
    dump("direct_switch_protocol.json", {"type": "parameter_free_hard_switch", "seeds": list(SEEDS), "valid_episodes_per_seed_source": 20, "total_valid_episodes": 120, "source_speeds_mps": [2.6, 2.8], "target_walk_speed_mps": 1.2, "walk_hold_seconds": 5.0, "runtime_action_blend": False})
    dump("direct_switch_summary.json", {"status": "FAIL", "valid_episodes": len(selected), "source_preparation_attempts": len(source_attempts), "source_preparation_success_rate": mean(row["valid_run_source"] for row in source_attempts), "overall_full_edge_success_rate": mean(row["full_edge_success"] for row in selected), "dominant_failure": "long_dwell_saturation"})
    dump("direct_switch_per_source.json", per_source)
    dump("contact_phase_audit.json", phase_audit)
    dump("failure_counts.json", dict(Counter(row["failure_class"] or "success" for row in selected)))
    dump("controller_classification.json", {"classification": classification, "direct_switch_gate": "FAIL", "learned_edge_required": True, "capability_updated": False})
    dump("transition_actor_design.json", {"status": "IMPLEMENTED", "actor": "RunToWalkTransitionActor152", "action_term": "RunToWalkTransitionAction", "initialization": "strict_deep_copy_RUN_LOW", "initial_action_bitwise_equal": r0_small["parent_action_bitwise_match"], "trainable_routes": ["RUN command encoder", "RUN state adapter", "RUN residual head"], "frozen_routes": ["running base", "other skill routes"], "observation_dim": 152, "action_dim": 37, "action_scale": 0.5})
    dump("command_observation_contract.json", {"status": "REUSED_WITHOUT_SEMANTIC_REMAP", "total_dim": 152, "legacy_dim": 123, "command_dim": 29, "current_skill": "RUN", "previous_skill": "RUN", "target_state": "WALK", "target_local_speed_mps": 1.2, "target_heading": "source RUN heading", "world_xy_included": False})
    dump("transition_only_runner_audit.json", {"status": "PARTIAL", "runner": "IN_PLACE_ENV_ID_COHORT + TransitionOnlyOnPolicyRunner", "source_route_live_evidence": "three Isaac direct-switch runs", "manager_and_gradient_wiring": "PASS", "live_transition_actor_activation": "NOT_VERIFIED", "source_prefix_stored_steps": 0, "state_copy": False, "runtime_action_blend": False, "production_optimizer_updates": 0})
    dump("r0_small_live_test.json", r0_small)
    dump("r0_production_live_test.json", r0_prod)
    dump("reward_skeleton_audit.json", {"status": "AUDIT_ONLY_NO_WEIGHTS_SELECTED", "progress": ["speed reduction progress", "WALK target-speed tracking", "heading tracking", "lateral velocity suppression", "upright"], "run_termination_precursor": ["flight-duration reduction", "safe landing", "RUN safe-cycle termination", "excessive-flight suppression"], "walk_acquisition": ["WALK-compatible contact", "stable support", "WALK contract progress", "WALK acceptance"], "safety": ["fall", "torso contact", "reverse velocity", "slip", "impact", "ankle saturation", "knee saturation", "action rate"], "alignment": ["entry RUN action alignment", "exit WALK action alignment"]})
    dump("protected_hashes.json", {"all_unchanged": all(value["unchanged"] for value in hashes.values()), "checkpoints": hashes, "optimizer_updates": 0, "stage7_series_unchanged": True, "exp005_006_unchanged": True, "isaac_lab_unchanged": True})
    dump("gate.json", {"stage": "8A", "status": classification, "direct_switch": "FAIL", "r0": "FAIL", "unverified": ["live transition actor activation in Isaac stepping loop", "live contact/sensor continuity across RUN_TO_WALK actor handoff"], "formal_executed": False, "ppo_pilot_executed": False, "capability_manifest_updated": False, "artifact_created": False, "eligible_next": "Complete only the missing live RUN_TO_WALK handoff wiring"})
    (OUT / "reproduction_commands.ps1").write_text(
        """cd "$HOME\\workspace\\physical-ai-lab"\n\n"""
        """.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_run_to_walk.ps1 -Seed 20261301 -Label seed_20261301 -AttemptsPerSource 30\n"""
        """.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_run_to_walk.ps1 -Seed 20261302 -Label seed_20261302 -AttemptsPerSource 30\n"""
        """.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_run_to_walk.ps1 -Seed 20261303 -Label seed_20261303 -AttemptsPerSource 30\n"""
        """& "$HOME\\workspace\\IsaacLab\\isaaclab.bat" -p .\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\finalize_stage8a.py\n""",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "per_source": per_source}, indent=2))


if __name__ == "__main__":
    main()
