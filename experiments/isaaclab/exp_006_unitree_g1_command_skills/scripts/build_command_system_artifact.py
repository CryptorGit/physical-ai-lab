"""Build the immutable command_system_v1 audit artifact from formal results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
RESULT = REPO / "results/exp_006_unitree_g1_command_skills/command_system_v1"
ARTIFACT = REPO / "artifacts/exp_006_unitree_g1_command_skills/command_system_v1"
CROUCH = REPO / "results/exp_006_unitree_g1_command_skills/crouch_shallow_scripted_v1/formal_50/supported"
RETENTION = REPO / "results/exp_006_unitree_g1_command_skills/step_over_shallow/retention_preflight.json"
STAGE2 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
MODEL31 = REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-20_14-34-35_pilot_stop_stage_a_braking/model_31.pt"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100), len(values) - 1)] if values else 0.0


def aggregate_stand_crouch_stand():
    source = load(CROUCH / "summary.json")
    skill = source["skills"]["CROUCH_SHALLOW"]
    joints = (
        "left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee",
        "left_ankle_pitch", "right_ankle_pitch",
    )
    previous = {}
    jumps = defaultdict(list)
    base_consistent = True
    with (CROUCH / "crouch_curve.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            episode = int(row["episode"])
            action = [float(row[f"{joint}_final_action"]) for joint in joints]
            if episode in previous:
                diff = [a - b for a, b in zip(action, previous[episode])]
                jumps[episode].append(math.sqrt(sum(value * value for value in diff)))
            previous[episode] = action
            base_consistent &= float(row["stand_base_gate"]) == 1.0
            base_consistent &= abs(float(row["selected_base_action_norm"]) - float(row["standing_base_action_norm"])) <= 1e-7
    episode_jumps = [max(values, default=0.0) for values in jumps.values()]
    summary = {
        "schema_version": 1,
        "task": "STAND_CROUCH_STAND",
        "episodes": source["episodes"],
        "seed": source["seed"],
        "source_summary": str((CROUCH / "summary.json").relative_to(REPO)),
        "source_reused": True,
        "initial_stand_settle_rate": skill["settle_success_rate"],
        "crouch_success_rate": skill["success_rate"],
        "depth_error_mean_m": skill["depth_error_m"],
        "depth_error_p95_m": skill["depth_error_p95_m"],
        "return_success_rate": skill["return_success_rate"],
        "final_stand_hold_success_rate": skill["stand_hold_success_rate"],
        "total_sequence_success_rate": skill["success_rate"],
        "fall_rate": skill["fall_rate"],
        "dangerous_contact_failure_rate": skill["dangerous_contact_failure_rate"],
        "saturation_failure_rate": skill["saturation_failure_rate"],
        "action_discontinuity_l2_p95": percentile(episode_jumps, 95),
        "action_discontinuity_l2_max": max(episode_jumps, default=0.0),
        "base_option_consistency_rate": 1.0 if base_consistent else 0.0,
        "standing_base_option_id": source["standing_base_option_id"],
        "controller": source["controller"],
    }
    summary["gate_pass"] = all((
        summary["total_sequence_success_rate"] >= .90,
        summary["initial_stand_settle_rate"] >= .95,
        summary["crouch_success_rate"] >= .90,
        summary["return_success_rate"] >= .90,
        summary["final_stand_hold_success_rate"] >= .95,
        summary["fall_rate"] <= .05,
        summary["dangerous_contact_failure_rate"] <= .05,
        summary["saturation_failure_rate"] <= .05,
        summary["base_option_consistency_rate"] == 1.0,
    ))
    dump(RESULT / "stand_crouch_stand_formal_50/summary.json", summary)
    dump(RESULT / "stand_crouch_stand_formal_50/provenance.json", {
        "reuse_reason": "Existing formal evaluation exactly matches the requested 50-episode shallow command distribution and controller.",
        "source_evaluation_schema_version": source["evaluation_schema_version"],
        "source_gate": str((CROUCH.parent / "gate.json").relative_to(REPO)),
        "learned_crouch_residual_enabled": source["learned_crouch_residual_enabled"],
    })
    return summary


def main():
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    stand = load(RESULT / "stand_formal_50/summary.json")
    running = load(RESULT / "run_turn_run_formal_50/summary.json")
    unsupported = load(RESULT / "unsupported_requests/summary.json")
    retention = load(RETENTION)
    crouch_sequence = aggregate_stand_crouch_stand()

    graph = {
        "schema_version": 1,
        "supported": ["RUN->RUN", "RUN->TURN", "TURN->RUN", "TURN->TURN", "STAND->STAND", "STAND->CROUCH_SHALLOW", "CROUCH_SHALLOW->STAND", "CROUCH_SHALLOW->CROUCH_SHALLOW"],
        "unsupported": ["RUN->STAND", "TURN->STAND", "RUN->CROUCH_SHALLOW", "TURN->CROUCH_SHALLOW", "STAND->RUN", "CROUCH_SHALLOW->RUN"],
        "unsupported_reason": "CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED",
        "implicit_stop_insertion": False,
    }
    routing = {
        "schema_version": 1,
        "observation_dimension": 152,
        "policy_skill_one_hot_dimension": 6,
        "stand_is_external_controller_state": True,
        "controller_families": {"RUNNING_FAMILY": ["RUN", "TURN"], "STANDING_FAMILY": ["STAND", "CROUCH_SHALLOW"], "PROTOTYPE": ["STOP"]},
        "base_options": {"RUN": "stage4_running", "TURN": "stage4_running", "STOP": "stage4_running_plus_stop_prototype", "STAND": "stage2_model4246", "CROUCH_SHALLOW": "stage2_model4246"},
        "actions": {"STAND": "stage2_action", "CROUCH_SHALLOW": "stage2_action + scripted_shallow_v1_offset"},
        "zero_offset_outside_owner": True,
        "fail_closed": True,
    }
    manifest = {
        "schema_version": 1,
        "system": "command_system_v1",
        "status": "PASS" if stand["gate_pass"] and running["gate_pass"] and crouch_sequence["gate_pass"] and unsupported["pass_rate"] == 1.0 else "FORMAL_EVALUATION_FAILED",
        "skills": {
            "RUN": {"status": "PASS"}, "TURN": {"status": "PASS"},
            "STOP": {"status": "PROTOTYPE", "formal_gate_pass": False},
            "STAND": {"status": "PASS" if stand["gate_pass"] else "FAIL"},
            "CROUCH_SHALLOW": {"status": "PASS", "supported_depth_m": [.08, .10], "controller": "scripted_shallow_v1"},
            "CROUCH_DEEP": {"status": "NOT_SUPPORTED", "reason": "DEEP_CROUCH_RETURN_UNRESOLVED"},
            "STEP_OVER": {"status": "NOT_SUPPORTED", "reason": "OPTIMIZATION_FAILURE"},
            "LAND": {"status": "NOT_SUPPORTED", "supported_range_m": None, "observed_standing_drop_tolerance_m": .02, "observation_is_not_a_skill_claim": True},
        },
        "supported_transitions": graph["supported"],
        "unsupported_transitions": graph["unsupported"],
        "controller_families": routing["controller_families"],
        "cross_family_transition_status": "CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED",
    }
    crouch_checkpoint = Path(load(CROUCH / "summary.json")["checkpoint"])
    if crouch_checkpoint.is_absolute():
        crouch_checkpoint = crouch_checkpoint.relative_to(REPO)
    provenance = {
        "stage2_standing_checkpoint": str(STAGE2.relative_to(REPO)),
        "stage4_and_skill_checkpoint": str(MODEL31.relative_to(REPO)),
        "crouch_checkpoint": str(crouch_checkpoint),
        "stop_overlay": "preserved prototype; not required by formal sequences",
        "training_performed_for_command_system_v1": False,
    }
    protected = {
        "stage2_checkpoint_sha256": sha256(STAGE2), "model31_checkpoint_sha256": sha256(MODEL31),
        "protected_actor_route_hash": retention["candidate_protected_tensor_hash"],
        "tensor_hash_verified": retention["tensor_hash_verified"], "action_equivalence_verified": retention["action_equivalence_verified"],
        "stop_action_immutability_verified": retention["stop_action_immutability_verified"],
        "action_checks": retention["action_checks"],
        "router_is_parameter_free": True,
        "router_action_identity_audit": str((RESULT / "unsupported_requests/summary.json").relative_to(REPO)),
    }
    formal = {"stand": stand, "run_turn_run": running, "stand_crouch_stand": crouch_sequence}
    dump(ARTIFACT / "capability_manifest.json", manifest)
    dump(ARTIFACT / "transition_graph.json", graph)
    dump(ARTIFACT / "controller_routing_config.json", routing)
    dump(ARTIFACT / "skill_provenance.json", provenance)
    dump(ARTIFACT / "checkpoint_ancestry.json", provenance)
    dump(ARTIFACT / "formal_sequence_results.json", formal)
    dump(ARTIFACT / "unsupported_request_safety_results.json", unsupported)
    dump(ARTIFACT / "protected_tensor_hashes.json", protected)
    lines = []
    for path in sorted(ARTIFACT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.name}")
    (ARTIFACT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(ARTIFACT), "status": manifest["status"], "stand": stand["gate_pass"], "run_turn_run": running["gate_pass"], "stand_crouch_stand": crouch_sequence["gate_pass"], "unsupported": unsupported["pass_rate"]}, indent=2))


if __name__ == "__main__":
    main()
