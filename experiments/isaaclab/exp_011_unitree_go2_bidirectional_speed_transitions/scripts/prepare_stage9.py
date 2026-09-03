"""Freeze Stage 9 provenance, telemetry contract, seeds, and offline tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage9_contact_kinematics_heading_diagnosis"
START = "6a43a6ca304a96ee04ca1ab7f5b827e9fdb04a18"
sys.path.insert(0, str(EXP / "src"))
from go2_bidirectional.contact_kinematics import run_unit_tests  # noqa: E402

CHECKPOINTS = {
    "official_parent": REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "stage4_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt",
    "stage7_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt",
}
EXPECTED = {
    "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
    "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
    "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
}
PROTOCOL_SHA = "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != START:
        raise RuntimeError(f"STARTING_HEAD_MISMATCH: {head}")
    actual = {key: sha(path) for key, path in CHECKPOINTS.items()}
    if actual != EXPECTED:
        raise RuntimeError(f"CHECKPOINT_PROVENANCE_FAIL: {actual}")
    status_now = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    stage9_markers = (
        "stage9_contact_kinematics_heading_diagnosis",
        "prepare_stage9.py", "probe_stage9_contact_api.py", "evaluate_stage9_contact_kinematics.py",
        "finalize_stage9.py", "contact_kinematics.py", "exp_011_go2_contact_kinematics_heading_report.md",
    )
    status = [row for row in status_now if not any(token in row for token in stage9_markers)]
    dump("starting_repository_state.json", {
        "starting_head": head,
        "starting_status": status,
        "unrelated_dirty_paths": [row[3:] for row in status],
    })
    dump("stage8_reference.json", {
        "classification": "LOW_SPEED_HEADING_MULTIPLE_CAUSES",
        "pilot_readiness": "PILOT_NOT_READY",
        "absolute_heading_observable": False,
        "yaw_rate_controllable": True,
        "stage7_selected_iteration": 50,
        "checkpoint_hashes": actual,
        "protected_stage7_fall": {"0.2": 0.02, "0.3": 0.0, "0.4": 0.02, "0.5": 0.0, "0.6": 0.0},
        "anchor_retention": "PASS",
    })
    dump("protocol.json", {
        "stage": 9,
        "target": "CONTACT_KINEMATICS_AND_LOW_SPEED_HEADING",
        "seed_root": 20268901,
        "deterministic_policy": True,
        "checkpoints": actual,
        "evaluation_protocol": "GO2_ENDPOINT_EVALUATION_V1",
        "evaluation_protocol_sha256": PROTOCOL_SHA,
        "physics_dt_s": 0.005,
        "control_dt_s": 0.020,
        "stable_contact": {
            "normal_force_gt_n": 5.0,
            "minimum_steps": 3,
            "exclude_onset_steps": 2,
            "exclude_release_steps": 2,
        },
        "tangential_speed_levels_mps": [0.02, 0.05, 0.10, 0.20, 0.30],
        "duration_levels_s": [0.04, 0.10, 0.20],
        "resolved_dynamic_friction": 0.6,
        "ppo_updates": 0,
        "reward_optimization": 0,
    })
    conditions = {
        "steady": {
            "values": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0],
            "episodes": 50, "duration_s": 8.0,
        },
        "low_transition": {
            "values": ["0->0.2", "0->0.4", "0->0.6", "0.6->0.4", "0.6->0.2", "0.6->0"],
            "episodes": 50,
        },
        "anchor_transition": {
            "values": ["0->1.2", "1.2->2.0", "2.0->1.2", "1.2->0"],
            "episodes": 20,
        },
    }
    dump("diagnostic_seed_manifest.json", {
        "seed_root": 20268901,
        "selection": "consecutive, pre-registered, no success filtering",
        "same_set_across_checkpoints": True,
        "conditions": conditions,
        "seeds": {
            "50_episode_conditions": list(range(20268901, 20268951)),
            "20_episode_conditions": list(range(20268901, 20268921)),
        },
    })
    probe_path = OUT / "contact_api_runtime_probe.json"
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    dump("contact_telemetry_source_audit.json", {
        "backend": "PhysX omni.physics.tensors.api.RigidContactView",
        "source_files": [
            "isaaclab_physx/sensors/contact_sensor/contact_sensor.py",
            "isaaclab_physx/test/sensors/test_contact_sensor.py",
        ],
        "runtime_probe": probe,
        "get_contact_data_fields_in_order": [
            {"name": "normal_force", "shape": "(total_contact_buffer, 1)", "unit": "N after dt conversion"},
            {"name": "world_contact_position", "shape": "(total_contact_buffer, 3)", "unit": "m"},
            {"name": "world_contact_normal", "shape": "(total_contact_buffer, 3)", "unit": "unit vector"},
            {"name": "separation", "shape": "(total_contact_buffer, 1)", "unit": "m"},
            {"name": "count", "shape": "(env*body, filter)", "unit": "count"},
            {"name": "start_index", "shape": "(env*body, filter)", "unit": "buffer index"},
        ],
        "get_friction_data_fields_in_order": [
            {"name": "friction_force_world", "unit": "N"},
            {"name": "friction_application_point_world", "unit": "m"},
            {"name": "count", "unit": "count"},
            {"name": "start_index", "unit": "buffer index"},
        ],
        "association": {
            "foot": "one dedicated contact view per FL/FR/RL/RR foot rigid body",
            "ground": "filter path /World/ground/terrain/GroundPlane/CollisionPlane",
            "unique": True,
        },
        "direct_relative_contact_velocity": "NOT_AVAILABLE",
        "contact_patch_id": "NOT_EXPOSED_BY_RUNTIME_GET_CONTACT_DATA",
        "manifold_id": "NOT_AVAILABLE",
        "body_pair_ids": "implicit in dedicated sensor body and single ground filter; not separately returned",
        "impulses": "raw impulse is not separately exposed by this runtime wrapper; force is returned after dt conversion",
        "friction": {
            "robot_static": 0.8, "robot_dynamic": 0.6,
            "ground_static": 1.0, "ground_dynamic": 1.0,
            "ground_combine_mode": "multiply",
            "resolved_static": 0.8, "resolved_dynamic": 0.6,
        },
    })
    dump("contact_kinematics_contract.json", {
        "frame": "world",
        "units": {"position": "m", "velocity": "m/s", "force": "N", "moment": "N*m"},
        "foot_surface_velocity": "v_b + omega_b cross (p_c - x_b)",
        "ground_surface_velocity": "zero (static ground plane)",
        "relative_velocity": "foot surface minus ground surface",
        "tangent_projection": "v_rel - dot(v_rel,n_hat)*n_hat",
        "force_weighting": "normal-force weighted mean and unweighted maximum",
        "friction_utilization": "|F_t|/(0.6*F_n+epsilon)",
        "yaw_moment": "sum cross(p_contact-root_com, force)_z",
        "normal_moment_point": "normal contact point",
        "friction_moment_point": "PhysX friction application point",
        "direct_physx_relative_velocity_comparison": "NOT_AVAILABLE",
    })
    tests = run_unit_tests()
    dump("contact_kinematics_unit_tests.json", tests)
    if not tests["all_pass"]:
        raise RuntimeError("CONTACT_KINEMATICS_UNIT_TEST_FAIL")


if __name__ == "__main__":
    main()
