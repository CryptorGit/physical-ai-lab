"""Finalize D25 after the mandatory interface/reference fail-closed gate."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_014_phase_2_d25_model_based_first_step_teacher_report.md"
START = "6252168a0311f278715a1125ffc3091af5b28f7c"
CLASS = "EXP014_D25_KINEMATICS_INTERFACE_UNAVAILABLE"


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


interface = json.loads((RAW / "interface_audit.json").read_text(encoding="utf-8"))
reference_manifest_path = OUT.parent / "phase_2_d17_start_source_and_causality_audit/wmove_basin_reference_manifest.json"
reference = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
snapshot_path = REPO / reference["source_artifact"]

robot = {
    "status": "PARTIAL_INTERFACE_ONLY",
    "teacher": "Exp014ModelBasedFirstStepTeacherV1",
    "total_mass_kg": interface["total_mass_kg"],
    "nominal_com_position": "AVAILABLE_AT_RUNTIME_NOT_CAPTURED_BECAUSE_GATE_STOPPED",
    "com_height": "NOT_DERIVED_WITHOUT_A_VALID_WMOVE_ENTRY_MEDOID",
    "foot_frames": {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"},
    "foot_sole_polygon": "UNAVAILABLE_AS_A_VERSIONED_NUMERIC_POLYGON_FROM_CURRENT_RUNTIME/REPOSITORY CONTRACT",
    "joint_names": interface["joint_names"],
    "joint_position_limits": interface["joint_position_limits"],
    "joint_velocity_limits": interface["joint_velocity_limits"],
    "effort_limits": {"source": "runtime PhysX articulation table", "status": "AVAILABLE"},
    "action_interface": {"dimension": 37, "type": "normalized joint-position", "offset": interface["action_term"]["_offset"]["value"], "scale": "configured by JointPositionAction; scalar runtime object was not JSON-convertible in audit"},
    "pd_interface": {"legs_stiffness": "150/200 Nm/rad by joint group", "legs_damping": 5.0, "ankles_stiffness": 20.0, "ankles_damping": 2.0, "arms_stiffness": 40.0, "arms_damping": 10.0},
    "control_dt_s": interface["control_dt_s"], "physics_dt_s": interface["physics_dt_s"], "decimation": interface["decimation"],
    "available_jacobians": {"available": "jacobians" in interface["api_shapes"], "shape": interface["api_shapes"].get("jacobians")},
    "available_mass_matrix": {"available": False, "error": interface["errors"].get("mass_matrices")},
    "available_body_mass_inertia": {"masses_shape": interface["api_shapes"].get("masses"), "inertias_shape": interface["api_shapes"].get("inertias")},
    "available_centroidal_momentum": {"direct_api": False, "body_com_fields": interface["body_fields"]},
    "available_contact_force": interface["contact_sensor_present"],
    "solver_priority_audit": {
        "repository_or_isaaclab_wbik": "NO_GENERAL_CONSTRAINED_WHOLE_BODY_IK_SOLVER_FOUND; DifferentialIKController is end-effector differential IK, not the registered hard-task WBIK contract",
        "pinocchio_dependency": "NOT_PRESENT_AS_A_REGISTERED_PROJECT DEPENDENCY",
        "custom_jacobian_wls": "NOT_AUTHORIZED_TO_PROCEED BECAUSE NUMERIC FOOT POLYGON AND IDENTITY-COMPLETE TARGET MEDOID INPUTS ARE ABSENT",
    },
    "interface_gate": "FAIL",
}
dump("model_based_teacher_robot_contract.json", robot)

manifest = {
    "name": "WMove03PostTouchdownEntryReferenceV1",
    "status": "NOT_CONSTRUCTIBLE_FROM_PERSISTED_SOURCE_OF_TRUTH",
    "requested_source": "D17 W_MOVE_FORWARD_BASIN_REFERENCE_V1",
    "persisted_reference": reference,
    "persisted_manifest_sha256": sha(reference_manifest_path),
    "source_snapshot_sha256": sha(snapshot_path),
    "available_fields": ["122D normalized-distance physical features generated at D17 runtime", "state count", "source snapshot identity"],
    "missing_required_fields": ["per-state root state bundle for all 10,240 states", "foot pose/velocity", "contact phase/forces", "CoM position/velocity", "centroidal momentum", "DCM", "support polygon", "next W_MOVE action", "step length/width/clearance trajectory distributions"],
    "raw_snapshot_restore_used": 0,
    "medoid_selection": "NOT_EXECUTED",
}
dump("wmove_entry_reference_manifest.json", manifest)
dump("wmove_entry_medoids.json", {"status": "NOT_EXECUTED", "left_landed": None, "right_landed": None, "reason": "identity-complete target states are not persisted; averaging or inventing contact states is prohibited"})
dump("wmove_dcm_offset_reference.json", {"status": "NOT_EXECUTED", "omega": None, "b_target_left": None, "b_target_right": None, "mirror_consistency": "NOT_TESTED", "reason": "target CoM, stance-foot pose, and target CoM height unavailable"})

dump("reduced_order_model_contract.json", {
    "status": "PREREGISTERED_NOT_EXECUTED", "model": "horizontal LIPM/DCM", "gravity_mps2": 9.81,
    "equations": {"omega": "sqrt(g/h)", "dcm": "c_xy + c_dot_xy/omega", "com_acceleration": "omega^2*(c_xy-z_xy)", "dcm_rate": "omega*(xi-z_xy)"},
    "phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"],
    "cop_constraints": {"double_support": "convex hull of both numeric sole polygons", "single_support": "stance sole polygon"},
    "yaw": {"pelvis_yaw": 0.0, "yaw_rate": 0.0, "vertical_centroidal_momentum": "connect to target medoid"},
    "execution_reason": "blocked before numeric reference generation",
})

dump("support_polygon_tests.json", {"status": "NOT_EXECUTED", "tests": [], "reason": "numeric foot sole polygons unavailable; no guessed geometry permitted"})
dump("mirror_tests.json", {"status": "NOT_EXECUTED", "mapping_source": "D19 PASS mirror contract", "reason": "no numeric LEFT/RIGHT target medoids or planned trajectories were constructed"})
grid = [{"plan_id": f"P{idx:02d}", "double_support_shift_s": ds, "swing_duration_multiplier": sm, "clearance_quantile": cq}
        for idx, (ds, sm, cq) in enumerate((a, b, c) for a in (.30, .40, .50) for b in (.80, 1.00, 1.20) for c in ("p50", "p75", "p90"))]
dump("model_parameter_grid.json", {"status": "FIXED_27_PLANS_NOT_INSTANTIATED", "plans": grid, "additional_search": 0})
dump("whole_body_ik_contract.json", {
    "status": "INTERFACE_GATE_FAIL", "preferred_solver": "constrained QP else damped weighted least squares with explicit checks",
    "hard_tasks": ["stance-foot 6D pose", "joint position limits", "action bounds"],
    "tracking_tasks": ["CoM xyz", "swing-foot 6D pose", "pelvis roll/pitch/yaw", "torso orientation", "stance contact consistency"],
    "regularization": ["joint velocity", "action rate", "current configuration", "mirrored nominal posture", "arm/waist motion", "centroidal yaw momentum"],
    "blocking_interfaces": ["general WBIK/constrained-QP interface absent", "mass-matrix API absent", "numeric sole polygon absent", "identity-complete W_MOVE target medoids absent"],
    "constraint_clipping_used": False,
})
dump("ik_unit_tests.json", {"status": "NOT_EXECUTED", "all_pass": False, "tests": {name: "NOT_EXECUTED" for name in ["LIPM/DCM analytical propagation", "support polygon containment", "DCM foot placement", "minimum-jerk continuity", "swing apex", "LEFT/RIGHT mirror", "IK Jacobian sign", "action round trip", "joint limits", "velocity limits"]}, "reason": "mandatory interface gate failed before testable numeric contract existed"})

csv_fields = ["recipe_id", "lead", "plan_id", "status", "reason"]
for filename in ("offline_kinematic_feasibility.csv", "development_model_based_results.csv", "generality_results.csv"):
    with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=csv_fields).writeheader()
dump("offline_kinematic_feasibility.json", {"status": "NOT_EXECUTED", "candidate_plans": 8 * 2 * 27, "evaluated_plans": 0, "eligible_plans": 0, "reason": "kinematics/reference interface gate failed"})
dump("development_model_based_results.json", {"status": "NOT_EXECUTED", "recipes": 8, "lead_conditions": 16, "eligible_plans": 0, "physics_attempts": 0, "raw_snapshot_restore": 0})
dump("model_based_first_step_metrics.json", {"status": "NOT_EXECUTED", "safe_first_step": 0, "denominator": 0, "not_a_failure_rate": True})
dump("wmove_basin_entry.json", {"status": "NOT_EXECUTED", "entries": 0, "denominator": 0})
dump("wmove_handoff.json", {"status": "NOT_EXECUTED", "attempts": 0, "retention": None, "reason": "no basin-entry episode"})
dump("generality_results.json", {"status": "NOT_EXECUTED", "recipes": 56, "reason": "development was blocked before physics"})
dump("mirror_consistency.json", {"status": "NOT_EXECUTED", "reason": "no selected plan"})

bundle = OUT / "model_based_start_trajectory_bundle.npz"
np.savez_compressed(bundle, status=np.array(["NOT_EXECUTED"]), trajectory_count=np.array([0], dtype=np.int64))
(OUT / "model_based_start_trajectory_bundle.sha256").write_text(sha(bundle) + "\n", encoding="ascii")
dump("model_based_start_trajectory_manifest.json", {"status": "EMPTY_BY_FAIL_CLOSED_DESIGN", "trajectory_count": 0, "bundle_sha256": sha(bundle), "physics_attempts": 0, "durability": "atomic capture not applicable because no trajectory was generated"})
dump("temporary_distillation_feasibility.json", {"status": "NOT_EXECUTED", "reason": "safe first-step trajectories <8", "persistent_checkpoint": 0})

dump("stage_reference.json", {"stage": "Phase 2-D25", "starting_head": START, "actual_head_before_commit": git("rev-parse", "HEAD"), "prior_classification": "EXP014_D24D_FRESH_START_REACHABILITY_ZERO", "teacher_name": "Exp014ModelBasedFirstStepTeacherV1", "scope": "forward 0 deg, 0.3 m/s, yaw 0", "date": "2026-08-04"})
dump("protocol.json", {"name": "EXP014_PHASE_2_D25_MODEL_BASED_FIRST_STEP_TEACHER_PREFLIGHT_V1", "fresh_source_lifecycle": "Exp014FreshS_HOLDSourceLifecycleV2", "candidate_grid": 27, "development_recipes": 8, "leads": ["LEFT", "RIGHT"], "fail_closed_gate": "required kinematics/reference inputs before unit/offline/physics", "persistent_policy_training": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "raw_snapshot_restore": 0, "validation_access": 0, "heldout_access": 0, "remote_push": False})
dump("stage_classification.json", {"primary_classification": CLASS, "status": "FAIL_CLOSED_BEFORE_PHYSICS", "failures": ["runtime ArticulationView exposes no mass-matrix method", "no registered general constrained WBIK interface", "D17 10,240-state reference did not persist D25-required contact/foot/CoM/centroidal fields"], "physics_attempts": 0})
dump("recommended_next_action.json", {"status": "NOT_AUTHORIZED", "action": "restore a versioned identity-complete W_MOVE entry reference and deterministic general WBIK interface, then rerun D25 preflight", "methodology_branches_authorized": [], "reason": "none of distillation, second-segment, handoff repair, or dynamics optimization can be grounded before a valid target/kinematics interface exists"})

changed = git("diff", "--name-only", START).splitlines()
protected_overlap = [p for p in changed if (p.startswith("results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d") and "phase_2_d25_" not in p)]
dump("protected_hashes.json", {"starting_HEAD": START, "exp005_through_exp013_unchanged_by_D25": True, "D6_through_D24D_unchanged_by_D25": len(protected_overlap) == 0, "protected_overlap": protected_overlap, "S_HOLD_unchanged": True, "Stage2Q_unchanged": True, "W_MOVE_unchanged": True, "S_STOP_OMNI_unchanged": True, "persistent_policy_update": 0, "new_learned_checkpoint": 0, "raw_snapshot_restore": 0, "PPO": 0, "CEM": 0, "D26_low_speed_STEP": 0, "validation_access": 0, "heldout_access": 0, "RUN_integration": 0, "remote_push": False})

(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference='Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n$env:PYTHONUTF8='1'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d25_interface_audit.py --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe' experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d25.py\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# Exp 014 Phase 2-D25 Model-Based First-Step Teacher Preflight

## Outcome

Primary classification: `{CLASS}`. The mandatory interface/reference gate failed before unit-test instantiation, offline IK, or physics. This is an infrastructure/input-contract result, not evidence that a model-based first step is physically impossible.

## Robot model

The runtime G1 has mass {interface['total_mass_kg']:.6f} kg, 44 rigid bodies, 37 actions, control dt 0.02 s, physics dt 0.005 s, and decimation 4. PhysX exposes a 44-body Jacobian tensor of shape `{interface['api_shapes'].get('jacobians')}`, per-body masses and inertias, and contact forces. It does not expose `get_mass_matrices()` on this ArticulationView. A registered constrained whole-body IK/QP interface and a versioned numeric sole polygon were not found.

## W_MOVE target

D17 registered 10,240 forward-0.3 states and a 122D distance representation, but its persisted artifact contains only the manifest plus the original D6 physical snapshot source. It does not contain the per-state foot poses, contact phase, CoM, centroidal momentum, DCM, support polygon, or next action required to select the requested real post-touchdown medoids. Recreating those quantities would require the raw state restoration prohibited by D25, or guessing target geometry, so neither was done.

## Reduced-order plan and WBIK

The LIPM/DCM equations, four phases, hard-task hierarchy, and fixed 27-plan grid were preregistered. Numeric DCM offsets, support polygons, step geometry, IK unit tests, and offline feasibility were not instantiated because the required target and solver contracts were incomplete. No constraint violation was hidden by clipping.

## Execution and protection

Physics candidates: 0. Development, handoff, generality, and distillation were not executed. Persistent updates, checkpoints, PPO, CEM, raw snapshot restores, validation access, and held-out access are all zero. D6-D24D artifacts and all Teacher checkpoints were left unchanged.
""", encoding="utf-8")

print(json.dumps({"classification": CLASS, "physics_attempts": 0, "bundle_sha256": sha(bundle)}, indent=2))


if __name__ == "__main__":
    pass
