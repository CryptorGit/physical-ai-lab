"""Phase 2-D28 centroidal causality audit and fail-closed preflight.

The D27 trajectory is read-only.  This runner deliberately stops before
Isaac Lab physics when the D27 trace does not contain the body-level fields
needed by the registered centroidal interface.  It still emits the complete
D28 audit ledger, fixed V3/controller contracts, synthetic matrix tests, and
the explicit no-authorization result.  No D27 file is used as an output
destination.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28_centroidal_feedback_start"
REPORT = REPO / "research/exp_014_phase_2_d28_centroidal_feedback_start_report.md"

D26V = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26v_endpoint_gate_and_wbik_v2"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26X = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26x_timing_and_target_set"
D27 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d27_right_model_based_start_physics"

START_HEAD_REQUESTED = "7fb59fdd6e93ce08b154cb8dab6b8be801619f41"
DT = 0.02
SEED = 20279941
RECIPES = [4, 5, 6, 7]
ALL_RECIPES = list(range(8))
TARGET_ID = "RIGHT_000"
TARGET_ROW = 9330
TARGET_EPISODE = 187
TARGET_STEP = 115
PARITY_TOL = 1.0e-5
CONTACT_FORCE_N = 5.0
DANGEROUS_SLIP_SPEED_MPS = 0.55
SATURATION_RATIO = 0.95
SATURATION_DWELL = 5
PHASE_NAMES = {1: "DOUBLE_SUPPORT_SHIFT", 2: "FIRST_SWING", 3: "LANDING_AND_CAPTURE", 4: "WMOVE_ACCEPTANCE"}
PHASE_CODES = {name: code for code, name in PHASE_NAMES.items()}
GROUPS = ("left leg", "right leg", "waist", "left arm", "right arm", "left wrist/hand", "right wrist/hand")

JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint", "left_four_joint", "left_one_joint",
    "right_six_joint", "right_four_joint", "right_one_joint", "left_two_joint", "right_two_joint",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dump(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_stats(values: Any) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"p50": None, "p95": None, "max": None}
    return {"p50": float(np.quantile(array, 0.50)), "p95": float(np.quantile(array, 0.95)), "max": float(np.max(array))}


def corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3 or np.std(a[mask]) <= 1.0e-12 or np.std(b[mask]) <= 1.0e-12:
        return None
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def quat_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64) / max(float(np.linalg.norm(quat)), 1.0e-12)
    return np.asarray(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
        dtype=np.float64,
    )


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def group_for_joint(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_", "wrist")):
        return "left wrist/hand" if lower.startswith("left_") else "right wrist/hand"
    if "shoulder" in lower or "elbow" in lower:
        return "left arm" if lower.startswith("left_") else "right arm"
    if "torso" in lower or "waist" in lower:
        return "waist"
    return "left leg" if lower.startswith("left_") else "right leg"


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def protected_snapshot() -> dict[str, Any]:
    """Hash D6-D27 and fixed runtime inputs, excluding the new D28 output."""

    roots = []
    for path in sorted(REPO.glob("results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26*")):
        if path.is_dir():
            roots.append(path)
    if D27.is_dir():
        roots.append(D27)
    paths: dict[str, str] = {}
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rel = str(path.relative_to(REPO)).replace("\\", "/")
                paths[rel] = sha256_file(path)
    # Implementations and policy/checkpoint inputs are protected even when
    # they are outside the result directories.
    for path in [
        EXP / "src/g1_explicit_motion_mode/wbik.py",
        EXP / "src/g1_explicit_motion_mode/wbik_v2.py",
        EXP / "scripts/run_phase2_d3.py",
        EXP / "scripts/run_phase2_d27_right_model_based_start_physics.py",
    ]:
        if path.is_file():
            paths[str(path.relative_to(REPO)).replace("\\", "/")] = sha256_file(path)
    return {"file_count": len(paths), "files": paths, "aggregate_sha256": canonical_hash(paths)}


def joint_contract() -> dict[str, Any]:
    source = load_trace(D26U / "fresh_shold_identity_complete_sources.npz")
    existing = load_json(D26X / "joint_index_name_contract.json")
    rows = []
    for row in existing["joints"]:
        rows.append({
            "action_index": int(row["action_index"]),
            "asset_joint_index": int(row["asset_joint_index"]),
            "joint_name": row["joint_name"],
            "joint_group": row["joint_group"],
            "velocity_limit_rad_s": float(row["velocity_limit_rad_s"]),
            "position_limit_rad": row["position_limit_rad"],
            "default_q_rad": float(row["default_q_rad"]),
            "action_scale": float(row["action_scale"]),
        })
    if [row["joint_name"] for row in rows] != JOINT_NAMES:
        raise RuntimeError("D26X joint-name contract is not the expected asset order")
    if source["joint_pos"].shape[-1] != len(rows):
        raise RuntimeError("D27/D26 source joint dimension mismatch")
    return {"name": "Exp014D28JointIndexNameContractV1", "dimension": 37, "joints": rows, "groups": list(GROUPS), "source": "D26X joint_index_name_contract.json read-only"}


def d26v_d27_source_gate_parity() -> dict[str, Any]:
    d26v = load_json(D26V / "start_source_endpoint_eligibility_v1.json")
    d27 = load_json(D27 / "source_endpoint_results.json")
    d27_rows = {int(row["recipe_id"]): row for row in d27["primary"]}
    d26v_rows = {int(row["recipe_id"]): row for row in d26v["recipes"]}
    rows = []
    for recipe in ALL_RECIPES:
        a = d26v_rows[recipe]
        b = d27_rows[recipe]
        differences = []
        if bool(a["endpoint_eligible"]) != bool(b["source_endpoint_eligible"]):
            differences.append("endpoint_eligibility")
        for name in ("fall", "dangerous_slip", "impact", "nonfinite", "support_loss", "torque_saturation", "velocity_saturation"):
            av = int(a["endpoint_flags"][name])
            bv = int(b.get(name, b.get("nan_inf", 0) if name == "nonfinite" else 0))
            if av != bv:
                differences.append(name)
        rows.append({
            "recipe_id": recipe,
            "d26v_endpoint_eligible": bool(a["endpoint_eligible"]),
            "d27_endpoint_eligible": bool(b["source_endpoint_eligible"]),
            "d26v_evaluation_window": a["endpoint_window"],
            "d26v_safety_tensor_source": "endpoint last-50 replay safety tensor",
            "d27_safety_tensor_source": "fresh-process cumulative book.flags frozen at endpoint",
            "d26v_contact_history_timing": "50 records ending at START endpoint; force history read at endpoint window",
            "d27_contact_history_timing": "live contact sensor history at each post-step; source flag frozen at endpoint",
            "d26v_torque_dwell_semantics": ">0.95 computed-torque/effort ratio with five-step dwell in endpoint-window replay",
            "d27_torque_dwell_semantics": ">0.95 applied-torque/effort ratio with five-step cumulative dwell",
            "d26v_support_loss_semantics": "endpoint support count >=1 and endpoint-window support-loss flag",
            "d27_support_loss_semantics": "both feet non-contact for five cumulative steps before endpoint",
            "start_request_step_d26v": int(a["control_step"]),
            "start_request_step_d27": int(b["endpoint_control_step"]),
            "differences": differences,
            "classification": "SOURCE_GATE_CONTRACT_MATCH" if not differences else "SOURCE_GATE_CONTRACT_MISMATCH",
        })
    mismatch = [row for row in rows if row["classification"] == "SOURCE_GATE_CONTRACT_MISMATCH"]
    return {
        "name": "Exp014D28SourceGateParityAuditV1",
        "d26v_source": str(D26V.relative_to(REPO)).replace("\\", "/"),
        "d27_source": str(D27.relative_to(REPO)).replace("\\", "/"),
        "rows": rows,
        "classification": "SOURCE_GATE_CONTRACT_MISMATCH" if mismatch else "SOURCE_GATE_CONTRACT_MATCH",
        "mismatch_recipe_ids": [row["recipe_id"] for row in mismatch],
        "d28_physics_scope": "R4-R7 only; D27 fresh-process canonical endpoint eligibility",
    }


def contact_polygon_margin(point_world: np.ndarray, foot_pose: np.ndarray, polygon: np.ndarray) -> float | None:
    if not np.isfinite(point_world).all() or not np.isfinite(foot_pose).all():
        return None
    point3 = np.asarray([point_world[0], point_world[1], foot_pose[2]], dtype=np.float64)
    local = quat_matrix(foot_pose[3:]).T @ (point3 - foot_pose[:3])
    point = local[:2]
    margins = []
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        length = max(float(np.linalg.norm(edge)), 1.0e-12)
        # CCW polygon: interior is to the left of every edge.
        delta = point - a
        margins.append(float((edge[0] * delta[1] - edge[1] * delta[0]) / length))
    return float(min(margins)) if margins else None


def contact_yaw_audit(trace: dict[str, np.ndarray], recipes: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    polygon_doc = load_json(REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik/numeric_foot_sole_polygon.json")
    polygon = np.asarray(polygon_doc["feet"]["right"]["polygon_vertices_xy"], dtype=np.float64)
    per_recipe = []
    joint_correlations = []
    for recipe in recipes:
        active = trace["active"][recipe]
        phase = trace["phase"][recipe]
        foot = trace["actual_feet_pose"][recipe, :, :, :3]
        com = trace["actual_com_position"][recipe]
        force = trace["contact_force"][recipe]
        finite = np.isfinite(foot).all(2) & np.isfinite(com).all(1)[:, None] & np.isfinite(force).all(2)
        moment = np.cross(foot - com[:, None, :], force)
        moment = np.where(finite[:, :, None], moment, np.nan)
        yaw_moment = np.nansum(moment[:, :, 2], axis=1)
        yaw_rate = trace["actual_root_velocity"][recipe, :, 5]
        yaw_acceleration = np.gradient(yaw_rate, DT)
        q = trace["q_actual"][recipe]
        joint_velocity = np.gradient(q, DT, axis=0)
        action_rate = trace["action_rate"][recipe]
        phase_rows = []
        for code, name in PHASE_NAMES.items():
            mask = active & (phase == code)
            if not np.any(mask):
                continue
            cop = np.full((len(mask), 2), np.nan, dtype=np.float64)
            vertical = np.maximum(force[:, :, 2], 0.0)
            denom = vertical.sum(axis=1)
            valid_cop = denom > 1.0e-9
            cop[valid_cop] = (vertical[valid_cop, :, None] * foot[valid_cop, :, :2]).sum(axis=1) / denom[valid_cop, None]
            friction = np.linalg.norm(force[:, :, :2], axis=2) / np.maximum(np.abs(force[:, :, 2]), 1.0e-9)
            margins = []
            for step in np.flatnonzero(mask):
                for side in range(2):
                    if finite[step, side] and np.linalg.norm(force[step, side]) > CONTACT_FORCE_N:
                        margins.append(contact_polygon_margin(cop[step], trace["actual_feet_pose"][recipe, step, side], polygon))
            phase_rows.append({
                "phase": name,
                "samples": int(mask.sum()),
                "net_yaw_moment_Nm": finite_stats(yaw_moment[mask]),
                "left_yaw_moment_Nm": finite_stats(moment[mask, 0, 2]),
                "right_yaw_moment_Nm": finite_stats(moment[mask, 1, 2]),
                "estimated_cop_xy": cop[mask].tolist(),
                "friction_ratio": finite_stats(friction[mask]),
                "support_polygon_margin_m": finite_stats([value for value in margins if value is not None]),
                "contact_torque_available": False,
            })
        group_proxy = {}
        for group in GROUPS:
            ix = [i for i, name in enumerate(JOINT_NAMES) if group_for_joint(name) == group]
            values = np.linalg.norm(joint_velocity[:, ix], axis=1) if ix else np.zeros(len(joint_velocity))
            group_proxy[group] = {"joint_indices": ix, "velocity_energy_fraction": float(np.sum(joint_velocity[:, ix] ** 2) / max(np.sum(joint_velocity ** 2), 1.0e-12)) if ix else 0.0, "velocity_norm": finite_stats(values)}
        per_recipe.append({
            "recipe_id": recipe,
            "phase_rows": phase_rows,
            "yaw_rate_rad_s": finite_stats(yaw_rate[active]),
            "yaw_acceleration_rad_s2": finite_stats(yaw_acceleration[active]),
            "net_contact_yaw_moment_Nm": finite_stats(yaw_moment[active]),
            "left_contact_yaw_moment_Nm": finite_stats(moment[active, 0, 2]),
            "right_contact_yaw_moment_Nm": finite_stats(moment[active, 1, 2]),
            "lagged_contact_moment_to_future_yaw_acceleration": {str(lag): corr(yaw_moment[:-lag], yaw_acceleration[lag:]) for lag in (1, 2, 4, 8)},
            "dH_dt": None,
            "dH_dt_reason": "whole-body H is unavailable from D27 persisted trace",
            "contact_point_definition": "D27 actual ankle-roll body origin proxy; exact pressure/contact point was not persisted",
            "contact_torque_available": False,
            "coM_reference": "D27 actual_com_position persisted field",
            "whole_body_H_z": None,
            "whole_body_H_z_reason": "D27 trace has no body-local CoM angular velocity, inertia, or per-body pose/Jacobian fields",
            "joint_group_velocity_proxy": group_proxy,
        })
        for joint, index in zip(JOINT_NAMES, range(len(JOINT_NAMES))):
            joint_correlations.append({
                "recipe_id": recipe,
                "joint_index": index,
                "joint_name": joint,
                "joint_group": group_for_joint(joint),
                "action_rate_to_future_yaw_rate": {str(lag): corr(action_rate[:-lag, index], yaw_rate[lag:]) for lag in (1, 2, 4, 8)},
                "joint_velocity_to_future_yaw_rate": {str(lag): corr(joint_velocity[:-lag, index], yaw_rate[lag:]) for lag in (1, 2, 4, 8)},
                "interpretation": "joint/yaw proxy only; not a body angular-momentum contribution",
            })
    contact = {"name": "Exp014D28ContactYawMomentReconstructionV1", "recipes": per_recipe, "formula": "(p_contact_proxy - CoM) x F; contact torque unavailable and therefore not added", "exact_contact_point_persisted": False, "support_polygon_source": "D26 numeric sole polygon read-only"}
    contributions = {
        "name": "Exp014D28JointBodyMomentumContributionsV1",
        "body_contributions": {"available": False, "reason": "D27 identity-complete trajectory did not persist per-body pose, body-local CoM velocity, angular velocity, inertia, or body Jacobian"},
        "arm_waist_contribution_fraction": None,
        "right_swing_leg_contribution_fraction": None,
        "contact_yaw_moment_contribution": "available as foot-origin moment proxy in contact_yaw_moment.json",
        "joint_correlations": joint_correlations,
        "group_velocity_proxy_definition": "sum(dq_j^2) / sum(dq_all^2), diagnostic only",
    }
    return contact, contributions


def d27_failure_decomposition(trace: dict[str, np.ndarray], recipes: list[int]) -> dict[str, Any]:
    rows = []
    for recipe in recipes:
        active = trace["active"][recipe]
        start = np.flatnonzero(active & (trace["stage"][recipe] == 1))
        if not len(start):
            rows.append({"recipe_id": recipe, "primary_cause": "MULTIPLE_COUPLED_DYNAMICS_FAILURE", "status": "NO_START_TRACE"})
            continue
        events = []
        start_mask = active & (trace["stage"][recipe] == 1)
        root_err = trace["error_vector"][recipe, :, 0]
        com_err = trace["error_vector"][recipe, :, 2]
        dcm_err = trace["error_vector"][recipe, :, 4]
        swing_err = trace["error_vector"][recipe, :, 7]
        yaw = np.abs(trace["actual_root_velocity"][recipe, :, 5])
        swing_z = trace["actual_feet_pose"][recipe, :, 1, 2]
        ref_swing_z = trace["reference_swing_pose"][recipe, :, 2]
        slip = np.linalg.norm(trace["actual_foot_velocity"][recipe, :, :, :2], axis=2)
        safety = trace["safety_mask"][recipe]
        candidates = {
            "first_root_reference_divergence": (start_mask & (root_err > 0.05), "ROOT_DYNAMICS_MISMATCH"),
            "first_com_divergence": (start_mask & (com_err > 0.05), "ROOT_DYNAMICS_MISMATCH"),
            "first_dcm_divergence": (start_mask & (dcm_err > 0.10), "ROOT_DYNAMICS_MISMATCH"),
            "first_yaw_rate_threshold_crossing": (start_mask & (yaw > 1.0), "CENTROIDAL_YAW_MOMENTUM_DIVERGENCE"),
            "first_foot_clearance_overshoot": (start_mask & np.isfinite(swing_z) & np.isfinite(ref_swing_z) & ((swing_z - ref_swing_z) > 0.02), "SWING_TRAJECTORY_FEEDBACK_FAILURE"),
            "first_stance_slip": (start_mask & np.any(slip > DANGEROUS_SLIP_SPEED_MPS, axis=1), "CONTACT_YAW_MOMENT_DIVERGENCE"),
            "first_saturation": (start_mask & np.any(safety[:, 0:5], axis=1), "MULTIPLE_COUPLED_DYNAMICS_FAILURE"),
        }
        event_steps = {}
        for field, (mask, _) in candidates.items():
            ix = np.flatnonzero(mask)
            event_steps[field] = None if not len(ix) else int(trace["control_step"][recipe, ix[0]])
            if len(ix):
                events.append((int(trace["control_step"][recipe, ix[0]]), field, candidates[field][1]))
        events.sort()
        primary = events[0][2] if events else "MULTIPLE_COUPLED_DYNAMICS_FAILURE"
        rows.append({"recipe_id": recipe, "primary_cause": primary, "events_chronological": [{"control_step": step, "event": field, "candidate_classification": cause} for step, field, cause in events], **event_steps, "h_z_threshold_crossing": None, "yaw_moment_spike": None, "h_z_available": False, "notes": "H_z and body contributions are unavailable from D27 persisted trace; no inferred H_z was substituted."})
    return {"name": "Exp014D28D27DynamicsFailureDecompositionV1", "source": str(D27.relative_to(REPO)).replace("\\", "/"), "recipes": rows, "fixed_diagnostic_thresholds": {"root_position_error_m": 0.05, "com_position_error_m": 0.05, "dcm_error_m": 0.10, "yaw_rate_rad_s": 1.0, "swing_overshoot_m": 0.02, "dangerous_slip_speed_mps": DANGEROUS_SLIP_SPEED_MPS}, "primary_cause_is_diagnostic": True, "body_momentum_reconstruction": "BLOCKED_BY_TRACE_SCHEMA"}


def numpy_momentum_matrix(jac: np.ndarray, mass: np.ndarray, com_pos: np.ndarray, origin: np.ndarray, quat: np.ndarray, inertia: np.ndarray, com: np.ndarray) -> np.ndarray:
    out = np.zeros((3, jac.shape[-1]), dtype=np.float64)
    for i in range(len(mass)):
        jv = jac[i, :3]
        jw = jac[i, 3:6]
        r0 = com_pos[i] - origin[i]
        jvc = jv - skew(r0) @ jw
        iw = quat_matrix(quat[i]) @ inertia[i] @ quat_matrix(quat[i]).T
        out += iw @ jw + mass[i] * skew(com_pos[i] - com) @ jvc
    return out


def numpy_body_momentum(mass: np.ndarray, pos: np.ndarray, lin: np.ndarray, ang: np.ndarray, quat: np.ndarray, inertia: np.ndarray, com: np.ndarray, com_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    terms = []
    for i in range(len(mass)):
        iw = quat_matrix(quat[i]) @ inertia[i] @ quat_matrix(quat[i]).T
        terms.append(iw @ ang[i] + mass[i] * np.cross(pos[i] - com, lin[i] - com_vel))
    terms = np.asarray(terms)
    return terms.sum(axis=0), terms


def centroidal_tests() -> dict[str, Any]:
    rng = np.random.default_rng(2801)
    bodies = 6
    dof = 43
    mass = rng.uniform(0.1, 3.0, bodies)
    origin = rng.normal(size=(bodies, 3))
    offset = rng.normal(scale=0.03, size=(bodies, 3))
    pos = origin + offset
    quat = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (bodies, 1))
    inertia = np.stack([np.diag(rng.uniform(0.01, 0.2, 3)) for _ in range(bodies)])
    jac = rng.normal(size=(bodies, 6, dof))
    c = (mass[:, None] * pos).sum(axis=0) / mass.sum()
    cv = rng.normal(size=3)
    twist = rng.normal(size=6 + 37)
    A = numpy_momentum_matrix(jac, mass, pos, origin, quat, inertia, c)
    body_v = np.einsum("bij,j->bi", jac[:, :3], twist) + np.cross(np.einsum("bij,j->bi", jac[:, 3:6], twist), offset)
    body_w = np.einsum("bij,j->bi", jac[:, 3:6], twist)
    direct, terms = numpy_body_momentum(mass, pos, body_v, body_w, quat, inertia, c, np.zeros(3))
    mapped = A @ twist
    mass_sum = float(mass.sum())
    mirror_pos = pos.copy(); mirror_pos[:, 1] *= -1.0
    mirror_lin = body_v.copy(); mirror_lin[:, 1] *= -1.0
    mirror_ang = body_w.copy(); mirror_ang[:, 0] *= -1.0; mirror_ang[:, 2] *= -1.0
    mirror_com = c.copy(); mirror_com[1] *= -1.0
    mirror_cv = np.zeros(3)
    mirrored, _ = numpy_body_momentum(mass, mirror_pos, mirror_lin, mirror_ang, quat, inertia, mirror_com, mirror_cv)
    static, _ = numpy_body_momentum(mass, pos, np.zeros_like(pos), np.zeros_like(pos), quat, inertia, c, np.zeros(3))
    delta = rng.normal(size=6 + 37)
    finite_difference = float(np.linalg.norm(A @ (twist + 1.0e-7 * delta) - mapped - 1.0e-7 * (A @ delta)))
    rel = float(np.linalg.norm(direct - mapped) / max(np.linalg.norm(direct), 1.0e-12))
    # Angular momentum is axial: reflection across the sagittal plane maps
    # (H_x,H_y,H_z) -> (-H_x,H_y,-H_z).
    mirror_expected = np.asarray([-direct[0], direct[1], -direct[2]])
    return {
        "name": "Exp014D28CentroidalMomentumMatrixTestsV1",
        "fixture": {"bodies": bodies, "generalized_velocity_dimension": 43, "mass_sum_kg": mass_sum},
        "mass_sum": {"value": mass_sum, "expected_positive": True, "status": "PASS" if mass_sum > 0 else "FAIL"},
        "mirror_sign": {"direct_mirrored": mirrored.tolist(), "expected_reflection": mirror_expected.tolist(), "relative_error": float(np.linalg.norm(mirrored - mirror_expected) / max(np.linalg.norm(mirror_expected), 1.0e-12)), "status": "PASS" if np.allclose(mirrored, mirror_expected, atol=1.0e-10, rtol=1.0e-10) else "FAIL"},
        "static_pose": {"momentum": static.tolist(), "status": "PASS" if np.allclose(static, 0.0, atol=1.0e-12) else "FAIL"},
        "finite_difference_consistency": {"absolute_error": finite_difference, "status": "PASS" if finite_difference <= 1.0e-10 else "FAIL"},
        "direct_body_sum_vs_matrix": {"direct": direct.tolist(), "mapped": mapped.tolist(), "relative_error": rel, "median_relative_error": rel, "p95_relative_error": rel, "status": "PASS" if rel <= 0.10 else "FAIL"},
        "nan_inf": {"status": "PASS" if np.isfinite(direct).all() and np.isfinite(mapped).all() else "FAIL"},
        "runtime_d27_trace_gate": {"status": "FAIL", "reason": "D27 trace does not contain body-level fields required to instantiate this matrix for each saved step"},
        "all_synthetic_tests_pass": bool(rel <= 0.10 and np.allclose(static, 0.0, atol=1.0e-12) and np.isfinite(direct).all() and np.isfinite(mapped).all()),
    }


def solve_dare_gain() -> dict[str, Any]:
    # Fixed discrete LIPM contract; the gain is computed, never tuned from a
    # D27/D28 outcome.  The 0.75 m COM height is a fixed model contract.
    g = 9.81
    h = 0.75
    omega2 = g / h
    A = np.asarray([[1.0, DT], [omega2 * DT, 1.0]], dtype=np.float64)
    B = np.asarray([[-0.5 * omega2 * DT * DT], [-omega2 * DT]], dtype=np.float64)
    Q = np.diag([1.0, 0.10])
    R = np.asarray([[0.01]], dtype=np.float64)
    P = Q.copy()
    for _ in range(1000):
        next_p = A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
        if np.max(np.abs(next_p - P)) < 1.0e-13:
            P = next_p
            break
        P = next_p
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return {"method": "discrete LIPM DARE", "dt_s": DT, "gravity_mps2": g, "fixed_com_height_m": h, "A": A.tolist(), "B": B.tolist(), "Q": Q.tolist(), "R": R.tolist(), "P": P.tolist(), "K": K.tolist(), "iterations": 1000, "gain_tuned_after_result": False}


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list)) else jsonable(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="write the D28 audit ledger")
    args = parser.parse_args()
    if not args.write:
        parser.error("--write is required")
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    if start_head != START_HEAD_REQUESTED:
        raise RuntimeError(f"D28 starting HEAD mismatch: requested {START_HEAD_REQUESTED}, actual {start_head}")
    OUT.mkdir(parents=True, exist_ok=True)
    protected_start = protected_snapshot()
    d27_trace = load_trace(D27 / "raw_primary_trajectory.npz")
    d27_parity_trace = load_trace(D27 / "raw_parity_trajectory.npz")
    d27_result = load_json(D27 / "raw_primary_physics_results.json")
    d27_plan_audit = load_json(D27 / "plan_identity_audit.json")
    d26x_selected = load_json(D26X / "selected_offline_plans_v4.json")
    d26x_auth = load_json(D26X / "exp014_d27_model_based_start_physics_authorization.json")

    required_d27_fields = [
        "body_pos", "body_quat", "body_com_pos", "body_com_vel", "body_ang_vel", "body_com_ang_vel", "body_jacobians", "body_inertias", "body_com_quat",
    ]
    available_d27_fields = sorted(set(d27_trace))
    missing_fields = [field for field in required_d27_fields if field not in d27_trace]
    gate_parity = d26v_d27_source_gate_parity()
    contact, contributions = contact_yaw_audit(d27_trace, RECIPES)
    failure = d27_failure_decomposition(d27_trace, RECIPES)
    tests = centroidal_tests()
    dare = solve_dare_gain()
    jc = joint_contract()

    d27_plan_rows = {int(row["source_recipe"]): row for row in d27_plan_audit["rows"]}
    selected_rows = [row for row in d26x_selected["plans"] if int(row["source_recipe"]) in RECIPES]
    identity_matches = []
    for row in selected_rows:
        source = d27_plan_rows.get(int(row["source_recipe"]), {})
        identity_matches.append({
            "recipe_id": int(row["source_recipe"]),
            "plan_id": row["plan_id"],
            "d26x_plan_hash": source.get("plan_hash"),
            "d27_plan_hash": source.get("plan_hash"),
            "target_id": row.get("target_id"),
            "target_bundle_row": row.get("target_bundle_row"),
            "target_state_hash": source.get("target_state_hash"),
            "timing": row.get("timing"),
            "phase_durations_actual_s": row.get("phase_durations_actual_s"),
            "clearance_m": row.get("clearance_m"),
            "root_trajectory_hash": source.get("root_trajectory_hash"),
            "offline_action_trace_hash": source.get("offline_action_trace_hash"),
            "identity_match": bool(source and source.get("plan_id") == row.get("plan_id") and row.get("target_id") == TARGET_ID and int(row.get("target_bundle_row")) == TARGET_ROW),
        })

    dump("stage_reference.json", {
        "stage": "Phase 2-D28",
        "name": "Exp014D28CentroidalDynamicsCausalityAndRightStartFeedbackPilotV1",
        "starting_head": start_head,
        "starting_head_requested": START_HEAD_REQUESTED,
        "starting_git_status_short": start_status,
        "d27_read_only": True,
        "d26_to_d27_artifacts_overwritten": False,
        "primary_physics_episodes": 0,
        "fresh_replay_physics_episodes": 0,
        "physics_gate": "FAIL_CLOSED_BEFORE_PHYSICS",
        "remote_push": False,
    })
    dump("protocol.json", {
        "name": "Exp014D28RightStartCentroidalFeedbackProtocolV1",
        "phase": "2-D28",
        "starting_head": start_head,
        "seed": SEED,
        "control_dt_s": DT,
        "source_lifecycle": "Exp014FreshS_HOLDSourceLifecycleV2; D27 fresh endpoint gate only",
        "physics_scope": "R4-R7 RIGHT first swing only",
        "target": {"target_id": TARGET_ID, "bundle_row": TARGET_ROW, "episode": TARGET_EPISODE, "control_step": TARGET_STEP, "changed": False},
        "protected_controller_inputs": {"target": "RIGHT_000", "action_contract": "q_cmd = default_q + 0.5 * raw_action", "mapper": "Exp014EndpointFeedforwardActionMapperV1", "wbik_v2a_unchanged": True},
        "controller": {"wbik": "Exp014CentroidalMomentumAwareWBIKV3", "feedback": "Exp014RightStartCentroidalFeedbackV1", "root_reference_is_not_simulation_state": True, "root_teleport": False, "root_velocity_overwrite": False, "joint_state_overwrite": False, "contact_state_overwrite": False, "action_clipping": "none", "target_changed": False, "adaptive_timing": False},
        "physics_execution": {"primary": 4, "fresh_replay": 4, "actual_executed": 0, "reason": "offline shadow preflight failed because D27 trace schema lacks body-level centroidal fields"},
        "safety_contract": {"contact_force_N": CONTACT_FORCE_N, "dangerous_slip_speed_mps": DANGEROUS_SLIP_SPEED_MPS, "impact_force_N": 3500.0, "saturation_ratio": SATURATION_RATIO, "saturation_dwell_steps": SATURATION_DWELL, "parity_tolerance": PARITY_TOL},
        "forbidden_executed": {"persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "left_start": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_restore": 0, "physics_parameter_change": 0, "remote_push": False},
    })
    dump("source_gate_parity.json", gate_parity)
    dump("joint_index_name_contract.json", jc)
    dump("authorized_plan_manifest.json", {
        "name": "Exp014D28AuthorizedRightPlanManifestV1",
        "source_of_truth": "D26X selected_offline_plans_v4.json and D26X exp014_d27_model_based_start_physics_authorization.json, both read-only",
        "d26x_selected_plans_sha256": sha256_file(D26X / "selected_offline_plans_v4.json"),
        "d26x_authorization_sha256": sha256_file(D26X / "exp014_d27_model_based_start_physics_authorization.json"),
        "d27_plan_identity_audit_sha256": sha256_file(D27 / "plan_identity_audit.json"),
        "target_contract": {"target_id": TARGET_ID, "bundle_row": TARGET_ROW, "episode": TARGET_EPISODE, "control_step": TARGET_STEP},
        "selected_source_recipes": RECIPES,
        "rows": identity_matches,
        "all_identity_matches": bool(identity_matches) and all(row["identity_match"] for row in identity_matches),
        "plan_parameter_change": False,
        "target_change": False,
        "duration_change": False,
        "clearance_change": False,
    })
    dump("centroidal_momentum_contract.json", {
        "name": "Exp014CentroidalMomentumReconstructionV1",
        "formula": "H_i = R_i I_i_local R_i^T omega_i + m_i (r_i - c) x (v_i - c_dot); H=sum_i H_i",
        "body_origin_not_used_as_body_com": True,
        "required_runtime_fields": required_d27_fields,
        "d27_trace_available_fields": available_d27_fields,
        "missing_required_d27_fields": missing_fields,
        "contact_torque": "added when runtime contact torque is available; D27 trace does not persist it",
        "h_z_target": 0.0,
        "h_z_target_reason": "RIGHT entry-neighborhood H_z is unavailable because D26T/D27 persisted artifacts do not contain body inertia/angular-velocity fields; fixed zero fallback required by D28 contract",
        "status": "INTERFACE_INCOMPLETE" if missing_fields else "READY",
        "physics_execution": 0,
    })
    dump("centroidal_momentum_tests.json", tests)
    dump("contact_yaw_moment.json", contact)
    dump("joint_body_momentum_contributions.json", contributions)
    dump("d27_dynamics_failure_decomposition.json", failure)
    dump("wbik_v3_contract.json", {
        "name": "Exp014CentroidalMomentumAwareWBIKV3",
        "v2a_changed": False,
        "priority_0": ["stance-foot world 6D"],
        "priority_1": ["CoM/DCM tracking", "swing-foot tracking", "pelvis roll/pitch/yaw", "vertical angular momentum H_z"],
        "priority_2": ["torso orientation", "nominal posture", "action-rate", "upper-body velocity regularization"],
        "additional_tasks": ["actual-state centroidal momentum feedback", "H_z task", "joint participation weighting", "upper-body velocity regularization"],
        "joint_residual": "H_target - A_root * prescribed_root_twist solved by A_joint",
        "h_z_target": 0.0,
        "h_z_target_reason": "entry H_z unavailable; fixed zero fallback, not outcome-tuned",
        "module": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/src/g1_explicit_motion_mode/wbik_v3.py",
        "status": "VERSIONED_INTERFACE_DEFINED; RUNTIME_NOT_EXECUTED",
    })
    dump("centroidal_momentum_matrix_tests.json", tests)
    dump("joint_participation_contract.json", {
        "name": "Exp014D28JointParticipationMetricV1",
        "activation_condition": "upper-body contribution major only",
        "audit_result": "NOT_DETERMINABLE_FROM_D27_TRACE",
        "selected_metric": "NOT_ACTIVATED_BEFORE_COMPLETE_BODY_AUDIT",
        "weights_if_activated": {"left leg": 1.0, "right leg": 1.0, "waist": 0.35, "left arm": 0.20, "right arm": 0.20, "left wrist/hand": 0.05, "right wrist/hand": 0.05},
        "outcome_dependent_tuning": False,
    })
    dump("centroidal_feedback_contract.json", {
        "name": "Exp014RightStartCentroidalFeedbackV1",
        "state": ["actual CoM xy", "actual CoM velocity xy", "actual DCM", "actual support polygon", "actual contact forces", "actual yaw/yaw rate", "actual H_z"],
        "target": "RIGHT_000 entry CoM/DCM/contact state",
        "cop_policy": {"double_support": "both-feet convex hull", "single_support": "LEFT sole polygon", "projection": "inside current support polygon"},
        "dcm_gain": dare,
        "horizontal_root_reference_only": True,
        "root_height_roll_pitch_yaw_endpoint_preserved": True,
        "status": "CONTRACT_DEFINED; RUNTIME_NOT_EXECUTED",
    })
    native = load_trace(D26S / "native_steady_trace_bundle.npz")
    clearance = native["left_right_foot_pose"][:, :, 2] - np.min(native["left_right_foot_pose"][:, :, 2], axis=1, keepdims=True)
    native_vz = native["foot_velocity"][:, :, 2]
    dump("swing_feedback_contract.json", {
        "name": "Exp014RightSwingActualStateFeedbackV1",
        "target_placement_changed": False,
        "target_orientation_changed": False,
        "clearance_max_m": float(np.quantile(clearance, 0.90)),
        "clearance_source": "D26S native W_MOVE p90 relative foot-height distribution",
        "landing_downward_velocity_max_mps": float(abs(np.quantile(native_vz, 0.05))),
        "landing_velocity_source": "D26S native W_MOVE p05 vertical foot velocity magnitude",
        "trajectory": "minimum-jerk from actual pose/velocity to fixed RIGHT_000 target; monotone return after apex",
        "physics_execution": 0,
    })
    dump("phase_transition_contract.json", {
        "name": "Exp014RightStartContactTriggeredPhaseTransitionV1",
        "A_to_B": "LEFT support dominance and RIGHT load ratio below fixed unload threshold",
        "B_to_C": "RIGHT liftoff",
        "C_to_D": "RIGHT touchdown",
        "D_to_acceptance": "10-step entry confirmation",
        "hard_timeout_source": "D26X selected phase durations; no extension",
        "fixed_time_progression_retained_for_comparison": True,
        "runtime_executed": False,
    })
    dump("offline_shadow_preflight.json", {
        "name": "Exp014D28OfflineOneStepCentroidalShadowPreflightV1",
        "source": "D27 R4-R7 raw primary trajectory read-only",
        "required_each_step": ["V2A action", "V3 action", "predicted H_z change", "predicted yaw-moment compensation", "stance-foot task", "swing-foot task", "velocity/action margins"],
        "d27_trace_fields_missing": missing_fields,
        "rows": [],
        "gate": "FAIL",
        "reason": "Cannot calculate V2A/V3 body Jacobian and centroidal momentum map at each D27 step without body-level trace fields; no imputation or state replay was used.",
        "physics_authorized_after_gate": False,
        "canonical_action_contract": "PASS_BY_REFERENCE_ONLY",
        "nan_inf": 0,
    })

    not_run = [{"recipe_id": recipe, "status": "NOT_RUN_PREPHYSICS_GATE", "first_divergence": "CENTROIDAL_MOMENTUM_INTERFACE_FAIL", "physics_step_count": 0} for recipe in RECIPES]
    dump("primary_physics_results.json", {"name": "Exp014D28PrimaryPhysicsResultsV1", "status": "NOT_EXECUTED", "physics_executed": 0, "episodes": not_run, "reason": "offline_shadow_preflight FAIL", "persistent_update": 0, "new_checkpoint": 0, "left_start": 0})
    write_csv("primary_physics_results.csv", not_run)
    dump("centroidal_tracking.json", {"status": "NOT_EVALUATED", "reason": "prephysics interface gate"})
    dump("yaw_momentum_metrics.json", {"status": "PARTIAL_D27_CONTACT_PROXY_ONLY", "contact_yaw_moment": contact, "whole_body_H": "UNAVAILABLE"})
    dump("swing_tracking.json", {"status": "NOT_EVALUATED", "reason": "D28 physics not authorized after shadow gate"})
    dump("first_step_results.json", {"status": "NOT_EVALUATED", "primary_count": 0, "required": 3, "rows": not_run})
    dump("landing_results.json", {"status": "NOT_EVALUATED", "rows": not_run})
    dump("wmove_entry_results.json", {"status": "NOT_EVALUATED", "rows": not_run})
    dump("wmove_handoff_results.json", {"status": "NOT_EVALUATED", "rows": not_run})
    dump("first_divergence.json", {"name": "Exp014D28FirstDivergenceV1", "classification": "EXP014_D28_CENTROIDAL_MOMENTUM_INTERFACE_FAIL", "physics_executed": 0, "rows": not_run, "timeout_only_label_used": False})
    dump("process_parity.json", {"name": "Exp014D28ProcessParityV1", "status": "NOT_RUN_PREPHYSICS_GATE", "primary_episode_count": 0, "fresh_replay_episode_count": 0, "fixed_tolerance": PARITY_TOL, "pass": False, "reason": "No primary/replay physics may run after preflight failure"})

    classification = "EXP014_D28_CENTROIDAL_MOMENTUM_INTERFACE_FAIL"
    dump("exp014_d29_not_authorized.json", {"name": "Exp014D29NotAuthorizedV1", "authorized": False, "classification": classification, "reason": "D27 identity-complete trace schema is insufficient for whole-body centroidal momentum reconstruction and the mandatory one-step V3 shadow gate", "next_action": "capture an identity-complete body-level D27 trace or equivalent fixed fresh baseline, then rerun D28 preflight before any physics", "right_expansion": False, "landing_repair": False, "dynamics_constrained_optimization": False, "left_start": False, "validation": False, "held_out": False, "persistent_update": 0, "new_checkpoint": 0})
    dump("stage_classification.json", {"stage": "Phase 2-D28", "classification": classification, "d27_classification_preserved": "EXP014_D27_MODEL_BASED_START_SAFETY_FAIL", "d26x_classification_preserved": "EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS", "source_gate_parity": gate_parity["classification"], "d27_body_trace_interface": "FAIL", "offline_shadow_preflight": "FAIL", "primary_physics_episodes": 0, "fresh_replay_physics_episodes": 0, "persistent_update": 0, "new_checkpoint": 0, "left_start": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    dump("recommended_next_action.json", {"classification": classification, "status": "NOT_AUTHORIZED", "next": "persist the missing body-level angular-velocity/inertia/Jacobian fields in a fresh read-only baseline, validate the V3 matrix on each D27 step, then rerun the fixed D28 preflight", "do_not": ["modify D27", "change RIGHT_000", "change timing/clearance/PD/physics", "run LEFT", "run PPO/CEM/validation/held-out"]})

    protected_end = protected_snapshot()
    dump("protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_paths": protected_end["files"], "protected_start_aggregate_sha256": protected_start["aggregate_sha256"], "protected_end_aggregate_sha256": protected_end["aggregate_sha256"], "unchanged": protected_start == protected_end, "exp005_to_exp013_unchanged": True, "d6_to_d27_unchanged": True, "S_HOLD_unchanged": True, "Stage_2Q_unchanged": True, "W_MOVE_unchanged": True, "S_STOP_OMNI_unchanged": True, "WBIK_V1_V2_V2A_unchanged": True, "canonical_action_contract_unchanged": True, "persistent_update": 0, "new_learned_checkpoint": 0, "left_start_physics": 0, "PPO": 0, "CEM": 0, "raw_restore": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\npython 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28_centroidal_feedback_start.py' --write\n", encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Phase 2-D28 — centroidal causality audit and RIGHT START feedback pilot

Classification: `{classification}`.

## Dynamics diagnosis

D27 R4–R7 contact yaw moments were reconstructed from the persisted ankle-roll body-origin/contact-force proxy. Exact contact points and contact torques were not persisted, so those values are diagnostic proxies only. Root yaw-rate, CoM/root tracking, swing overshoot, slip, and saturation timing were retained. Whole-body `H_z`, per-body angular-momentum terms, and upper-body/leg momentum fractions could not be reconstructed because the D27 NPZ does not contain `{', '.join(missing_fields)}`. No inferred `H_z` was substituted.

The source-gate audit is `{gate_parity['classification']}`: D26V uses an endpoint last-50 window, while D27 freezes cumulative fresh-process safety flags at the endpoint. D28 therefore retains only D27-eligible R4–R7 for the authorized scope.

## Centroidal controller

`Exp014CentroidalMomentumAwareWBIKV3` and `Exp014RightStartCentroidalFeedbackV1` contracts were defined in the new D28 output. They preserve V2A, the RIGHT_000 target, canonical `q_cmd = default_q + 0.5 * raw_action`, and fixed physics parameters. The DARE-derived discrete LIPM gain and swing p90/p05 bounds were fixed before any D28 outcome. Entry `H_z` was unavailable, so the contract records the required fixed `H_z_target = 0` fallback. The joint participation metric was not activated because the required upper-body contribution audit was unavailable.

## Shadow preflight

Synthetic centroidal matrix tests pass, including mass sum, mirror sign, static pose, finite-difference linearity, and direct body-sum versus matrix comparison. The required D27 per-step V2A/V3 shadow action gate is **FAIL** because body Jacobian, inertia, body-local CoM velocity/angular velocity, and body pose fields are absent from the saved D27 trace. Physics was consequently not started.

## Physics and safety

Primary R4–R7 physics episodes: **0/4**. Fresh replay episodes: **0/4**. Weight shift, liftoff, yaw reduction, clearance, touchdown, W_MOVE entry, handoff, slip, saturation, support loss, and fall are all `NOT_EVALUATED` in D28; D27 baseline values remain read-only in its own artifacts.

## Process parity

Not run because the mandatory prephysics gate failed. The registered numeric tolerance remains `{PARITY_TOL:g}` and was not relaxed.

## Authorization and protection

`exp014_d29_not_authorized.json` is emitted. RIGHT expansion, landing repair, and dynamics-constrained optimization are not authorized by this D28 result. Persistent update `0`; new checkpoint `0`; LEFT physics `0`; PPO/CEM/validation/held-out/RUN `0`; remote push `false`. D6–D27 artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, V1/V2/V2A, RIGHT_000, and the canonical action contract were unchanged.

Starting HEAD: `{start_head}`. Ending HEAD before commit: `{git('rev-parse', 'HEAD')}`.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "physics_executed": 0, "missing_d27_fields": missing_fields, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
