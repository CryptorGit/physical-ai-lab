"""Phase 2-D27: authorized RIGHT-first model-based START physics diagnostic.

This runner is deliberately narrow.  It reads the D26X authorization and
selected exact-medoid plans, verifies their identity before simulation, then
executes two independent fresh Isaac Lab processes: one primary set of eight
recipe lifecycles and one process-parity replay.  It never writes a policy,
checkpoint, dataset, optimizer, or protected D26X artifact.

The controller route is:

    fresh reset recipe -> frozen S_HOLD lifecycle -> fixed D26X plan
    -> current-state WBIK V2A -> endpoint feedforward mapper -> q_cmd
    -> canonical normalized action -> hard W_MOVE handoff after entry gate

There is no root/state teleport, adaptive timing, contact-triggered stretching,
action clipping, action blending, LEFT execution, or RUN integration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d27_right_model_based_start_physics"
REPORT = REPO / "research/exp_014_phase_2_d27_right_model_based_start_physics_report.md"

D26X = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26x_timing_and_target_set"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D25 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher"

P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
P1 = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"

DT = 0.02
SEED = 20279941
RECIPES = list(range(8))
N_ENVS = 8
CONFIRMATION_STEPS = 50
ADDITIONAL_HOLD_STEPS = 50
MAX_STEPS = 320
WMOVE_STEPS = 75
WMOVE_SPEED = 0.30
RIGHT_TARGET_ROW = 9330
RIGHT_TARGET_ID = "RIGHT_000"
RIGHT_ENTRY_EPISODE = 187
RIGHT_ENTRY_CONTROL_STEP = 115

CONTACT_FORCE_N = 5.0
DANGEROUS_SLIP_SPEED_MPS = 0.55
IMPACT_FORCE_N = 3500.0
SATURATION_RATIO = 0.95
SATURATION_DWELL_STEPS = 5
FORWARD_DISPLACEMENT_M = 0.03
FIRST_STEP_YAW_RATE_RAD_S = 0.15
ENTRY_FORWARD_ERROR_MPS = 0.12
ENTRY_LATERAL_VELOCITY_MPS = 0.08
ENTRY_YAW_RATE_RAD_S = 0.10
ENTRY_CONFIRMATION_STEPS = 10
PELVIS_ROLL_PITCH_SAFE_RAD = 0.80

# These are fixed before either physics process.  They are not adjusted from
# a replay outcome.  Isaac GPU execution can be numerically, but not bitwise,
# reproducible across fresh processes, so comparison first attempts bitwise
# equality and then uses this pre-registered tolerance.
PARITY_ABS_TOL = 1.0e-5
PARITY_REL_TOL = 1.0e-5

PHASES = ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
PHASE_CODE = {"": 0, "DOUBLE_SUPPORT_SHIFT": 1, "FIRST_SWING": 2, "LANDING_AND_CAPTURE": 3, "WMOVE_ACCEPTANCE": 4, "WMOVE": 5}
STAGE_CODE = {"S_HOLD": 0, "START": 1, "WMOVE": 2, "TERMINAL": 3}
SAFETY_NAMES = ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nonfinite")


def minimum_jerk(s: float) -> float:
    """Exact D26 endpoint-mapper scalar, kept local to avoid mutating D26X."""
    u = min(1.0, max(0.0, float(s)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# D3 imports the frozen task registration and supplies only the reset-recipe
# lifecycle.  D26X is imported for read-only reference reconstruction; main()
# is never called.
d3 = load_module("exp014_d27_d3_read_only", EXP / "scripts/run_phase2_d3.py")
d26x = load_module("exp014_d27_d26x_read_only", EXP / "scripts/finalize_phase2_d26x.py")

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def tensor_np(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        result = value.detach().cpu().numpy()
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return np.asarray(result)


def torch64(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)


def mean_percentiles(values: list[float] | np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {"p50": None, "p95": None, "max": None}
    return {"p50": float(np.quantile(array, 0.50)), "p95": float(np.quantile(array, 0.95)), "max": float(np.max(array))}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def protected_snapshot() -> dict[str, Any]:
    """Hash D6-D26X and protected policy/checkpoint inputs without D27 files."""
    paths = set(d26x.protected_file_candidates())
    tracked = git("ls-files").splitlines()
    for relative in tracked:
        normalized = relative.replace("\\", "/")
        if "/phase_2_d26x_" in normalized:
            paths.add(REPO / relative)
    for path in (P0, P1, WMOVE):
        if path.exists():
            paths.add(path)
    files = {}
    for path in sorted(paths):
        if path.is_file():
            files[str(path.relative_to(REPO)).replace("\\", "/")] = sha256_file(path)
    return {"file_count": len(files), "files": files, "aggregate_sha256": canonical_hash(files)}


def source_target_hash(source: dict[str, np.ndarray], native: dict[str, np.ndarray], recipe: int) -> str:
    fields = {
        "root_pose": source["root_pose"][recipe],
        "root_velocity": source["root_velocity"][recipe],
        "joint_pos": source["joint_pos"][recipe],
        "joint_vel": source["joint_vel"][recipe],
        "body_pos_w": source["body_pos_w"][recipe],
        "body_quat_w": source["body_quat_w"][recipe],
        "com_position_w": source["com_position_w"][recipe],
        "com_velocity_w": source["com_velocity_w"][recipe],
        "dcm": source["dcm"][recipe],
        "current_action": source["current_action"][recipe],
        "previous_action": source["previous_action"][recipe],
        "contact_force": source["contact_force"][recipe],
    }
    return canonical_hash({name: np.asarray(value).tolist() for name, value in fields.items()})


def target_state_hash(native: dict[str, np.ndarray], row: int) -> str:
    names = ("root_pose", "root_velocity", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "com_position", "com_velocity", "dcm", "current_action", "previous_action", "contact_force")
    return canonical_hash({name: np.asarray(native[name][row]).tolist() for name in names if name in native})


def physical_entry_features(native: dict[str, np.ndarray], rows: np.ndarray, side: str) -> np.ndarray:
    """D26T physical-only feature, explicitly excluding command/history."""
    rows = np.asarray(rows, dtype=int)
    col = 0 if side == "LEFT" else 1
    root = native["root_velocity"][rows]
    jp = native["joint_pos"][rows]
    jv = native["joint_vel"][rows]
    com = native["com_position"][rows]
    cv = native["com_velocity"][rows]
    fp = native["left_right_foot_pose"][rows]
    fv = native["foot_velocity"][rows]
    force = native["contact_force"][rows]
    rp = native["root_pose"][rows]
    rel_com = com[:, :2] - fp[:, col, :2]
    rel_foot = fp[:, :, :2] - rp[:, None, :2]
    return np.concatenate((root, jp, jv, rel_com, cv[:, :2], rel_foot.reshape(len(rows), -1), fv.reshape(len(rows), -1), force.reshape(len(rows), -1)), axis=1)


def build_entry_distance_contract(native: dict[str, np.ndarray]) -> dict[str, Any]:
    manifest = json.loads((D26T / "entry_neighborhood_manifest.json").read_text(encoding="utf-8"))
    refs = [row for row in manifest["references"] if row["side"] == "RIGHT"]
    rows = np.asarray([row["bundle_row"] for row in refs], dtype=int)
    medoid_row = next(int(row["bundle_row"]) for row in refs if int(row["rank"]) == 0)
    features = physical_entry_features(native, rows, "RIGHT")
    medoid = physical_entry_features(native, np.asarray([medoid_row]), "RIGHT")[0]
    center = np.median(features, axis=0)
    mad = np.median(np.abs(features - center), axis=0) * 1.4826
    iqr = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    scale = np.maximum(np.maximum(mad, iqr / 1.349), 1.0e-6)
    distances = np.linalg.norm((features - medoid[None, :]) / scale[None, :], axis=1)
    return {
        "name": "Exp014RightEntryPhysicalStateDistanceV1",
        "source": "D26T validated RIGHT references; train-only robust scale",
        "feature_definition": "D26T physical feature with previous_action/history and command dimensions excluded",
        "feature_dimensions": int(features.shape[1]),
        "reference_rows": rows.tolist(),
        "medoid_row": medoid_row,
        "center": center.tolist(),
        "robust_scale": scale.tolist(),
        "medoid_feature": medoid.tolist(),
        "entry_neighborhood_p95": float(np.quantile(distances, 0.95)),
        "distance_formula": "L2((physical_feature(actual)-physical_feature(RIGHT_000))/robust_scale)",
        "command_and_history_in_distance": False,
    }


def actual_entry_feature(snapshot: dict[str, np.ndarray], side: str) -> np.ndarray:
    col = 0 if side == "LEFT" else 1
    root = snapshot["root_velocity"][None, :]
    jp = snapshot["joint_pos"][None, :]
    jv = snapshot["joint_vel"][None, :]
    com = snapshot["com_position"][None, :]
    cv = snapshot["com_velocity"][None, :]
    fp = snapshot["feet_pose"][None, :, :3]
    fv = snapshot["foot_velocity"][None, :]
    force = snapshot["contact_force"][None, :]
    rp = snapshot["root_pose"][None, :]
    rel_com = com[:, :2] - fp[:, col, :2]
    rel_foot = fp[:, :, :2] - rp[:, None, :2]
    return np.concatenate((root, jp, jv, rel_com, cv[:, :2], rel_foot.reshape(1, -1), fv.reshape(1, -1), force.reshape(1, -1)), axis=1)[0]


def resolve_selected_plans(source: dict[str, np.ndarray], native: dict[str, np.ndarray], geometry: dict[str, Any], default_q: np.ndarray, action_scale: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_artifact = json.loads((D26X / "selected_offline_plans_v4.json").read_text(encoding="utf-8"))
    authorization = json.loads((D26X / "exp014_d27_model_based_start_physics_authorization.json").read_text(encoding="utf-8"))
    exact = json.loads((D26X / "exact_medoid_timing_replay.json").read_text(encoding="utf-8"))
    path_contract = json.loads((D26X / "path_time_separation_contract.json").read_text(encoding="utf-8"))
    if authorization.get("allowed_scope") != "RIGHT" or not authorization.get("authorized"):
        raise RuntimeError("D27 authorization is not RIGHT-authorized")
    selected = selected_artifact.get("plans", [])
    if len(selected) != 8 or any(row.get("lead_side") != "RIGHT" for row in selected):
        raise RuntimeError("D27 requires exactly eight RIGHT selected plans")
    exact_rows = [row for row in exact["rows"] if row.get("plan_id") in {item["plan_id"] for item in selected}]
    records = path_contract["records"]
    resolved = []
    audit_rows = []
    for item in sorted(selected, key=lambda row: int(row["source_recipe"])):
        recipe = int(item["source_recipe"])
        if item.get("target_id") != RIGHT_TARGET_ID or int(item.get("target_bundle_row")) != RIGHT_TARGET_ROW:
            raise RuntimeError(f"selected plan target mismatch for R{recipe}")
        matches = []
        for record in records:
            if record.get("source_recipe") != recipe or record.get("lead_side") != "RIGHT" or record.get("target_id") != RIGHT_TARGET_ID:
                continue
            candidate = d26x.timing_candidates(record).get(item["timing"])
            if candidate is None:
                continue
            if candidate["control_duration_s"] == item["phase_durations_actual_s"] and abs(float(candidate["total_duration_s"]) - float(item["total_transition_duration_s"])) <= 1.0e-12:
                matches.append((record, candidate))
        if not matches:
            raise RuntimeError(f"no deterministic D26X path record for {item['plan_id']}")
        # D26X generated path_records in geometry-ledger order and selected the
        # first equal-rank/equal-timing candidate.  Preserve that order; do not
        # choose a record by looking at a physics or replay outcome.
        record, candidate = matches[0]
        target = d26x.aligned_target_for_row(source, recipe, native, "RIGHT", RIGHT_TARGET_ROW)
        summary, offline_rows, extra = d26x.rollout_timed(
            source,
            native,
            recipe,
            target,
            "RIGHT",
            candidate["control_duration_s"],
            record["base_geometric_phase_durations_s"],
            geometry,
            float(record["clearance_m"]),
            default_q,
            action_scale,
        )
        reference_hash = canonical_hash({
            "phase": extra["refs"]["phase_names"],
            "root_pose": np.concatenate((extra["refs"]["root_position"], np.asarray([d26x.d26v.matrix_quat(rotation) for rotation in extra["refs"]["root_rotation"]])) , axis=1).tolist(),
            "root_velocity": np.asarray(extra["refs"]["root_velocity"]).tolist(),
            "com_position": np.asarray(extra["refs"]["com_position"]).tolist(),
            "com_velocity": np.asarray(extra["refs"]["com_velocity"]).tolist(),
            "dcm": np.asarray(extra["refs"]["dcm"]).tolist(),
            "swing_position": [np.asarray(row["swing_position"]).tolist() for row in extra["refs"]["foot_refs"]],
        })
        actions = np.asarray([row["normalized_action"] for row in offline_rows], dtype=np.float64)
        q_cmds = np.asarray([row["q_cmd"] for row in offline_rows], dtype=np.float64)
        selected_trace = next((row for row in exact_rows if row.get("plan_id") == item["plan_id"] and int(row.get("source_recipe")) == recipe and row.get("timing") == item["timing"] and abs(float(row.get("total_transition_duration_s")) - float(item["total_transition_duration_s"])) <= 1.0e-12 and abs(float(row.get("max_planned_joint_velocity_ratio")) - float(item.get("max_planned_joint_velocity_ratio"))) <= 1.0e-9), None)
        if selected_trace is None:
            raise RuntimeError(f"selected plan trace missing for {item['plan_id']}")
        reference_match = len(offline_rows) == int(selected_trace["total_steps"])
        action_match = False
        q_cmd_match = False
        if reference_match:
            selected_actions = np.asarray([row["normalized_action"] for row in selected_trace["step_rows"]], dtype=np.float64)
            selected_q_cmds = np.asarray([row["q_cmd"] for row in selected_trace["step_rows"]], dtype=np.float64)
            action_match = bool(np.allclose(actions, selected_actions, atol=2.0e-10, rtol=2.0e-10))
            q_cmd_match = bool(np.allclose(q_cmds, selected_q_cmds, atol=2.0e-10, rtol=2.0e-10))
        plan_hash = canonical_hash(item)
        target_hash = target_state_hash(native, RIGHT_TARGET_ROW)
        row = {
            "plan_id": item["plan_id"],
            "source_recipe": recipe,
            "lead_side": "RIGHT",
            "target_id": RIGHT_TARGET_ID,
            "target_bundle_row": RIGHT_TARGET_ROW,
            "target_episode": RIGHT_ENTRY_EPISODE,
            "target_control_step": RIGHT_ENTRY_CONTROL_STEP,
            "target_state_hash": target_hash,
            "plan_hash": plan_hash,
            "timing": item["timing"],
            "phase_durations_requested_s": item["phase_durations_requested_s"],
            "phase_durations_actual_s": item["phase_durations_actual_s"],
            "total_transition_duration_s": item["total_transition_duration_s"],
            "clearance_m": float(record["clearance_m"]),
            "source_geometry_plan_id": record["plan_id"],
            "path_record_hash": canonical_hash({key: value for key, value in record.items() if key != "path"}),
            "root_trajectory_hash": reference_hash,
            "offline_action_trace_hash": array_hash(actions),
            "offline_q_cmd_trace_hash": array_hash(q_cmds),
            "offline_action_trace_matches_d26x": action_match,
            "offline_q_cmd_trace_matches_d26x": q_cmd_match,
            "offline_reference_step_count_matches_d26x": reference_match,
            "target_medoid_control": True,
            "compatibility_rank": item.get("target_compatibility_rank"),
            "mandatory_gates_pass_d26x": bool(item.get("mandatory_gates_pass")),
        }
        audit_rows.append(row)
        resolved.append({
            "identity": row,
            "item": item,
            "target": target,
            "record": record,
            "candidate": candidate,
            "refs": extra["refs"],
            "offline_rows": offline_rows,
            "offline_actions": actions,
            "offline_q_cmds": q_cmds,
            "source_offset": d26x.endpoint_offsets(source, native, RIGHT_TARGET_ROW, recipe, "RIGHT", default_q)[0],
            "target_offset": d26x.endpoint_offsets(source, native, RIGHT_TARGET_ROW, recipe, "RIGHT", default_q)[1],
        })
    if not all(row["offline_action_trace_matches_d26x"] and row["offline_q_cmd_trace_matches_d26x"] and row["offline_reference_step_count_matches_d26x"] for row in audit_rows):
        detail = [{key: row[key] for key in ("plan_id", "source_geometry_plan_id", "clearance_m", "offline_reference_step_count_matches_d26x", "offline_action_trace_matches_d26x", "offline_q_cmd_trace_matches_d26x")} for row in audit_rows]
        raise RuntimeError(f"D26X selected plan identity replay mismatch; fail closed before physics: {detail}")
    return resolved, {
        "name": "Exp014D27AuthorizedPlanIdentityV1",
        "d26x_authorization_sha256": sha256_file(D26X / "exp014_d27_model_based_start_physics_authorization.json"),
        "d26x_selected_plans_sha256": sha256_file(D26X / "selected_offline_plans_v4.json"),
        "d26x_classification": "EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS",
        "allowed_scope": "RIGHT",
        "target_contract": {"target_id": RIGHT_TARGET_ID, "target_bundle_row": RIGHT_TARGET_ROW, "episode": RIGHT_ENTRY_EPISODE, "control_step": RIGHT_ENTRY_CONTROL_STEP},
        "wbik": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2A",
        "action_mapper": "Exp014EndpointFeedforwardActionMapperV1",
        "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; actor/wrapper/action-term clipping none",
        "rows": audit_rows,
        "physics_executed_before_identity_gate": 0,
    }


def joint_contract(contract: dict[str, Any], source: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray) -> dict[str, Any]:
    names = contract["joint_names"]
    rows = []
    for index, name in enumerate(names):
        rows.append({
            "action_index": index,
            "asset_joint_index": index,
            "joint_name": name,
            "joint_group": d26x.joint_group(name),
            "velocity_limit_rad_s": float(np.median(source["joint_velocity_limits"][:, index])),
            "position_limit_rad": [float(np.min(source["joint_position_limits"][:, index, 0])), float(np.max(source["joint_position_limits"][:, index, 1]))],
            "default_q_rad": float(default_q[index]),
            "action_scale": float(action_scale[index]),
        })
    return {"name": "Exp014D27JointIndexNameContractV1", "dimension": 37, "joints": rows, "groups": ["left leg", "right leg", "waist", "left arm", "right arm", "left wrist/hand", "right wrist/hand"], "velocity_limits_changed": False, "position_limits_changed": False}


def build_protocol(start_head: str, start_status: list[str], source: dict[str, np.ndarray], native: dict[str, np.ndarray], entry_contract: dict[str, Any], plan_audit: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "name": "Exp014D27RightModelBasedStartPhysicsDiagnosticV1",
        "phase": "2-D27",
        "mode": mode,
        "starting_head": start_head,
        "starting_git_status_short": start_status,
        "d26x_read_only": True,
        "d26x_classification_preserved": "EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS",
        "source_lifecycle": {"name": "Exp014FreshS_HOLDSourceLifecycleV2", "recipes": RECIPES, "reset": "D3 reset recipe, not raw snapshot restore", "confirmation_steps": CONFIRMATION_STEPS, "additional_hold_steps": ADDITIONAL_HOLD_STEPS, "endpoint_gate_all_eight_required": True},
        "authorized_scope": "RIGHT first swing only",
        "primary_episodes": 8,
        "parity_episodes": 8,
        "seed": SEED,
        "control_dt_s": DT,
        "target": {"target_id": RIGHT_TARGET_ID, "bundle_row": RIGHT_TARGET_ROW, "episode": RIGHT_ENTRY_EPISODE, "control_step": RIGHT_ENTRY_CONTROL_STEP, "target_changed": False},
        "controller": {"root_is_reference_only": True, "current_state_feedback": True, "wbik": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2A", "mapper": "Exp014EndpointFeedforwardActionMapperV1", "q_cmd": "default_q + 0.5 * raw_action", "actor_clipping": "none", "wrapper_clipping": "none", "action_term_clipping": "none", "action_blending": False, "root_teleport": False, "root_velocity_overwrite": False, "joint_state_overwrite": False, "contact_state_overwrite": False},
        "timing": {"source_of_truth": "D26X selected_offline_plans_v4.json", "adaptive_stretching": False, "contact_triggered_change": False, "plan_parameter_change": False},
        "safety": {"contact_force_N": CONTACT_FORCE_N, "dangerous_slip_speed_mps": DANGEROUS_SLIP_SPEED_MPS, "dangerous_slip_dwell_steps": SATURATION_DWELL_STEPS, "impact_force_N": IMPACT_FORCE_N, "joint_velocity_saturation_ratio": SATURATION_RATIO, "torque_saturation_ratio": SATURATION_RATIO, "saturation_dwell_steps": SATURATION_DWELL_STEPS, "support_loss_dwell_steps": SATURATION_DWELL_STEPS, "roll_pitch_safe_rad": PELVIS_ROLL_PITCH_SAFE_RAD},
        "first_step_gate": {"right_unload": True, "left_support_dominance": True, "right_liftoff": True, "right_touchdown": True, "forward_pelvis_displacement_m_gt": FORWARD_DISPLACEMENT_M, "yaw_rate_abs_rad_s_le": FIRST_STEP_YAW_RATE_RAD_S, "landing_deadline": "selected plan landing time + 0.20 s"},
        "entry_gate": {"forward_velocity_error_mps_le": ENTRY_FORWARD_ERROR_MPS, "lateral_velocity_abs_mps_le": ENTRY_LATERAL_VELOCITY_MPS, "yaw_rate_abs_rad_s_le": ENTRY_YAW_RATE_RAD_S, "physical_distance": entry_contract, "confirmation_steps": ENTRY_CONFIRMATION_STEPS, "command_history_in_distance": False},
        "wmove_handoff": {"hard_switch": True, "speed_mps": WMOVE_SPEED, "lateral_mps": 0.0, "yaw_rad_s": 0.0, "steps": WMOVE_STEPS, "checkpoint": str(WMOVE.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": sha256_file(WMOVE)},
        "parity": {"bitwise_attempted": True, "numeric_abs_tolerance": PARITY_ABS_TOL, "numeric_rel_tolerance": PARITY_REL_TOL},
        "protected_inputs": {"source_bundle_sha256": sha256_file(D26U / "fresh_shold_identity_complete_sources.npz"), "native_bundle_sha256": sha256_file(D26S / "native_steady_trace_bundle.npz"), "entry_manifest_sha256": sha256_file(D26T / "entry_neighborhood_manifest.json"), "p0_sha256": sha256_file(P0), "wmove_sha256": sha256_file(WMOVE)},
        "plan_identity": plan_audit,
        "forbidden_executed": {"left_start_physics": 0, "bilateral_start": 0, "persistent_policy_update": 0, "new_learned_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0, "physics_parameter_change": 0},
    }


def get_jacobians(world) -> torch.Tensor:
    raw = world.robot.root_physx_view.get_jacobians()
    if torch.is_tensor(raw):
        result = raw
    else:
        import warp as wp

        result = wp.to_torch(raw)
    result = result.detach()
    if result.ndim != 4 or tuple(result.shape[1:]) != (44, 6, 43):
        raise RuntimeError(f"unexpected PhysX Jacobian shape {tuple(result.shape)}")
    return result


def batched_masses(world) -> torch.Tensor:
    masses = world.robot.root_physx_view.get_masses()
    if not torch.is_tensor(masses):
        masses = torch.as_tensor(masses, dtype=torch.float32, device=world.device)
    masses = masses.to(world.device)
    if masses.ndim == 1:
        masses = masses[None, :].expand(N_ENVS, -1)
    elif masses.shape[0] == 1:
        masses = masses.expand(N_ENVS, -1)
    return masses


def roll_pitch_magnitude(quaternion: np.ndarray) -> float:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return float(math.sqrt(roll * roll + pitch * pitch))


def read_runtime(world, masses: torch.Tensor) -> dict[str, np.ndarray]:
    data = world.robot.data
    n = N_ENVS
    root_pos = tensor_np(data.root_pos_w[:n], np.float64)
    root_quat = tensor_np(data.root_quat_w[:n], np.float64)
    root_lin = tensor_np(data.root_lin_vel_w[:n], np.float64)
    root_ang = tensor_np(data.root_ang_vel_w[:n], np.float64)
    root_pose = np.concatenate((root_pos, root_quat), axis=1)
    root_velocity = np.concatenate((root_lin, root_ang), axis=1)
    body_pos = tensor_np(data.body_pos_w[:n], np.float64)
    body_quat = tensor_np(data.body_quat_w[:n], np.float64)
    body_lin = tensor_np(data.body_lin_vel_w[:n], np.float64)
    body_ang = tensor_np(data.body_ang_vel_w[:n], np.float64)
    body_com = tensor_np(data.body_com_pos_w[:n], np.float64)
    body_com_vel = tensor_np(data.body_com_lin_vel_w[:n], np.float64)
    mass = tensor_np(masses[:n], np.float64)
    total_mass = np.maximum(mass.sum(axis=1), 1.0e-9)
    com = (body_com * mass[:, :, None]).sum(axis=1) / total_mass[:, None]
    com_velocity = (body_com_vel * mass[:, :, None]).sum(axis=1) / total_mass[:, None]
    omega = np.sqrt(9.81 / np.maximum(com[:, 2], 0.1))
    dcm = com[:, :2] + com_velocity[:, :2] / omega[:, None]
    rf = [int(index) for index in tensor_np(world.rf).reshape(-1)]
    sf = [int(index) for index in tensor_np(world.sf).reshape(-1)]
    feet_pose = np.concatenate((body_pos[:, rf], body_quat[:, rf]), axis=2)
    foot_velocity = body_lin[:, rf]
    force_history = tensor_np(world.sensor.data.net_forces_w_history[:n, :, sf, :], np.float64)
    force = force_history[:, -1]
    contact = np.linalg.norm(force, axis=2) > CONTACT_FORCE_N
    velocity_limits = tensor_np(data.joint_vel_limits[:n], np.float64)
    if velocity_limits.ndim == 3:
        velocity_limits = np.abs(velocity_limits[..., 1])
    effort_limits = tensor_np(data.joint_effort_limits[:n], np.float64)
    if effort_limits.ndim == 3:
        effort_limits = np.max(np.abs(effort_limits), axis=2)
    else:
        effort_limits = np.abs(effort_limits)
    applied = getattr(data, "applied_torque", None)
    computed = getattr(data, "computed_torque", None)
    applied_torque = np.zeros_like(tensor_np(data.joint_pos[:n], np.float64)) if applied is None else tensor_np(applied[:n], np.float64)
    computed_torque = applied_torque.copy() if computed is None else tensor_np(computed[:n], np.float64)
    q = tensor_np(data.joint_pos[:n], np.float64)
    dq = tensor_np(data.joint_vel[:n], np.float64)
    joint_ratio = np.abs(dq) / np.maximum(np.abs(velocity_limits), 1.0e-6)
    torque_ratio = np.abs(applied_torque) / np.maximum(np.abs(effort_limits), 1.0e-6)
    pos_limits = getattr(data, "soft_joint_pos_limits", getattr(data, "joint_pos_limits", None))
    pos_limits_np = tensor_np(pos_limits[:n], np.float64)
    base_lin_b = tensor_np(data.root_lin_vel_b[:n], np.float64)
    base_ang_b = tensor_np(data.root_ang_vel_b[:n], np.float64)
    return {
        "root_pose": root_pose,
        "root_velocity": root_velocity,
        "joint_pos": q,
        "joint_vel": dq,
        "body_pos": body_pos,
        "body_quat": body_quat,
        "body_lin_vel": body_lin,
        "body_ang_vel": body_ang,
        "body_com_pos": body_com,
        "body_com_vel": body_com_vel,
        "com_position": com,
        "com_velocity": com_velocity,
        "dcm": dcm,
        "feet_pose": feet_pose,
        "foot_velocity": foot_velocity,
        "contact_force": force,
        "contact": contact,
        "joint_velocity_limits": velocity_limits,
        "joint_position_limits": pos_limits_np,
        "effort_limits": effort_limits,
        "applied_torque": applied_torque,
        "computed_torque": computed_torque,
        "joint_velocity_ratio": joint_ratio,
        "torque_ratio": torque_ratio,
        "base_lin_vel_b": base_lin_b,
        "base_ang_vel_b": base_ang_b,
        "roll_pitch": np.asarray([roll_pitch_magnitude(item) for item in root_quat]),
        "total_mass": total_mass,
    }


def finite_snapshot(snapshot: dict[str, np.ndarray]) -> np.ndarray:
    keys = ("root_pose", "root_velocity", "joint_pos", "joint_vel", "body_pos", "body_quat", "com_position", "com_velocity", "dcm", "feet_pose", "foot_velocity", "contact_force", "applied_torque", "computed_torque")
    return np.asarray([np.isfinite(snapshot[key]).all() for key in keys], dtype=bool)


class TraceBuffer:
    def __init__(self, default_q: np.ndarray) -> None:
        self.max_steps = MAX_STEPS
        self.n = N_ENVS
        self.default_q = np.asarray(default_q, dtype=np.float64)
        self.arrays: dict[str, np.ndarray] = {
            "active": np.zeros((self.n, self.max_steps), dtype=np.bool_),
            "control_step": np.full((self.n, self.max_steps), -1, dtype=np.int32),
            "stage": np.full((self.n, self.max_steps), -1, dtype=np.int8),
            "phase": np.full((self.n, self.max_steps), -1, dtype=np.int8),
            "phase_progress": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "reference_valid": np.zeros((self.n, self.max_steps), dtype=np.bool_),
            "reference_root_pose": np.full((self.n, self.max_steps, 7), np.nan, dtype=np.float64),
            "reference_root_velocity": np.full((self.n, self.max_steps, 6), np.nan, dtype=np.float64),
            "reference_com_position": np.full((self.n, self.max_steps, 3), np.nan, dtype=np.float64),
            "reference_com_velocity": np.full((self.n, self.max_steps, 3), np.nan, dtype=np.float64),
            "reference_dcm": np.full((self.n, self.max_steps, 2), np.nan, dtype=np.float64),
            "reference_stance_pose": np.full((self.n, self.max_steps, 7), np.nan, dtype=np.float64),
            "reference_swing_pose": np.full((self.n, self.max_steps, 7), np.nan, dtype=np.float64),
            "actual_root_pose_current": np.full((self.n, self.max_steps, 7), np.nan, dtype=np.float64),
            "actual_root_pose": np.full((self.n, self.max_steps, 7), np.nan, dtype=np.float64),
            "actual_root_velocity": np.full((self.n, self.max_steps, 6), np.nan, dtype=np.float64),
            "actual_com_position": np.full((self.n, self.max_steps, 3), np.nan, dtype=np.float64),
            "actual_com_velocity": np.full((self.n, self.max_steps, 3), np.nan, dtype=np.float64),
            "actual_dcm": np.full((self.n, self.max_steps, 2), np.nan, dtype=np.float64),
            "actual_feet_pose": np.full((self.n, self.max_steps, 2, 7), np.nan, dtype=np.float64),
            "actual_foot_velocity": np.full((self.n, self.max_steps, 2, 3), np.nan, dtype=np.float64),
            "contact_force": np.full((self.n, self.max_steps, 2, 3), np.nan, dtype=np.float64),
            "contact": np.zeros((self.n, self.max_steps, 2), dtype=np.bool_),
            "q_actual_current": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "q_actual": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "q_cmd": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "action": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "previous_action": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "action_rate": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "joint_velocity_ratio": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "torque_ratio": np.full((self.n, self.max_steps, 37), np.nan, dtype=np.float64),
            "error_vector": np.full((self.n, self.max_steps, 12), np.nan, dtype=np.float64),
            "forward_velocity_error": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "lateral_velocity": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "yaw_rate": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "roll_pitch": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "entry_distance": np.full((self.n, self.max_steps), np.nan, dtype=np.float64),
            "entry_condition": np.zeros((self.n, self.max_steps), dtype=np.bool_),
            "safety_mask": np.zeros((self.n, self.max_steps, len(SAFETY_NAMES)), dtype=np.bool_),
        }

    def append(self, recipe: int, step: int, stage: str, phase: str, progress: float | None, reference: dict[str, np.ndarray] | None, current: dict[str, np.ndarray], post: dict[str, np.ndarray], action: np.ndarray, previous_action: np.ndarray, q_cmd: np.ndarray | None, errors: dict[str, float] | None, entry_distance: float | None, entry_condition: bool, safety: dict[str, bool]) -> None:
        if step >= self.max_steps:
            return
        a = self.arrays
        a["active"][recipe, step] = True
        a["control_step"][recipe, step] = step + 1
        a["stage"][recipe, step] = STAGE_CODE[stage]
        a["phase"][recipe, step] = PHASE_CODE.get(phase, 0)
        if progress is not None:
            a["phase_progress"][recipe, step] = progress
        if reference is not None:
            a["reference_valid"][recipe, step] = True
            for name in ("root_pose", "root_velocity", "com_position", "com_velocity", "dcm"):
                a[f"reference_{name}"][recipe, step] = reference[name]
            a["reference_stance_pose"][recipe, step] = np.concatenate((reference["stance_position"], d26x.d26v.matrix_quat(reference["stance_rotation"])))
            a["reference_swing_pose"][recipe, step] = np.concatenate((reference["swing_position"], d26x.d26v.matrix_quat(reference["swing_rotation"])))
        a["actual_root_pose_current"][recipe, step] = current["root_pose"][recipe]
        a["actual_root_pose"][recipe, step] = post["root_pose"][recipe]
        a["actual_root_velocity"][recipe, step] = post["root_velocity"][recipe]
        a["actual_com_position"][recipe, step] = post["com_position"][recipe]
        a["actual_com_velocity"][recipe, step] = post["com_velocity"][recipe]
        a["actual_dcm"][recipe, step] = post["dcm"][recipe]
        a["actual_feet_pose"][recipe, step] = post["feet_pose"][recipe]
        a["actual_foot_velocity"][recipe, step] = post["foot_velocity"][recipe]
        a["contact_force"][recipe, step] = post["contact_force"][recipe]
        a["contact"][recipe, step] = post["contact"][recipe]
        a["q_actual_current"][recipe, step] = current["joint_pos"][recipe]
        a["q_actual"][recipe, step] = post["joint_pos"][recipe]
        a["q_cmd"][recipe, step] = d26x.q_cmd_from_action(action[recipe], self.default_q) if q_cmd is None else q_cmd
        a["action"][recipe, step] = action[recipe]
        a["previous_action"][recipe, step] = previous_action[recipe]
        a["action_rate"][recipe, step] = action[recipe] - previous_action[recipe]
        a["joint_velocity_ratio"][recipe, step] = post["joint_velocity_ratio"][recipe]
        a["torque_ratio"][recipe, step] = post["torque_ratio"][recipe]
        if errors is not None:
            names = ("root_position_error_m", "root_orientation_error_rad", "com_position_error_m", "com_velocity_error_mps", "dcm_error_m", "stance_position_error_m", "stance_orientation_error_rad", "swing_position_error_m", "swing_orientation_error_rad", "joint_target_error_rad", "contact_timing_error", "pelvis_roll_pitch_rad")
            a["error_vector"][recipe, step] = np.asarray([errors.get(name, np.nan) for name in names], dtype=np.float64)
        if entry_distance is not None:
            a["entry_distance"][recipe, step] = entry_distance
        a["entry_condition"][recipe, step] = entry_condition
        a["safety_mask"][recipe, step] = np.asarray([safety[name] for name in SAFETY_NAMES], dtype=bool)

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        payload = dict(self.arrays)
        payload["recipe_ids"] = np.asarray(RECIPES, dtype=np.int64)
        payload["phase_names"] = np.asarray(PHASES, dtype="U32")
        payload["safety_names"] = np.asarray(SAFETY_NAMES, dtype="U32")
        np.savez_compressed(path, **payload)
        dump(path.with_suffix(".metadata.json"), metadata)


def load_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def error_metrics(reference: dict[str, np.ndarray], post: dict[str, np.ndarray], recipe: int, q_cmd: np.ndarray, target_support: np.ndarray, phase: str) -> dict[str, float]:
    ref_root = reference["root_pose"]
    ref_root_rot = d26x.d26v.quat_matrix(ref_root[3:])
    actual_root = post["root_pose"][recipe]
    actual_rot = d26x.d26v.quat_matrix(actual_root[3:])
    root_rot_error = d26x.d26v.so3_log_np(ref_root_rot @ actual_rot.T)
    ref_stance_rot = reference["stance_rotation"]
    ref_swing_rot = reference["swing_rotation"]
    stance = post["feet_pose"][recipe, 0]
    swing = post["feet_pose"][recipe, 1]
    stance_rot_error = d26x.d26v.so3_log_np(ref_stance_rot @ d26x.d26v.quat_matrix(stance[3:]).T)
    swing_rot_error = d26x.d26v.so3_log_np(ref_swing_rot @ d26x.d26v.quat_matrix(swing[3:]).T)
    ref_contact = np.asarray(target_support, dtype=bool)
    contact_error = float(np.any(post["contact"][recipe] != ref_contact))
    return {
        "root_position_error_m": float(np.linalg.norm(ref_root[:3] - actual_root[:3])),
        "root_orientation_error_rad": float(np.linalg.norm(root_rot_error)),
        "com_position_error_m": float(np.linalg.norm(reference["com_position"] - post["com_position"][recipe])),
        "com_velocity_error_mps": float(np.linalg.norm(reference["com_velocity"] - post["com_velocity"][recipe])),
        "dcm_error_m": float(np.linalg.norm(reference["dcm"] - post["dcm"][recipe])),
        "stance_position_error_m": float(np.linalg.norm(reference["stance_position"] - stance[:3])),
        "stance_orientation_error_rad": float(np.linalg.norm(stance_rot_error)),
        "swing_position_error_m": float(np.linalg.norm(reference["swing_position"] - swing[:3])),
        "swing_orientation_error_rad": float(np.linalg.norm(swing_rot_error)),
        "joint_target_error_rad": float(np.linalg.norm(q_cmd - post["joint_pos"][recipe])),
        "contact_timing_error": contact_error,
        "pelvis_roll_pitch_rad": float(np.linalg.norm(root_rot_error[:2])),
    }


def set_command(world, wmove_mask: np.ndarray) -> None:
    command = world.term.external_override
    command[:, :3] = 0.0
    for index, enabled in enumerate(wmove_mask.tolist()):
        if enabled:
            command[index, 0] = WMOVE_SPEED
    world.term._update_command()


def step_world(world, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    # Keep the frozen explicit state machine at its S_HOLD-compatible zero
    # command.  W_MOVE reads the original 123D observation and does not use the
    # appended explicit-mode fields.
    world.state.advance(torch.zeros((N_ENVS, 3), device=world.device), torch.ones(N_ENVS, device=world.device), DT)
    result = world.wrapped.step(action)
    obs, reward, done, extras = result
    if isinstance(obs, dict):
        obs = obs["policy"]
    return obs, reward, done.bool(), extras


def build_reference(plan: dict[str, Any], step: int) -> dict[str, np.ndarray]:
    raw = d26x.reference_for_timed_step(plan["refs"], step, plan["target"], plan["source"], plan["identity"]["source_recipe"]) if "source" in plan else None
    if raw is None:
        raise RuntimeError("plan source payload missing")
    reference = {key: tensor_np(value, np.float64) for key, value in raw.items()}
    # D26X keeps DCM in the timing-reference bundle for diagnostics; the V2A
    # solver does not consume it, so append it only to the runtime reference
    # record used by tracking/error accounting.
    reference["dcm"] = np.asarray(plan["refs"]["dcm"][step], dtype=np.float64)
    return reference


def safety_update(snapshot: dict[str, np.ndarray], done: np.ndarray, timeout: np.ndarray, book: dict[str, Any], global_step: int) -> dict[str, np.ndarray]:
    n = N_ENVS
    contact = snapshot["contact"]
    foot_speed = np.linalg.norm(snapshot["foot_velocity"][..., :2], axis=2)
    slip_now = np.any(contact & (foot_speed > DANGEROUS_SLIP_SPEED_MPS), axis=1)
    book["streak_slip"] = np.where(slip_now, book["streak_slip"] + 1, 0)
    dangerous_now = book["streak_slip"] >= SATURATION_DWELL_STEPS
    impact_now = np.max(np.abs(snapshot["contact_force"]), axis=(1, 2)) > IMPACT_FORCE_N
    velocity_now = np.max(snapshot["joint_velocity_ratio"], axis=1) > SATURATION_RATIO
    torque_now = np.max(snapshot["torque_ratio"], axis=1) > SATURATION_RATIO
    book["streak_velocity"] = np.where(velocity_now, book["streak_velocity"] + 1, 0)
    book["streak_torque"] = np.where(torque_now, book["streak_torque"] + 1, 0)
    velocity_sat_now = book["streak_velocity"] >= SATURATION_DWELL_STEPS
    torque_sat_now = book["streak_torque"] >= SATURATION_DWELL_STEPS
    support_now = ~np.any(contact, axis=1)
    book["streak_support"] = np.where(support_now, book["streak_support"] + 1, 0)
    support_loss_now = book["streak_support"] >= SATURATION_DWELL_STEPS
    finite_now = np.asarray([finite_snapshot({key: value[i] for key, value in snapshot.items()}) .all() for i in range(n)], dtype=bool)
    nonfinite_now = ~finite_now
    fall_now = np.asarray(done, dtype=bool) & ~np.asarray(timeout, dtype=bool)
    now = {"fall": fall_now, "dangerous_slip": dangerous_now, "impact": impact_now, "velocity_saturation": velocity_sat_now, "torque_saturation": torque_sat_now, "support_loss": support_loss_now, "nonfinite": nonfinite_now}
    for name in SAFETY_NAMES:
        first = now[name] & ~book["flags"][name]
        indices = np.flatnonzero(first)
        for index in indices:
            if book["first_safety"][name][index] < 0:
                book["first_safety"][name][index] = global_step + 1
        book["flags"][name] |= now[name]
    return now


def source_gate_row(book: dict[str, Any], recipe: int, endpoint: dict[str, Any] | None) -> dict[str, Any]:
    # Freeze the lifecycle gate at the exact endpoint.  A source that passes
    # S_HOLD must not become ineligible because a different authorized START
    # episode continues running later in the same vectorized process.
    frozen = book.get("source_flags", [None] * N_ENVS)[recipe]
    if frozen is None:
        flags = {name: bool(book["flags"][name][recipe]) for name in SAFETY_NAMES}
    else:
        flags = {name: bool(frozen[name]) for name in SAFETY_NAMES}
    valid = bool(endpoint is not None and book["confirmation_end"][recipe] >= 0 and book["endpoint_step"][recipe] >= 0 and endpoint["support_valid"] and not any(flags.values()))
    return {
        "recipe_id": recipe,
        "seed": SEED,
        "reset_to_stand": "PASS" if book["confirmation_end"][recipe] >= 0 else "FAIL",
        "confirmation_50_steps": "PASS" if book["confirmation_end"][recipe] >= 0 else "FAIL",
        "additional_hold_1s": "PASS" if book["endpoint_step"][recipe] >= 0 else "FAIL",
        "confirmation_end_step": int(book["confirmation_end"][recipe]),
        "endpoint_control_step": int(book["endpoint_step"][recipe]),
        "endpoint_support_valid": bool(endpoint and endpoint["support_valid"]),
        "fall": int(flags["fall"]),
        "dangerous_slip": int(flags["dangerous_slip"]),
        "impact": int(flags["impact"]),
        "velocity_saturation": int(flags["velocity_saturation"]),
        "torque_saturation": int(flags["torque_saturation"]),
        "support_loss": int(flags["support_loss"]),
        "nan_inf": int(flags["nonfinite"]),
        "source_endpoint_hash": None if endpoint is None else endpoint["hash"],
        "source_endpoint_eligible": valid,
        "failure": None if valid else "SOURCE_ENDPOINT_INELIGIBLE",
    }


def empty_episode(recipe: int, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "recipe_id": recipe,
        "plan_id": plan["identity"]["plan_id"],
        "source_endpoint_eligible": False,
        "first_divergence": None,
        "first_step": {"pass": False},
        "entry": {"pass": False},
        "handoff": {"pass": False},
        "safety": {name: False for name in SAFETY_NAMES},
        "phase_rows": {phase: [] for phase in PHASES},
        "wmove_rows": [],
        "event_steps": {"right_unload": None, "right_liftoff": None, "right_touchdown": None, "left_support_dominance": None},
        "entry_samples": [],
        "entry_streak_max": 0,
        "entry_confirmation_step": None,
        "handoff_step": None,
        "plan_complete": False,
    }


def first_divergence_name(name: str, phase: str) -> str:
    if name == "fall":
        return "FALL"
    if name == "dangerous_slip":
        return "SUPPORT_FOOT_SLIP"
    if name == "impact":
        return "LANDING_IMPACT_FAILURE" if phase == "LANDING_AND_CAPTURE" else "LANDING_IMPACT_FAILURE"
    if name == "velocity_saturation":
        return "JOINT_VELOCITY_SATURATION"
    if name == "torque_saturation":
        return "TORQUE_SATURATION"
    if name == "support_loss":
        return "SUPPORT_LOSS"
    if name == "nonfinite":
        return "NUMERICAL_FAILURE"
    return "NUMERICAL_FAILURE"


def source_hash_live(snapshot: dict[str, np.ndarray], recipe: int, action: np.ndarray, previous_action: np.ndarray) -> str:
    fields = {name: snapshot[name][recipe].tolist() for name in ("root_pose", "root_velocity", "joint_pos", "joint_vel", "body_pos", "body_quat", "com_position", "com_velocity", "dcm", "feet_pose", "contact_force", "contact")}
    fields["action"] = action[recipe].tolist()
    fields["previous_action"] = previous_action[recipe].tolist()
    return canonical_hash(fields)


def solve_runtime_action(world, runtime: dict[str, np.ndarray], plan: dict[str, Any], plan_step: int, default_q: np.ndarray, action_scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    recipe = int(plan["identity"]["source_recipe"])
    reference = build_reference(plan, plan_step)
    data = world.robot.data
    jacobians = tensor_np(get_jacobians(world)[recipe], np.float64)
    masses = tensor_np(batched_masses(world)[recipe], np.float64)
    q_min = runtime["joint_position_limits"][recipe, :, 0]
    q_max = runtime["joint_position_limits"][recipe, :, 1]
    solution = d26x.wbik_v2.solve_prescribed_floating_base(
        root_pose=torch64(runtime["root_pose"][recipe]),
        root_velocity=torch64(runtime["root_velocity"][recipe]),
        joint_position=torch64(runtime["joint_pos"][recipe]),
        joint_velocity=torch64(runtime["joint_vel"][recipe]),
        body_position=torch64(runtime["body_pos"][recipe]),
        body_quaternion=torch64(runtime["body_quat"][recipe]),
        body_jacobians=torch64(jacobians),
        body_com_position=torch64(runtime["body_com_pos"][recipe]),
        body_masses=torch64(masses),
        com_position=torch64(runtime["com_position"][recipe]),
        reference={key: torch64(value) for key, value in reference.items()},
        stance_body_index=24,
        swing_body_index=25,
        q_min=torch64(q_min),
        q_max=torch64(q_max),
        velocity_limits=torch64(runtime["joint_velocity_limits"][recipe]),
        default_q=torch64(default_q),
        action_scale=torch64(action_scale),
    )
    if not bool(solution["solver_diagnostics"]["finite"]) or str(solution["status"]) in ("NUMERICAL_FAILURE", "ACTIVE_SET_NONCONVERGENCE"):
        raise RuntimeError(str(solution["status"]))
    alpha = minimum_jerk(float(plan_step + 1) / float(plan["refs"]["total_steps"]))
    q_des = tensor_np(solution["q_des"], np.float64)
    q_cmd = q_des + (1.0 - alpha) * plan["source_offset"] + alpha * plan["target_offset"]
    action = (q_cmd - default_q) / action_scale
    if not np.isfinite(action).all() or not np.isfinite(q_cmd).all():
        raise RuntimeError("NUMERICAL_FAILURE")
    return action, q_cmd, solution, reference


def summarize_phase_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("root_position_error_m", "root_orientation_error_rad", "com_position_error_m", "com_velocity_error_mps", "dcm_error_m", "stance_position_error_m", "stance_orientation_error_rad", "swing_position_error_m", "swing_orientation_error_rad", "joint_target_error_rad", "contact_timing_error", "pelvis_roll_pitch_rad")
    return {name: mean_percentiles([row["errors"][name] for row in rows if name in row.get("errors", {})]) for name in metrics}


def run_physics(args, plans: list[dict[str, Any]], source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray, entry_contract: dict[str, Any]) -> dict[str, Any]:
    cfg, agent = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = N_ENVS
    cfg.seed = SEED
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    primary_protected = protected_snapshot()
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        world = d3.StandWorld(wrapped, d3.load_resets(), torch.zeros(680, device=wrapped.unwrapped.device))
        if agent.clip_actions is not None:
            raise RuntimeError(f"D27 requires wrapper clipping none, got {agent.clip_actions}")
        stand_actor = d3.initialize("P0_STAND_PARENT", world.device)[0].eval()
        wmove_actor = FrozenGaitActor(WMOVE).to(world.device).eval()
        masses = batched_masses(world)
        world.restore(torch.as_tensor(RECIPES, dtype=torch.long, device=world.device))

        trace = TraceBuffer(default_q)
        book: dict[str, Any] = {
            "stage": ["S_HOLD"] * N_ENVS,
            "plan_step": [0] * N_ENVS,
            "wmove_step": [0] * N_ENVS,
            "confirmation": np.zeros(N_ENVS, dtype=np.int64),
            "confirmation_end": np.full(N_ENVS, -1, dtype=np.int64),
            "endpoint_step": np.full(N_ENVS, -1, dtype=np.int64),
            "endpoint": [None] * N_ENVS,
            "endpoint_hash": [None] * N_ENVS,
            "flags": {name: np.zeros(N_ENVS, dtype=bool) for name in SAFETY_NAMES},
            "source_flags": [None] * N_ENVS,
            "first_safety": {name: np.full(N_ENVS, -1, dtype=np.int64) for name in SAFETY_NAMES},
            "streak_slip": np.zeros(N_ENVS, dtype=np.int64),
            "streak_velocity": np.zeros(N_ENVS, dtype=np.int64),
            "streak_torque": np.zeros(N_ENVS, dtype=np.int64),
            "streak_support": np.zeros(N_ENVS, dtype=np.int64),
            "first_divergence": [None] * N_ENVS,
            "previous_contact": [None] * N_ENVS,
        }
        episodes = [empty_episode(i, plans[i]) for i in RECIPES]
        runtime = read_runtime(world, masses)
        previous_action = tensor_np(world.env.action_manager.prev_action[:N_ENVS], np.float64)
        lifecycle_hashes = [None] * N_ENVS
        plan_runtime_by_recipe = {int(plan["identity"]["source_recipe"]): plan for plan in plans}
        for step in range(MAX_STEPS):
            runtime_current = read_runtime(world, masses)
            wmove_mask = np.asarray([stage == "WMOVE" for stage in book["stage"]], dtype=bool)
            set_command(world, wmove_mask)
            base_obs = world.env.observation_manager.compute()["policy"]
            start_obs = d3.build_observation_141(base_obs, world.state)
            action = np.zeros((N_ENVS, 37), dtype=np.float64)
            q_cmds: list[np.ndarray | None] = [None] * N_ENVS
            references: list[dict[str, np.ndarray] | None] = [None] * N_ENVS
            errors_pre: list[dict[str, float] | None] = [None] * N_ENVS
            phases = [""] * N_ENVS
            progress = [None] * N_ENVS
            wmove_actions = None
            with torch.inference_mode():
                stand_actions = tensor_np(stand_actor.mean(start_obs), np.float64)
                wmove_actions = tensor_np(wmove_actor(base_obs, torch.zeros(N_ENVS, device=world.device)), np.float64)
            for recipe in RECIPES:
                stage = book["stage"][recipe]
                if stage == "S_HOLD":
                    action[recipe] = stand_actions[recipe]
                elif stage == "START":
                    plan = plan_runtime_by_recipe[recipe]
                    plan_step = int(book["plan_step"][recipe])
                    phases[recipe] = plan["refs"]["phase_names"][plan_step]
                    progress[recipe] = float((plan_step + 1) / plan["refs"]["total_steps"])
                    try:
                        one_action, one_q_cmd, solution, reference = solve_runtime_action(world, runtime_current, plan, plan_step, default_q, action_scale)
                        action[recipe] = one_action
                        q_cmds[recipe] = one_q_cmd
                        references[recipe] = reference
                    except Exception as exc:
                        failure = {"classification": "WBIK_RUNTIME_FAILURE", "control_step": step + 1, "phase": phases[recipe], "detail": str(exc)}
                        if book["first_divergence"][recipe] is None:
                            book["first_divergence"][recipe] = failure
                            episodes[recipe]["first_divergence"] = failure
                        book["stage"][recipe] = "TERMINAL"
                elif stage == "WMOVE":
                    action[recipe] = wmove_actions[recipe]
                    phases[recipe] = "WMOVE"
                    progress[recipe] = float((book["wmove_step"][recipe] + 1) / WMOVE_STEPS)
                else:
                    action[recipe] = 0.0
            action_t = torch.as_tensor(action, dtype=torch.float32, device=world.device)
            previous_action_current = tensor_np(world.env.action_manager.prev_action[:N_ENVS], np.float64)
            post_obs, _, done, extras = step_world(world, action_t)
            timeout_value = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            timeout = tensor_np(timeout_value[:N_ENVS], bool)
            runtime_post = read_runtime(world, masses)
            now_safety = safety_update(runtime_post, tensor_np(done[:N_ENVS], bool), timeout, book, step)
            safety_row = {name: bool(book["flags"][name][recipe]) for name in SAFETY_NAMES}
            for recipe in RECIPES:
                stage_before = "S_HOLD" if book["stage"][recipe] == "S_HOLD" else ("START" if phases[recipe] in PHASES else ("WMOVE" if phases[recipe] == "WMOVE" else "TERMINAL"))
                phase = phases[recipe]
                reference = references[recipe]
                errors = None
                entry_distance = None
                entry_condition = False
                if stage_before == "START" and reference is not None:
                    plan = plan_runtime_by_recipe[recipe]
                    target_support = np.asarray(plan["target"]["target_support_configuration"], dtype=bool)
                    errors = error_metrics(reference, runtime_post, recipe, q_cmds[recipe], target_support, phase)
                    episodes[recipe]["phase_rows"][phase].append({"global_step": step + 1, "plan_step": int(book["plan_step"][recipe]), "errors": errors, "velocity_ratio_max": float(np.max(runtime_post["joint_velocity_ratio"][recipe])), "torque_ratio_max": float(np.max(runtime_post["torque_ratio"][recipe])), "actual_contact": runtime_post["contact"][recipe].tolist(), "reference_contact": target_support.tolist()})
                    feature = actual_entry_feature({key: value[recipe] for key, value in runtime_post.items()}, "RIGHT")
                    medoid = np.asarray(entry_contract["medoid_feature"], dtype=np.float64)
                    scale = np.asarray(entry_contract["robust_scale"], dtype=np.float64)
                    entry_distance = float(np.linalg.norm((feature - medoid) / scale))
                    forward_error = abs(float(runtime_post["base_lin_vel_b"][recipe, 0]) - WMOVE_SPEED)
                    lateral_velocity = abs(float(runtime_post["base_lin_vel_b"][recipe, 1]))
                    yaw_rate = abs(float(runtime_post["base_ang_vel_b"][recipe, 2]))
                    phase_match = bool(np.all(runtime_post["contact"][recipe] == target_support))
                    entry_condition = bool(phase in ("LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE") and phase_match and entry_distance <= float(entry_contract["entry_neighborhood_p95"]) and forward_error <= ENTRY_FORWARD_ERROR_MPS and lateral_velocity <= ENTRY_LATERAL_VELOCITY_MPS and yaw_rate <= ENTRY_YAW_RATE_RAD_S)
                    episodes[recipe]["entry_samples"].append({"global_step": step + 1, "plan_step": int(book["plan_step"][recipe]), "phase": phase, "distance": entry_distance, "p95": float(entry_contract["entry_neighborhood_p95"]), "forward_velocity_error": forward_error, "lateral_velocity": lateral_velocity, "yaw_rate": yaw_rate, "phase_match": phase_match, "condition": entry_condition})
                    if entry_condition:
                        episodes[recipe]["entry_streak_max"] = max(episodes[recipe]["entry_streak_max"], episodes[recipe].get("entry_streak", 0) + 1)
                        episodes[recipe]["entry_streak"] = episodes[recipe].get("entry_streak", 0) + 1
                    else:
                        episodes[recipe]["entry_streak"] = 0
                    if episodes[recipe].get("entry_streak", 0) >= ENTRY_CONFIRMATION_STEPS and episodes[recipe]["entry_confirmation_step"] is None:
                        episodes[recipe]["entry_confirmation_step"] = step + 1
                        episodes[recipe]["entry"]["pass"] = True
                elif stage_before == "WMOVE":
                    episodes[recipe]["wmove_rows"].append({"global_step": step + 1, "step": int(book["wmove_step"][recipe]) + 1, "forward_error": abs(float(runtime_post["base_lin_vel_b"][recipe, 0]) - WMOVE_SPEED), "lateral_velocity": abs(float(runtime_post["base_lin_vel_b"][recipe, 1])), "yaw_rate": abs(float(runtime_post["base_ang_vel_b"][recipe, 2])), "contact": runtime_post["contact"][recipe].tolist(), "safety": safety_row})
                trace.append(recipe, step, stage_before, phase, progress[recipe], reference, runtime_current, runtime_post, action, previous_action_current, q_cmds[recipe], errors, entry_distance, entry_condition, safety_row)

                if stage_before == "S_HOLD":
                    good = float(np.linalg.norm(runtime_post["base_lin_vel_b"][recipe, :2])) <= 0.08 and abs(float(runtime_post["base_ang_vel_b"][recipe, 2])) <= 0.08
                    if book["confirmation_end"][recipe] < 0:
                        book["confirmation"][recipe] = int(book["confirmation"][recipe] + 1) if good else 0
                        if book["confirmation"][recipe] >= CONFIRMATION_STEPS:
                            book["confirmation_end"][recipe] = step + 1
                    elif step + 1 >= book["confirmation_end"][recipe] + ADDITIONAL_HOLD_STEPS and book["endpoint_step"][recipe] < 0:
                        endpoint_support = bool(np.any(runtime_post["contact"][recipe]))
                        endpoint_hash = source_hash_live(runtime_post, recipe, action, previous_action_current)
                        endpoint = {"support_valid": endpoint_support, "hash": endpoint_hash, "root_pose": runtime_post["root_pose"][recipe].tolist(), "root_velocity": runtime_post["root_velocity"][recipe].tolist(), "joint_pos": runtime_post["joint_pos"][recipe].tolist(), "joint_vel": runtime_post["joint_vel"][recipe].tolist(), "contact": runtime_post["contact"][recipe].tolist(), "contact_force": runtime_post["contact_force"][recipe].tolist(), "base_speed": float(np.linalg.norm(runtime_post["base_lin_vel_b"][recipe, :2])), "yaw_rate": float(abs(runtime_post["base_ang_vel_b"][recipe, 2]))}
                        book["endpoint_step"][recipe] = step + 1
                        book["endpoint"][recipe] = endpoint
                        book["endpoint_hash"][recipe] = endpoint_hash
                        book["source_flags"][recipe] = {name: bool(book["flags"][name][recipe]) for name in SAFETY_NAMES}
                        source_row = source_gate_row(book, recipe, endpoint)
                        if source_row["source_endpoint_eligible"]:
                            book["stage"][recipe] = "START"
                            episodes[recipe]["source_endpoint_eligible"] = True
                            episodes[recipe]["source_endpoint"] = source_row
                            episodes[recipe]["source_root_xy"] = runtime_post["root_pose"][recipe, :2].tolist()
                        else:
                            episodes[recipe]["source_endpoint"] = source_row
                            episodes[recipe]["first_divergence"] = {"classification": "SOURCE_ENDPOINT_INELIGIBLE", "control_step": step + 1, "phase": "S_HOLD", "detail": "fresh S_HOLD source gate failed"}
                            book["first_divergence"][recipe] = episodes[recipe]["first_divergence"]
                            book["stage"][recipe] = "TERMINAL"

                if stage_before == "START":
                    # Safety is a hard termination for model-based START.  It
                    # is checked before event progression or handoff.
                    new_safety = [name for name in SAFETY_NAMES if bool(now_safety[name][recipe]) and book["first_safety"][name][recipe] == step + 1]
                    if new_safety and book["first_divergence"][recipe] is None:
                        selected_safety = next(name for name in SAFETY_NAMES if name in new_safety)
                        book["first_divergence"][recipe] = {"classification": first_divergence_name(selected_safety, phase), "control_step": step + 1, "phase": phase, "detail": "canonical safety termination"}
                        episodes[recipe]["first_divergence"] = book["first_divergence"][recipe]
                        book["stage"][recipe] = "TERMINAL"
                    if book["stage"][recipe] == "START":
                        episode = episodes[recipe]
                        contact = runtime_post["contact"][recipe]
                        left_force = float(np.linalg.norm(runtime_post["contact_force"][recipe, 0]))
                        right_force = float(np.linalg.norm(runtime_post["contact_force"][recipe, 1]))
                        if episode["event_steps"]["right_unload"] is None and right_force <= CONTACT_FORCE_N:
                            episode["event_steps"]["right_unload"] = step + 1
                        if episode["event_steps"]["left_support_dominance"] is None and contact[0] and left_force > right_force:
                            episode["event_steps"]["left_support_dominance"] = step + 1
                        had_contact = episode["event_steps"]["right_unload"] is not None
                        if had_contact and episode["event_steps"]["right_liftoff"] is None and not contact[1]:
                            episode["event_steps"]["right_liftoff"] = step + 1
                        if episode["event_steps"]["right_liftoff"] is not None and episode["event_steps"]["right_touchdown"] is None and contact[1]:
                            episode["event_steps"]["right_touchdown"] = step + 1
                            episode["landing"] = {"step": step + 1, "pose_error": errors, "vertical_velocity": float(runtime_post["foot_velocity"][recipe, 1, 2]), "impact_force": float(np.max(np.abs(runtime_post["contact_force"][recipe, 1]))), "dcm_error": None if errors is None else errors["dcm_error_m"], "support_transfer": runtime_post["contact"][recipe].tolist()}
                        heading = d26x.d26v.quat_matrix(source["root_pose"][recipe, 3:])[:2, 0]
                        displacement = float(np.dot(runtime_post["root_pose"][recipe, :2] - runtime_current["root_pose"][recipe, :2], heading))
                        episode["forward_displacement_m"] = max(float(episode.get("forward_displacement_m", 0.0)), float(np.dot(runtime_post["root_pose"][recipe, :2] - np.asarray(episode.get("source_root_xy", runtime_current["root_pose"][recipe, :2])), heading)))
                        episode["max_yaw_rate"] = max(float(episode.get("max_yaw_rate", 0.0)), abs(float(runtime_post["base_ang_vel_b"][recipe, 2])))
                        episode["max_roll_pitch"] = max(float(episode.get("max_roll_pitch", 0.0)), float(runtime_post["roll_pitch"][recipe]))
                        plan = plan_runtime_by_recipe[recipe]
                        book["plan_step"][recipe] += 1
                        elapsed = book["plan_step"][recipe] * DT
                        landing_deadline = sum(float(plan["identity"]["phase_durations_actual_s"][phase_name]) for phase_name in ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE")) + 0.20
                        if episode["event_steps"]["right_touchdown"] is not None:
                            touchdown_elapsed = (episode["event_steps"]["right_touchdown"] - int(book["endpoint_step"][recipe])) * DT
                            episode["touchdown_before_deadline"] = bool(touchdown_elapsed <= landing_deadline + 1.0e-9)
                        if episode["entry"]["pass"]:
                            book["stage"][recipe] = "WMOVE"
                            book["wmove_step"][recipe] = 0
                            episode["handoff_step"] = step + 2
                            episode["handoff"]["scheduled"] = True
                        elif book["plan_step"][recipe] >= int(plan["refs"]["total_steps"]):
                            episode["plan_complete"] = True
                            episode["first_divergence"] = {"classification": "WMOVE_BASIN_ENTRY_FAILURE", "control_step": step + 1, "phase": phase, "detail": "fixed D26X plan ended without ten-step entry confirmation"}
                            book["first_divergence"][recipe] = episode["first_divergence"]
                            book["stage"][recipe] = "TERMINAL"

                if stage_before == "WMOVE":
                    episode = episodes[recipe]
                    new_safety = [name for name in SAFETY_NAMES if bool(now_safety[name][recipe]) and book["first_safety"][name][recipe] == step + 1]
                    if new_safety and book["first_divergence"][recipe] is None:
                        selected_safety = next(name for name in SAFETY_NAMES if name in new_safety)
                        book["first_divergence"][recipe] = {"classification": first_divergence_name(selected_safety, "WMOVE"), "control_step": step + 1, "phase": "WMOVE", "detail": "handoff/retention canonical safety termination"}
                        episode["first_divergence"] = book["first_divergence"][recipe]
                    book["wmove_step"][recipe] += 1
                    if book["wmove_step"][recipe] >= WMOVE_STEPS or new_safety:
                        episode["handoff"]["completed_steps"] = len(episode["wmove_rows"])
                        book["stage"][recipe] = "TERMINAL"
                        if episode["first_divergence"] is None:
                            episode["handoff"]["pass"] = True

                previous_action = action.copy()
            if all(stage == "TERMINAL" for stage in book["stage"]):
                break

        for recipe in RECIPES:
            episode = episodes[recipe]
            episode["safety"] = {name: bool(book["flags"][name][recipe]) for name in SAFETY_NAMES}
            if episode.get("source_endpoint") is None:
                endpoint = book["endpoint"][recipe]
                episode["source_endpoint"] = source_gate_row(book, recipe, endpoint)
            if episode["source_endpoint"]["source_endpoint_eligible"]:
                touchdown = episode["event_steps"]["right_touchdown"]
                first = {
                    "right_unload": episode["event_steps"]["right_unload"] is not None,
                    "left_support_dominance": episode["event_steps"]["left_support_dominance"] is not None,
                    "right_liftoff": episode["event_steps"]["right_liftoff"] is not None,
                    "right_touchdown": touchdown is not None,
                    "forward_pelvis_displacement_m": float(episode.get("forward_displacement_m", 0.0)),
                    "forward_pelvis_displacement_pass": float(episode.get("forward_displacement_m", 0.0)) > FORWARD_DISPLACEMENT_M,
                    "yaw_rate_max_rad_s": float(episode.get("max_yaw_rate", 0.0)),
                    "yaw_rate_pass": float(episode.get("max_yaw_rate", 0.0)) <= FIRST_STEP_YAW_RATE_RAD_S,
                    "pelvis_roll_pitch_max_rad": float(episode.get("max_roll_pitch", 0.0)),
                    "pelvis_roll_pitch_pass": float(episode.get("max_roll_pitch", 0.0)) <= PELVIS_ROLL_PITCH_SAFE_RAD,
                    "touchdown_before_deadline": bool(episode.get("touchdown_before_deadline", False)),
                    "safety_pass": not any(episode["safety"].values()),
                }
                first["pass"] = bool(all(first[name] for name in ("right_unload", "left_support_dominance", "right_liftoff", "right_touchdown", "forward_pelvis_displacement_pass", "yaw_rate_pass", "pelvis_roll_pitch_pass", "touchdown_before_deadline", "safety_pass")))
                episode["first_step"] = first
            else:
                episode["first_step"] = {"pass": False}
            if episode["entry"].get("pass") and episode["first_step"].get("pass"):
                rows = episode["wmove_rows"]
                if rows:
                    forward = np.asarray([row["forward_error"] for row in rows])
                    lateral = np.asarray([row["lateral_velocity"] for row in rows])
                    yaw = np.asarray([row["yaw_rate"] for row in rows])
                    expected_left_only = any(bool(row["contact"][0]) and not bool(row["contact"][1]) for row in rows[2:])
                    entry["wmove_retention_stats"] = {"forward_error": mean_percentiles(forward), "lateral_velocity": mean_percentiles(lateral), "yaw_rate": mean_percentiles(yaw), "mean_forward_error": float(np.mean(forward)), "mean_lateral_velocity": float(np.mean(lateral)), "mean_yaw_rate": float(np.mean(yaw)), "phase_alternation": expected_left_only, "expected_next_side": "LEFT", "safety_pass": not any(episode["safety"].values())}
                    episode["handoff"]["pass"] = bool(np.mean(forward) <= ENTRY_FORWARD_ERROR_MPS and np.mean(lateral) <= ENTRY_LATERAL_VELOCITY_MPS and np.mean(yaw) <= ENTRY_YAW_RATE_RAD_S and expected_left_only and not any(episode["safety"].values()))
                else:
                    episode["handoff"]["pass"] = False
            episode["entry"]["confirmation_steps"] = ENTRY_CONFIRMATION_STEPS
            episode["entry"]["samples"] = episode["entry_samples"]
            episode["source_endpoint_eligible"] = bool(episode["source_endpoint"].get("source_endpoint_eligible"))
            if episode["first_divergence"] is None and not episode["first_step"].get("pass") and episode["source_endpoint_eligible"]:
                episode["first_divergence"] = {"classification": "SWING_TRACKING_FAILURE", "control_step": None, "phase": None, "detail": "fixed first-step acceptance gate not satisfied"}

        source_rows = [episodes[i]["source_endpoint"] for i in RECIPES]
        identity_rows = [plan_runtime_by_recipe[i]["identity"] for i in RECIPES]
        raw_result = {
            "mode": args.run,
            "seed": SEED,
            "recipe_ids": RECIPES,
            "source_endpoint_results": source_rows,
            "episodes": episodes,
            "identity_rows": identity_rows,
            "source_lifecycle_hashes": [row.get("source_endpoint_hash") for row in source_rows],
            "source_first_safety_steps": [{name: int(book["first_safety"][name][i]) for name in SAFETY_NAMES} for i in RECIPES],
            "reference_trace_hash": [array_hash(trace.arrays["reference_root_pose"][i, trace.arrays["active"][i]]) for i in RECIPES],
            "action_trace_hash": [array_hash(trace.arrays["action"][i, trace.arrays["active"][i]]) for i in RECIPES],
            "physics_state_trace_hash": [array_hash(trace.arrays["actual_root_pose"][i, trace.arrays["active"][i]]) for i in RECIPES],
            "contact_event_hash": [array_hash(trace.arrays["contact"][i, trace.arrays["active"][i]]) for i in RECIPES],
            "protected_start": primary_protected,
            "physics_executed": 1,
            "persistent_update": 0,
            "new_checkpoint": 0,
            "left_start_physics": 0,
            "wmove_handoff_speed_mps": WMOVE_SPEED,
        }
        trace_path = OUT / f"raw_{args.run}_trajectory.npz"
        trace.save(trace_path, {"mode": args.run, "seed": SEED, "recipe_ids": RECIPES, "plan_ids": [plan_runtime_by_recipe[i]["identity"]["plan_id"] for i in RECIPES]})
        dump(OUT / f"raw_{args.run}_physics_results.json", raw_result)
        wrapped.close()
    return raw_result


def parity_compare(primary_result: dict[str, Any], parity_result: dict[str, Any], primary_trace: dict[str, np.ndarray], parity_trace: dict[str, np.ndarray]) -> dict[str, Any]:
    identity_keys = ("source_lifecycle_hashes", "reference_trace_hash", "action_trace_hash", "physics_state_trace_hash", "contact_event_hash")
    identity = {key: primary_result.get(key) == parity_result.get(key) for key in identity_keys}
    continuous = ("reference_root_pose", "reference_root_velocity", "reference_com", "actual_root_pose", "actual_root_velocity", "actual_com_position", "actual_com_velocity", "actual_dcm", "actual_feet_pose", "actual_foot_velocity", "contact_force", "q_actual", "q_cmd", "action", "joint_velocity_ratio", "torque_ratio", "error_vector")
    numeric_rows = []
    bitwise = True
    tolerance = True
    for key in primary_trace:
        if key not in parity_trace or primary_trace[key].shape != parity_trace[key].shape:
            numeric_rows.append({"field": key, "shape_match": False, "bitwise": False, "within_tolerance": False})
            bitwise = False
            tolerance = False
            continue
        a, b = primary_trace[key], parity_trace[key]
        exact = bool(np.array_equal(a, b))
        if np.issubdtype(a.dtype, np.floating):
            close = bool(np.allclose(a, b, atol=PARITY_ABS_TOL, rtol=PARITY_REL_TOL, equal_nan=True))
        else:
            close = exact
        if np.issubdtype(a.dtype, np.number) and a.size:
            delta = np.abs(a - b)
            max_abs_delta = float(np.nanmax(delta)) if np.isfinite(delta).any() else None
        else:
            max_abs_delta = 0.0
        numeric_rows.append({"field": key, "shape_match": True, "bitwise": exact, "within_tolerance": close, "max_abs_delta": max_abs_delta})
        bitwise &= exact
        tolerance &= close
    class_match = [primary_result["episodes"][i].get("first_step", {}).get("pass") == parity_result["episodes"][i].get("first_step", {}).get("pass") and primary_result["episodes"][i].get("entry", {}).get("pass") == parity_result["episodes"][i].get("entry", {}).get("pass") and primary_result["episodes"][i].get("handoff", {}).get("pass") == parity_result["episodes"][i].get("handoff", {}).get("pass") and primary_result["episodes"][i].get("first_divergence") == parity_result["episodes"][i].get("first_divergence") for i in RECIPES]
    return {"name": "Exp014D27ProcessParityV1", "primary_mode": "primary", "fresh_replay_mode": "parity", "identity_hash_match": identity, "trace_fields": numeric_rows, "bitwise_match": bool(bitwise), "fixed_tolerance_match": bool(tolerance), "classification_match_per_recipe": class_match, "result_classification_match": all(class_match), "pass": bool(all(identity.values()) and tolerance and all(class_match)), "tolerance_contract": {"absolute": PARITY_ABS_TOL, "relative": PARITY_REL_TOL, "outcome_adjustment": False}}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value) for key, value in row.items()})


def make_tracking(primary_trace: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metrics = ("root_position_error_m", "root_orientation_error_rad", "com_position_error_m", "com_velocity_error_mps", "dcm_error_m", "stance_position_error_m", "stance_orientation_error_rad", "swing_position_error_m", "swing_orientation_error_rad", "joint_target_error_rad", "contact_timing_error", "pelvis_roll_pitch_rad")
    for recipe in RECIPES:
        active = primary_trace["active"][recipe] & (primary_trace["stage"][recipe] == STAGE_CODE["START"]) & primary_trace["reference_valid"][recipe]
        phase_codes = primary_trace["phase"][recipe]
        for phase in PHASES:
            mask = active & (phase_codes == PHASE_CODE[phase])
            if not np.any(mask):
                continue
            values = primary_trace["error_vector"][recipe, mask]
            row = {"recipe_id": recipe, "phase": phase, "samples": int(np.sum(mask))}
            for index, metric in enumerate(metrics):
                stat = mean_percentiles(values[:, index])
                row[f"{metric}_p50"] = stat["p50"]
                row[f"{metric}_p95"] = stat["p95"]
                row[f"{metric}_max"] = stat["max"]
            row["joint_position_tracking_p50"] = float(np.quantile(np.linalg.norm(primary_trace["q_cmd"][recipe, mask] - primary_trace["q_actual"][recipe, mask], axis=1), 0.50))
            row["joint_position_tracking_p95"] = float(np.quantile(np.linalg.norm(primary_trace["q_cmd"][recipe, mask] - primary_trace["q_actual"][recipe, mask], axis=1), 0.95))
            row["joint_position_tracking_max"] = float(np.max(np.linalg.norm(primary_trace["q_cmd"][recipe, mask] - primary_trace["q_actual"][recipe, mask], axis=1)))
            row["action_rate_l2_p50"] = float(np.quantile(np.linalg.norm(primary_trace["action_rate"][recipe, mask], axis=1), 0.50))
            row["action_rate_l2_p95"] = float(np.quantile(np.linalg.norm(primary_trace["action_rate"][recipe, mask], axis=1), 0.95))
            row["action_rate_l2_max"] = float(np.max(np.linalg.norm(primary_trace["action_rate"][recipe, mask], axis=1)))
            rows.append(row)
    return rows, {"name": "Exp014D27ActualReferenceTrackingV1", "rows": rows, "percentiles": ["p50", "p95", "max"], "source": "primary fresh lifecycle only", "reference_vs_actual": "post-physics actual state against fixed current-step reference", "command_history_excluded_from_entry_distance": True}


def write_phase_artifacts(primary_result: dict[str, Any], primary_trace: dict[str, np.ndarray], plan_audit: dict[str, Any], entry_contract: dict[str, Any], source: dict[str, np.ndarray], native: dict[str, np.ndarray], start_head: str, start_status: list[str], parity: dict[str, Any]) -> str:
    episodes = primary_result["episodes"]
    first_count = sum(bool(row.get("first_step", {}).get("pass")) for row in episodes)
    entry_count = sum(bool(row.get("entry", {}).get("pass")) for row in episodes)
    handoff_count = sum(bool(row.get("handoff", {}).get("pass")) for row in episodes)
    safety_failures = sum(any(row.get("safety", {}).values()) for row in episodes if row.get("source_endpoint_eligible"))
    wbik_failures = sum(row.get("first_divergence", {}).get("classification") == "WBIK_RUNTIME_FAILURE" for row in episodes if row.get("first_divergence"))
    if not parity["pass"]:
        classification = "EXP014_D27_RUNTIME_PARITY_FAIL"
    elif wbik_failures:
        classification = "EXP014_D27_WBIK_RUNTIME_FAIL"
    elif safety_failures:
        classification = "EXP014_D27_MODEL_BASED_START_SAFETY_FAIL"
    elif first_count < 2:
        classification = "EXP014_D27_POSITION_REFERENCE_DYNAMICS_FAIL"
    elif first_count >= 6 and entry_count < 4:
        classification = "EXP014_D27_RIGHT_FIRST_STEP_PASS_WMOVE_ENTRY_FAIL"
    elif entry_count >= 4 and handoff_count < math.ceil(0.75 * max(entry_count, 1)):
        classification = "EXP014_D27_RIGHT_WMOVE_ENTRY_PASS_HANDOFF_FAIL"
    elif first_count >= 6 and entry_count >= 4 and handoff_count >= math.ceil(0.75 * entry_count):
        classification = "EXP014_D27_RIGHT_MODEL_BASED_START_ROUTE_PASS"
    else:
        classification = "EXP014_D27_MULTIPLE_FAILURES"

    phase_a = {"name": "Exp014D27PhaseAWeightShiftV1", "episodes": []}
    phase_b = {"name": "Exp014D27PhaseBRightFirstSwingV1", "episodes": []}
    phase_c = {"name": "Exp014D27PhaseCLandingAndCaptureV1", "episodes": []}
    phase_d = {"name": "Exp014D27PhaseDWMOVEAcceptanceV1", "entry_contract": entry_contract, "episodes": []}
    post_contract, post_default_q, post_action_scale = d26x.source_contract()
    post_plans, _ = resolve_selected_plans(source, native, d26x.load_wmove_geometry(), post_default_q, post_action_scale)
    post_plan_by_recipe = {int(plan["identity"]["source_recipe"]): plan for plan in post_plans}

    def finite_stat(values: Any) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        return mean_percentiles(array[np.isfinite(array)])

    def clean_points(values: np.ndarray) -> list[list[float | None]]:
        array = np.asarray(values, dtype=np.float64)
        return [[float(value) if np.isfinite(value) else None for value in row] for row in array]

    for episode in episodes:
        rid = int(episode["recipe_id"])
        active = primary_trace["active"][rid]
        phases = primary_trace["phase"][rid]
        start_indices = np.flatnonzero(active & (primary_trace["stage"][rid] == STAGE_CODE["START"]))
        a_mask = active & (phases == PHASE_CODE["DOUBLE_SUPPORT_SHIFT"])
        b_mask = active & (phases == PHASE_CODE["FIRST_SWING"])
        c_mask = active & (phases == PHASE_CODE["LANDING_AND_CAPTURE"])
        d_mask = active & (phases == PHASE_CODE["WMOVE_ACCEPTANCE"])
        force = primary_trace["contact_force"][rid]
        load = np.divide(np.linalg.norm(force[:, 0], axis=1), np.maximum(np.linalg.norm(force[:, 0], axis=1) + np.linalg.norm(force[:, 1], axis=1), 1.0e-9))
        lateral = primary_trace["actual_root_pose"][rid, :, 1]
        source_y = lateral[np.flatnonzero(active)[0]] if np.any(active) else np.nan
        plan = post_plan_by_recipe[rid]
        n_ref = min(len(start_indices), int(plan["refs"]["total_steps"]))
        zmp_reference = np.asarray(plan["refs"]["zmp"][:n_ref], dtype=np.float64)
        if n_ref:
            start_force = force[start_indices[:n_ref]]
            start_feet = primary_trace["actual_feet_pose"][rid, start_indices[:n_ref], :, :3]
            start_weights = np.linalg.norm(start_force, axis=2)
            weight_sum = start_weights.sum(axis=1)
            actual_cop = np.full((n_ref, 2), np.nan, dtype=np.float64)
            valid_cop = weight_sum > 1.0e-9
            actual_cop[valid_cop] = (start_weights[valid_cop, :, None] * start_feet[valid_cop, :, :2]).sum(axis=1) / weight_sum[valid_cop, None]
            stance_slip = np.linalg.norm(primary_trace["actual_foot_velocity"][rid, start_indices[:n_ref], :, :2], axis=2)
        else:
            actual_cop = np.empty((0, 2), dtype=np.float64)
            stance_slip = np.empty((0, 2), dtype=np.float64)
        a_start_mask = np.asarray([phases[index] == PHASE_CODE["DOUBLE_SUPPORT_SHIFT"] for index in start_indices[:n_ref]], dtype=bool)
        phase_a["episodes"].append({
            "recipe_id": rid,
            "samples": int(np.sum(a_mask)),
            "left_force_N": finite_stat(np.linalg.norm(force[a_mask, 0], axis=1)),
            "right_force_N": finite_stat(np.linalg.norm(force[a_mask, 1], axis=1)),
            "left_load_ratio": finite_stat(load[a_mask]),
            "com_lateral_displacement_m": None if not np.any(a_mask) else float(np.max(np.abs(primary_trace["actual_com_position"][rid, a_mask, 1] - source_y))),
            "root_lateral_displacement_m": None if not np.any(a_mask) else float(np.max(np.abs(primary_trace["actual_root_pose"][rid, a_mask, 1] - source_y))),
            "zmp_reference_xy": clean_points(zmp_reference),
            "zmp_reference_x": finite_stat(zmp_reference[:, 0] if n_ref else []),
            "zmp_reference_y": finite_stat(zmp_reference[:, 1] if n_ref else []),
            "estimated_actual_cop_xy": clean_points(actual_cop),
            "estimated_actual_cop_x": finite_stat(actual_cop[:, 0] if n_ref else []),
            "estimated_actual_cop_y": finite_stat(actual_cop[:, 1] if n_ref else []),
            "yaw_rate_rad_s": finite_stat(np.abs(primary_trace["actual_root_velocity"][rid, a_mask, 5])),
            "lz_diagnostic": {"available": False, "reason": "D27 trace contract captures yaw rate and contact moment proxies, but not a PhysX articulated angular-momentum tensor"},
            "stance_foot_slip_mps": finite_stat(stance_slip[a_start_mask, 0] if n_ref else []),
            "safety": episode["safety"],
        })
        b_foot_z = primary_trace["actual_feet_pose"][rid, :, 1, 2]
        source_foot_z = b_foot_z[np.flatnonzero(active)[0]] if np.any(active) else np.nan
        phase_b["episodes"].append({"recipe_id": rid, "samples": int(np.sum(b_mask)), "right_unload_step": episode["event_steps"]["right_unload"], "right_liftoff_step": episode["event_steps"]["right_liftoff"], "right_clearance_m": None if not np.any(b_mask) else float(np.max(b_foot_z[b_mask] - source_foot_z)), "left_stance_error_p95_m": None if not np.any(b_mask) else float(np.quantile(primary_trace["error_vector"][rid, b_mask, 5], 0.95)), "root_forward_velocity_mps": None if not np.any(b_mask) else finite_stat(primary_trace["actual_root_velocity"][rid, b_mask, 0]), "com_forward_velocity_mps": None if not np.any(b_mask) else finite_stat(primary_trace["actual_com_velocity"][rid, b_mask, 0]), "yaw_rate_rad_s": None if not np.any(b_mask) else finite_stat(np.abs(primary_trace["actual_root_velocity"][rid, b_mask, 5])), "joint_velocity_ratio_max": None if not np.any(b_mask) else finite_stat(np.max(primary_trace["joint_velocity_ratio"][rid, b_mask], axis=1)), "torque_ratio_max": None if not np.any(b_mask) else finite_stat(np.max(primary_trace["torque_ratio"][rid, b_mask], axis=1)), "safety": episode["safety"]})
        phase_c["episodes"].append({"recipe_id": rid, "samples": int(np.sum(c_mask)), "right_touchdown_step": episode["event_steps"]["right_touchdown"], "landing": episode.get("landing"), "landing_vertical_velocity_mps": None if episode.get("landing") is None else episode["landing"].get("vertical_velocity"), "support_transfer": None if episode.get("landing") is None else episode["landing"].get("support_transfer"), "dcm_error": None if not np.any(c_mask) else finite_stat(primary_trace["error_vector"][rid, c_mask, 4]), "impact_force_N": None if not np.any(c_mask) else finite_stat(np.max(np.abs(force[c_mask, 1]), axis=1)), "contact_force_N": None if not np.any(c_mask) else finite_stat(np.linalg.norm(force[c_mask, 1], axis=1)), "safety": episode["safety"]})
        entry_dist = primary_trace["entry_distance"][rid, d_mask | c_mask]
        entry_samples = episode.get("entry_samples", [])
        phase_d["episodes"].append({"recipe_id": rid, "samples": int(np.sum(d_mask | c_mask)), "entry_distance": finite_stat(entry_dist), "entry_p95": float(entry_contract["entry_neighborhood_p95"]), "entry_confirmation_step": episode.get("entry_confirmation_step"), "entry_pass": bool(episode.get("entry", {}).get("pass")), "samples_detail": entry_samples, "velocity_error_mps": finite_stat([row.get("forward_velocity_error", np.nan) for row in entry_samples]), "lateral_velocity_mps": finite_stat([row.get("lateral_velocity", np.nan) for row in entry_samples]), "yaw_rate_rad_s": finite_stat([row.get("yaw_rate", np.nan) for row in entry_samples]), "contact_phase": [row.get("phase_match") for row in entry_samples], "expected_next_side": "LEFT", "action_continuity": finite_stat(np.linalg.norm(primary_trace["action_rate"][rid, d_mask | c_mask], axis=1)), "safety": episode["safety"]})

    tracking_rows, tracking_json = make_tracking(primary_trace)
    write_csv(OUT / "actual_reference_tracking.csv", tracking_rows)
    dump(OUT / "actual_reference_tracking.json", tracking_json)
    dump(OUT / "phase_a_weight_shift.json", phase_a)
    dump(OUT / "phase_b_first_swing.json", phase_b)
    dump(OUT / "phase_c_landing.json", phase_c)
    dump(OUT / "phase_d_wmove_acceptance.json", phase_d)

    first_rows = []
    entry_rows = []
    handoff_rows = []
    divergence_rows = []
    for episode in episodes:
        first_rows.append({"recipe_id": episode["recipe_id"], "plan_id": episode["plan_id"], **episode.get("first_step", {})})
        entry_rows.append({"recipe_id": episode["recipe_id"], "plan_id": episode["plan_id"], **episode.get("entry", {})})
        handoff_rows.append({"recipe_id": episode["recipe_id"], "plan_id": episode["plan_id"], **episode.get("handoff", {})})
        divergence_rows.append({"recipe_id": episode["recipe_id"], **(episode.get("first_divergence") or {"classification": None})})
    dump(OUT / "first_step_results.json", {"name": "Exp014D27FirstStepResultsV1", "safe_first_step_count": first_count, "required": 6, "rows": first_rows})
    dump(OUT / "wmove_entry_results.json", {"name": "Exp014D27WMOVEEntryResultsV1", "entry_count": entry_count, "required": 4, "rows": entry_rows})
    dump(OUT / "wmove_handoff_results.json", {"name": "Exp014D27WMOVEHandoffResultsV1", "handoff_pass_count": handoff_count, "entry_count": entry_count, "required_retention_fraction": 0.75, "rows": handoff_rows})
    dump(OUT / "first_divergence.json", {"name": "Exp014D27FirstDivergenceV1", "rows": divergence_rows, "one_primary_failure_per_episode": True, "timeout_only_label_used": False})
    dump(OUT / "primary_physics_results.json", {"name": "Exp014D27PrimaryPhysicsResultsV1", "episodes": episodes, "safe_first_step_count": first_count, "entry_count": entry_count, "handoff_pass_count": handoff_count, "classification": classification, "primary_only_denominator": 8, "physics_executed": 1})
    csv_rows = []
    for episode in episodes:
        csv_rows.append({"recipe_id": episode["recipe_id"], "plan_id": episode["plan_id"], "source_endpoint_eligible": episode["source_endpoint_eligible"], "first_step_pass": episode.get("first_step", {}).get("pass", False), "entry_pass": episode.get("entry", {}).get("pass", False), "handoff_pass": episode.get("handoff", {}).get("pass", False), "first_divergence": None if not episode.get("first_divergence") else episode["first_divergence"].get("classification"), "fall": episode["safety"]["fall"], "dangerous_slip": episode["safety"]["dangerous_slip"], "impact": episode["safety"]["impact"], "velocity_saturation": episode["safety"]["velocity_saturation"], "torque_saturation": episode["safety"]["torque_saturation"], "support_loss": episode["safety"]["support_loss"]})
    write_csv(OUT / "primary_physics_results.csv", csv_rows)

    dump(OUT / "process_parity.json", parity)
    full_pass = classification == "EXP014_D27_RIGHT_MODEL_BASED_START_ROUTE_PASS"
    if full_pass:
        auth = {"name": "Exp014D28RightStartTeacherExpansionAuthorizationV1", "authorized": True, "classification": classification, "scope": "RIGHT first swing only", "selected_sources": RECIPES, "fixed_target_contract": {"target_id": RIGHT_TARGET_ID, "bundle_row": RIGHT_TARGET_ROW, "episode": RIGHT_ENTRY_EPISODE, "control_step": RIGHT_ENTRY_CONTROL_STEP}, "fixed_plan_contract": plan_audit, "fixed_timing_contract": "D26X selected source-specific FAST/NOMINAL plans", "wbik": "V2A", "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; no clipping", "endpoint_mapper": "Exp014EndpointFeedforwardActionMapperV1", "trajectory_capture": True, "remaining_unauthorized": ["LEFT first swing", "bilateral START claim", "validation 102", "held-out", "final S_START authorization"], "persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0}
        dump(OUT / "exp014_d28_right_start_teacher_expansion_authorization.json", auth)
        not_auth_path = OUT / "exp014_d28_not_authorized.json"
        if not_auth_path.exists():
            not_auth_path.unlink()
    else:
        not_auth = {"name": "Exp014D28NotAuthorizedV1", "authorized": False, "classification": classification, "reason": "D27 full RIGHT route gate did not pass", "first_step_count": first_count, "entry_count": entry_count, "handoff_pass_count": handoff_count, "process_parity_pass": parity["pass"], "next_mapping": "landing-to-WMOVE capture repair" if classification == "EXP014_D27_RIGHT_FIRST_STEP_PASS_WMOVE_ENTRY_FAIL" else "dynamics-constrained centroidal trajectory optimization or MPC feedback" if classification == "EXP014_D27_POSITION_REFERENCE_DYNAMICS_FAIL" else "no D28 authorization", "persistent_update": 0, "new_checkpoint": 0, "left_physics": 0}
        dump(OUT / "exp014_d28_not_authorized.json", not_auth)
        auth_path = OUT / "exp014_d28_right_start_teacher_expansion_authorization.json"
        if auth_path.exists():
            auth_path.unlink()

    interpretation = {
        "classification": classification,
        "safe_first_step_count": first_count,
        "entry_count": entry_count,
        "handoff_pass_count": handoff_count,
        "process_parity": parity["pass"],
        "next_action": "expand RIGHT model-based Teacher to remaining train-only sources" if classification == "EXP014_D27_RIGHT_MODEL_BASED_START_ROUTE_PASS" else "build model-based landing-to-W_MOVE capture segment" if classification == "EXP014_D27_RIGHT_FIRST_STEP_PASS_WMOVE_ENTRY_FAIL" else "dynamics-constrained centroidal trajectory optimization or MPC feedback" if classification == "EXP014_D27_POSITION_REFERENCE_DYNAMICS_FAIL" else "retain RIGHT-only diagnostic scope; no D28 authorization",
    }
    dump(OUT / "stage_classification.json", {"classification": classification, "classification_precedence": ["runtime parity", "WBIK runtime", "model-based safety", "offline-to-physics dynamics", "first-step/entry", "handoff", "full route"], "d26x_classification_unchanged": "EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS", "physics_scope": "RIGHT first swing only", "primary_episode_count": 8, "parity_episode_count": 8, "safe_first_step_count": first_count, "entry_basin_count": entry_count, "handoff_retention_pass_count": handoff_count, "interpretation": interpretation, "persistent_update": 0, "new_checkpoint": 0, "remote_push": False})
    dump(OUT / "recommended_next_action.json", interpretation)

    protected_start = primary_result["protected_start"]
    protected_end = protected_snapshot()
    protected_unchanged = protected_start["aggregate_sha256"] == protected_end["aggregate_sha256"] and protected_start["files"] == protected_end["files"]
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_paths": protected_end["files"], "protected_start_aggregate_sha256": protected_start["aggregate_sha256"], "protected_end_aggregate_sha256": protected_end["aggregate_sha256"], "unchanged": protected_unchanged, "exp005_to_exp013_unchanged": protected_unchanged, "d6_to_d26x_unchanged": protected_unchanged, "S_HOLD_unchanged": protected_unchanged, "Stage_2Q_unchanged": protected_unchanged, "W_MOVE_unchanged": protected_unchanged, "S_STOP_OMNI_unchanged": protected_unchanged, "WBIK_V1_V2_V2A_unchanged": protected_unchanged, "persistent_update": 0, "new_learned_checkpoint": 0, "left_start_physics": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "raw_restore": 0, "remote_push": False})

    bundle_src = OUT / "raw_primary_trajectory.npz"
    bundle_dst = OUT / "model_based_start_trajectory_bundle.npz"
    shutil.copyfile(bundle_src, bundle_dst)
    dump(OUT / "model_based_start_trajectory_manifest.json", {"name": "Exp014D27ModelBasedStartTrajectoryBundleV1", "source": "primary eight fresh S_HOLD lifecycles", "bundle": bundle_dst.name, "bundle_sha256": sha256_file(bundle_dst), "recipe_ids": RECIPES, "plan_ids": [row["plan_id"] for row in plan_audit["rows"]], "identity_complete": True, "root_reference_is_not_simulation_state": True, "left_physics": 0, "wmove_steps_per_eligible_handoff": WMOVE_STEPS})
    (OUT / "model_based_start_trajectory_bundle.sha256").write_text(sha256_file(bundle_dst) + "\n", encoding="ascii")
    contract_for_manifest, default_q_for_manifest, action_scale_for_manifest = d26x.source_contract()
    dump(OUT / "authorized_plan_manifest.json", {"name": "Exp014D27AuthorizedRightPlanManifestV1", "authorization": json.loads((D26X / "exp014_d27_model_based_start_physics_authorization.json").read_text(encoding="utf-8")), "selected_offline_plans": plan_audit["rows"], "joint_contract": joint_contract(contract_for_manifest, source, default_q_for_manifest, action_scale_for_manifest), "physics_scope": "RIGHT first swing only", "primary_count": 8, "parity_count": 8})
    dump(OUT / "source_endpoint_results.json", {"name": "Exp014D27SourceEndpointResultsV1", "primary": primary_result["source_endpoint_results"], "source_gate": "all eight endpoint-eligible required; ineligible source does not execute plan", "parity_source_endpoint_hashes": primary_result.get("source_lifecycle_hashes")})
    dump(OUT / "plan_identity_audit.json", plan_audit)
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D27", "starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "starting_git_status_short": start_status, "primary_and_parity_processes": ["primary", "parity"], "D26X_read_only": True, "D26X_artifacts_not_overwritten": True, "remote_push": False, "physics_executed_primary": 8, "physics_executed_parity": 8, "persistent_policy_update": 0, "new_checkpoint": 0, "left_start_physics": 0})
    dump(OUT / "protocol.json", build_protocol(start_head, start_status, source, native, entry_contract, plan_audit, "finalized_after_primary_and_parity"))
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d27_right_model_based_start_physics.py' --headless --run primary\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d27_right_model_based_start_physics.py' --headless --run parity\n", encoding="utf-8")

    source_pass = sum(bool(row["source_endpoint_eligible"]) for row in episodes)
    first_detail = "; ".join(f"R{row['recipe_id']} {row.get('first_step', {}).get('pass', False)}" for row in episodes)
    report_text = f"""# EXP014 Phase 2-D27 RIGHT model-based START physics

Classification: `{classification}`.

## Source and plans

The authorized scope was RIGHT first swing only. Eight fixed D26X plans were identity-checked before physics, with D26X classification preserved as `EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS`. The exact selected plan rows and source-specific timing classes are in `authorized_plan_manifest.json`; target `RIGHT_000` is episode {RIGHT_ENTRY_EPISODE}, control step {RIGHT_ENTRY_CONTROL_STEP}. No plan, target, clearance, root reference, or duration was changed.

Fresh lifecycle endpoint gate: **{source_pass}/8** eligible. Ineligible sources were fail-closed as `SOURCE_ENDPOINT_INELIGIBLE` and did not execute START.

## Weight shift

Phase A diagnostics are in `phase_a_weight_shift.json`. The primary first-step results are {first_count}/8 (`{first_detail}`); load transfer, left/right contact force, CoM/root displacement, and safety are retained per recipe.

## First swing

Phase B records RIGHT unload, liftoff, clearance, LEFT stance error, forward velocity, yaw, and saturation. The gate used the unchanged contact (>5 N), slip, impact, saturation, support-loss, fall, NaN/Inf, forward displacement, yaw, and roll/pitch contracts.

## Landing

Phase C records RIGHT touchdown, landing pose error, vertical velocity, impact force, support transfer, and DCM error. A hard safety event terminated the model-based episode and did not hand off to W_MOVE.

## W_MOVE entry

Entry used the pre-fixed physical-only D26T feature distance: command/history dimensions were excluded, the RIGHT p95 threshold was fixed before physics, and no new or interpolated state was created. Entry acceptance required the fixed velocity/yaw limits, target support phase, and ten-step continuous confirmation. Accepted entries: **{entry_count}/8**.

## Handoff

Only accepted entries hard-switched to the frozen W_MOVE checkpoint at 0.3 m/s for 75 control steps; no action blending was used. Handoff-retention passes: **{handoff_count}/{entry_count}** accepted entries. Handoff action jump, cosine, joint-target jump, continuity, velocity/yaw retention, next-side alternation, and safety are in `wmove_handoff_results.json` and `phase_d_wmove_acceptance.json`.

## Tracking and first divergence

`actual_reference_tracking.csv/.json` reports p50/p95/max root, CoM, DCM, stance/swing foot, joint-target, action-rate, and contact timing errors per source and phase. `first_divergence.json` records one primary failure label per episode; timeout alone was not used as a classification.

## Process parity

The first eight episodes ran in `primary`; the same eight recipes and fixed plans ran in an independent fresh `parity` process. Parity classification: **{parity['pass']}**. Bitwise equality was attempted first; if needed, the pre-registered absolute/relative tolerance was {PARITY_ABS_TOL:g}/{PARITY_REL_TOL:g}. No result-dependent tolerance was introduced.

## Authorization

`exp014_d28_right_start_teacher_expansion_authorization.json` is present only if the full RIGHT route gate passed; otherwise `exp014_d28_not_authorized.json` records the fail-closed result. LEFT first swing, bilateral START, validation, held-out, final S_START authorization, PPO/CEM, persistent update, and new checkpoints remain unauthorized.

## Protection and repository

Protected hashes are in `protected_hashes.json`; protected inputs remained unchanged: {json.dumps(bool(json.loads((OUT / 'protected_hashes.json').read_text(encoding='utf-8'))['unchanged']))}. Persistent update: `0`; new learned checkpoint: `0`; LEFT physics: `0`; remote push: `false`.

Starting HEAD: `{start_head}`. Ending HEAD before commit: `{git('rev-parse', 'HEAD')}`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text, encoding="utf-8")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("primary", "parity"), required=True)
    parser.add_argument("--preflight", action="store_true", help="resolve and verify D26X identity without launching physics")
    parser.add_argument("--finalize-existing", action="store_true", help="finalize already captured primary/parity raw artifacts without physics")
    add_launcher_args(parser)
    args, hydra = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    source = d26x.load_npz(D26U / "fresh_shold_identity_complete_sources.npz")
    native = d26x.load_npz(D26S / "native_steady_trace_bundle.npz")
    contract, default_q, action_scale = d26x.source_contract()
    geometry = d26x.load_wmove_geometry()
    entry_contract = build_entry_distance_contract(native)
    plans, plan_audit = resolve_selected_plans(source, native, geometry, default_q, action_scale)
    if args.preflight:
        print(json.dumps({"preflight": "PASS", "plans": [row["plan_id"] for row in plan_audit["rows"]], "physics_executed": 0}, indent=2), flush=True)
        return
    if args.finalize_existing:
        if args.run != "parity":
            raise RuntimeError("--finalize-existing requires --run parity")
        primary_path = OUT / "raw_primary_physics_results.json"
        primary_trace_path = OUT / "raw_primary_trajectory.npz"
        parity_path = OUT / "raw_parity_physics_results.json"
        parity_trace_path = OUT / "raw_parity_trajectory.npz"
        if not all(path.exists() for path in (primary_path, primary_trace_path, parity_path, parity_trace_path)):
            raise RuntimeError("finalize-existing requires completed primary and parity raw artifacts")
        primary_result = json.loads(primary_path.read_text(encoding="utf-8"))
        parity_result = json.loads(parity_path.read_text(encoding="utf-8"))
        parity = parity_compare(primary_result, parity_result, load_trace(primary_trace_path), load_trace(parity_trace_path))
        classification = write_phase_artifacts(primary_result, load_trace(primary_trace_path), plan_audit, entry_contract, source, native, start_head, start_status, parity)
        print(json.dumps({"run": "parity", "classification": classification, "parity": parity["pass"]}, indent=2), flush=True)
        return
    for plan in plans:
        plan["source"] = source
    if args.run == "primary":
        dump(OUT / "stage_reference.json", {"stage": "Phase 2-D27", "starting_head": start_head, "starting_git_status_short": start_status, "D26X_read_only": True, "physics_executed": 0, "primary_process_started": True, "parity_process_started": False, "remote_push": False})
        dump(OUT / "plan_identity_audit.json", plan_audit)
        dump(OUT / "protocol.json", build_protocol(start_head, start_status, source, native, entry_contract, plan_audit, "primary"))
        dump(OUT / "authorized_plan_manifest.json", {"name": "Exp014D27AuthorizedRightPlanManifestV1", "selected_offline_plans": plan_audit["rows"], "physics_scope": "RIGHT first swing only", "primary_count": 8, "parity_count": 8})
        dump(OUT / "source_endpoint_results.json", {"name": "Exp014D27SourceEndpointResultsV1", "status": "PRIMARY_PENDING", "gate": "all eight endpoint-eligible required"})
        result = run_physics(args, plans, source, native, default_q, action_scale, entry_contract)
        print(json.dumps({"run": "primary", "source_endpoint_hashes": result["source_lifecycle_hashes"], "first_step": [row.get("first_step", {}).get("pass") for row in result["episodes"]]}, indent=2), flush=True)
        return
    primary_path = OUT / "raw_primary_physics_results.json"
    primary_trace_path = OUT / "raw_primary_trajectory.npz"
    if not primary_path.exists() or not primary_trace_path.exists():
        raise RuntimeError("parity requires completed primary process artifacts")
    primary_result = json.loads(primary_path.read_text(encoding="utf-8"))
    primary_trace = load_trace(primary_trace_path)
    dump(OUT / "protocol.json", build_protocol(start_head, start_status, source, native, entry_contract, plan_audit, "parity"))
    parity_result = run_physics(args, plans, source, native, default_q, action_scale, entry_contract)
    parity_trace = load_trace(OUT / "raw_parity_trajectory.npz")
    parity = parity_compare(primary_result, parity_result, primary_trace, parity_trace)
    classification = write_phase_artifacts(primary_result, primary_trace, plan_audit, entry_contract, source, native, start_head, start_status, parity)
    print(json.dumps({"run": "parity", "classification": classification, "parity": parity["pass"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
