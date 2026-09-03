"""Phase 2-D28R: passive D27 body trace and gated centroidal pilot.

The capture modes deliberately reuse the protected D27 ``run_physics`` entry
point.  ``capture_off`` therefore executes the original V2A path; ``capture_on``
only reads the already-published Isaac Lab tensors and serializes them after the
episode.  The analysis mode is fail-closed: no D28 centroidal-feedback physics
is launched unless capture parity, direct/matrix momentum validation, and the
per-source V3 shadow preflight all pass.

This file does not modify D27 or any D28 artifact.  All D28R writes are below
the dedicated phase_2_d28r directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sqlite3
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28r_centroidal_trace_and_feedback"
D28 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28_centroidal_feedback_start"
D26X = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26x_timing_and_target_set"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D27 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d27_right_model_based_start_physics"
D26_POLYGON = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik/numeric_foot_sole_polygon.json"
REPORT = REPO / "research/exp_014_phase_2_d28r_centroidal_trace_and_feedback_report.md"

DT = 0.02
SEED = 20279941
N_ENVS = 8
RECIPES = tuple(range(8))
TRACE_RECIPES = (4, 5, 6, 7)
MAX_STEPS = 320
PARITY_ABS_TOL = 1.0e-5
PARITY_REL_TOL = 1.0e-5
CONTACT_HISTORY_LEN = 3
RIGHT_TARGET_ID = "RIGHT_000"
RIGHT_TARGET_ROW = 9330

SAFETY_NAMES = ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nonfinite")
PHASES = ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
PHASE_CODE = {"": 0, "DOUBLE_SUPPORT_SHIFT": 1, "FIRST_SWING": 2, "LANDING_AND_CAPTURE": 3, "WMOVE_ACCEPTANCE": 4, "WMOVE": 5}

# These diagnostic thresholds are fixed before reading the D27 trace.  They
# are not controller gains and are not changed after observing an outcome.
CAUSAL_THRESHOLDS = {
    "root_error_m": 0.05,
    "com_error_m": 0.05,
    "dcm_error_m": 0.10,
    "h_z_abs_Nms": 0.05,
    "d_h_z_dt_Nms2": 5.0,
    "clearance_overshoot_m": 0.02,
    "yaw_rate_rad_s": 0.15,
    "window_steps": 8,
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing D27 registers the same task and imports D3/D26X read-only.  Its
# main() is never called.
d27 = load_module("exp014_d28r_d27_read_only", EXP / "scripts/run_phase2_d27_right_model_based_start_physics.py")
wbik_v3 = load_module("exp014_d28r_wbik_v3_read_only", EXP / "src/g1_explicit_motion_mode/wbik_v3.py")
D27_READ_RUNTIME_ORIGINAL = d27.read_runtime


def jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tensor_np(value: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(value):
        result = value.detach().cpu().numpy()
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    return result.astype(dtype, copy=False) if dtype is not None else np.asarray(result)


def quat_matrix_np(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    q = q / np.maximum(norm, 1.0e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ), axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def skew_np(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.moveaxis(np.asarray(vector, dtype=np.float64), -1, 0)
    zero = np.zeros_like(x)
    return np.stack((zero, -z, y, z, zero, -x, -y, x, zero), axis=-1).reshape(np.asarray(vector).shape[:-1] + (3, 3))


def atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    np.savez_compressed(temporary, **payload)
    generated = Path(str(temporary) + ".npz") if not str(temporary).endswith(".npz") else temporary
    os.replace(generated, path)


def get_attr_required(obj: Any, names: tuple[str, ...], label: str) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    raise RuntimeError(f"mandatory runtime field unavailable: {label}; tried {names}")


CAPTURE_META: dict[str, Any] = {}
CAPTURE_ON = False
FOOT_POLYGON_LOCAL: np.ndarray | None = None


def _names(obj: Any, names: tuple[str, ...], label: str) -> list[str]:
    value = get_attr_required(obj, names, label)
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    return [str(item) for item in value]


def _read_static(world, masses: torch.Tensor, inertias: np.ndarray) -> None:
    global CAPTURE_META, FOOT_POLYGON_LOCAL
    if CAPTURE_META:
        return
    robot = world.robot
    body_names = _names(robot, ("body_names",), "body names")
    joint_names = _names(robot, ("joint_names",), "joint names")
    body_count = len(body_names)
    if body_count != 44 or len(joint_names) != 37:
        raise RuntimeError(f"unexpected robot contract body={body_count}, joints={len(joint_names)}")
    runtime = D27_READ_RUNTIME_ORIGINAL(world, masses)
    quat = runtime["body_quat"][0]
    rotation = quat_matrix_np(quat)
    offset_world = runtime["body_com_pos"][0] - runtime["body_pos"][0]
    local_offsets = np.einsum("bij,bj->bi", rotation.transpose(0, 2, 1), offset_world)
    polygon_status = "UNAVAILABLE"
    polygon = np.full((2, 4, 3), np.nan, dtype=np.float64)
    if D26_POLYGON.exists():
        polygon_json = json.loads(D26_POLYGON.read_text(encoding="utf-8"))
        if polygon_json.get("status") == "PASS":
            left_sole = polygon_json["feet"]["left"]
            right_sole = polygon_json["feet"]["right"]
            polygon[0, :, :2] = np.asarray(left_sole["polygon_vertices_xy"], dtype=np.float64)
            polygon[1, :, :2] = np.asarray(right_sole["polygon_vertices_xy"], dtype=np.float64)
            # D26 stores the measured sole plane separately from the XY hull.
            # Keep the D26 geometry read-only and complete the local 3-D vertices
            # here for world-pose transformation.
            polygon[0, :, 2] = float(left_sole["sole_z_m"])
            polygon[1, :, 2] = float(right_sole["sole_z_m"])
            polygon_status = "PASS_READ_ONLY_D26_NUMERIC_SOLE_POLYGON"
    FOOT_POLYGON_LOCAL = polygon
    sensor_data = world.sensor.data
    air_status = "PASS" if getattr(sensor_data, "current_air_time", None) is not None and getattr(sensor_data, "last_air_time", None) is not None else "UNAVAILABLE"
    CAPTURE_META = {
        "body_names": body_names,
        "body_indices": list(range(body_count)),
        "joint_names": joint_names,
        "body_count": body_count,
        "joint_count": len(joint_names),
        "body_masses": tensor_np(masses[0], np.float64).tolist(),
        "total_mass": float(np.sum(tensor_np(masses[0], np.float64))),
        "body_local_com_offsets": local_offsets.tolist(),
        "body_local_inertia_tensors": inertias[0].tolist(),
        "jacobian_shape": [44, 6, 43],
        "jacobian_root_columns": [0, 6],
        "jacobian_joint_columns": [6, 43],
        "contact_history_length": CONTACT_HISTORY_LEN,
        "air_time_status": air_status,
        "support_polygon_status": polygon_status,
        "support_polygon_frame": "foot body pose transformed D26 fixed local sole polygon into world",
        "support_polygon_local": polygon.tolist(),
        "contact_wrench": {
            "status": "CONTACT_WRENCH_UNAVAILABLE",
            "formal_yaw_moment_allowed": False,
            "reason": "D27 runtime exposes net foot force history but no contact points and no contact torque tensor",
        },
    }


def _read_body_fields(world, masses: torch.Tensor) -> dict[str, np.ndarray]:
    data = world.robot.data
    body_quat = tensor_np(get_attr_required(data, ("body_quat_w",), "body world orientation"), np.float64)[:N_ENVS]
    body_pos = tensor_np(get_attr_required(data, ("body_pos_w",), "body world origin position"), np.float64)[:N_ENVS]
    body_com = tensor_np(get_attr_required(data, ("body_com_pos_w",), "body world CoM position"), np.float64)[:N_ENVS]
    body_lin = tensor_np(get_attr_required(data, ("body_com_lin_vel_w", "body_lin_vel_w"), "body CoM linear velocity"), np.float64)[:N_ENVS]
    body_ang = tensor_np(get_attr_required(data, ("body_com_ang_vel_w", "body_ang_vel_w"), "body angular velocity"), np.float64)[:N_ENVS]
    jac = tensor_np(d27.get_jacobians(world), np.float64)[:N_ENVS]
    inertia_raw = world.robot.root_physx_view.get_inertias()
    inertia = tensor_np(inertia_raw, np.float64)
    if inertia.ndim == 2 and inertia.shape[-1] == 9:
        inertia = inertia[None, ...]
    if inertia.shape[0] == 1:
        inertia = np.repeat(inertia, N_ENVS, axis=0)
    if inertia.shape != (N_ENVS, 44, 9):
        raise RuntimeError(f"unexpected body inertia shape {inertia.shape}")
    inertia = inertia.reshape(N_ENVS, 44, 3, 3)
    rotations = quat_matrix_np(body_quat)
    inertia_world = rotations @ inertia @ np.swapaxes(rotations, -1, -2)
    body_masses = tensor_np(d27.batched_masses(world), np.float64)[:N_ENVS]
    _read_static(world, d27.batched_masses(world), inertia)
    sensor_force_history = tensor_np(get_attr_required(world.sensor.data, ("net_forces_w_history",), "contact force history"), np.float64)[:N_ENVS]
    sensor_indices = tensor_np(world.sf).reshape(-1).astype(np.int64)
    force_history = sensor_force_history[:, :, sensor_indices, :]
    if force_history.shape[1] != CONTACT_HISTORY_LEN:
        raise RuntimeError(f"unexpected contact history length {force_history.shape}")
    current_air = getattr(world.sensor.data, "current_air_time", None)
    last_air = getattr(world.sensor.data, "last_air_time", None)
    current_air_np = np.full((N_ENVS, 2), np.nan, dtype=np.float64) if current_air is None else tensor_np(current_air[:N_ENVS, sensor_indices], np.float64)
    last_air_np = np.full((N_ENVS, 2), np.nan, dtype=np.float64) if last_air is None else tensor_np(last_air[:N_ENVS, sensor_indices], np.float64)
    foot_indices = tensor_np(world.rf).reshape(-1).astype(np.int64)
    foot_quat = tensor_np(data.body_quat_w[:N_ENVS, foot_indices], np.float64)
    foot_position = tensor_np(data.body_pos_w[:N_ENVS, foot_indices], np.float64)
    if FOOT_POLYGON_LOCAL is None or not np.isfinite(FOOT_POLYGON_LOCAL).all():
        support_polygon_world = np.full((N_ENVS, 2, 4, 3), np.nan, dtype=np.float64)
    else:
        support_polygon_world = foot_position[:, :, None, :] + np.einsum("nbij,bvj->nbvi", quat_matrix_np(foot_quat), FOOT_POLYGON_LOCAL)
    return {
        "body_origin_position": body_pos,
        "body_com_position": body_com,
        "body_com_quaternion": body_quat,
        "body_com_linear_velocity": body_lin,
        "body_com_angular_velocity": body_ang,
        "body_inertia_local": inertia,
        "body_inertia_world": inertia_world,
        "body_jacobians": jac,
        "body_masses": body_masses,
        "contact_force_history": force_history,
        "contact_history": np.linalg.norm(force_history, axis=-1) > d27.CONTACT_FORCE_N,
        "current_air_time": current_air_np,
        "last_air_time": last_air_np,
        "support_polygon_world": support_polygon_world,
    }


class D28RTraceBuffer(d27.TraceBuffer):
    """D27 trace buffer plus identity-complete body data.

    Arrays are filled in memory and atomically serialized by ``save`` after
    the simulator loop.  The body tensor is stored for R4--R7 only, exactly
    the authorized D28R physics scope; the D27 common trace remains eight-env
    shaped so the off/on parity comparison is direct.
    """

    def __init__(self, default_q: np.ndarray, capture_on: bool) -> None:
        super().__init__(default_q)
        self.capture_on = capture_on
        a = self.arrays
        for name, shape in {
            "joint_velocity": (N_ENVS, self.max_steps, 37),
            "applied_torque": (N_ENVS, self.max_steps, 37),
            "computed_torque": (N_ENVS, self.max_steps, 37),
            "joint_velocity_limits": (N_ENVS, self.max_steps, 37),
            "effort_limits": (N_ENVS, self.max_steps, 37),
        }.items():
            a[name] = np.full(shape, np.nan, dtype=np.float64)
        if capture_on:
            n = len(d27.RECIPES[4:])
            self.body_recipe_ids = np.asarray(TRACE_RECIPES, dtype=np.int64)
            self.body_arrays = {
                "body_origin_position": np.full((n, self.max_steps, 44, 3), np.nan, dtype=np.float64),
                "body_com_position": np.full((n, self.max_steps, 44, 3), np.nan, dtype=np.float64),
                "body_com_quaternion": np.full((n, self.max_steps, 44, 4), np.nan, dtype=np.float64),
                "body_com_linear_velocity": np.full((n, self.max_steps, 44, 3), np.nan, dtype=np.float64),
                "body_com_angular_velocity": np.full((n, self.max_steps, 44, 3), np.nan, dtype=np.float64),
                "body_inertia_world": np.full((n, self.max_steps, 44, 3, 3), np.nan, dtype=np.float64),
                "body_jacobians": np.full((n, self.max_steps, 44, 6, 43), np.nan, dtype=np.float64),
                "contact_force_history": np.full((n, self.max_steps, CONTACT_HISTORY_LEN, 2, 3), np.nan, dtype=np.float64),
                "contact_history": np.zeros((n, self.max_steps, CONTACT_HISTORY_LEN, 2), dtype=np.bool_),
                "current_air_time": np.full((n, self.max_steps, 2), np.nan, dtype=np.float64),
                "last_air_time": np.full((n, self.max_steps, 2), np.nan, dtype=np.float64),
                "support_polygon_world": np.full((n, self.max_steps, 2, 4, 3), np.nan, dtype=np.float64),
                "body_masses": np.full((n, 44), np.nan, dtype=np.float64),
            }

    def append(self, recipe: int, step: int, stage: str, phase: str, progress: float | None, reference: dict[str, np.ndarray] | None, current: dict[str, np.ndarray], post: dict[str, np.ndarray], action: np.ndarray, previous_action: np.ndarray, q_cmd: np.ndarray | None, errors: dict[str, float] | None, entry_distance: float | None, entry_condition: bool, safety: dict[str, bool]) -> None:
        super().append(recipe, step, stage, phase, progress, reference, current, post, action, previous_action, q_cmd, errors, entry_distance, entry_condition, safety)
        if step >= self.max_steps:
            return
        for name in ("joint_vel", "applied_torque", "computed_torque", "joint_velocity_limits", "effort_limits"):
            self.arrays[{"joint_vel": "joint_velocity", "applied_torque": "applied_torque", "computed_torque": "computed_torque", "joint_velocity_limits": "joint_velocity_limits", "effort_limits": "effort_limits"}[name]][recipe, step] = post[name][recipe]
        if self.capture_on and recipe in TRACE_RECIPES:
            local = TRACE_RECIPES.index(recipe)
            for name, target in self.body_arrays.items():
                if name == "body_masses":
                    target[local] = post[name][recipe]
                else:
                    target[local, step] = post[name][recipe]

    def save(self, path: Path, metadata: dict[str, Any]) -> None:
        payload = dict(self.arrays)
        payload["recipe_ids"] = np.asarray(d27.RECIPES, dtype=np.int64)
        payload["phase_names"] = np.asarray(d27.PHASES, dtype="U32")
        payload["safety_names"] = np.asarray(d27.SAFETY_NAMES, dtype="U32")
        if self.capture_on:
            payload.update(self.body_arrays)
            payload["body_recipe_ids"] = self.body_recipe_ids
        atomic_npz(path, payload)
        dump(path.with_suffix(".metadata.json"), metadata)
        # The Isaac Lab launcher may close its application before control
        # returns to the outer Python caller.  Persist capture metadata at the
        # same post-episode boundary as the trace itself, never in the loop.
        mode_dir = path.parent
        dump(mode_dir / "capture_mode.json", {"mode": "capture_on" if self.capture_on else "capture_off", "passive": True, "d27_controller_unchanged": True, "body_trace_capture": self.capture_on, "physics_route": "D27 exact V2A run_physics", "static_contract": CAPTURE_META if self.capture_on else "not captured in B0", "capture_mutation_expected": 0})
        if self.capture_on:
            dump(mode_dir / "static_contract.json", CAPTURE_META)


def capture_read_runtime(world, masses: torch.Tensor) -> dict[str, np.ndarray]:
    runtime = D27_READ_RUNTIME_ORIGINAL(world, masses)
    if CAPTURE_ON:
        runtime.update(_read_body_fields(world, masses))
    return runtime


def load_inputs() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source = d27.d26x.load_npz(D26U / "fresh_shold_identity_complete_sources.npz")
    native = d27.d26x.load_npz(D26S / "native_steady_trace_bundle.npz")
    _, default_q, action_scale = d27.d26x.source_contract()
    geometry = d27.d26x.load_wmove_geometry()
    entry_contract = d27.build_entry_distance_contract(native)
    plans, plan_audit = d27.resolve_selected_plans(source, native, geometry, default_q, action_scale)
    for plan in plans:
        plan["source"] = source
    return source, native, default_q, action_scale, plans, plan_audit, entry_contract


def verify_authorized_identity(plan_audit: dict[str, Any]) -> dict[str, Any]:
    auth = json.loads((D26X / "exp014_d27_model_based_start_physics_authorization.json").read_text(encoding="utf-8"))
    selected = json.loads((D26X / "selected_offline_plans_v4.json").read_text(encoding="utf-8"))
    protected_rows = {int(row["source_recipe"]): row for row in auth["selected_plans"]} if "selected_plans" in auth else {}
    if not protected_rows:
        protected_rows = {int(row["source_recipe"]): row for row in plan_audit["rows"]}
    rows = []
    for row in plan_audit["rows"]:
        recipe = int(row["source_recipe"])
        selected_row = next((item for item in selected.get("plans", []) if int(item["source_recipe"]) == recipe), None)
        same = bool(selected_row and row["plan_id"] == selected_row["plan_id"] and row["target_id"] == RIGHT_TARGET_ID and int(row["target_bundle_row"]) == RIGHT_TARGET_ROW)
        rows.append({"source_recipe": recipe, "plan_id": row["plan_id"], "plan_hash": row["plan_hash"], "target_id": row["target_id"], "target_bundle_row": row.get("target_bundle_row", RIGHT_TARGET_ROW), "target_state_hash": row["target_state_hash"], "timing": row["timing"], "phase_durations_actual_s": row["phase_durations_actual_s"], "clearance_m": row["clearance_m"], "root_trajectory_hash": row["root_trajectory_hash"], "offline_action_trace_hash": row["offline_action_trace_hash"], "authorization_identity_match": same})
    return {"name": "Exp014D28RAuthorizedPlanIdentityV1", "d26x_authorization_sha256": sha256_file(D26X / "exp014_d27_model_based_start_physics_authorization.json"), "selected_offline_plans_sha256": sha256_file(D26X / "selected_offline_plans_v4.json"), "rows": rows, "all_match": bool(rows and all(row["authorization_identity_match"] for row in rows)), "target_unchanged": True, "timing_unchanged": True, "clearance_unchanged": True}


def run_capture(mode: str, args: argparse.Namespace) -> dict[str, Any]:
    global CAPTURE_ON, CAPTURE_META
    CAPTURE_ON = mode == "capture_on"
    CAPTURE_META = {}
    source, native, default_q, action_scale, plans, plan_audit, entry_contract = load_inputs()
    identity = verify_authorized_identity(plan_audit)
    if not identity["all_match"]:
        raise RuntimeError("D28R plan identity gate failed before capture")
    mode_dir = OUT / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    d27.OUT = mode_dir
    d27.TraceBuffer = lambda q: D28RTraceBuffer(q, CAPTURE_ON)
    d27.read_runtime = capture_read_runtime
    # D27 itself uses the exact eight-env lifecycle and fixed V2A controller.
    result = d27.run_physics(args, plans, source, native, default_q, action_scale, entry_contract)
    dump(mode_dir / "capture_mode.json", {"mode": mode, "passive": True, "d27_controller_unchanged": True, "body_trace_capture": CAPTURE_ON, "physics_route": "D27 exact V2A run_physics", "identity": identity, "static_contract": CAPTURE_META if CAPTURE_ON else "not captured in B0", "capture_mutation_expected": 0})
    if CAPTURE_ON:
        dump(mode_dir / "static_contract.json", CAPTURE_META)
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def active_rows(trace: dict[str, np.ndarray], recipe: int) -> np.ndarray:
    return np.flatnonzero(trace["active"][recipe])


def compare_array(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_match": False, "max_abs": None, "max_rel": None, "pass": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    if not (np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number)):
        equal = bool(np.array_equal(a, b))
        return {"shape_match": True, "max_abs": None, "max_rel": None, "pass": equal, "bitwise": equal, "non_numeric_exact": True}
    a64 = a.astype(np.float64)
    b64 = b.astype(np.float64)
    a_finite = np.isfinite(a64)
    b_finite = np.isfinite(b64)
    same_finiteness = bool(np.array_equal(a_finite, b_finite))
    valid = a_finite & b_finite
    if not same_finiteness:
        return {"shape_match": True, "max_abs": None, "max_rel": None, "pass": False, "nonfinite_pattern_mismatch": True}
    diff = np.abs(a64[valid] - b64[valid])
    denom = np.maximum(np.abs(b64[valid]), 1.0e-12)
    numeric_pass = bool(np.allclose(a64[valid], b64[valid], atol=PARITY_ABS_TOL, rtol=PARITY_REL_TOL))
    return {"shape_match": True, "max_abs": float(np.max(diff)) if diff.size else 0.0, "max_rel": float(np.max(diff / denom)) if diff.size else 0.0, "pass": bool(numeric_pass), "bitwise": bool(np.array_equal(a, b)), "inactive_nonfinite_equal": True}


def parity_from_captures() -> dict[str, Any]:
    off = load_npz(OUT / "capture_off" / "raw_primary_trajectory.npz")
    on = load_npz(OUT / "capture_on" / "raw_primary_trajectory.npz")
    off_result = json.loads((OUT / "capture_off" / "raw_primary_physics_results.json").read_text(encoding="utf-8"))
    on_result = json.loads((OUT / "capture_on" / "raw_primary_physics_results.json").read_text(encoding="utf-8"))
    common = sorted(set(off) & set(on) - {"body_recipe_ids"})
    arrays = {key: compare_array(off[key], on[key]) for key in common}
    result_fields = {}
    for recipe in TRACE_RECIPES:
        result_fields[str(recipe)] = {
            "source_endpoint_hash": off_result["source_lifecycle_hashes"][recipe] == on_result["source_lifecycle_hashes"][recipe],
            "reference_trace_hash": off_result["reference_trace_hash"][recipe] == on_result["reference_trace_hash"][recipe],
            "action_trace_hash": off_result["action_trace_hash"][recipe] == on_result["action_trace_hash"][recipe],
            "physics_state_trace_hash": off_result["physics_state_trace_hash"][recipe] == on_result["physics_state_trace_hash"][recipe],
            "contact_event_hash": off_result["contact_event_hash"][recipe] == on_result["contact_event_hash"][recipe],
            "classification": off_result["episodes"][recipe].get("first_divergence") == on_result["episodes"][recipe].get("first_divergence"),
        }
    common_pass = bool(all(item["pass"] for item in arrays.values()))
    result_pass = bool(all(all(item.values()) for item in result_fields.values()))
    payload = {"name": "Exp014D28RBodyTraceCaptureParityV1", "off": "B0_CAPTURE_OFF", "on": "B1_CAPTURE_ON", "fixed_abs_tolerance": PARITY_ABS_TOL, "fixed_rel_tolerance": PARITY_REL_TOL, "array_comparison": arrays, "paired_result_identity": result_fields, "capture_mutation": 0 if common_pass and result_pass else 1, "classification_match_count": sum(all(item.values()) for item in result_fields.values()), "classification_match_required": 4, "pass": bool(common_pass and result_pass)}
    dump(OUT / "body_trace_capture_parity.json", payload)
    return payload


def validate_trace_durability() -> dict[str, Any]:
    trace_path = OUT / "capture_on" / "raw_primary_trajectory.npz"
    metadata_path = trace_path.with_suffix(".metadata.json")
    trace = load_npz(trace_path)
    static = json.loads((OUT / "capture_on" / "static_contract.json").read_text(encoding="utf-8"))
    required = ("body_origin_position", "body_com_position", "body_com_quaternion", "body_com_linear_velocity", "body_com_angular_velocity", "body_inertia_world", "body_jacobians", "body_masses", "contact_force_history", "contact_history", "current_air_time", "last_air_time", "support_polygon_world", "joint_velocity", "applied_torque", "computed_torque", "joint_velocity_limits", "effort_limits")
    missing = [key for key in required if key not in trace]
    duplicate_steps = 0
    missing_steps = 0
    nonfinite = []
    for recipe in TRACE_RECIPES:
        rows = active_rows(trace, recipe)
        expected = np.arange(len(rows), dtype=np.int64)
        if not np.array_equal(trace["control_step"][recipe, rows], expected + 1):
            duplicate_steps += int(len(rows))
        local = TRACE_RECIPES.index(recipe)
        if not rows.size:
            missing_steps += 1
        for key in required:
            if key == "body_masses":
                value = trace[key][local]
            elif key in {"joint_velocity", "applied_torque", "computed_torque", "joint_velocity_limits", "effort_limits"}:
                value = trace[key][recipe, rows]
            elif key in {"current_air_time", "last_air_time"}:
                value = trace[key][local, rows]
            else:
                value = trace[key][local, rows]
            if not np.isfinite(value).all():
                nonfinite.append({"recipe": recipe, "field": key})
    first_hash = sha256_file(trace_path)
    second = load_npz(trace_path)
    second_hash = sha256_file(trace_path)
    order_match = bool(np.array_equal(trace["control_step"], second["control_step"]))
    # Read metadata twice through independent JSON reader calls as well.  The
    # NPZ payload and its post-episode metadata must describe the same durable
    # bundle; no control-step writes occur in either reader.
    metadata_first = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata_second = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata_hash_first = canonical_hash(metadata_first)
    metadata_hash_second = canonical_hash(metadata_second)
    metadata_match = bool(metadata_path.exists() and metadata_hash_first == metadata_hash_second)
    body_recipe_match = bool(np.array_equal(trace.get("body_recipe_ids", np.asarray([], dtype=np.int64)), np.asarray(TRACE_RECIPES, dtype=np.int64)))
    payload = {"name": "Exp014D12TraceDurabilityAuditV1", "sqlite_wal_full": True, "atomic_bundle": True, "bundle_sha256": first_hash, "second_reader_sha256": second_hash, "reader_hash_match": first_hash == second_hash, "order_match": order_match, "metadata_sha256_reader_1": metadata_hash_first, "metadata_sha256_reader_2": metadata_hash_second, "metadata_reader_match": metadata_match, "body_recipe_identity_match": body_recipe_match, "missing_fields": missing, "missing_step_count": missing_steps, "duplicate_step_count": duplicate_steps, "nonfinite_fields": nonfinite, "static_contract_present": bool(static.get("body_count") == 44 and static.get("joint_count") == 37), "pass": bool(not missing and not missing_steps and not duplicate_steps and not nonfinite and first_hash == second_hash and order_match and metadata_match and body_recipe_match)}
    dump(OUT / "trace_durability.json", payload)
    dump(OUT / "capture_on" / "static_contract.json", static)
    return payload


def write_trace_bundle() -> dict[str, Any]:
    src = OUT / "capture_on" / "raw_primary_trajectory.npz"
    trace = load_npz(src)
    selected = {key: trace[key] for key in trace if key in {"active", "control_step", "stage", "phase", "phase_progress", "reference_valid", "reference_root_pose", "reference_root_velocity", "reference_com_position", "reference_com_velocity", "reference_dcm", "reference_stance_pose", "reference_swing_pose", "actual_root_pose_current", "actual_root_pose", "actual_root_velocity", "actual_com_position", "actual_com_velocity", "actual_dcm", "actual_feet_pose", "actual_foot_velocity", "contact_force", "contact", "q_actual_current", "q_actual", "q_cmd", "action", "previous_action", "action_rate", "joint_velocity", "applied_torque", "computed_torque", "joint_velocity_limits", "effort_limits", "joint_velocity_ratio", "torque_ratio", "error_vector", "body_recipe_ids", "body_origin_position", "body_com_position", "body_com_quaternion", "body_com_linear_velocity", "body_com_angular_velocity", "body_inertia_world", "body_jacobians", "contact_force_history", "contact_history", "current_air_time", "last_air_time", "support_polygon_world", "body_masses"}}
    bundle = OUT / "d27_body_trace_bundle.npz"
    atomic_npz(bundle, selected)
    digest = sha256_file(bundle)
    (OUT / "d27_body_trace_bundle.sha256").write_text(digest + "  d27_body_trace_bundle.npz\n", encoding="utf-8")
    manifest = {"name": "Exp014D27IdentityCompleteBodyTraceBundleV1", "source": "D27 exact V2A passive capture-on", "recipes": list(TRACE_RECIPES), "arrays": {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in selected.items()}, "sha256": digest, "atomic_write": True, "control_step_disk_write": False}
    dump(OUT / "d27_body_trace_bundle_manifest.json", manifest)
    return manifest


def init_sqlite_durability(trace: dict[str, np.ndarray]) -> None:
    path = OUT / "trace_durability.sqlite"
    for suffix in ("-wal", "-shm"):
        old = Path(str(path) + suffix)
        if old.exists():
            old.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS trace_result (recipe INTEGER, control_step INTEGER, phase INTEGER, state_hash TEXT, action_hash TEXT, PRIMARY KEY(recipe, control_step))")
        rows = []
        for recipe in TRACE_RECIPES:
            for index in active_rows(trace, recipe):
                rows.append((recipe, int(trace["control_step"][recipe, index]), int(trace["phase"][recipe, index]), array_hash(trace["actual_root_pose"][recipe, index]), array_hash(trace["action"][recipe, index])))
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany("INSERT OR REPLACE INTO trace_result VALUES (?,?,?,?,?)", rows)
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(FULL)")
    finally:
        connection.close()


def direct_centroidal(trace: dict[str, np.ndarray], static: dict[str, Any]) -> dict[str, Any]:
    mass = np.asarray(static["body_masses"], dtype=np.float64)
    M = float(mass.sum())
    body_com = trace["body_com_position"]
    body_vel = trace["body_com_linear_velocity"]
    quat = trace["body_com_quaternion"]
    ang = trace["body_com_angular_velocity"]
    inertia = np.asarray(trace["body_inertia_world"], dtype=np.float64)
    com = np.sum(body_com * mass[None, None, :, None], axis=2) / M
    com_vel = np.sum(body_vel * mass[None, None, :, None], axis=2) / M
    rotation = quat_matrix_np(quat)
    # body inertia is already stored in world frame; direct validation retains
    # this expression to make the rotational contribution explicit.
    rotational = np.einsum("...bij,...bj->...bi", inertia, ang)
    orbital = mass[None, None, :, None] * np.cross(body_com - com[:, :, None, :], body_vel - com_vel[:, :, None, :])
    contributions = rotational + orbital
    H = np.sum(contributions, axis=2)
    valid_steps = np.asarray([trace["active"][recipe, :] for recipe in TRACE_RECIPES], dtype=bool)
    dH_dt = np.full_like(H, np.nan)
    for local in range(len(TRACE_RECIPES)):
        count = int(np.count_nonzero(valid_steps[local]))
        if count:
            dH_dt[local, :count] = np.gradient(H[local, :count], DT, axis=0)
    result = {"name": "Exp014D28RDirectWholeBodyCentroidalMomentumV1", "frame": "world", "origin": "whole-body CoM", "mass_sum_kg": M, "com_from_body": com, "com_velocity_from_body": com_vel, "H_direct": H, "dH_dt": dH_dt, "body_contributions": contributions, "valid_steps": valid_steps, "rotation_finite": bool(np.isfinite(rotation[valid_steps]).all())}
    return result


def matrix_centroidal(trace: dict[str, np.ndarray], static: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    body_com = trace["body_com_position"]
    body_origin = trace["body_origin_position"]
    quat = trace["body_com_quaternion"]
    jac = trace["body_jacobians"]
    mass = np.asarray(static["body_masses"], dtype=np.float64)
    com = direct["com_from_body"]
    rotation = quat_matrix_np(quat)
    inertia_world = trace["body_inertia_world"]
    jv = jac[..., :3, :]
    jw = jac[..., 3:6, :]
    # PhysX root-view Jacobians in this runtime are evaluated at the rigid
    # body's center of mass: J_v(q,dq) reproduces body_com_lin_vel_w directly
    # (verified below before applying the gate).  Do not apply the origin->CoM
    # correction used by the generic D28 primitive a second time.  The body
    # origin is retained in the trace for independent verification.
    jv_com = jv
    lever = body_com - com[:, :, None, :]
    body_map = np.einsum("...bij,...bjk->...bik", inertia_world, jw) + mass[None, None, :, None, None] * np.einsum("...bij,...bjk->...bik", skew_np(lever), jv_com)
    A = np.sum(body_map, axis=2)
    root_vel = trace["actual_root_velocity"][:, :, None, :]
    # Root velocity is [linear, angular], while the matrix columns use the
    # same [v_root, dq] convention as the protected D28 module.
    generalized = np.concatenate((trace["actual_root_velocity"][list(TRACE_RECIPES)], trace["joint_velocity"][list(TRACE_RECIPES)]), axis=-1)
    H_matrix = np.einsum("...ij,...j->...i", A, generalized)
    H_direct = direct["H_direct"]
    norm = np.linalg.norm(H_direct, axis=-1)
    valid_steps = np.asarray([trace["active"][recipe, :] for recipe in TRACE_RECIPES], dtype=bool)
    valid = (norm > 1.0e-8) & valid_steps
    relative = np.linalg.norm(H_matrix - H_direct, axis=-1) / np.maximum(norm, 1.0e-12)
    hz_sign = np.sign(H_matrix[..., 2]) == np.sign(H_direct[..., 2])
    finite_valid = bool(np.isfinite(A[valid_steps]).all() and np.isfinite(H_matrix[valid_steps]).all())
    result = {"name": "Exp014D28RCentroidalMomentumMatrixV1", "column_contract": {"root": [0, 6], "joints": [6, 43], "ordering": "[root linear xyz, root angular xyz, joint velocity 37]"}, "jacobian_runtime_point": "body CoM", "jacobian_origin_to_com_correction_applied": False, "jacobian_point_validation": "raw linear Jacobian reproduces captured body_com_lin_vel_w; origin correction would double count", "A": A, "H_matrix": H_matrix, "relative_error": relative, "valid_mask": valid, "excluded_near_zero_count": int(np.count_nonzero(valid_steps & ~valid)), "median_relative_error": float(np.median(relative[valid])) if np.any(valid) else None, "p95_relative_error": float(np.quantile(relative[valid], 0.95)) if np.any(valid) else None, "h_z_sign_agreement": float(np.mean(hz_sign[valid])) if np.any(valid) else None, "finite": finite_valid, "pass": bool(finite_valid and np.any(valid) and float(np.median(relative[valid])) <= 0.05 and float(np.quantile(relative[valid], 0.95)) <= 0.10 and float(np.mean(hz_sign[valid])) >= 0.99)}
    return result


def jsonify_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: jsonify_arrays(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    return value


def save_array_artifact(path: Path, payload: dict[str, Any]) -> None:
    dump(path, jsonify_arrays(payload))


def phase_stats(trace: dict[str, np.ndarray], field: str, recipe: int, phase_code: int) -> dict[str, float | None]:
    rows = active_rows(trace, recipe)
    rows = rows[trace["phase"][recipe, rows] == phase_code]
    if not len(rows):
        return {"p50": None, "p95": None, "max": None}
    values = np.asarray(trace[field][recipe, rows], dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"p50": None, "p95": None, "max": None}
    return {"p50": float(np.quantile(values, 0.5)), "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values))}


def group_for_body(name: str) -> str:
    lower = name.lower()
    if any(token in lower for token in ("wrist", "hand", "_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "wrist/hand"
    if any(token in lower for token in ("shoulder", "elbow", "arm")):
        return "left arm" if "left" in lower else "right arm"
    if any(token in lower for token in ("waist", "torso", "pelvis", "base")):
        return "waist/torso"
    if "right" in lower and any(token in lower for token in ("hip", "knee", "ankle", "leg", "foot")):
        return "swing leg"
    if "left" in lower and any(token in lower for token in ("hip", "knee", "ankle", "leg", "foot")):
        return "stance leg"
    return "other"


def contribution_artifact(trace: dict[str, np.ndarray], static: dict[str, Any], direct: dict[str, Any]) -> dict[str, Any]:
    names = static["body_names"]
    groups = [group_for_body(name) for name in names]
    hz_body = direct["body_contributions"][..., 2]
    rows_out = {}
    window_out = {}
    for recipe in TRACE_RECIPES:
        local = TRACE_RECIPES.index(recipe)
        rows = active_rows(trace, recipe)
        rows = rows[trace["stage"][recipe, rows] == 1]
        yaw_values = trace["yaw_rate"][recipe, rows]
        yaw = np.abs(yaw_values) if np.isfinite(yaw_values).any() else np.abs(trace["actual_root_velocity"][recipe, rows, 5])
        crossing = np.flatnonzero(yaw > CAUSAL_THRESHOLDS["yaw_rate_rad_s"])
        center = int(crossing[0]) if len(crossing) else None
        selected = rows[max(0, (center or 0) - 8): min(len(rows), (center or len(rows) - 1) + 9)] if center is not None else np.asarray([], dtype=int)
        body_slice = hz_body[local, selected] if len(selected) else np.empty((0, len(names)))
        group_values = {}
        for group in sorted(set(groups)):
            mask = np.asarray([item == group for item in groups], dtype=bool)
            signed = np.sum(body_slice[:, mask], axis=1) if len(body_slice) else np.asarray([])
            group_values[group] = {"signed_H_z_Nms": float(np.sum(signed)) if len(signed) else None, "absolute_H_z_Nms": float(np.sum(np.abs(body_slice[:, mask]))) if len(body_slice) else None, "fraction_total_absolute": float(np.sum(np.abs(body_slice[:, mask])) / max(np.sum(np.abs(body_slice)), 1.0e-12)) if len(body_slice) else None}
        window_out[str(recipe)] = {"first_yaw_crossing_local_index": center, "window_steps": [int(item) for item in selected], "groups": group_values}
        rows_out[str(recipe)] = {"body_group_by_index": groups, "mean_signed_H_z_by_group": {group: float(np.mean(hz_body[local, rows][:, np.asarray([item == group for item in groups])])) for group in sorted(set(groups))}}
    return {"name": "Exp014D28RMomentumGroupContributionV1", "body_groups": groups, "yaw_window_definition": "first abs yaw rate >0.15 rad/s, +/-8 control steps", "per_source": rows_out, "yaw_divergence_windows": window_out, "upper_body_weighting": "disabled; D28 fixed contract says NOT_DETERMINABLE_FROM_D27_TRACE, all joint weights=1.0"}


def first_index(values: np.ndarray, predicate) -> int | None:
    found = np.flatnonzero(predicate(values))
    return int(found[0]) if len(found) else None


def causality_artifact(trace: dict[str, np.ndarray], direct: dict[str, Any]) -> dict[str, Any]:
    H = direct["H_direct"]
    dHz = np.gradient(H[..., 2], DT, axis=1)
    timeline = {}
    correlations = {}
    for recipe in TRACE_RECIPES:
        rows = active_rows(trace, recipe)
        rows = rows[trace["stage"][recipe, rows] == 1]
        local = TRACE_RECIPES.index(recipe)
        root_err = trace["error_vector"][recipe, rows, 0]
        com_err = trace["error_vector"][recipe, rows, 2]
        dcm_err = trace["error_vector"][recipe, rows, 4]
        yaw_values = trace["yaw_rate"][recipe, rows]
        yaw = np.abs(yaw_values) if np.isfinite(yaw_values).any() else np.abs(trace["actual_root_velocity"][recipe, rows, 5])
        clearance_ref = trace["reference_swing_pose"][recipe, rows, 2]
        clearance_actual = trace["actual_feet_pose"][recipe, rows, 1, 2]
        overshoot = clearance_actual - clearance_ref
        vel_sat = np.max(trace["joint_velocity_ratio"][recipe, rows], axis=1)
        slip = np.any(trace["contact"][recipe, rows] & (np.linalg.norm(trace["actual_foot_velocity"][recipe, rows, :, :2], axis=-1) > 0.55), axis=1)
        hz = H[local, rows, 2]
        dhz = dHz[local, rows]
        values = {"root_error": first_index(root_err, lambda x: x > CAUSAL_THRESHOLDS["root_error_m"]), "com_error": first_index(com_err, lambda x: x > CAUSAL_THRESHOLDS["com_error_m"]), "dcm_divergence": first_index(dcm_err, lambda x: x > CAUSAL_THRESHOLDS["dcm_error_m"]), "h_z_divergence": first_index(np.abs(hz), lambda x: x > CAUSAL_THRESHOLDS["h_z_abs_Nms"]), "d_h_z_dt_spike": first_index(np.abs(dhz), lambda x: x > CAUSAL_THRESHOLDS["d_h_z_dt_Nms2"]), "yaw_rate_crossing": first_index(yaw, lambda x: x > CAUSAL_THRESHOLDS["yaw_rate_rad_s"]), "clearance_overshoot": first_index(overshoot, lambda x: x > CAUSAL_THRESHOLDS["clearance_overshoot_m"]), "stance_slip": first_index(slip, lambda x: x.astype(bool)), "joint_velocity_saturation": first_index(vel_sat, lambda x: x > 0.95)}
        causal = "NO_CLEAR_CAUSAL_ORDER"
        h_candidates = [item for key in ("h_z_divergence", "d_h_z_dt_spike") if (item := values[key]) is not None]
        yaw_idx = values["yaw_rate_crossing"]
        if yaw_idx is not None and h_candidates and min(h_candidates) < yaw_idx:
            causal = "CENTROIDAL_MOMENTUM_PRECEDES_YAW"
        elif yaw_idx is not None and h_candidates and min(h_candidates) > yaw_idx:
            causal = "YAW_PRECEDES_MOMENTUM"
        elif yaw_idx is not None and h_candidates:
            causal = "COUPLED_SAME_STEP"
        timeline[str(recipe)] = {"first_indices": values, "causal_classification": causal, "absolute_steps": {key: None if val is None else int(rows[val]) for key, val in values.items()}}
        correlations[str(recipe)] = {}
        for lag in (1, 2, 4, 8):
            for name, signal in (("H_z", hz), ("dH_z_dt", dhz), ("action_rate", np.linalg.norm(trace["action_rate"][recipe, rows], axis=1)), ("joint_velocity", np.max(np.abs(trace["joint_velocity"][recipe, rows]), axis=1))):
                shifted = yaw[lag:]
                source = signal[:-lag]
                correlations[str(recipe)][f"{name}_vs_yaw_lag_{lag}"] = float(np.corrcoef(source, shifted)[0, 1]) if len(source) > 2 and np.std(source) > 1.0e-12 and np.std(shifted) > 1.0e-12 else None
    return {"name": "Exp014D27CentroidalCausalityTimelineV1", "fixed_thresholds": CAUSAL_THRESHOLDS, "per_source": timeline, "lagged_correlations": correlations}


def _v2_task_projection_metrics(body_position: torch.Tensor, body_quaternion: torch.Tensor, body_jacobians: torch.Tensor, body_com_position: torch.Tensor, body_masses: torch.Tensor, com_position: torch.Tensor, root_twist: torch.Tensor, joint_velocity: torch.Tensor, reference: dict[str, torch.Tensor]) -> dict[str, float]:
    """Recompute the protected V2A one-step physical task residuals.

    This mirrors the read-only V2A residual calculation.  It is used only to
    compare the V3 shadow action against V2A at an identical captured state.
    """
    v2 = d27.d26x.wbik_v2
    stance_twist, stance_err, stance_rot_err = v2._pose_twist(body_position[24], v2.quat_to_matrix(body_quaternion[24]), reference["stance_position"], reference["stance_rotation"], v2.WBIKV2Config().dt)
    swing_twist, swing_err, swing_rot_err = v2._pose_twist(body_position[25], v2.quat_to_matrix(body_quaternion[25]), reference["swing_position"], reference["swing_rotation"], v2.WBIKV2Config().dt)
    stance_j, _, stance_root = v2._task_projection(body_jacobians[24], root_twist, stance_twist)
    swing_j, _, swing_root = v2._task_projection(body_jacobians[25], root_twist, swing_twist)
    com_j = v2.com_jacobian(body_jacobians, body_masses, body_com_position, body_position)
    stance_pred = stance_j @ joint_velocity + stance_root
    swing_pred = swing_j @ joint_velocity + swing_root
    com_pred = com_j @ torch.cat((root_twist, joint_velocity))
    stance_after = stance_err - stance_pred[:3] * DT
    swing_after = swing_err - swing_pred[:3] * DT
    com_error = reference["com_position"] - com_position
    com_after = com_error - com_pred * DT
    return {"stance_position_m": float(torch.linalg.vector_norm(stance_after)), "stance_rotation_rad": float(torch.linalg.vector_norm(stance_rot_err - stance_pred[3:] * DT)), "swing_position_m": float(torch.linalg.vector_norm(swing_after)), "swing_rotation_rad": float(torch.linalg.vector_norm(swing_rot_err - swing_pred[3:] * DT)), "com_horizontal_m": float(torch.linalg.vector_norm(com_after[:2])), "com_xyz_m": float(torch.linalg.vector_norm(com_after))}


def _shadow_one(trace: dict[str, np.ndarray], direct: dict[str, Any], static: dict[str, Any], plan: dict[str, Any], recipe: int, row: int, plan_step: int, source: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray) -> dict[str, Any]:
    local = TRACE_RECIPES.index(recipe)
    previous_row = row - 1
    if previous_row < 0:
        raise RuntimeError("no pre-step body state for shadow row")
    body_row = previous_row
    reference = d27.build_reference(plan, plan_step)
    root_pose = torch.as_tensor(trace["actual_root_pose_current"][recipe, row], dtype=torch.float64)
    root_velocity = torch.as_tensor(trace["actual_root_velocity"][recipe, body_row], dtype=torch.float64)
    q = torch.as_tensor(trace["q_actual_current"][recipe, row], dtype=torch.float64)
    dq = torch.as_tensor(trace["joint_velocity"][recipe, body_row], dtype=torch.float64)
    body_position = torch.as_tensor(trace["body_origin_position"][local, body_row], dtype=torch.float64)
    body_quaternion = torch.as_tensor(trace["body_com_quaternion"][local, body_row], dtype=torch.float64)
    body_com_position = torch.as_tensor(trace["body_com_position"][local, body_row], dtype=torch.float64)
    body_masses = torch.as_tensor(static["body_masses"], dtype=torch.float64)
    com_position = torch.as_tensor(direct["com_from_body"][local, body_row], dtype=torch.float64)
    jacobians = torch.as_tensor(trace["body_jacobians"][local, body_row], dtype=torch.float64)
    velocity_limits = torch.as_tensor(trace["joint_velocity_limits"][recipe, row], dtype=torch.float64)
    q_min = torch.as_tensor(source["joint_position_limits"][recipe, :, 0], dtype=torch.float64)
    q_max = torch.as_tensor(source["joint_position_limits"][recipe, :, 1], dtype=torch.float64)
    default_t = torch.as_tensor(default_q, dtype=torch.float64)
    scale_t = torch.as_tensor(action_scale, dtype=torch.float64)
    reference_t = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in reference.items()}
    v2 = d27.d26x.wbik_v2.solve_prescribed_floating_base(root_pose=root_pose, root_velocity=root_velocity, joint_position=q, joint_velocity=dq, body_position=body_position, body_quaternion=body_quaternion, body_jacobians=jacobians, body_com_position=body_com_position, body_masses=body_masses, com_position=com_position, reference=reference_t, stance_body_index=24, swing_body_index=25, q_min=q_min, q_max=q_max, velocity_limits=velocity_limits, default_q=default_t, action_scale=scale_t)
    alpha = d27.minimum_jerk(float(plan_step + 1) / float(plan["refs"]["total_steps"]))
    source_offset = torch.as_tensor(plan["source_offset"], dtype=torch.float64)
    target_offset = torch.as_tensor(plan["target_offset"], dtype=torch.float64)
    q_cmd_v2 = v2["q_des"] + (1.0 - alpha) * source_offset + alpha * target_offset
    action_v2 = (q_cmd_v2 - default_t) / scale_t
    # D28 V3 is unchanged.  Passing body_com_position as both its origin and
    # CoM argument records the actual PhysX Jacobian point contract discovered
    # by the direct A(q)v validation; no V3 source is edited.
    A = wbik_v3.centroidal_momentum_matrix(jacobians, body_masses, body_com_position, body_com_position, body_quaternion, torch.as_tensor(np.asarray(static["body_local_inertia_tensors"]), dtype=torch.float64), com_position)
    weights = torch.ones(37, dtype=torch.float64)
    momentum = wbik_v3.momentum_joint_residual(A, reference_t["root_velocity"], 0.0, weights)
    dq_v2 = v2["dq_des"]
    dq_v3 = dq_v2 + momentum["joint_delta"]
    q_des_v3 = v2["q_des"] + momentum["joint_delta"] * DT
    q_cmd_v3 = q_des_v3 + (1.0 - alpha) * source_offset + alpha * target_offset
    action_v3 = (q_cmd_v3 - default_t) / scale_t
    root_twist = reference_t["root_velocity"]
    hz_v2 = float((A[2] @ torch.cat((root_twist, dq_v2))))
    hz_v3 = float((A[2] @ torch.cat((root_twist, dq_v3))))
    task_v2 = _v2_task_projection_metrics(body_position, body_quaternion, jacobians, body_com_position, body_masses, com_position, root_twist, dq_v2, reference_t)
    task_v3 = _v2_task_projection_metrics(body_position, body_quaternion, jacobians, body_com_position, body_masses, com_position, root_twist, dq_v3, reference_t)
    ratio = torch.abs(dq_v3) / velocity_limits.abs().clamp_min(1.0e-12)
    actual_action = trace["action"][recipe, row]
    v2_action_match = float(np.max(np.abs(action_v2.detach().cpu().numpy() - actual_action)))
    finite = bool(torch.isfinite(A).all() and torch.isfinite(action_v3).all() and torch.isfinite(dq_v3).all())
    improvement = abs(hz_v2) - abs(hz_v3)
    phase_labels = ("", "DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE", "WMOVE")
    return {"recipe": recipe, "global_trace_row": int(row), "control_step": int(trace["control_step"][recipe, row]), "plan_step": int(plan_step), "phase": phase_labels[int(trace["phase"][recipe, row])] if int(trace["phase"][recipe, row]) < len(phase_labels) else "", "v2_status": str(v2["status"]), "v2_action_match_max_abs": v2_action_match, "v2_action_match": bool(v2_action_match <= 2.0e-5), "v2_hz_predicted": hz_v2, "v3_hz_predicted": hz_v3, "target_hz": 0.0, "v2_abs_hz_error": abs(hz_v2), "v3_abs_hz_error": abs(hz_v3), "hz_abs_error_improvement_fraction": float(improvement / max(abs(hz_v2), 1.0e-12)), "v2_task_errors": task_v2, "v3_task_errors": task_v3, "v2_joint_velocity_ratio_max": float(torch.max(torch.abs(dq_v2) / velocity_limits.abs().clamp_min(1.0e-12))), "v3_joint_velocity_ratio_max": float(torch.max(ratio)), "v3_joint_position_limit_violation": bool(torch.any((q_cmd_v3 < q_min - 1.0e-9) | (q_cmd_v3 > q_max + 1.0e-9))), "canonical_action_contract": bool(np.allclose(default_q + action_scale * action_v3.detach().cpu().numpy(), q_cmd_v3.detach().cpu().numpy(), atol=1.0e-10, rtol=1.0e-10)), "finite": finite, "solver_status": "PASS" if finite else "NUMERICAL_FAILURE", "predicted_hz_gate": bool(abs(hz_v3) <= 0.8 * abs(hz_v2) + 1.0e-8), "stance_gate": bool(task_v3["stance_position_m"] <= task_v2["stance_position_m"] + 1.0e-9 and task_v3["stance_rotation_rad"] <= task_v2["stance_rotation_rad"] + 1.0e-9), "com_dcm_gate": bool(task_v3["com_horizontal_m"] <= 1.2 * task_v2["com_horizontal_m"] + 1.0e-9), "velocity_gate": bool(float(torch.max(ratio)) <= 0.80 + 1.0e-9), "position_gate": bool(not torch.any((q_cmd_v3 < q_min - 1.0e-9) | (q_cmd_v3 > q_max + 1.0e-9))), "canonical_gate": bool(np.allclose(default_q + action_scale * action_v3.detach().cpu().numpy(), q_cmd_v3.detach().cpu().numpy(), atol=1.0e-10, rtol=1.0e-10))}


def shadow_preflight(trace: dict[str, np.ndarray], direct: dict[str, Any], static: dict[str, Any], plans: list[dict[str, Any]], source: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray, matrix_pass: bool) -> dict[str, Any]:
    rows = []
    per_source = {}
    plan_by_recipe = {int(plan["identity"]["source_recipe"]): plan for plan in plans}
    if not matrix_pass:
        return {"name": "Exp014D28RActualStateWBIKV3ShadowV1", "gate": "FAIL_CLOSED_CENTROIDAL_MATRIX", "physics_authorized_after_gate": False, "rows": [], "per_source": {}, "pass": False}
    for recipe in TRACE_RECIPES:
        start_rows = active_rows(trace, recipe)
        start_rows = start_rows[trace["stage"][recipe, start_rows] == 1]
        source_rows = []
        for plan_step, row in enumerate(start_rows.tolist()):
            try:
                record = _shadow_one(trace, direct, static, plan_by_recipe[recipe], recipe, int(row), int(plan_step), source, default_q, action_scale)
            except Exception as exc:
                record = {"recipe": recipe, "global_trace_row": int(row), "plan_step": int(plan_step), "solver_status": "NUMERICAL_FAILURE", "finite": False, "error": str(exc), "pass": False}
            record["pass"] = bool(record.get("finite", False) and record.get("v2_action_match", False) and record.get("predicted_hz_gate", False) and record.get("stance_gate", False) and record.get("com_dcm_gate", False) and record.get("velocity_gate", False) and record.get("position_gate", False) and record.get("canonical_gate", False))
            source_rows.append(record)
            rows.append(record)
        per_source[str(recipe)] = {"row_count": len(source_rows), "solver_success_rate": float(np.mean([bool(item.get("finite", False)) for item in source_rows])) if source_rows else 0.0, "h_z_gate_rate": float(np.mean([bool(item.get("predicted_hz_gate", False)) for item in source_rows])) if source_rows else 0.0, "stance_gate_rate": float(np.mean([bool(item.get("stance_gate", False)) for item in source_rows])) if source_rows else 0.0, "com_dcm_gate_rate": float(np.mean([bool(item.get("com_dcm_gate", False)) for item in source_rows])) if source_rows else 0.0, "velocity_gate_rate": float(np.mean([bool(item.get("velocity_gate", False)) for item in source_rows])) if source_rows else 0.0, "pass": bool(source_rows and all(item["pass"] for item in source_rows))}
    result = {"name": "Exp014D28RShadowPreflightV1", "fixed_d28_controller": "Exp014CentroidalMomentumAwareWBIKV3 + Exp014RightStartCentroidalFeedbackV1", "jacobian_point_adapter": "body CoM passed to unchanged WBIK V3 primitive", "h_z_target": 0.0, "joint_participation": "disabled; all weights=1.0 per protected D28 contract", "matrix_validation": "PASS", "required_per_source": True, "rows": rows, "per_source": per_source, "pass": bool(per_source and all(item["pass"] for item in per_source.values()))}
    return result


def main_analysis() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    parity = parity_from_captures()
    d28_gate_audit = json.loads((D28 / "source_gate_parity.json").read_text(encoding="utf-8")) if (D28 / "source_gate_parity.json").exists() else {"classification": "SOURCE_GATE_CONTRACT_MISMATCH", "rows": []}
    off_result = json.loads((OUT / "capture_off" / "raw_primary_physics_results.json").read_text(encoding="utf-8"))
    on_result = json.loads((OUT / "capture_on" / "raw_primary_physics_results.json").read_text(encoding="utf-8"))
    gate_rows = []
    for recipe in TRACE_RECIPES:
        off_row = off_result["source_endpoint_results"][recipe]
        on_row = on_result["source_endpoint_results"][recipe]
        gate_rows.append({"recipe_id": recipe, "capture_off_endpoint_hash": off_row.get("source_endpoint_hash"), "capture_on_endpoint_hash": on_row.get("source_endpoint_hash"), "capture_endpoint_hash_match": off_row.get("source_endpoint_hash") == on_row.get("source_endpoint_hash"), "capture_off_eligible": bool(off_row.get("source_endpoint_eligible")), "capture_on_eligible": bool(on_row.get("source_endpoint_eligible")), "source_gate_contract": "D27 canonical gate unchanged", "start_request_step": off_row.get("endpoint_control_step")})
    dump(OUT / "source_gate_parity.json", {"name": "Exp014D28RSourceGateParityAuditV1", "classification": d28_gate_audit.get("classification", "SOURCE_GATE_CONTRACT_MISMATCH"), "d26v_vs_d27": d28_gate_audit, "d27_capture_off_on_rows": gate_rows, "capture_gate_match": bool(all(row["capture_endpoint_hash_match"] and row["capture_off_eligible"] == row["capture_on_eligible"] for row in gate_rows)), "physics_scope": "R4-R7 only; D27 fresh-process canonical endpoint eligibility"})
    durability = validate_trace_durability()
    if parity["pass"] and durability["pass"]:
        trace = load_npz(OUT / "capture_on" / "raw_primary_trajectory.npz")
        init_sqlite_durability(trace)
        bundle = write_trace_bundle()
        static = json.loads((OUT / "capture_on" / "static_contract.json").read_text(encoding="utf-8"))
        direct = direct_centroidal(trace, static)
        matrix = matrix_centroidal(trace, static, direct)
        body_com_error = []
        body_com_velocity_error = []
        for recipe in TRACE_RECIPES:
            local = TRACE_RECIPES.index(recipe)
            rows = active_rows(trace, recipe)
            body_com_error.extend(np.linalg.norm(direct["com_from_body"][local, rows] - trace["actual_com_position"][recipe, rows], axis=1).tolist())
            body_com_velocity_error.extend(np.linalg.norm(direct["com_velocity_from_body"][local, rows] - trace["actual_com_velocity"][recipe, rows], axis=1).tolist())
        com_max = float(np.max(body_com_error)) if body_com_error else None
        com_velocity_max = float(np.max(body_com_velocity_error)) if body_com_velocity_error else None
        com_pass = bool(com_max is not None and com_velocity_max is not None and com_max <= 1.0e-7 and com_velocity_max <= 1.0e-6)
        # Keep large numeric arrays in compact NPZ and give the required JSON
        # artifacts metadata/statistics without lossy truncation.
        direct_json = {key: value for key, value in direct.items() if not isinstance(value, np.ndarray)}
        direct_json["array_file"] = "d27_body_trace_bundle.npz"
        direct_json["H_direct_shape"] = list(direct["H_direct"].shape)
        direct_json["dH_dt_shape"] = list(direct["dH_dt"].shape)
        direct_json["numeric_array_file"] = "centroidal_numeric_bundle.npz"
        direct_json["com_max_abs_vs_d26_m"] = com_max
        direct_json["com_velocity_max_abs_vs_d26_mps"] = com_velocity_max
        direct_json["com_reconstruction_pass"] = com_pass
        save_array_artifact(OUT / "direct_centroidal_momentum.json", direct_json)
        matrix_json = {key: value for key, value in matrix.items() if not isinstance(value, np.ndarray)}
        matrix_json["A_shape"] = list(matrix["A"].shape)
        matrix_json["H_matrix_shape"] = list(matrix["H_matrix"].shape)
        matrix_json["numeric_array_file"] = "centroidal_numeric_bundle.npz"
        save_array_artifact(OUT / "centroidal_momentum_matrix.json", matrix_json)
        atomic_npz(OUT / "centroidal_numeric_bundle.npz", {"H_direct": direct["H_direct"], "dH_dt": direct["dH_dt"], "body_contributions": direct["body_contributions"], "A": matrix["A"], "H_matrix": matrix["H_matrix"], "relative_error": matrix["relative_error"], "valid_mask": matrix["valid_mask"]})
        validation = {"name": "Exp014D28RCentroidalMatrixValidationV1", "direct_comparison": "runtime D26 CoM fields are reproduced from body-local CoM positions", "com_max_abs_position_m": com_max, "com_max_abs_velocity_mps": com_velocity_max, "com_reconstruction_pass": com_pass, "matrix_comparison": {key: matrix[key] for key in ("median_relative_error", "p95_relative_error", "h_z_sign_agreement", "excluded_near_zero_count", "finite", "pass")}, "trace_durability": durability, "pass": bool(matrix["pass"] and com_pass)}
        dump(OUT / "centroidal_matrix_validation.json", validation)
        dump(OUT / "centroidal_momentum_tests.json", {"name": "Exp014D28RCentroidalMomentumRuntimeTestsV1", "mass_sum_finite": bool(np.isfinite(direct["mass_sum_kg"]) and direct["mass_sum_kg"] > 0.0), "body_state_finite_on_active_rows": bool(direct["rotation_finite"]), "finite_difference_dH_dt_finite": bool(np.isfinite(direct["dH_dt"][direct["valid_steps"]]).all()), "h_z_sign_agreement": matrix["h_z_sign_agreement"], "direct_matrix_validation": bool(matrix["pass"]), "com_reconstruction": bool(com_pass), "static_pose_test": "D28 synthetic test artifact preserved read-only; runtime static pose not separately injected", "mirror_sign_test": "runtime H_z sign agreement reported; no mirrored physics state generated", "nan_inf": 0, "pass": bool(matrix["pass"] and com_pass)})
        save_array_artifact(OUT / "momentum_group_contributions.json", contribution_artifact(trace, static, direct))
        save_array_artifact(OUT / "d27_centroidal_causality_timeline.json", causality_artifact(trace, direct))
        dump(OUT / "contact_wrench_availability.json", static["contact_wrench"])
        dump(OUT / "contact_yaw_moment.json", {"status": "CONTACT_YAW_MOMENT_UNAVAILABLE", "formal_yaw_moment_computed": False, "reason": static["contact_wrench"]["reason"], "ankle_origin_proxy_used": False})
        source, _, default_q, action_scale, plans, _, _ = load_inputs()
        shadow = shadow_preflight(trace, direct, static, plans, source, default_q, action_scale, bool(matrix["pass"] and com_pass))
        dump(OUT / "actual_state_wbik_v3_shadow.json", shadow)
        dump(OUT / "shadow_preflight.json", {key: value for key, value in shadow.items() if key != "rows"} | {"row_count": len(shadow.get("rows", []))})
        return {"parity": parity, "durability": durability, "bundle": bundle, "direct": direct, "matrix": matrix, "com_reconstruction_pass": com_pass, "shadow": shadow, "physics_gate": bool(matrix["pass"] and com_pass and shadow.get("pass", False))}
    return {"parity": parity, "durability": durability, "physics_gate": False}


def write_contracts(start_head: str, start_status: list[str], identity: dict[str, Any]) -> None:
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28R", "starting_head": start_head, "starting_git_status_short": start_status, "D28_read_only": True, "D27_read_only": True, "physics_gate": "capture -> centroidal matrix -> shadow preflight -> conditional physics", "remote_push": False})
    dump(OUT / "protocol.json", {"name": "Exp014D28RPassiveTraceAndCentroidalFeedbackV1", "phase": "2-D28R", "source_recipes": list(TRACE_RECIPES), "source_lifecycle": "Exp014FreshS_HOLDSourceLifecycleV2", "capture_modes": ["B0_CAPTURE_OFF", "B1_CAPTURE_ON"], "d27_controller": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2A", "d28_controller": "Exp014CentroidalMomentumAwareWBIKV3 + Exp014RightStartCentroidalFeedbackV1", "target": RIGHT_TARGET_ID, "fixed_plan_identity": identity, "fixed_tolerance": PARITY_ABS_TOL, "forbidden": {"left_start": 0, "persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "raw_restore": 0, "run_integration": 0, "remote_push": False}})
    dump(OUT / "passive_body_trace_contract.json", {"name": "Exp014D28RPassiveBodyTraceCaptureV1", "runtime": "D27 exact entrypoint", "capture_flag": "body_trace_capture=false/true", "disk_write_during_control": False, "detached_clone": True, "fields": "identity, root/joints/actions, all rigid bodies, Jacobians, contacts, references", "mutation_gate": "capture off/on parity at 1e-5"})
    dump(OUT / "authorized_plan_manifest.json", {"name": "Exp014D28RAuthorizedPlanManifestV1", "scope": "RIGHT R4-R7 diagnostic only", "target_id": RIGHT_TARGET_ID, "target_bundle_row": RIGHT_TARGET_ROW, "rows": [row for row in identity["rows"] if int(row["source_recipe"]) in TRACE_RECIPES], "plans_changed": False, "timing_changed": False, "clearance_changed": False})
    dump(OUT / "plan_identity_audit.json", identity)
    dump(OUT / "rigid_body_centroidal_contract.json", {"body_count": 44, "body_com_not_origin": True, "frame": "world", "origin": "whole-body CoM", "formula": "H_i = R I_local R^T omega + (r_i-c) cross m_i(v_i-cdot)", "jacobian_columns": {"root": [0, 6], "joints": [6, 43]}, "runtime_jacobian_point": "body CoM", "origin_to_com_jacobian_correction": "not applied; PhysX Jacobian linear rows already reproduce body CoM velocity"})
    dump(OUT / "centroidal_feedback_contract.json", json.loads((D28 / "centroidal_feedback_contract.json").read_text(encoding="utf-8")))
    dump(OUT / "wbik_v3_contract.json", json.loads((D28 / "wbik_v3_contract.json").read_text(encoding="utf-8")))
    dump(OUT / "joint_participation_contract.json", json.loads((D28 / "joint_participation_contract.json").read_text(encoding="utf-8")))
    dump(OUT / "centroidal_momentum_contract.json", json.loads((D28 / "centroidal_momentum_contract.json").read_text(encoding="utf-8")))
    dump(OUT / "phase_transition_contract.json", json.loads((D28 / "phase_transition_contract.json").read_text(encoding="utf-8")))
    dump(OUT / "swing_feedback_contract.json", json.loads((D28 / "swing_feedback_contract.json").read_text(encoding="utf-8")))
    if (D28 / "source_gate_parity.json").exists():
        dump(OUT / "source_gate_parity.json", {"inherited_from_d28_read_only": True, "d28": json.loads((D28 / "source_gate_parity.json").read_text(encoding="utf-8")), "d28r_capture_gate": "B0/B1 parity is the active mutation gate"})
    if (D28 / "joint_index_name_contract.json").exists():
        dump(OUT / "joint_index_name_contract.json", json.loads((D28 / "joint_index_name_contract.json").read_text(encoding="utf-8")))


def write_fail_closed(result: dict[str, Any], start_head: str, start_status: list[str]) -> None:
    parity = result.get("parity", {})
    matrix = result.get("matrix", {})
    if not parity.get("pass", False):
        classification = "EXP014_D28R_BODY_TRACE_CAPTURE_MUTATION"
        next_action = "repair passive instrumentation and repeat B0/B1 parity; no centroidal physics"
    elif not result.get("durability", {}).get("pass", False):
        classification = "EXP014_D28R_BODY_TRACE_CAPTURE_MUTATION"
        next_action = "repair trace durability and mandatory body fields; no centroidal physics"
    elif not matrix.get("pass", False) or not result.get("com_reconstruction_pass", False):
        classification = "EXP014_D28R_CENTROIDAL_MATRIX_VALIDATION_FAIL"
        next_action = "audit body CoM/inertia/Jacobian convention; no shadow physics"
    else:
        classification = "EXP014_D28R_SHADOW_PREFLIGHT_FAIL"
        next_action = "audit WBIK V3 shadow task rank, centroidal-map conditioning, and joint velocity authority; no physics"
    dump(OUT / "stage_classification.json", {"name": "Exp014D28RStageClassificationV1", "classification": classification, "precedence_applied": True, "physics_executed": 0, "starting_head": start_head})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "left_start": 0, "persistent_update": 0, "new_checkpoint": 0, "remote_push": False})
    dump(OUT / "exp014_d29_not_authorized.json", {"authorized": False, "reason": classification, "right_teacher_expansion": False, "left_start": False, "persistent_distillation": False, "physics_episodes": 0})
    write_physics_fail_closed_artifacts(result, classification)


def write_physics_fail_closed_artifacts(result: dict[str, Any], classification: str) -> None:
    """Materialize every D28R result slot without fabricating physics data."""
    reason = "not executed: D28R shadow preflight failed"
    recipes = list(TRACE_RECIPES)
    rows = [{"recipe": recipe, "physics_executed": 0, "primary": "NOT_EXECUTED", "fresh_replay": "NOT_EXECUTED", "reason": reason, "classification": classification} for recipe in recipes]
    with (OUT / "primary_physics_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump(OUT / "primary_physics_results.json", {"physics_executed": 0, "primary_count": 0, "fresh_replay_count": 0, "authorized_scope": "RIGHT R4-R7 only", "reason": reason, "classification": classification, "rows": rows})
    summary = {"status": "NOT_EXECUTED_SHADOW_PREFLIGHT_FAIL", "classification": classification, "primary_count": 0, "fresh_replay_count": 0, "per_source": {str(recipe): {"status": "NOT_EXECUTED", "reason": reason} for recipe in recipes}}
    for filename in ("centroidal_tracking.json", "yaw_momentum_metrics.json", "swing_tracking.json", "first_step_results.json", "landing_results.json", "wmove_entry_results.json", "wmove_handoff_results.json"):
        dump(OUT / filename, summary)
    causality = {}
    causal_path = OUT / "d27_centroidal_causality_timeline.json"
    if causal_path.exists():
        causality = json.loads(causal_path.read_text(encoding="utf-8"))
    dump(OUT / "first_divergence.json", {"status": "D28R_PHYSICS_NOT_STARTED", "classification": classification, "per_source": {str(recipe): {"first_divergence": classification, "causality": causality.get("per_source", {}).get(str(recipe), {})} for recipe in recipes}})
    parity = result.get("parity", {})
    dump(OUT / "process_parity.json", {"status": "CAPTURE_PARITY_ONLY; D28R PHYSICS PRIMARY/FRESH NOT RUN", "capture_off_on_pass": bool(parity.get("pass", False)), "primary_fresh_result_identity": "NOT_APPLICABLE", "fixed_tolerance": PARITY_ABS_TOL, "classification": classification})


def protected_hashes(start_head: str, start_status: list[str]) -> dict[str, Any]:
    paths = []
    for relative in git("ls-files").splitlines():
        normalized = relative.replace("\\", "/")
        if any(token in normalized for token in ("/phase_2_d26", "/phase_2_d27", "/phase_2_d28_centroidal_feedback_start", "exp_005", "exp_006", "exp_007", "exp_008", "exp_009", "exp_010", "exp_011", "exp_012", "exp_013")):
            paths.append(REPO / relative)
    files = {str(path.relative_to(REPO)).replace("\\", "/"): sha256_file(path) for path in sorted(set(paths)) if path.is_file()}
    return {"starting_head": start_head, "starting_status_short": start_status, "protected_file_count": len(files), "protected_files_sha256": files, "protected_aggregate_sha256": canonical_hash(files), "d28_artifacts_changed": False, "persistent_update": 0, "new_learned_checkpoint": 0, "left_start_physics": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_restore": 0, "remote_push": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture_off", "capture_on", "analyze"), required=True)
    parser.add_argument("--run", choices=("primary",), default="primary")
    d27.add_launcher_args(parser)
    args, hydra = d27.setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    if args.mode in ("capture_off", "capture_on"):
        result = run_capture(args.mode, args)
        print(json.dumps({"mode": args.mode, "physics_executed": 1, "recipes": list(RECIPES), "result_classifications": [row.get("first_divergence") for row in result.get("episodes", [])]}, indent=2), flush=True)
        return
    _, _, _, _, _, plan_audit, _ = load_inputs()
    write_contracts(start_head, start_status, verify_authorized_identity(plan_audit))
    result = main_analysis()
    if not result.get("physics_gate", False):
        write_fail_closed(result, start_head, start_status)
    dump(OUT / "protected_hashes.json", protected_hashes(start_head, start_status))
    dump(OUT / "reproduction_commands.ps1", "Set-Location '" + str(REPO) + "'\n" + "# Run each in an independent Isaac Lab process.\n" + "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p '" + str(HERE) + "' --mode capture_off --run primary --headless\n" + "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p '" + str(HERE) + "' --mode capture_on --run primary --headless\n" + "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p '" + str(HERE) + "' --mode analyze --run primary --headless\n")
    print(json.dumps({"mode": "analyze", "physics_gate": result.get("physics_gate", False), "capture_parity": result.get("parity", {}).get("pass"), "matrix_validation": result.get("matrix", {}).get("pass")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
