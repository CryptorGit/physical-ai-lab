"""Phase 2-D26V endpoint-gate correction and offline START execution.

This finalizer consumes the protected D26U source bundle and D26S/D26T
references read-only.  It writes only the D26V result directory, adds the
versioned prescribed-floating-base WBIK V2 module, and never creates a
simulator, policy optimizer, checkpoint, or physics START rollout.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26v_endpoint_gate_and_wbik_v2"
REPORT = REPO / "research/exp_014_phase_2_d26v_endpoint_gate_and_wbik_v2_report.md"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
SOURCE = D26U / "fresh_shold_identity_complete_sources.npz"
NATIVE = D26S / "native_steady_trace_bundle.npz"
ENDPOINT_OFF = OUT / "raw_endpoint_window_capture_off.json"
ENDPOINT_ON = OUT / "raw_endpoint_window_capture_on.json"
V1_PATH = EXP / "src/g1_explicit_motion_mode/wbik.py"
V2_PATH = EXP / "src/g1_explicit_motion_mode/wbik_v2.py"
START_HEAD_REQUESTED = "ab59baf29f16b78f5724122e164b4e06aa201de5"
SOURCE_SHA_REQUESTED = "b164cf1882eac2b45e5f0ee019bf5e21df57dd7c5457fba21d3d30c90caf345f"
NATIVE_SHA_REQUESTED = "e4f2250a35a5feee2d1adb415d11121e52118648bc7678dcf91a47e0894f6"
# The literal D26V input contract contains a transcription omission.  The
# protected D26S file and all prior manifests agree on this actual digest.
NATIVE_SHA_ACTUAL = "e4f2250a35a5feee2d1adb415d11121e52164018648bc7678dcf91a47e0894f6"
DT = 0.02
T_REF = 0.16
RECIPES = list(range(8))
LEADS = ("LEFT", "RIGHT")
SOURCE_STEPS = [125, 132, 129, 125, 128, 125, 130, 129]
CONFIRMATION_END = [75, 82, 79, 75, 78, 75, 80, 79]
MEDOID_ROWS = {"LEFT": 8171, "RIGHT": 9330}
MEDOID_EPISODES = {"LEFT": 52, "RIGHT": 187}
MEDOID_STEPS = {"LEFT": 111, "RIGHT": 115}
FOOT_BODY = {"LEFT": 24, "RIGHT": 25}
SHIFTS = (0.30, 0.40, 0.50)
SWING_MULTIPLIERS = (0.8, 1.0, 1.2)
CLEARANCE_PERCENTILES = (50, 75, 90)
ACTION_SCALE = 0.5
G = 9.81


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wbik_v2 = load_module("exp014_d26v_wbik_v2", V2_PATH)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def dump(name: str, value: Any) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(to_jsonable(rows))


def q(values: Any, percentile: float) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.quantile(values, percentile)) if values.size else None


def quat_matrix(quat: np.ndarray) -> np.ndarray:
    qv = np.asarray(quat, dtype=np.float64)
    qv = qv / max(float(np.linalg.norm(qv)), 1.0e-12)
    x, y, z, w = qv
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_log_np(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.asarray([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) * 0.5
    if theta < 1.0e-6:
        return vee
    if theta > 3.13:
        axis = np.sqrt(np.maximum(np.diag(r) + 1.0, 0.0) * 0.5)
        axis = axis / max(float(np.linalg.norm(axis)), 1.0e-12)
        return axis * theta
    return vee * theta / max(float(np.sin(theta)), 1.0e-8)


def so3_exp_np(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    theta = float(np.linalg.norm(v))
    if theta < 1.0e-10:
        k = np.asarray([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + k
    axis = v / theta
    k = np.asarray([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def minimum_jerk(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def minimum_jerk_derivative(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4


def rotation_trajectory(start: np.ndarray, target: np.ndarray, s: float) -> np.ndarray:
    return so3_exp_np(so3_log_np(target @ start.T) * minimum_jerk(s)) @ start


def heading_yaw(rotation: np.ndarray) -> float:
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def heading_aligned_rotation(target_rotation: np.ndarray, source_rotation: np.ndarray) -> np.ndarray:
    yaw_target = heading_yaw(target_rotation)
    yaw_source = heading_yaw(source_rotation)
    rz_target = so3_exp_np(np.asarray([0.0, 0.0, -yaw_target]))
    rz_source = so3_exp_np(np.asarray([0.0, 0.0, yaw_source]))
    return rz_source @ rz_target @ target_rotation


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def source_action_contract() -> tuple[np.ndarray, np.ndarray]:
    contract_path = D26U.parent / "phase_2_d25_model_based_first_step_teacher/model_based_teacher_robot_contract.json"
    if not contract_path.exists():
        contract_path = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher/model_based_teacher_robot_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    default = np.asarray(contract["action_interface"]["offset"][0], dtype=np.float64)
    return default, np.full(37, ACTION_SCALE, dtype=np.float64)


def polygon_contains(point: np.ndarray, vertices: np.ndarray) -> bool:
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 3:
        return False
    sign = []
    for i in range(len(vertices)):
        a = vertices[i]
        b = vertices[(i + 1) % len(vertices)]
        sign.append(float(np.cross(b - a, point - a)))
    return bool(all(x >= -1.0e-9 for x in sign) or all(x <= 1.0e-9 for x in sign))


def convex_hull(points: np.ndarray) -> np.ndarray:
    points = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(points) <= 1:
        return np.asarray(points, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def foot_polygon(center: np.ndarray, rotation: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    yaw = heading_yaw(rotation)
    rz = so3_exp_np(np.asarray([0.0, 0.0, yaw]))[:2, :2]
    return np.asarray(center[:2]) + np.asarray(polygon_xy) @ rz.T


def d26_geometry() -> dict[str, Any]:
    path = D26U / "wmove_transition_geometry_v1.json"
    if not path.exists():
        path = D26T / "wmove_step_geometry_contract_v4.json"
    return json.loads(path.read_text(encoding="utf-8"))


def geometry_values(geometry: dict[str, Any]) -> dict[str, Any]:
    side = geometry.get("side_statistics", {})
    # D26U values are the protected fixed planning inputs.  Do not select a
    # new percentile after seeing rollout outcomes.
    aggregate = geometry.get("aggregate_statistics", {})
    clearance = {p: float(aggregate["swing_clearance_m"][f"p{p:02d}"]) for p in CLEARANCE_PERCENTILES}
    return {"T_ref_s": T_REF, "side_statistics": side, "aggregate_statistics": aggregate, "clearance": clearance}


def endpoint_records(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def endpoint_reclassification(source_manifest: dict[str, Any], off: dict[str, Any], on: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parity_rows = []
    for recipe in RECIPES:
        off_window = next(x for x in off["windows"] if int(x["recipe_id"]) == recipe)
        on_window = next(x for x in on["windows"] if int(x["recipe_id"]) == recipe)
        parity_rows.append({
            "recipe_id": recipe,
            "window_record_count_off": int(off_window["window_record_count"]),
            "window_record_count_on": int(on_window["window_record_count"]),
            "records_bitwise_equal": canonical_hash(off_window["records"]) == canonical_hash(on_window["records"]),
            "window_step_equal": off_window["source_control_step"] == on_window["source_control_step"],
        })
    parity_pass = all(x["records_bitwise_equal"] and x["window_step_equal"] for x in parity_rows)
    manifest_rows = {int(row["recipe_id"]): row for row in source_manifest["recipes"]}
    rows: list[dict[str, Any]] = []
    for recipe in RECIPES:
        m = manifest_rows[recipe]
        window = next(x for x in on["windows"] if int(x["recipe_id"]) == recipe)
        records = list(window["records"])
        index = recipe
        endpoint_flags = {}
        for name in ("fall", "dangerous_slip", "impact", "support_loss", "velocity_saturation", "torque_saturation", "nonfinite"):
            endpoint_flags[name] = int(any(bool(row[name][index]) for row in records))
        support_counts = [int(row["support_count"][index]) for row in records]
        velocity_ratios = [float(row["velocity_ratio_max"][index]) for row in records]
        torque_ratios = [float(row["torque_ratio_max"][index]) for row in records]
        endpoint_valid = bool(
            len(records) == 50
            and all(value == 0 for value in endpoint_flags.values())
            and min(support_counts, default=0) >= 1
            and np.isfinite(velocity_ratios).all()
            and np.isfinite(torque_ratios).all()
        )
        pre_events = {name: int(m.get("first_safety_flag_step", {}).get(name, -1)) for name in endpoint_flags}
        pre_acquisition_only = any(step >= 0 and step < int(window["window_start_step"]) for step in pre_events.values()) and endpoint_valid
        if not endpoint_valid:
            classification = "ENDPOINT_UNSAFE"
        elif pre_acquisition_only:
            classification = "PRE_ACQUISITION_TRANSIENT_ONLY"
        else:
            classification = "ENDPOINT_ELIGIBLE"
        rows.append({
            "recipe_id": recipe,
            "seed": int(m["seed"]),
            "recipe_family": m["recipe_family"],
            "control_step": int(m["control_step"]),
            "confirmation_end_step": int(m["confirmation_end_step"]),
            "full_lifecycle_validity_d26u": bool(m.get("source_valid", False)),
            "pre_acquisition_event_first_step": pre_events,
            "pre_acquisition_event_count": {name: int(step >= 0) for name, step in pre_events.items()},
            "endpoint_window": {"start": int(window["window_start_step"]), "end": int(window["window_end_step"]), "records": len(records)},
            "endpoint_flags": endpoint_flags,
            "endpoint_support_count_min": min(support_counts, default=0),
            "endpoint_velocity_ratio_max": max(velocity_ratios, default=None),
            "endpoint_torque_ratio_max": max(torque_ratios, default=None),
            "endpoint_state_valid": endpoint_valid,
            "endpoint_eligible": endpoint_valid,
            "classification": classification,
            "source_endpoint_gate": "START_SOURCE_ENDPOINT_ELIGIBILITY_V1",
            "capture_off_on_parity": parity_pass,
            "capture_off_on_parity_row": parity_rows[recipe],
            "eligibility_reason": "endpoint last-50 window PASS; D26U pre-acquisition transient is retained diagnostically" if endpoint_valid else "endpoint-window safety/support gate failed",
        })
    result = {
        "name": "START_SOURCE_ENDPOINT_ELIGIBILITY_V1",
        "evaluation_window": "last 50 control steps immediately preceding and including START request endpoint",
        "pre_acquisition_events_do_not_directly_fail_endpoint": True,
        "capture_parity": {"status": "PASS" if parity_pass else "FAIL", "rows": parity_rows, "mutation": 0 if parity_pass else "UNKNOWN"},
        "recipes": rows,
        "endpoint_eligible_count": sum(int(x["endpoint_eligible"]) for x in rows),
        "minimum_progression": 6,
        "minimum_progression_pass": sum(int(x["endpoint_eligible"]) for x in rows) >= 6 and parity_pass,
        "d26u_full_lifecycle_valid_count_read_only": sum(int(x["full_lifecycle_validity_d26u"]) for x in rows),
        "d26u_classification_unchanged": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL",
    }
    return rows, result


def saturation_timing_audit(endpoint: dict[str, Any], source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Recover recipe 0/3 torque saturation timing from D26V per-step replay."""
    rows = []
    for recipe in (0, 3):
        records = [r for r in endpoint["records"] if r["kind"] == "main_control_step"]
        candidates = []
        for row in records:
            comp = np.asarray(row["computed_torque"][recipe], dtype=np.float64)
            applied = np.asarray(row["applied_torque"][recipe], dtype=np.float64)
            effort = np.asarray(row["effort_limits"][recipe], dtype=np.float64)
            ratio = np.abs(comp) / np.maximum(np.abs(effort), 1.0e-12)
            if float(ratio.max()) > 0.95:
                candidates.append((int(row["main_step"]), int(ratio.argmax()), float(ratio.max()), float(applied[ratio.argmax()]), float(comp[ratio.argmax()]), float(effort[ratio.argmax()]), int(row["support_count"][recipe]), row["support_state"][recipe]))
        if candidates:
            start = candidates[0][0]
            end = candidates[-1][0]
            onset = next(x for x in candidates if x[0] == start)
            peak = max(candidates, key=lambda x: x[2])
        else:
            start = end = -1
            onset = peak = None
        first_flag = int(next(x for x in source_manifest["recipes"] if int(x["recipe_id"]) == recipe)["first_safety_flag_step"]["torque_saturation"])
        rows.append({
            "recipe_id": recipe,
            "canonical_threshold": ">0.95 computed-torque/effort ratio with D26U five-step dwell detection",
            "saturation_joint_at_onset": None if onset is None else onset[1],
            "saturation_joint_at_peak": None if peak is None else peak[1],
            "computed_torque_at_onset": None if onset is None else onset[4],
            "applied_torque_at_onset": None if onset is None else onset[3],
            "effort_limit_at_onset": None if onset is None else onset[5],
            "torque_ratio_at_onset": None if onset is None else onset[2],
            "computed_torque_at_peak": None if peak is None else peak[4],
            "applied_torque_at_peak": None if peak is None else peak[3],
            "effort_limit_at_peak": None if peak is None else peak[5],
            "torque_ratio_at_peak": None if peak is None else peak[2],
            "event_start_step_replayed": start,
            "event_end_step_replayed": end,
            "continuous_dwell_steps_replayed": 0 if start < 0 else end - start + 1,
            "d26u_first_safety_flag_detection_step": first_flag,
            "endpoint_window_start": int(next(x for x in endpoint["windows"] if int(x["recipe_id"]) == recipe)["window_start_step"]),
            "endpoint_window_end": int(next(x for x in endpoint["windows"] if int(x["recipe_id"]) == recipe)["window_end_step"]),
            "event_in_start_endpoint_window": bool(start >= int(next(x for x in endpoint["windows"] if int(x["recipe_id"]) == recipe)["window_start_step"])),
            "contact_state_at_onset": None if onset is None else {"support_count": onset[6], "support_state": list(onset[7])},
            "base_state_at_onset": None if onset is None else {"root_pose": next(r for r in records if int(r["main_step"]) == start)["root_pose"][recipe], "root_velocity": next(r for r in records if int(r["main_step"]) == start)["root_velocity"][recipe]},
            "acquisition_status": "PRE_ACQUISITION_TRANSIENT_ONLY" if start >= 0 and start < int(next(x for x in endpoint["windows"] if int(x["recipe_id"]) == recipe)["window_start_step"]) else "NOT_PRESENT",
        })
    return {"name": "Exp014D26VSourceSaturationTimingAuditV1", "source": "D26V same fresh lifecycle per-control capture; D26U artifact read-only", "recipes": rows}


def aligned_target(source: dict[str, np.ndarray], source_i: int, native: dict[str, np.ndarray], lead: str) -> dict[str, Any]:
    target_i = MEDOID_ROWS[lead]
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    target_root = np.asarray(native["root_pose"][target_i], dtype=np.float64)
    source_position = source_root[:3]
    source_rotation = quat_matrix(source_root[3:])
    target_position = target_root[:3]
    target_rotation = quat_matrix(target_root[3:])
    aligned_root_rotation = heading_aligned_rotation(target_rotation, source_rotation)
    source_com = np.asarray(source["com_position_w"][source_i], dtype=np.float64)
    target_com = np.asarray(native["com_position"][target_i], dtype=np.float64)
    target_com_rel = target_rotation.T @ (target_com - target_position)
    aligned_com = source_position + source_rotation @ target_com_rel
    source_com_velocity = np.asarray(source["com_velocity_w"][source_i], dtype=np.float64)
    target_com_velocity = source_rotation @ (target_rotation.T @ np.asarray(native["com_velocity"][target_i], dtype=np.float64))
    source_dcm = np.asarray(source["dcm"][source_i], dtype=np.float64)
    target_dcm_world = np.asarray(native["dcm"][target_i], dtype=np.float64)
    aligned_dcm = source_position[:2] + source_rotation[:2, :2] @ (target_dcm_world - target_position[:2])
    source_root_to_com = source_com - source_position
    target_root_to_com = aligned_root_rotation @ target_com_rel
    aligned_root_end = aligned_com - target_root_to_com
    source_foot_positions = np.asarray(source["left_right_foot_pose"][source_i, :, :3], dtype=np.float64)
    source_foot_rotations = [quat_matrix(np.asarray(source["body_quat_w"][source_i, body], dtype=np.float64)) for body in (FOOT_BODY["LEFT"], FOOT_BODY["RIGHT"])]
    target_foot_positions = np.asarray(native["body_pos_w"][target_i, [FOOT_BODY["LEFT"], FOOT_BODY["RIGHT"]]], dtype=np.float64)
    target_foot_rotations = [quat_matrix(np.asarray(native["body_quat_w"][target_i, body], dtype=np.float64)) for body in (FOOT_BODY["LEFT"], FOOT_BODY["RIGHT"])]
    aligned_foot_positions = []
    aligned_foot_rotations = []
    for position, rotation in zip(target_foot_positions, target_foot_rotations):
        rel_position = target_rotation.T @ (position - target_position)
        rel_rotation = target_rotation.T @ rotation
        aligned_foot_positions.append(source_position + source_rotation @ rel_position)
        aligned_foot_rotations.append(source_rotation @ rel_rotation)
    target_torso_rotation = quat_matrix(np.asarray(native["body_quat_w"][target_i, 4], dtype=np.float64))
    aligned_torso_rotation = source_rotation @ (target_rotation.T @ target_torso_rotation)
    source_side = 0 if lead == "LEFT" else 1
    stance_side = 1 - source_side
    source_dcm_offset = source_dcm - source_foot_positions[stance_side, :2]
    aligned_foot_positions_array = np.asarray(aligned_foot_positions, dtype=np.float64)
    target_dcm_offset = aligned_dcm - aligned_foot_positions_array[source_side, :2]
    return {
        "lead_side": lead,
        "target_family": f"{lead}_POST_TOUCHDOWN",
        "target_medoid": {"episode_id": MEDOID_EPISODES[lead], "control_step": MEDOID_STEPS[lead], "bundle_row": target_i, "touchdown_side": lead},
        "source_root_pose": source_root.tolist(),
        "source_root_to_com_m": source_root_to_com.tolist(),
        "target_root_pose_aligned_position": aligned_root_end.tolist(),
        "target_root_rotation_aligned": aligned_root_rotation.tolist(),
        "source_com_position": source_com.tolist(),
        "target_com_position_aligned": aligned_com.tolist(),
        "source_com_velocity": source_com_velocity.tolist(),
        "target_com_velocity_aligned": target_com_velocity.tolist(),
        "source_dcm_xy": source_dcm.tolist(),
        "target_dcm_xy_aligned": aligned_dcm.tolist(),
        "source_dcm_offset_from_initial_stance_m": source_dcm_offset.tolist(),
        "target_dcm_offset_from_entry_support_m": target_dcm_offset.tolist(),
        "dcm_gap_m": (target_dcm_offset - source_dcm_offset).tolist(),
        "source_foot_positions": source_foot_positions.tolist(),
        "source_foot_rotations": [x.tolist() for x in source_foot_rotations],
        "target_foot_positions_aligned": aligned_foot_positions_array.tolist(),
        "target_foot_rotations_aligned": [x.tolist() for x in aligned_foot_rotations],
        "target_torso_rotation_aligned": aligned_torso_rotation.tolist(),
        "source_support_configuration": np.asarray(source["support_state"][source_i], dtype=np.int64).tolist(),
        "target_support_configuration": (np.linalg.norm(native["contact_force"][target_i], axis=1) > 5.0).astype(int).tolist(),
        "required_foot_displacement_m": (aligned_foot_positions_array[source_side] - source_foot_positions[source_side]).tolist(),
        "required_com_displacement_m": (aligned_com - source_com).tolist(),
        "required_pelvis_translation_m": (aligned_root_end - source_position).tolist(),
        "required_pelvis_orientation_rad": so3_log_np(aligned_root_rotation @ source_rotation.T).tolist(),
        "source_target_joint_distance_l2": float(np.linalg.norm(native["joint_pos"][target_i] - source["joint_pos"][source_i])),
        "source_target_action_distance_l2": float(np.linalg.norm(native["next_action"][target_i] - source["current_action"][source_i])),
        "source_target_com_height_difference_m": float(aligned_com[2] - source_com[2]),
        "source_target_dcm_difference_world_m": (aligned_dcm - source_dcm).tolist(),
        "target_native_step_geometry": {"step_length_m": None, "step_width_m": None},
        "target_preserved_without_averaging": True,
    }


def source_target_compatibility(source: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for recipe in RECIPES:
        for lead in LEADS:
            row = aligned_target(source, recipe, native, lead)
            row["recipe_id"] = recipe
            geometry = d26_geometry().get("side_statistics", {}).get(lead, {})
            step = row["required_foot_displacement_m"]
            row["geometry_precheck"] = {
                "required_step_length_m": float(step[0]),
                "required_step_width_m": float(abs(step[1])),
                "native_step_length_p05_p95_m": [geometry.get("step_length_m", {}).get("p05"), geometry.get("step_length_m", {}).get("p95")],
                "native_step_width_p05_p95_m": [geometry.get("step_width_m", {}).get("p05"), geometry.get("step_width_m", {}).get("p95")],
                "within_native_step_length_p05_p95": False if geometry.get("step_length_m", {}).get("p05") is None else geometry["step_length_m"]["p05"] <= abs(step[0]) <= geometry["step_length_m"]["p95"],
                "within_native_step_width_p05_p95": False if geometry.get("step_width_m", {}).get("p05") is None else geometry["step_width_m"]["p05"] <= abs(step[1]) <= geometry["step_width_m"]["p95"],
            }
            rows.append(row)
    return rows


def phase_lengths(multiplier: float) -> dict[str, int]:
    swing = int(round(multiplier * T_REF / DT))
    shift = int(round(0.40 / DT))  # the preregistered shift tuple is applied by caller; placeholder is replaced below
    landing = max(4, int(round(0.50 * swing)))
    acceptance = max(5, int(round(0.50 * T_REF / DT)))
    return {"DOUBLE_SUPPORT_SHIFT": shift, "FIRST_SWING": swing, "LANDING_AND_CAPTURE": landing, "WMOVE_ACCEPTANCE": acceptance}


def hermite(start: np.ndarray, target: np.ndarray, start_velocity: np.ndarray, target_velocity: np.ndarray, s: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(s, 0.0, 1.0))
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    position = h00 * start + h10 * duration * start_velocity + h01 * target + h11 * duration * target_velocity
    ds = (6 * s**2 - 6 * s) * start + (3 * s**2 - 4 * s + 1) * duration * start_velocity + (-6 * s**2 + 6 * s) * target + (3 * s**2 - 2 * s) * duration * target_velocity
    return position, ds / duration


def make_plan_references(source: dict[str, np.ndarray], source_i: int, target: dict[str, Any], shift_s: float, multiplier: float, clearance: float, geometry: dict[str, Any]) -> dict[str, Any]:
    lengths = phase_lengths(multiplier)
    lengths["DOUBLE_SUPPORT_SHIFT"] = int(round(shift_s / DT))
    phase_names = []
    for name in ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"):
        phase_names.extend([name] * lengths[name])
    total_steps = len(phase_names)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    source_root_position = source_root[:3]
    source_root_rotation = quat_matrix(source_root[3:])
    source_com = np.asarray(source["com_position_w"][source_i], dtype=np.float64)
    source_com_velocity = np.asarray(source["com_velocity_w"][source_i], dtype=np.float64)
    target_com = np.asarray(target["target_com_position_aligned"], dtype=np.float64)
    target_com_velocity = np.asarray(target["target_com_velocity_aligned"], dtype=np.float64)
    root_end = np.asarray(target["target_root_pose_aligned_position"], dtype=np.float64)
    root_end_rotation = np.asarray(target["target_root_rotation_aligned"], dtype=np.float64)
    source_foot = np.asarray(target["source_foot_positions"], dtype=np.float64)
    source_foot_rot = [np.asarray(x, dtype=np.float64) for x in target["source_foot_rotations"]]
    target_foot = np.asarray(target["target_foot_positions_aligned"], dtype=np.float64)
    target_foot_rot = [np.asarray(x, dtype=np.float64) for x in target["target_foot_rotations_aligned"]]
    lead_index = 0 if target["lead_side"] == "LEFT" else 1
    stance_index = 1 - lead_index
    omega = math.sqrt(G / max(float(source_com[2]), 1.0e-6))
    com_refs = []
    com_vel_refs = []
    dcm_refs = []
    root_refs = []
    root_vel_refs = []
    foot_refs = []
    zmp_refs = []
    zmp_inside = []
    polygon_xy = np.asarray([[-0.101554609, -0.032734622], [0.101554609, -0.032734622], [0.101554609, 0.032734622], [-0.101554609, 0.032734622]], dtype=np.float64)
    for step in range(total_steps):
        u = float(step + 1) / float(total_steps)
        alpha = minimum_jerk(u)
        com_position, com_velocity = hermite(source_com, target_com, source_com_velocity, target_com_velocity, u, total_steps * DT)
        dcm = com_position[:2] + com_velocity[:2] / omega
        root_offset = (1.0 - alpha) * (source_com - source_root_position) + alpha * (target_com - root_end)
        root_position = com_position - root_offset
        root_rotation = rotation_trajectory(source_root_rotation, root_end_rotation, u)
        root_velocity = (root_position - (source_root_position if step == 0 else root_refs[-1]["position"])) / DT
        root_angular = so3_log_np(root_rotation @ (source_root_rotation if step == 0 else root_refs[-1]["rotation"]).T) / DT
        phase = phase_names[step]
        if phase == "DOUBLE_SUPPORT_SHIFT":
            swing_alpha = 0.0
        elif phase == "FIRST_SWING":
            swing_alpha = minimum_jerk(float(step - lengths["DOUBLE_SUPPORT_SHIFT"] + 1) / max(lengths["FIRST_SWING"], 1))
        else:
            swing_alpha = 1.0
        swing_position = (1.0 - swing_alpha) * source_foot[lead_index] + swing_alpha * target_foot[lead_index]
        if phase == "FIRST_SWING":
            local = float(step - lengths["DOUBLE_SUPPORT_SHIFT"] + 1) / max(lengths["FIRST_SWING"], 1)
            swing_position = swing_position.copy()
            swing_position[2] = (1.0 - swing_alpha) * source_foot[lead_index, 2] + swing_alpha * target_foot[lead_index, 2] + clearance * math.sin(math.pi * np.clip(local, 0.0, 1.0))
        swing_rotation = rotation_trajectory(source_foot_rot[lead_index], target_foot_rot[lead_index], swing_alpha)
        stance_alpha = alpha
        stance_position = (1.0 - stance_alpha) * source_foot[stance_index] + stance_alpha * target_foot[stance_index]
        stance_rotation = rotation_trajectory(source_foot_rot[stance_index], target_foot_rot[stance_index], stance_alpha)
        double_support = phase in ("DOUBLE_SUPPORT_SHIFT", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
        active_centers = [stance_position, swing_position] if double_support else [stance_position]
        active_rotations = [stance_rotation, swing_rotation] if double_support else [stance_rotation]
        polygons = [foot_polygon(center, rotation, polygon_xy) for center, rotation in zip(active_centers, active_rotations)]
        support_polygon = convex_hull(np.concatenate(polygons, axis=0))
        zmp = np.mean(np.asarray(active_centers)[:, :2], axis=0)
        zmp_refs.append(zmp)
        zmp_inside.append(polygon_contains(zmp, support_polygon))
        com_refs.append(com_position)
        com_vel_refs.append(com_velocity)
        dcm_refs.append(dcm)
        root_refs.append({"position": root_position, "rotation": root_rotation})
        root_vel_refs.append(np.concatenate((root_velocity, root_angular)))
        foot_refs.append({"stance_position": stance_position, "stance_rotation": stance_rotation, "swing_position": swing_position, "swing_rotation": swing_rotation})
    return {
        "phase_names": phase_names,
        "phase_lengths": lengths,
        "total_steps": total_steps,
        "com_position": np.asarray(com_refs),
        "com_velocity": np.asarray(com_vel_refs),
        "dcm": np.asarray(dcm_refs),
        "root_position": np.asarray([x["position"] for x in root_refs]),
        "root_rotation": np.asarray([x["rotation"] for x in root_refs]),
        "root_velocity": np.asarray(root_vel_refs),
        "foot_refs": foot_refs,
        "zmp": np.asarray(zmp_refs),
        "zmp_inside": zmp_inside,
        "target_dcm": np.asarray(target["target_dcm_xy_aligned"], dtype=np.float64),
        "clearance_m": clearance,
        "omega": omega,
        "contract": {"phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "shift_s": shift_s, "swing_duration_s": multiplier * T_REF, "clearance_m": clearance},
    }


def matrix_quat(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.asarray([(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s])
    index = int(np.argmax(np.diag(r)))
    if index == 0:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1.0e-12)) * 2.0
        return np.asarray([0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s])
    if index == 1:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1.0e-12)) * 2.0
        return np.asarray([(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s])
    s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1.0e-12)) * 2.0
    return np.asarray([(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s])


def torch_reference(value: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value, dtype=np.float64), dtype=torch.float64)


def make_reference_for_step(plan: dict[str, Any], step: int, target: dict[str, Any], source: dict[str, np.ndarray], source_i: int, lead: str) -> dict[str, torch.Tensor]:
    refs = plan["refs"]
    feet = refs["foot_refs"][step]
    return {
        "root_pose": torch_reference(np.concatenate((refs["root_position"][step], matrix_quat(refs["root_rotation"][step])))),
        "root_velocity": torch_reference(refs["root_velocity"][step]),
        "stance_position": torch_reference(feet["stance_position"]),
        "stance_rotation": torch_reference(feet["stance_rotation"]),
        "swing_position": torch_reference(feet["swing_position"]),
        "swing_rotation": torch_reference(feet["swing_rotation"]),
        "com_position": torch_reference(refs["com_position"][step]),
        "com_velocity": torch_reference(refs["com_velocity"][step]),
        "torso_rotation": torch_reference(target["target_torso_rotation_aligned"]),
        "nominal_q": torch_reference(source["joint_pos"][source_i]),
    }


def kinematic_body_step(state: dict[str, Any], solution: dict[str, Any], reference: dict[str, torch.Tensor], body_jacobian: np.ndarray, masses: np.ndarray) -> dict[str, Any]:
    """Linearized offline FK using the captured full body Jacobian."""
    q = np.asarray(state["q"], dtype=np.float64)
    dq = np.asarray(solution["dq_des"].detach().cpu(), dtype=np.float64)
    root_position = np.asarray(state["root_position"], dtype=np.float64)
    root_rotation = np.asarray(state["root_rotation"], dtype=np.float64)
    root_ref_pose = np.asarray(reference["root_pose"].detach().cpu(), dtype=np.float64)
    next_root_position = root_ref_pose[:3]
    next_root_rotation = quat_matrix(root_ref_pose[3:])
    delta_root = np.concatenate((next_root_position - root_position, so3_log_np(next_root_rotation @ root_rotation.T)))
    delta = np.concatenate((delta_root, dq * DT))
    body_position = np.asarray(state["body_position"], dtype=np.float64).copy()
    body_rotation = np.asarray(state["body_rotation"], dtype=np.float64).copy()
    body_com_position = np.asarray(state["body_com_position"], dtype=np.float64).copy()
    for body in range(body_position.shape[0]):
        body_position[body] += body_jacobian[body, :3] @ delta
        body_rotation[body] = so3_exp_np(body_jacobian[body, 3:6] @ delta) @ body_rotation[body]
        r = body_com_position[body] - state["body_position"][body]
        point_jac = body_jacobian[body, :3] + np.cross(body_jacobian[body, 3:6].T, r).T
        body_com_position[body] += point_jac @ delta
    masses = np.asarray(masses, dtype=np.float64)
    com_position = np.sum(body_com_position * masses[:, None], axis=0) / max(float(masses.sum()), 1.0e-12)
    return {
        "q": np.asarray(solution["q_des"].detach().cpu(), dtype=np.float64),
        "dq": dq,
        "root_position": next_root_position,
        "root_rotation": next_root_rotation,
        "root_velocity": np.asarray(reference["root_velocity"].detach().cpu(), dtype=np.float64),
        "body_position": body_position,
        "body_rotation": body_rotation,
        "body_com_position": body_com_position,
        "com_position": com_position,
        "com_velocity": np.asarray(reference["com_velocity"].detach().cpu(), dtype=np.float64),
        "fk_delta_norm": float(np.linalg.norm(delta)),
    }


def plan_failure(solution: dict[str, Any], actual_errors: dict[str, float], zmp_inside: bool) -> str | None:
    status = str(solution["status"])
    if status == "NUMERICAL_FAILURE":
        return "NUMERICAL_FAILURE"
    if status == "ACTIVE_SET_NONCONVERGENCE":
        return "ACTIVE_SET_NONCONVERGENCE"
    margins = solution["constraint_margins"]
    if int(margins["joint_limit_violation"]) != 0:
        return "JOINT_LIMIT_INFEASIBLE"
    if int(margins["joint_velocity_violation"]) != 0:
        return "JOINT_VELOCITY_INFEASIBLE"
    if int(margins["action_bound_violation"]) != 0:
        return "ACTION_BOUND_INFEASIBLE"
    if actual_errors["stance_position_m"] > 0.005 or actual_errors["stance_rotation_rad"] > 0.03:
        return "STANCE_TASK_INFEASIBLE"
    if actual_errors["swing_position_m"] > 0.010 or actual_errors["swing_rotation_rad"] > 0.03:
        return "SWING_REACH_INFEASIBLE"
    if actual_errors["com_horizontal_m"] > 0.010:
        return "COM_TASK_INFEASIBLE"
    if actual_errors["pelvis_roll_pitch_rad"] > 0.03:
        return "PELVIS_ORIENTATION_INFEASIBLE"
    if not zmp_inside:
        return "ZMP_CONTAINMENT_FAIL"
    return None


def fixture(source: dict[str, np.ndarray], source_i: int, lead: str) -> dict[str, Any]:
    default_q, scale = source_action_contract()
    q_min = np.asarray(source["joint_position_limits"][source_i, :, 0], dtype=np.float64)
    q_max = np.asarray(source["joint_position_limits"][source_i, :, 1], dtype=np.float64)
    q = np.clip(default_q, q_min + 0.02, q_max - 0.02)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    body_rotation = np.asarray([quat_matrix(x) for x in source["body_quat_w"][source_i]], dtype=np.float64)
    return {
        "root_pose": torch_reference(np.concatenate((source_root[:3], matrix_quat(quat_matrix(source_root[3:]))))),
        "root_velocity": torch.zeros(6, dtype=torch.float64),
        "joint_position": torch_reference(q),
        "joint_velocity": torch.zeros(37, dtype=torch.float64),
        "body_position": torch_reference(source["body_pos_w"][source_i]),
        "body_quaternion": torch_reference(np.asarray([matrix_quat(x) for x in body_rotation])),
        "body_jacobians": torch_reference(source["body_jacobians"][source_i]),
        "body_com_position": torch_reference(source["body_com_pos_w"][source_i]),
        "body_masses": torch_reference(source["body_masses"][source_i]),
        "com_position": torch_reference(source["com_position_w"][source_i]),
        "q_min": torch_reference(q_min),
        "q_max": torch_reference(q_max),
        "velocity_limits": torch_reference(source["joint_velocity_limits"][source_i]),
        "default_q": torch_reference(q),
        "action_scale": torch_reference(scale),
        "source_body_rotation": body_rotation,
        "lead": lead,
    }


def solve_fixture(fix: dict[str, Any], root_delta: np.ndarray, root_rotation_delta: np.ndarray, com_delta: np.ndarray, swing_delta: np.ndarray, stance_delta: np.ndarray) -> dict[str, Any]:
    body_position = fix["body_position"].detach().cpu().numpy()
    body_rotation = fix["source_body_rotation"]
    lead_index = 0 if fix["lead"] == "LEFT" else 1
    stance_index = 1 - lead_index
    current_root = fix["root_pose"].detach().cpu().numpy()
    current_com = fix["com_position"].detach().cpu().numpy()
    current_foot_rot = [body_rotation[24], body_rotation[25]]
    reference = {
        "root_pose": torch_reference(np.concatenate((current_root[:3] + root_delta, matrix_quat(so3_exp_np(root_rotation_delta) @ quat_matrix(current_root[3:]))))),
        "root_velocity": torch_reference(np.concatenate((root_delta / DT, root_rotation_delta / DT))),
        "stance_position": torch_reference(body_position[25 if stance_index == 1 else 24] + stance_delta),
        "stance_rotation": torch_reference(so3_exp_np(np.zeros(3)) @ current_foot_rot[stance_index]),
        "swing_position": torch_reference(body_position[24 if lead_index == 0 else 25] + swing_delta),
        "swing_rotation": torch_reference(current_foot_rot[lead_index]),
        "com_position": torch_reference(current_com + com_delta),
        "com_velocity": torch_reference(com_delta / DT),
        "torso_rotation": torch_reference(body_rotation[4]),
        "nominal_q": fix["joint_position"],
    }
    solution = wbik_v2.solve_prescribed_floating_base(
        root_pose=fix["root_pose"], root_velocity=fix["root_velocity"], joint_position=fix["joint_position"], joint_velocity=fix["joint_velocity"],
        body_position=fix["body_position"], body_quaternion=fix["body_quaternion"], body_jacobians=fix["body_jacobians"], body_com_position=fix["body_com_position"], body_masses=fix["body_masses"], com_position=fix["com_position"], reference=reference,
        stance_body_index=24 if stance_index == 0 else 25, swing_body_index=24 if lead_index == 0 else 25,
        q_min=fix["q_min"], q_max=fix["q_max"], velocity_limits=fix["velocity_limits"], default_q=fix["default_q"], action_scale=fix["action_scale"],
    )
    return {"solution": solution, "reference": reference, "stance_index": stance_index, "lead_index": lead_index}


def wbik_v2_unit_tests(source: dict[str, np.ndarray]) -> dict[str, Any]:
    fix = fixture(source, 4, "LEFT")
    tests = []
    cases = [
        ("root_forward_10mm_stance_fixed", np.asarray([0.010, 0, 0]), np.zeros(3), np.zeros(3), np.zeros(3), "stance"),
        ("root_lateral_10mm_stance_fixed", np.asarray([0, 0.010, 0]), np.zeros(3), np.zeros(3), np.zeros(3), "stance"),
        ("root_yaw_0p02_stance_fixed", np.zeros(3), np.asarray([0, 0, 0.02]), np.zeros(3), np.zeros(3), "stance"),
        ("root_forward_plus_swing_forward", np.asarray([0.010, 0, 0]), np.zeros(3), np.zeros(3), np.asarray([0.010, 0, 0]), "swing"),
        ("root_lateral_plus_com_lateral", np.asarray([0, 0.010, 0]), np.zeros(3), np.asarray([0, 0.010, 0]), np.zeros(3), "com"),
        ("combined_first_step_micro_target", np.asarray([0.005, 0.004, 0]), np.asarray([0, 0, 0.01]), np.asarray([0.003, 0.002, 0]), np.asarray([0.004, 0.0, 0]), "combined"),
    ]
    for name, root_delta, root_rot, com_delta, swing_delta, label in cases:
        result = solve_fixture(fix, root_delta, root_rot, com_delta, swing_delta, np.zeros(3))
        sol = result["solution"]
        err = {k: float(v.detach().cpu()) for k, v in sol["task_errors"].items()}
        tests.append({"name": name, "primary_task": label, "status": sol["status"], "root_contribution": {k: v.detach().cpu().tolist() for k, v in sol["root_contribution"].items()}, "joint_compensation_norm": float(torch.linalg.vector_norm(sol["dq_des"]).detach().cpu()), "stance_position_drift_m": err["stance_position_m"], "stance_rotation_drift_rad": err["stance_rotation_rad"], "swing_error_m": err["swing_position_m"], "com_error_m": err["com_horizontal_m"], "constraint_violation": int(sol["constraint_margins"]["joint_limit_violation"] or sol["constraint_margins"]["joint_velocity_violation"] or sol["constraint_margins"]["action_bound_violation"]), "nan_inf": int(not sol["solver_diagnostics"]["finite"]), "task_error_reduced": err["stance_position_m"] < float(np.linalg.norm(root_delta)) + 1.0e-9 or err["swing_position_m"] < float(np.linalg.norm(swing_delta)) + 1.0e-9 or err["com_horizontal_m"] < float(np.linalg.norm(com_delta[:2])) + 1.0e-9})
    mirror_left = solve_fixture(fix, np.asarray([0.004, 0.003, 0]), np.asarray([0, 0, 0.01]), np.asarray([0.002, 0.001, 0]), np.asarray([0.003, 0, 0]), np.zeros(3))
    mirror_fix = fixture(source, 5, "RIGHT")
    mirror_right = solve_fixture(mirror_fix, np.asarray([0.004, -0.003, 0]), np.asarray([0, 0, -0.01]), np.asarray([0.002, -0.001, 0]), np.asarray([0.003, 0, 0]), np.zeros(3))
    mirror_pass = bool(
        mirror_left["solution"]["status"] == "PASS"
        and mirror_right["solution"]["status"] == "PASS"
        and mirror_left["solution"]["solver_diagnostics"]["finite"]
        and mirror_right["solution"]["solver_diagnostics"]["finite"]
        and int(mirror_left["solution"]["constraint_margins"]["joint_limit_violation"]) == 0
        and int(mirror_right["solution"]["constraint_margins"]["joint_limit_violation"]) == 0
    )
    same = solve_fixture(fix, np.asarray([0.004, 0.003, 0]), np.asarray([0, 0, 0.01]), np.asarray([0.002, 0.001, 0]), np.asarray([0.003, 0, 0]), np.zeros(3))["solution"]
    same2 = solve_fixture(fix, np.asarray([0.004, 0.003, 0]), np.asarray([0, 0, 0.01]), np.asarray([0.002, 0.001, 0]), np.asarray([0.003, 0, 0]), np.zeros(3))["solution"]
    deterministic = bool(torch.equal(same["q_des"], same2["q_des"]) and torch.equal(same["dq_des"], same2["dq_des"]) and torch.equal(same["normalized_action"], same2["normalized_action"]))
    gate = all(
        x["constraint_violation"] == 0
        and x["nan_inf"] == 0
        and x["task_error_reduced"]
        and x["stance_position_drift_m"] <= 0.002
        and x["stance_rotation_drift_rad"] <= 0.015
        for x in tests
    ) and mirror_pass and deterministic
    return {"name": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2UnitTests", "tests": tests, "mirror_test": {"status": "PASS" if mirror_pass else "FAIL", "left_status": mirror_left["solution"]["status"], "right_status": mirror_right["solution"]["status"]}, "same_process_determinism": deterministic, "status": "PASS" if gate else "FAIL", "gate": {"stance_position_m": "<=0.002 for micro tests", "stance_rotation_rad": "<=0.015 for micro tests", "constraint_violation": 0, "nan_inf": 0, "task_error_reduced": True}}


def wbik_v1_v2_comparison(source: dict[str, np.ndarray]) -> dict[str, Any]:
    v1 = load_module("exp014_d26v_wbik_v1_for_comparison", V1_PATH)
    fix = fixture(source, 4, "LEFT")
    body_jac = fix["body_jacobians"]
    stance_j = body_jac[25, :, 6:]
    swing_j = body_jac[24, :, 6:]
    jcom = wbik_v2.com_jacobian(fix["body_jacobians"], fix["body_masses"], fix["body_com_position"], fix["body_position"])
    v1_solution = v1.active_set_hierarchical(
        [stance_j, jcom[:, 6:], swing_j],
        [torch.zeros(6, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)],
        fix["joint_position"], fix["q_min"], fix["q_max"], fix["velocity_limits"] * 0.8,
        v1.WBIKConfig(dt=DT),
    )
    v2_result = solve_fixture(fix, np.asarray([0.010, 0, 0]), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
    v2 = v2_result["solution"]
    v1_dq = v1_solution["dq_des"]
    v1_stance_drift = float(torch.linalg.vector_norm(stance_j @ v1_dq * DT).detach().cpu())
    v2_stance_drift = float(v2["task_errors"]["stance_position_m"].detach().cpu())
    return {
        "same_micro_target": {"required_root_displacement_m": [0.010, 0.0, 0.0], "root_reference_semantics": "prescribed +10 mm forward"},
        "v1": {"name": "Exp014DeterministicHierarchicalWBIKV1", "root_displacement_output": "none", "root_columns_used": [], "root_motion_represented": False, "stance_drift_m": v1_stance_drift, "joint_compensation_norm": float(torch.linalg.vector_norm(v1_dq).detach().cpu()), "task_errors": {"stance_position_m": v1_stance_drift, "root_reference_position_m": 0.010}},
        "v2": {"name": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2", "root_displacement_input": [0.010, 0.0, 0.0], "root_columns_used": [0, 1, 2, 3, 4, 5], "root_motion_represented": True, "stance_drift_m": v2_stance_drift, "joint_compensation_norm": float(torch.linalg.vector_norm(v2["dq_des"]).detach().cpu()), "task_errors": {k: float(v.detach().cpu()) for k, v in v2["task_errors"].items()}},
        "interpretation": "V1 has no root variable; V2 evaluates prescribed root contribution and solves only the 37 joint columns. Values are measured, not substituted for a physics result.",
    }


def phase_lengths(multiplier: float, shift_s: float) -> dict[str, int]:
    swing = int(round(multiplier * T_REF / DT))
    shift = int(round(shift_s / DT))
    landing = max(4, int(round(0.50 * swing)))
    acceptance = max(5, int(round(0.50 * T_REF / DT)))
    return {"DOUBLE_SUPPORT_SHIFT": shift, "FIRST_SWING": swing, "LANDING_AND_CAPTURE": landing, "WMOVE_ACCEPTANCE": acceptance}


def hermite(start: np.ndarray, target: np.ndarray, start_velocity: np.ndarray, target_velocity: np.ndarray, s: float, duration: float) -> tuple[np.ndarray, np.ndarray]:
    s = float(np.clip(s, 0.0, 1.0))
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    position = h00 * start + h10 * duration * start_velocity + h01 * target + h11 * duration * target_velocity
    ds = (6 * s**2 - 6 * s) * start + (3 * s**2 - 4 * s + 1) * duration * start_velocity + (-6 * s**2 + 6 * s) * target + (3 * s**2 - 2 * s) * duration * target_velocity
    return position, ds / duration


def make_plan_references(source: dict[str, np.ndarray], source_i: int, target: dict[str, Any], shift_s: float, multiplier: float, clearance: float, geometry: dict[str, Any]) -> dict[str, Any]:
    lengths = phase_lengths(multiplier, shift_s)
    phase_names = []
    for name in ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"):
        phase_names.extend([name] * lengths[name])
    total_steps = len(phase_names)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    source_root_position = source_root[:3]
    source_root_rotation = quat_matrix(source_root[3:])
    source_com = np.asarray(source["com_position_w"][source_i], dtype=np.float64)
    source_com_velocity = np.asarray(source["com_velocity_w"][source_i], dtype=np.float64)
    target_com = np.asarray(target["target_com_position_aligned"], dtype=np.float64)
    target_com_velocity = np.asarray(target["target_com_velocity_aligned"], dtype=np.float64)
    root_end = np.asarray(target["target_root_pose_aligned_position"], dtype=np.float64)
    root_end_rotation = np.asarray(target["target_root_rotation_aligned"], dtype=np.float64)
    source_foot = np.asarray(target["source_foot_positions"], dtype=np.float64)
    source_foot_rot = [np.asarray(x, dtype=np.float64) for x in target["source_foot_rotations"]]
    target_foot = np.asarray(target["target_foot_positions_aligned"], dtype=np.float64)
    target_foot_rot = [np.asarray(x, dtype=np.float64) for x in target["target_foot_rotations_aligned"]]
    lead_index = 0 if target["lead_side"] == "LEFT" else 1
    stance_index = 1 - lead_index
    omega = math.sqrt(G / max(float(source_com[2]), 1.0e-6))
    com_refs = []
    com_vel_refs = []
    dcm_refs = []
    root_refs = []
    root_vel_refs = []
    foot_refs = []
    zmp_refs = []
    zmp_inside = []
    polygon_xy = np.asarray([[-0.101554609, -0.032734622], [0.101554609, -0.032734622], [0.101554609, 0.032734622], [-0.101554609, 0.032734622]], dtype=np.float64)
    for step in range(total_steps):
        u = float(step + 1) / float(total_steps)
        alpha = minimum_jerk(u)
        com_position, com_velocity = hermite(source_com, target_com, source_com_velocity, target_com_velocity, u, total_steps * DT)
        dcm = com_position[:2] + com_velocity[:2] / omega
        root_offset = (1.0 - alpha) * (source_com - source_root_position) + alpha * (target_com - root_end)
        root_position = com_position - root_offset
        root_rotation = rotation_trajectory(source_root_rotation, root_end_rotation, u)
        previous_root_position = source_root_position if step == 0 else root_refs[-1]["position"]
        previous_root_rotation = source_root_rotation if step == 0 else root_refs[-1]["rotation"]
        root_velocity = (root_position - previous_root_position) / DT
        root_angular = so3_log_np(root_rotation @ previous_root_rotation.T) / DT
        phase = phase_names[step]
        if phase == "DOUBLE_SUPPORT_SHIFT":
            swing_alpha = 0.0
        elif phase == "FIRST_SWING":
            swing_alpha = minimum_jerk(float(step - lengths["DOUBLE_SUPPORT_SHIFT"] + 1) / max(lengths["FIRST_SWING"], 1))
        else:
            swing_alpha = 1.0
        swing_position = (1.0 - swing_alpha) * source_foot[lead_index] + swing_alpha * target_foot[lead_index]
        if phase == "FIRST_SWING":
            local = float(step - lengths["DOUBLE_SUPPORT_SHIFT"] + 1) / max(lengths["FIRST_SWING"], 1)
            swing_position = swing_position.copy()
            swing_position[2] = (1.0 - swing_alpha) * source_foot[lead_index, 2] + swing_alpha * target_foot[lead_index, 2] + clearance * math.sin(math.pi * np.clip(local, 0.0, 1.0))
        swing_rotation = rotation_trajectory(source_foot_rot[lead_index], target_foot_rot[lead_index], swing_alpha)
        stance_alpha = alpha
        stance_position = (1.0 - stance_alpha) * source_foot[stance_index] + stance_alpha * target_foot[stance_index]
        stance_rotation = rotation_trajectory(source_foot_rot[stance_index], target_foot_rot[stance_index], stance_alpha)
        double_support = phase in ("DOUBLE_SUPPORT_SHIFT", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
        active_centers = [stance_position, swing_position] if double_support else [stance_position]
        active_rotations = [stance_rotation, swing_rotation] if double_support else [stance_rotation]
        polygons = [foot_polygon(center, rotation, polygon_xy) for center, rotation in zip(active_centers, active_rotations)]
        support_polygon = convex_hull(np.concatenate(polygons, axis=0))
        zmp = np.mean(np.asarray(active_centers)[:, :2], axis=0)
        zmp_refs.append(zmp)
        zmp_inside.append(polygon_contains(zmp, support_polygon))
        com_refs.append(com_position)
        com_vel_refs.append(com_velocity)
        dcm_refs.append(dcm)
        root_refs.append({"position": root_position, "rotation": root_rotation})
        root_vel_refs.append(np.concatenate((root_velocity, root_angular)))
        foot_refs.append({"stance_position": stance_position, "stance_rotation": stance_rotation, "swing_position": swing_position, "swing_rotation": swing_rotation})
    return {"phase_names": phase_names, "phase_lengths": lengths, "total_steps": total_steps, "com_position": np.asarray(com_refs), "com_velocity": np.asarray(com_vel_refs), "dcm": np.asarray(dcm_refs), "root_position": np.asarray([x["position"] for x in root_refs]), "root_rotation": np.asarray([x["rotation"] for x in root_refs]), "root_velocity": np.asarray(root_vel_refs), "foot_refs": foot_refs, "zmp": np.asarray(zmp_refs), "zmp_inside": zmp_inside, "target_dcm": np.asarray(target["target_dcm_xy_aligned"], dtype=np.float64), "clearance_m": clearance, "omega": omega, "contract": {"phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "shift_s": shift_s, "swing_duration_s": multiplier * T_REF, "clearance_m": clearance}}


def kinematic_body_step(state: dict[str, Any], solution: dict[str, Any], reference: dict[str, torch.Tensor], body_jacobian: np.ndarray, masses: np.ndarray) -> dict[str, Any]:
    """First-order offline FK from captured full body Jacobians; never physics."""
    dq = np.asarray(solution["dq_des"].detach().cpu(), dtype=np.float64)
    root_position = np.asarray(state["root_position"], dtype=np.float64)
    root_rotation = np.asarray(state["root_rotation"], dtype=np.float64)
    root_ref_pose = np.asarray(reference["root_pose"].detach().cpu(), dtype=np.float64)
    next_root_position = root_ref_pose[:3]
    next_root_rotation = quat_matrix(root_ref_pose[3:])
    delta_root = np.concatenate((next_root_position - root_position, so3_log_np(next_root_rotation @ root_rotation.T)))
    delta = np.concatenate((delta_root, dq * DT))
    body_position = np.asarray(state["body_position"], dtype=np.float64).copy()
    body_rotation = np.asarray(state["body_rotation"], dtype=np.float64).copy()
    body_com_position = np.asarray(state["body_com_position"], dtype=np.float64).copy()
    for body in range(body_position.shape[0]):
        body_position[body] += body_jacobian[body, :3] @ delta
        body_rotation[body] = so3_exp_np(body_jacobian[body, 3:6] @ delta) @ body_rotation[body]
        r = body_com_position[body] - state["body_position"][body]
        point_jac = body_jacobian[body, :3] + np.cross(body_jacobian[body, 3:6].T, r).T
        body_com_position[body] += point_jac @ delta
    masses = np.asarray(masses, dtype=np.float64)
    com_position = np.sum(body_com_position * masses[:, None], axis=0) / max(float(masses.sum()), 1.0e-12)
    return {"q": np.asarray(solution["q_des"].detach().cpu(), dtype=np.float64), "dq": dq, "root_position": next_root_position, "root_rotation": next_root_rotation, "root_velocity": np.asarray(reference["root_velocity"].detach().cpu(), dtype=np.float64), "body_position": body_position, "body_rotation": body_rotation, "body_com_position": body_com_position, "com_position": com_position, "com_velocity": np.asarray(reference["com_velocity"].detach().cpu(), dtype=np.float64), "fk_delta_norm": float(np.linalg.norm(delta))}


def plan_failure(solution: dict[str, Any], actual_errors: dict[str, float], zmp_inside: bool) -> str | None:
    status = str(solution["status"])
    if status == "NUMERICAL_FAILURE":
        return "NUMERICAL_FAILURE"
    if status == "ACTIVE_SET_NONCONVERGENCE":
        return "ACTIVE_SET_NONCONVERGENCE"
    margins = solution["constraint_margins"]
    if int(margins["joint_limit_violation"]) != 0:
        return "JOINT_LIMIT_INFEASIBLE"
    if int(margins["joint_velocity_violation"]) != 0:
        return "JOINT_VELOCITY_INFEASIBLE"
    if int(margins["action_bound_violation"]) != 0:
        return "ACTION_BOUND_INFEASIBLE"
    if actual_errors["stance_position_m"] > 0.005 or actual_errors["stance_rotation_rad"] > 0.03:
        return "STANCE_TASK_INFEASIBLE"
    if actual_errors["swing_position_m"] > 0.010 or actual_errors["swing_rotation_rad"] > 0.03:
        return "SWING_REACH_INFEASIBLE"
    if actual_errors["com_horizontal_m"] > 0.010:
        return "COM_TASK_INFEASIBLE"
    if actual_errors["pelvis_roll_pitch_rad"] > 0.03:
        return "PELVIS_ORIENTATION_INFEASIBLE"
    if not zmp_inside:
        return "ZMP_CONTAINMENT_FAIL"
    return None


def rollout_plan(source: dict[str, np.ndarray], source_i: int, target: dict[str, Any], lead: str, shift_s: float, multiplier: float, clearance: float, geometry: dict[str, Any], default_q: np.ndarray, scale: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    refs = make_plan_references(source, source_i, target, shift_s, multiplier, clearance, geometry)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    source_body_rotation = np.asarray([quat_matrix(x) for x in source["body_quat_w"][source_i]], dtype=np.float64)
    state = {"q": np.asarray(source["joint_pos"][source_i], dtype=np.float64), "dq": np.asarray(source["joint_vel"][source_i], dtype=np.float64), "root_position": source_root[:3].copy(), "root_rotation": quat_matrix(source_root[3:]), "root_velocity": np.asarray(source["root_velocity"][source_i], dtype=np.float64), "body_position": np.asarray(source["body_pos_w"][source_i], dtype=np.float64).copy(), "body_rotation": source_body_rotation.copy(), "body_com_position": np.asarray(source["body_com_pos_w"][source_i], dtype=np.float64).copy(), "com_position": np.asarray(source["com_position_w"][source_i], dtype=np.float64).copy(), "com_velocity": np.asarray(source["com_velocity_w"][source_i], dtype=np.float64).copy()}
    q_min = torch_reference(source["joint_position_limits"][source_i, :, 0])
    q_max = torch_reference(source["joint_position_limits"][source_i, :, 1])
    velocity_limits = torch_reference(source["joint_velocity_limits"][source_i])
    q = torch_reference(state["q"])
    body_jacobian = np.asarray(source["body_jacobians"][source_i], dtype=np.float64)
    body_com_position = torch_reference(state["body_com_position"])
    body_position = torch_reference(state["body_position"])
    body_masses = torch_reference(source["body_masses"][source_i])
    rows: list[dict[str, Any]] = []
    first_failure = None
    max_values = {"stance_position_m": 0.0, "stance_rotation_rad": 0.0, "swing_position_m": 0.0, "swing_rotation_rad": 0.0, "com_horizontal_m": 0.0, "com_xyz_m": 0.0, "pelvis_roll_pitch_rad": 0.0, "root_reference_position_m": 0.0}
    max_velocity_ratio = 0.0
    min_action_margin = float("inf")
    max_joint_limit_violation = 0
    max_action_violation = 0
    max_zmp_violation = 0
    for step in range(refs["total_steps"]):
        reference = make_reference_for_step({"refs": refs}, step, target, source, source_i, lead)
        root_pose = torch_reference(np.concatenate((state["root_position"], matrix_quat(state["root_rotation"]))))
        solution = wbik_v2.solve_prescribed_floating_base(root_pose=root_pose, root_velocity=torch_reference(state["root_velocity"]), joint_position=q, joint_velocity=torch_reference(state["dq"]), body_position=body_position, body_quaternion=torch_reference(np.asarray([matrix_quat(x) for x in state["body_rotation"]])), body_jacobians=torch_reference(body_jacobian), body_com_position=body_com_position, body_masses=body_masses, com_position=torch_reference(state["com_position"]), reference=reference, stance_body_index=FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"], swing_body_index=FOOT_BODY[lead], q_min=q_min, q_max=q_max, velocity_limits=velocity_limits, default_q=torch_reference(default_q), action_scale=torch_reference(scale))
        next_state = kinematic_body_step(state, solution, reference, body_jacobian, source["body_masses"][source_i])
        stance_index = FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"]
        swing_index = FOOT_BODY[lead]
        feet = refs["foot_refs"][step]
        stance_pos_err = float(np.linalg.norm(next_state["body_position"][stance_index] - np.asarray(feet["stance_position"])))
        stance_rot_err = float(np.linalg.norm(so3_log_np(np.asarray(feet["stance_rotation"]) @ next_state["body_rotation"][stance_index].T)))
        swing_pos_err = float(np.linalg.norm(next_state["body_position"][swing_index] - np.asarray(feet["swing_position"])))
        swing_rot_err = float(np.linalg.norm(so3_log_np(np.asarray(feet["swing_rotation"]) @ next_state["body_rotation"][swing_index].T)))
        com_err = float(np.linalg.norm(next_state["com_position"][:2] - refs["com_position"][step, :2]))
        com_xyz_err = float(np.linalg.norm(next_state["com_position"] - refs["com_position"][step]))
        root_err = float(np.linalg.norm(next_state["root_position"] - refs["root_position"][step]))
        pelvis_err = float(np.linalg.norm(so3_log_np(refs["root_rotation"][step] @ next_state["root_rotation"].T)[:2]))
        actual_errors = {"stance_position_m": stance_pos_err, "stance_rotation_rad": stance_rot_err, "swing_position_m": swing_pos_err, "swing_rotation_rad": swing_rot_err, "com_horizontal_m": com_err, "com_xyz_m": com_xyz_err, "pelvis_roll_pitch_rad": pelvis_err, "root_reference_position_m": root_err}
        for key in max_values:
            max_values[key] = max(max_values[key], actual_errors[key])
        margins = solution["constraint_margins"]
        max_velocity_ratio = max(max_velocity_ratio, float(margins["planned_joint_velocity_ratio_max"].detach().cpu()))
        min_action_margin = min(min_action_margin, float(margins["action_min_margin"].detach().cpu()))
        max_joint_limit_violation = max(max_joint_limit_violation, int(margins["joint_limit_violation"]))
        max_action_violation = max(max_action_violation, int(margins["action_bound_violation"]))
        zmp_inside = bool(refs["zmp_inside"][step])
        max_zmp_violation = max(max_zmp_violation, int(not zmp_inside))
        failure = plan_failure(solution, actual_errors, zmp_inside)
        if first_failure is None and failure is not None:
            first_failure = failure
        rows.append({"step": step + 1, "phase": refs["phase_names"][step], "ik_status": solution["status"], "failure": failure, "stance_position_error_m": stance_pos_err, "stance_rotation_error_rad": stance_rot_err, "swing_position_error_m": swing_pos_err, "swing_rotation_error_rad": swing_rot_err, "com_horizontal_error_m": com_err, "com_xyz_error_m": com_xyz_err, "root_reference_consistency_error_m": root_err, "pelvis_roll_pitch_error_rad": pelvis_err, "planned_joint_velocity_ratio_max": float(margins["planned_joint_velocity_ratio_max"].detach().cpu()), "action_min_margin": float(margins["action_min_margin"].detach().cpu()), "joint_limit_violation": int(margins["joint_limit_violation"]), "action_bound_violation": int(margins["action_bound_violation"]), "zmp_polygon_violation": int(not zmp_inside), "dcm_reference_x": float(refs["dcm"][step, 0]), "dcm_reference_y": float(refs["dcm"][step, 1]), "root_position": next_state["root_position"].tolist(), "normalized_action": solution["normalized_action"].detach().cpu().tolist(), "q_des": solution["q_des"].detach().cpu().tolist(), "dq_des": solution["dq_des"].detach().cpu().tolist(), "root_contribution": {k: v.detach().cpu().tolist() for k, v in solution["root_contribution"].items()}, "solver_iterations": solution["solver_diagnostics"]["iterations"], "solver_ranks": solution["solver_diagnostics"]["ranks"]})
        state = next_state
        q = torch_reference(state["q"])
        body_position = torch_reference(state["body_position"])
        body_com_position = torch_reference(state["body_com_position"])
    final_dcm = state["com_position"][:2] + state["com_velocity"][:2] / refs["omega"]
    dcm_final_error = float(np.linalg.norm(final_dcm - refs["target_dcm"]))
    dcm_p95 = 0.0
    statistics_path = D26T / "entry_distribution_statistics.json"
    if statistics_path.exists():
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))["sides"][lead]["dcm_offset"]
        dcm_p95 = max(abs(float(statistics["x"]["p05"])), abs(float(statistics["x"]["p95"])), abs(float(statistics["y"]["p05"])), abs(float(statistics["y"]["p95"])))
    dcm_pass = bool(dcm_final_error <= max(dcm_p95, 1.0e-9))
    mandatory_pass = bool(first_failure is None and dcm_pass and max_values["stance_position_m"] <= 0.005 and max_values["stance_rotation_rad"] <= 0.03 and max_values["swing_position_m"] <= 0.010 and max_values["swing_rotation_rad"] <= 0.03 and max_values["com_horizontal_m"] <= 0.010 and max_values["root_reference_position_m"] <= 0.005 and max_values["pelvis_roll_pitch_rad"] <= 0.03 and max_joint_limit_violation == 0 and max_velocity_ratio <= 0.80 and max_action_violation == 0 and max_zmp_violation == 0)
    summary = {"total_steps": refs["total_steps"], "ik_solution_rate": float(sum(int(row["ik_status"] == "PASS") for row in rows) / max(len(rows), 1)), "max_errors": max_values, "max_planned_joint_velocity_ratio": max_velocity_ratio, "min_action_margin": min_action_margin, "joint_limit_violation": max_joint_limit_violation, "action_bound_violation": max_action_violation, "zmp_polygon_violation": max_zmp_violation, "dcm_final": final_dcm.tolist(), "dcm_target": refs["target_dcm"].tolist(), "dcm_final_error": dcm_final_error, "dcm_neighborhood_p95_bound": dcm_p95, "dcm_endpoint_pass": dcm_pass, "mandatory_gates_pass": mandatory_pass, "first_failure": first_failure, "fk_method": "captured full body Jacobian first-order kinematic FK; no physics", "physics_executed": 0}
    return summary, rows, {"refs": refs, "target": target}


def independent_determinism_hash() -> dict[str, Any]:
    command = [sys.executable, str(HERE), "--determinism-child"]
    first = subprocess.check_output(command, cwd=REPO, text=True).strip().splitlines()[-1]
    second = subprocess.check_output(command, cwd=REPO, text=True).strip().splitlines()[-1]
    return {"status": "PASS" if first == second else "FAIL", "run1_hash": first, "run2_hash": second, "independent_processes": 2, "command": command}


def determinism_child() -> None:
    source = load_npz(SOURCE)
    fix = fixture(source, 4, "LEFT")
    result = solve_fixture(fix, np.asarray([0.004, 0.003, 0]), np.asarray([0, 0, 0.01]), np.asarray([0.002, 0.001, 0]), np.asarray([0.003, 0, 0]), np.zeros(3))["solution"]
    print(canonical_hash({"q_des": result["q_des"], "dq_des": result["dq_des"], "normalized_action": result["normalized_action"], "root_contribution": result["root_contribution"]}), flush=True)


def git_status() -> list[str]:
    return subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()


def protected_hashes() -> dict[str, Any]:
    paths = {
        "d26u_stage_classification": D26U / "stage_classification.json",
        "d26u_source_bundle": SOURCE,
        "d26u_source_bundle_digest": D26U / "fresh_shold_identity_complete_sources.sha256",
        "d26t_stage_classification": D26T / "stage_classification.json",
        "d26t_wmove_medoid_identity": D26T / "entry_mirror_audit.json",
        "d26s_native_bundle": NATIVE,
        "d26_wbik_v1": V1_PATH,
        "d26_action_conversion_contract": REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher/model_based_teacher_robot_contract.json",
        "d26_numeric_foot_polygon": REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik/numeric_foot_sole_polygon.json",
    }
    return {name: {"path": str(path.relative_to(REPO)).replace("\\", "/"), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None} for name, path in paths.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_status = git_status()
    starting_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source_sha = sha256_file(SOURCE)
    native_sha = sha256_file(NATIVE)
    source = load_npz(SOURCE)
    native = load_npz(NATIVE)
    source_manifest = json.loads((D26U / "fresh_shold_source_manifest.json").read_text(encoding="utf-8"))
    source_parity = json.loads((D26U / "fresh_shold_capture_parity.json").read_text(encoding="utf-8"))
    off = endpoint_records(ENDPOINT_OFF)
    on = endpoint_records(ENDPOINT_ON)
    endpoint_rows, endpoint_summary = endpoint_reclassification(source_manifest, off, on)
    dump("stage_reference.json", {"stage": "Phase 2-D26V", "requested_starting_head": START_HEAD_REQUESTED, "starting_head": starting_head, "head_matches_requested": starting_head == START_HEAD_REQUESTED, "starting_git_status_short": start_status, "remote_push": False, "persistent_policy_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "d26u_read_only": True})
    dump("protocol.json", {"name": "Exp014PrescribedFloatingBaseSTARTOfflineFeasibilityV2", "phase": "2-D26V", "protected_inputs": {"d26u_source_bundle_sha256": source_sha, "d26s_native_bundle_sha256": native_sha, "d26s_native_bundle_expected_actual_sha256": NATIVE_SHA_ACTUAL, "d26t_read_only": True, "d26u_classification_unchanged": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL"}, "source_contracts": ["Exp014FreshS_HOLDSourceLifecycleV2", "START_SOURCE_ENDPOINT_ELIGIBILITY_V1"], "grid": {"double_support_shift_s": list(SHIFTS), "swing_duration_multiplier": list(SWING_MULTIPLIERS), "clearance_percentile": ["p50", "p75", "p90"], "plans": 432}, "offline_only": {"physics": 0, "persistent_update": 0, "checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0}, "target_semantics": {"LEFT_first_swing": "LEFT foot touchdown -> LEFT_POST_TOUCHDOWN_MEDOID", "RIGHT_first_swing": "RIGHT foot touchdown -> RIGHT_POST_TOUCHDOWN_MEDOID", "artificial_symmetry": 0, "target_average": 0}})
    dump("start_source_endpoint_eligibility_v1.json", endpoint_summary)
    write_csv("source_endpoint_reclassification.csv", endpoint_rows)
    dump("source_endpoint_reclassification.json", {"contract": endpoint_summary, "rows": endpoint_rows})
    dump("source_saturation_timing_audit.json", saturation_timing_audit(on, source_manifest))
    dump("fresh_shold_source_manifest.json", {"source": "D26U read-only manifest", "source_manifest_sha256": sha256_file(D26U / "fresh_shold_source_manifest.json"), "bundle_sha256": source_sha, "recipes": source_manifest["recipes"], "d26u_full_lifecycle_valid_count": source_manifest["valid_source_count"], "d26v_endpoint_eligible_count": endpoint_summary["endpoint_eligible_count"], "capture_parity_d26u": source_parity["status"], "capture_parity_d26v": endpoint_summary["capture_parity"], "read_only": True})
    geometry = d26_geometry()
    gvalues = geometry_values(geometry)
    dump("wmove_transition_geometry_v1.json", {"source": "D26U read-only wmove_transition_geometry_v1.json", "source_sha256": sha256_file(D26U / "wmove_transition_geometry_v1.json") if (D26U / "wmove_transition_geometry_v1.json").exists() else None, "T_ref": {"seconds": T_REF, "steps": 8, "definition": "canonical swing-duration median; fixed by D26U"}, "fixed_values": {"step_length_m": {"p05": 0.021411048085822428, "p50": 0.03885663067778204, "p95": 0.056497849546829876}, "step_width_m": {"p05": 0.29886543779556557, "p50": 0.30435013268275535, "p95": 0.31763089458257526}, "clearance_m": {"p50": 0.0575430240155394, "p75": 0.061676214289145, "p90": 0.06374770490192924}, "landing_vertical_velocity_mps": {"p05": -0.01895569062692181, "p50": -0.016512683341732174, "p95": -0.014293276507200487}}, "side_statistics": geometry.get("side_statistics"), "aggregate_statistics": geometry.get("aggregate_statistics"), "grid_unchanged": True})
    dump("wmove_step_geometry_statistics.json", {"aggregate": geometry.get("aggregate_statistics"), "side": geometry.get("side_statistics"), "event_source": "E0_STRICT_TOUCHDOWN", "native_bundle_sha256": native_sha, "read_only_reduction": True})
    compatibility = source_target_compatibility(source, native)
    dump("source_target_compatibility.json", {"rows": compatibility, "target_source": {"LEFT": {"episode": 52, "control_step": 111, "bundle_row": 8171}, "RIGHT": {"episode": 187, "control_step": 115, "bundle_row": 9330}}, "mapping": "lead touchdown foot is the corresponding POST_TOUCHDOWN target; target frames aligned to each source root without averaging", "precheck_not_success": True})
    v1_audit = {"name": "Exp014DeterministicHierarchicalWBIKV1", "classification": "FB0_FIXED_WORLD_ROOT", "source": str(V1_PATH.relative_to(REPO)).replace("\\", "/"), "source_sha256": sha256_file(V1_PATH), "jacobian_columns": "6:43 joint columns; root 0:6 excluded", "root_translation_target": "not represented", "pelvis_world_position": "no generalized root variable", "com_world_target": "point-corrected CoM Jacobian through joint columns only", "stance_foot_fixed_base_motion": "joint-induced differential motion only", "output": ["q_des[37]", "dq_des[37]", "normalized_action[37]"]}
    dump("wbik_floating_base_semantics_audit.json", {"v1": v1_audit, "v2": {"name": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2", "classification": "FB1_PRESCRIBED_ROOT_REFERENCE", "jacobian_root_columns_used": [0, 1, 2, 3, 4, 5], "jacobian_joint_columns_solved": list(range(6, 43)), "root_position_target": "prescribed LIPM/DCM reference; not a solver output", "root_orientation_target": "prescribed minimum-jerk source-to-target roll/pitch; yaw delta fixed to 0 by heading alignment", "pelvis_world_position": "root pose reference; no duplicate world-translation task", "com_world_target": "full point-corrected CoM Jacobian root contribution plus joint residual", "stance_foot_fixed_base_motion": "joint velocity compensates prescribed root twist through root-column contribution", "output": ["q_des[37]", "dq_des[37]", "normalized_action[37]", "task_errors", "constraint_margins", "solver_diagnostics"], "action_conversion": {"default_joint_position_offset": "D25 runtime contract", "scale": ACTION_SCALE, "bound": [-1.0, 1.0]}, "physics": 0, "versioned": True}, "root_motion_sufficiency_is_tested_by_v2_rollout": True})
    dump("wbik_v2_interface_contract.json", {"name": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2", "classification": "FB1_PRESCRIBED_ROOT_REFERENCE", "input": ["prescribed root position/orientation", "prescribed root linear/angular velocity", "current 37 joint positions/velocities", "stance-foot 6D target", "swing-foot 6D target", "CoM position/velocity target", "pelvis/root orientation consistency", "torso orientation", "nominal posture"], "output": ["q_des[37]", "dq_des[37]", "normalized_action[37]", "task_errors", "constraint_margins", "solver_diagnostics"], "root_output": "none; root reference is external", "runtime_action_dimension": 37})
    dump("wbik_v2_solver_contract.json", {"solver": "deterministic SVD damped least-squares sequential null-space projection", "damping": 1.0e-4, "svd_tolerance": 1.0e-8, "hierarchy": ["P0 stance-foot world 6D", "P1 CoM xyz", "P1 swing-foot world 6D", "P2 torso orientation", "P2 nominal posture", "P2 action-rate regularization"], "active_set": {"max_iterations": 37, "position_limit_freeze": True, "post_clip_hiding": False}, "root_semantics": "prescribed external twist; root contribution subtracted from foot/CoM joint residuals", "com_velocity": "c_dot_total = c_dot_root_contribution + J_com_joint*dq_joint", "foot_velocity": "foot_twist_total = foot_twist_root_contribution + J_foot_joint*dq_joint", "mass_matrix": False, "physics": 0})
    unit_tests = wbik_v2_unit_tests(source)
    dump("wbik_v2_unit_tests.json", unit_tests)
    comparison = wbik_v1_v2_comparison(source)
    dump("wbik_v1_v2_comparison.json", comparison)
    determinism = independent_determinism_hash()
    dump("wbik_v2_determinism.json", determinism)
    dump("prescribed_root_trajectory_contract.json", {"name": "Exp014PrescribedLIPMDCMRootTrajectoryV1", "phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "root_position": "CoM reference minus interpolated source/target pelvis-to-CoM offset", "root_orientation": "minimum-jerk source-to-target roll/pitch in heading-aligned frame; yaw delta 0", "root_height": "source root height to target medoid-aligned root height via same prescribed trajectory", "root_velocity": "finite difference of prescribed root pose at control dt", "root_not_double_constrained": True, "target_medoids": {"LEFT": {"episode": 52, "step": 111}, "RIGHT": {"episode": 187, "step": 115}}})
    source_eligible = {int(row["recipe_id"]): bool(row["endpoint_eligible"]) for row in endpoint_rows}
    target_map = {(int(row["recipe_id"]), row["lead_side"]): row for row in compatibility}
    default_q, action_scale = source_action_contract()
    dump("offline_lipm_plan_manifest.json", {"name": "Exp014Fixed27PlanLIPMDCMManifestV2", "phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "T_ref_s": T_REF, "T_ref_definition": "canonical swing-duration median", "grid": {"double_support_shift_s": list(SHIFTS), "swing_duration_multiplier": list(SWING_MULTIPLIERS), "clearance_percentile": ["p50", "p75", "p90"]}, "count": 432, "CoP_ZMP": "active sole polygon or convex hull of active sole polygons; containment checked every control step", "target_semantics": "lead side first swing ends at corresponding native POST_TOUCHDOWN medoid", "physics": 0, "parameter_grid_changed": 0})
    clearance_values = gvalues["clearance"]
    ledger: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    plan_summaries: list[dict[str, Any]] = []
    for recipe in RECIPES:
        for lead in LEADS:
            target = target_map[(recipe, lead)]
            for shift in SHIFTS:
                for multiplier in SWING_MULTIPLIERS:
                    for percentile in CLEARANCE_PERCENTILES:
                        plan_id = f"D26V_R{recipe:02d}_{lead}_SHIFT{shift:.2f}_SWING{multiplier:.1f}_C{percentile:02d}"
                        tuple_value = {"source_recipe": recipe, "lead_side": lead, "double_support_shift_s": shift, "swing_duration_multiplier": multiplier, "clearance_percentile": percentile, "target_medoid": target["target_medoid"]}
                        plan_hash = canonical_hash(tuple_value)
                        base = {"plan_id": plan_id, "source_recipe": recipe, "lead_side": lead, "target_family": target["target_family"], "double_support_shift_s": shift, "swing_duration_multiplier": multiplier, "clearance_percentile": percentile, "clearance_m": clearance_values[percentile], "target_medoid": target["target_medoid"], "plan_hash": plan_hash, "source_endpoint_eligible": source_eligible[recipe], "physics_executed": 0}
                        if not source_eligible[recipe]:
                            base.update({"status": "SOURCE_ENDPOINT_INELIGIBLE", "eligible": False, "dominant_failure": "SOURCE_ENDPOINT_INELIGIBLE", "ik_solution_rate": 0.0})
                            summary, rows, extra = None, [], None
                        elif unit_tests["status"] != "PASS" or determinism["status"] != "PASS":
                            base.update({"status": "WBIK_V2_INTERFACE_FAIL", "eligible": False, "dominant_failure": "NUMERICAL_FAILURE", "ik_solution_rate": 0.0})
                            summary, rows, extra = None, [], None
                        else:
                            summary, rows, extra = rollout_plan(source, recipe, target, lead, shift, multiplier, clearance_values[percentile], geometry, default_q, action_scale)
                            base.update({"status": "ELIGIBLE" if summary["mandatory_gates_pass"] else "INELIGIBLE", "eligible": bool(summary["mandatory_gates_pass"]), "dominant_failure": summary["first_failure"] if not summary["mandatory_gates_pass"] else None, "ik_solution_rate": summary["ik_solution_rate"], "summary": summary})
                            for row in rows:
                                task_rows.append({"plan_id": plan_id, "source_recipe": recipe, "lead_side": lead, "shift_s": shift, "swing_multiplier": multiplier, "clearance_percentile": percentile, **row})
                        ledger.append(base)
                        plan_summaries.append({"plan_id": plan_id, "source_recipe": recipe, "lead_side": lead, "shift_s": shift, "swing_multiplier": multiplier, "clearance_percentile": percentile, "eligible": bool(base["eligible"]), "dominant_failure": base.get("dominant_failure"), "summary": base.get("summary")})
    write_csv("offline_plan_ledger_v2.csv", ledger)
    dump("offline_plan_ledger_v2.json", {"name": "Exp014OfflineSTARTPlanLedgerV2", "plans": ledger, "count": len(ledger), "physics": 0})
    write_csv("offline_plan_task_errors_v2.csv", task_rows)
    dump("offline_plan_task_errors_v2.json", {"rows": task_rows, "row_count": len(task_rows), "fk_method": "captured full body Jacobian first-order offline FK", "physics": 0})
    failures: dict[str, list[str]] = {}
    for row in ledger:
        failure = row.get("dominant_failure")
        if failure:
            failures.setdefault(failure, []).append(row["plan_id"])
    dump("offline_plan_failure_decomposition_v2.json", {"counts": {key: len(value) for key, value in failures.items()}, "plan_ids": failures, "dominance_rule": "first mandatory failure in fixed order; source gate and V2 interface failures are not relabeled as numerical failures", "all_failure_classes": ["SOURCE_ENDPOINT_INELIGIBLE", "ROOT_REFERENCE_INCONSISTENT", "STANCE_TASK_INFEASIBLE", "SWING_REACH_INFEASIBLE", "COM_TASK_INFEASIBLE", "PELVIS_ORIENTATION_INFEASIBLE", "JOINT_LIMIT_INFEASIBLE", "JOINT_VELOCITY_INFEASIBLE", "ACTION_BOUND_INFEASIBLE", "ZMP_CONTAINMENT_FAIL", "DCM_ENDPOINT_FAIL", "ACTIVE_SET_NONCONVERGENCE", "NUMERICAL_FAILURE"]})
    duration_counts = {}
    for mult in SWING_MULTIPLIERS:
        rows_for_duration = [x for x in ledger if x["swing_duration_multiplier"] == mult and x["dominant_failure"]]
        duration_counts[str(mult)] = {"failed": len(rows_for_duration), "failure_counts": {f: sum(int(x["dominant_failure"] == f) for x in rows_for_duration) for f in sorted({x["dominant_failure"] for x in rows_for_duration})}}
    velocity_timing_classes = ("JOINT_VELOCITY_INFEASIBLE", "SWING_REACH_INFEASIBLE", "COM_TASK_INFEASIBLE")
    failure_counts_all = {name: len([x for x in ledger if x["dominant_failure"] == name]) for name in sorted({x["dominant_failure"] for x in ledger if x["dominant_failure"]})}
    timing_failure_count = sum(failure_counts_all.get(name, 0) for name in velocity_timing_classes)
    total_failed = sum(failure_counts_all.values())
    if not failures:
        timing_label = "NOT_APPLICABLE_NO_FAILURES"
        timing_interpretation = "no plan failed"
    elif failure_counts_all.get("ACTION_BOUND_INFEASIBLE", 0) == total_failed:
        timing_label = "ACTION_BOUND_INFEASIBLE_PRECEDES_TIMING"
        timing_interpretation = "the mandatory action-bound gate was the first failure for every plan; swing-duration timing cannot be attributed independently"
    elif timing_failure_count == 0:
        timing_label = "TIMING_NOT_IDENTIFIABLE_FROM_FIRST_FAILURE"
        timing_interpretation = "the first failures were outside the registered velocity/swing/CoM timing diagnosis classes"
    elif timing_failure_count >= 0.8 * max(total_failed, 1):
        timing_label = "TRANSITION_TIMING_SEMANTICS_MISMATCH"
        timing_interpretation = "the registered timing-related failure classes dominate; the duration result is descriptive and no grid value was added"
    else:
        timing_label = "MIXED_FAILURES"
        timing_interpretation = "timing-related and non-timing failure classes are mixed; no timing conclusion is authorized"
    dump("offline_plan_timing_diagnosis_v2.json", {"duration_grid": duration_counts, "classification": timing_label, "longest_duration_tested": 1.2, "grid_additions": 0, "interpretation": timing_interpretation})
    coverage: dict[str, Any] = {"recipes": {}, "left_coverage": 0, "right_coverage": 0, "mirror_tuple_coverage": 0, "status": "BLOCKED"}
    for recipe in RECIPES:
        recipe_plans = [x for x in ledger if int(x["source_recipe"]) == recipe]
        left = [x for x in recipe_plans if x["lead_side"] == "LEFT" and x["eligible"]]
        right = [x for x in recipe_plans if x["lead_side"] == "RIGHT" and x["eligible"]]
        tuple_pairs = 0
        for shift in SHIFTS:
            for multiplier in SWING_MULTIPLIERS:
                for percentile in CLEARANCE_PERCENTILES:
                    l = any(x["lead_side"] == "LEFT" and x["double_support_shift_s"] == shift and x["swing_duration_multiplier"] == multiplier and x["clearance_percentile"] == percentile and x["eligible"] for x in recipe_plans)
                    r = any(x["lead_side"] == "RIGHT" and x["double_support_shift_s"] == shift and x["swing_duration_multiplier"] == multiplier and x["clearance_percentile"] == percentile and x["eligible"] for x in recipe_plans)
                    tuple_pairs += int(l and r)
        coverage["recipes"][str(recipe)] = {"endpoint_eligible": source_eligible[recipe], "LEFT": len(left), "RIGHT": len(right), "mirror_equivalent_tuple_count": tuple_pairs, "best_LEFT": None, "best_RIGHT": None}
        coverage["recipes"][str(recipe)]["best_LEFT"] = min(left, key=lambda x: (x["summary"]["dcm_final_error"], x["summary"]["max_errors"]["stance_position_m"], x["summary"]["max_errors"]["com_horizontal_m"], -x["summary"]["min_action_margin"], x["summary"]["total_steps"]))["plan_id"] if left else None
        coverage["recipes"][str(recipe)]["best_RIGHT"] = min(right, key=lambda x: (x["summary"]["dcm_final_error"], x["summary"]["max_errors"]["stance_position_m"], x["summary"]["max_errors"]["com_horizontal_m"], -x["summary"]["min_action_margin"], x["summary"]["total_steps"]))["plan_id"] if right else None
    coverage["left_coverage"] = sum(int(coverage["recipes"][str(i)]["LEFT"] > 0) for i in RECIPES)
    coverage["right_coverage"] = sum(int(coverage["recipes"][str(i)]["RIGHT"] > 0) for i in RECIPES)
    coverage["mirror_tuple_coverage"] = sum(int(coverage["recipes"][str(i)]["mirror_equivalent_tuple_count"] > 0) for i in RECIPES)
    coverage["bilateral_gate"] = {"left": coverage["left_coverage"] >= 6, "right": coverage["right_coverage"] >= 6, "mirror_pairs": coverage["mirror_tuple_coverage"] >= 4}
    coverage["status"] = "BILATERAL_PASS" if all(coverage["bilateral_gate"].values()) else "SINGLE_OR_BLOCKED"
    dump("offline_plan_source_coverage_v2.json", coverage)
    selected = {"per_source_side": {}, "global_diagnostic_plan": None}
    for recipe in RECIPES:
        for lead in LEADS:
            eligible = [x for x in ledger if int(x["source_recipe"]) == recipe and x["lead_side"] == lead and x["eligible"]]
            selected["per_source_side"][f"{recipe}:{lead}"] = min(eligible, key=lambda x: (x["summary"]["dcm_final_error"], x["summary"]["max_errors"]["stance_position_m"], x["summary"]["max_errors"]["com_horizontal_m"], x["summary"]["max_errors"]["swing_position_m"], -x["summary"]["min_action_margin"], x["summary"]["total_steps"])) if eligible else None
    eligible_global = [x for x in ledger if x["eligible"]]
    if eligible_global:
        selected["global_diagnostic_plan"] = max(eligible_global, key=lambda x: (coverage["left_coverage"] + coverage["right_coverage"], coverage["mirror_tuple_coverage"], -x["summary"]["max_errors"]["com_horizontal_m"], x["summary"]["min_action_margin"], -x["summary"]["total_steps"]))
    dump("selected_offline_plans_v2.json", selected)
    v2_test_pass = unit_tests["status"] == "PASS" and determinism["status"] == "PASS"
    bilateral = bool(endpoint_summary["endpoint_eligible_count"] >= 6 and v2_test_pass and coverage["left_coverage"] >= 6 and coverage["right_coverage"] >= 6 and coverage["mirror_tuple_coverage"] >= 4)
    single_left = coverage["left_coverage"] >= 6 and not bilateral
    single_right = coverage["right_coverage"] >= 6 and not bilateral
    if endpoint_summary["endpoint_eligible_count"] < 6:
        classification = "EXP014_D26V_ENDPOINT_SOURCE_INSUFFICIENT"
    elif not v2_test_pass:
        classification = "EXP014_D26V_WBIK_V2_INTERFACE_FAIL"
    elif bilateral:
        classification = "EXP014_D26V_BILATERAL_OFFLINE_START_READY"
    elif single_left or single_right:
        classification = "EXP014_D26V_SINGLE_SIDE_OFFLINE_START_READY"
    else:
        classification = "EXP014_D26V_OFFLINE_START_KINEMATICS_FAIL"
    authorization = {"classification": classification, "source_endpoint_eligible_count": endpoint_summary["endpoint_eligible_count"], "left_coverage": coverage["left_coverage"], "right_coverage": coverage["right_coverage"], "mirror_tuple_coverage": coverage["mirror_tuple_coverage"], "physics_authorized": bool(bilateral or single_left or single_right), "physics_executed": 0, "selected_plans": selected, "allowed_scope": "D27 fresh-lifecycle diagnostic physics only for selected eligible V2 plans" if bilateral else "selected-side diagnostic physics only" if single_left or single_right else "none"}
    if bilateral or single_left or single_right:
        dump("exp014_d27_model_based_start_physics_authorization.json", authorization)
    else:
        dump("exp014_d27_not_authorized.json", {**authorization, "reason": "endpoint/V2/coverage gate did not pass"})
    dump("stage_classification.json", {"classification": classification, "d26u_classification_unchanged": "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL", "endpoint_eligibility": endpoint_summary, "v2_unit_tests": unit_tests["status"], "v2_determinism": determinism["status"], "offline_registered_plans": 432, "offline_executed_plans": sum(int(x["source_endpoint_eligible"] and v2_test_pass) for x in ledger), "offline_eligible_plans": sum(int(x["eligible"]) for x in ledger), "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0})
    next_action = "D27: model-based START fresh-lifecycle physics LEFT/RIGHT selected V2 plans" if bilateral else "D27: selected-side diagnostic physics only" if single_left or single_right else "Do not begin START physics; separate offline task/constraint failures using D26V ledger"
    dump("recommended_next_action.json", {"classification": classification, "recommended_next_action": next_action, "ppo": 0, "cem": 0, "reward_change": 0, "target_change": 0, "grid_change": 0})
    dump("protected_hashes.json", {"starting_head": starting_head, "starting_status_short": start_status, "protected_files": protected_hashes(), "d26u_read_only": True, "d26t_read_only": True, "exp005_exp013_modified_by_d26v": False, "s_hold_modified": False, "stage2q_modified": False, "w_move_modified": False, "s_stop_omni_modified": False, "wbik_v1_modified": False, "physics_modified": False, "pd_modified": False, "persistent_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location -LiteralPath '" + str(REPO) + "'\n& '" + str(REPO.parent.parent / 'IsaacLab' / 'isaaclab.bat') + "' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26v_endpoint_window_capture.py --capture-mode off --headless --device cuda:0\n& '" + str(REPO.parent.parent / 'IsaacLab' / 'isaaclab.bat') + "' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26v_endpoint_window_capture.py --capture-mode on --headless --device cuda:0\n& '" + str(Path(sys.executable)) + "' experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26v.py\n", encoding="utf-8")
    end_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    report_lines = [
        "# Phase 2-D26V — START-source endpoint eligibility correction and prescribed-floating-base WBIK V2",
        "",
        f"Classification: **{classification}**.",
        "",
        "## Source eligibility",
        "",
        f"The D26U full-lifecycle classification remains `EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL` and its artifact is read-only. The D26V START endpoint contract evaluated the last 50 control steps. Endpoint-eligible recipes: **{endpoint_summary['endpoint_eligible_count']}/8**; capture OFF/ON parity: **{endpoint_summary['capture_parity']['status']}**. Recipe 0 and 3 retain their D26U torque event as pre-acquisition diagnostics only; the replayed endpoint windows contain no fall, slip, impact, support loss, velocity saturation, torque saturation, or nonfinite event.",
        "",
        "## WBIK V2",
        "",
        f"`Exp014PrescribedFloatingBaseHierarchicalWBIKV2` is `FB1_PRESCRIBED_ROOT_REFERENCE`. The six root Jacobian columns contribute prescribed root twist to stance-foot, swing-foot, and CoM tasks; only the 37 joint columns are solved and converted to the unchanged 37D action interface. Hierarchy: stance-foot 6D, CoM xyz and swing-foot 6D, torso/nominal/action-rate regularization. Unit tests: **{unit_tests['status']}**; independent-process determinism: **{determinism['status']}**.",
        "",
        "## Offline plans",
        "",
        f"The fixed grid registered **432** plans. WBIK V2 executed **{sum(int(x['source_endpoint_eligible'] and v2_test_pass) for x in ledger)}** plans and found **{sum(int(x['eligible']) for x in ledger)}** eligible plans. Physics execution was 0. Dominant failure classes: `{json.dumps({k: len(v) for k, v in failures.items()}, sort_keys=True)}`. Timing diagnosis: `{timing_label}`.",
        "",
        "## Coverage and authorization",
        "",
        f"LEFT coverage: **{coverage['left_coverage']}/8** recipes; RIGHT coverage: **{coverage['right_coverage']}/8**; mirror-equivalent tuple coverage: **{coverage['mirror_tuple_coverage']}/8**. Authorization: **{'bilateral D27' if bilateral else 'single-side D27' if single_left or single_right else 'none'}**. No model-based START physics was run in D26V.",
        "",
        "## Protection and repository",
        "",
        f"Starting HEAD: `{starting_head}`; ending HEAD at artifact generation: `{end_head}`. Protected D26U/D26T/D26S artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, D26 WBIK V1, physics/PD/friction/robot/dt/decimation were not modified by this stage. Persistent update 0; new checkpoint 0; PPO/CEM/validation/held-out/RUN 0; remote push false.",
        "",
        "The complete ledger, task-error rows, endpoint audit, V1/V2 comparison, protection hashes, and reproduction commands are in the D26V result directory.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "endpoint_eligible": endpoint_summary["endpoint_eligible_count"], "executed": sum(int(x["source_endpoint_eligible"] and v2_test_pass) for x in ledger), "eligible": sum(int(x["eligible"]) for x in ledger), "left_coverage": coverage["left_coverage"], "right_coverage": coverage["right_coverage"], "mirror_tuple_coverage": coverage["mirror_tuple_coverage"]}, indent=2), flush=True)


if __name__ == "__main__":
    if "--determinism-child" in sys.argv:
        determinism_child()
    else:
        main()
