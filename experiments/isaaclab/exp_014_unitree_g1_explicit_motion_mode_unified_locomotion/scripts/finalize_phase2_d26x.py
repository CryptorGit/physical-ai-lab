"""Phase 2-D26X geometric-path, timing, and W_MOVE entry-target audit.

This script is intentionally offline-only.  It consumes D26U/D26T/D26W
artifacts read-only, reuses the versioned D26 WBIK V2A implementation, and
writes only the D26X result directory plus the D26X research report.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26x_timing_and_target_set"
REPORT = REPO / "research/exp_014_phase_2_d26x_timing_and_target_set_report.md"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26V = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26v_endpoint_gate_and_wbik_v2"
D26W = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26w_action_semantics_and_feedforward"
D25 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher"
SOURCE = D26U / "fresh_shold_identity_complete_sources.npz"
NATIVE = D26S / "native_steady_trace_bundle.npz"
START_HEAD_REQUESTED = "003c564f77decc4550f7731499b0325809301b3d"
DT = 0.02
T_REF = 0.16
VELOCITY_RATIO_LIMIT = 0.80
SAFE_MARGIN = 1.10
ACTION_SCALE = 0.5
G = 9.81
RECIPES = list(range(8))
SIDES = ("LEFT", "RIGHT")
PHASES = ("DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
TIMING_FACTORS = {"FAST": 1.00, "NOMINAL": 1.25, "SLOW": 1.50}
TIMING_ORDER = {"FAST": 0, "NOMINAL": 1, "SLOW": 2}
HARD_MAX = {
    "DOUBLE_SUPPORT_SHIFT": 1.00,
    "FIRST_SWING": 1.00,
    "LANDING_AND_CAPTURE": 0.60,
    "WMOVE_ACCEPTANCE": 0.60,
}
TOTAL_HARD_MAX = 2.50
FOOT_BODY = {"LEFT": 24, "RIGHT": 25}
MEDOID_ROWS = {"LEFT": 8171, "RIGHT": 9330}
MEDOID_EPISODES = {"LEFT": 52, "RIGHT": 187}
MEDOID_STEPS = {"LEFT": 111, "RIGHT": 115}
CLEARANCE_P50 = 50
CLEARANCE_P75 = 75


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Importing the existing finalizer is read-only; its main() is not called.
d26v = load_module(
    "exp014_d26x_d26v_read_only",
    EXP / "scripts/finalize_phase2_d26v.py",
)
wbik_v2 = d26v.wbik_v2


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(
        json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(to_jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else to_jsonable(value) for key, value in row.items()})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def source_contract() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    contract = read_json(D25 / "model_based_teacher_robot_contract.json")
    default_q = np.asarray(contract["action_interface"]["offset"][0], dtype=np.float64)
    action_scale = np.full(37, ACTION_SCALE, dtype=np.float64)
    return contract, default_q, action_scale


def joint_group(name: str) -> str:
    lower = name.lower()
    side = "left" if lower.startswith("left_") else "right"
    if any(token in lower for token in ("hip", "knee", "ankle")):
        return f"{side} leg"
    if "torso" in lower:
        return "waist"
    if any(token in lower for token in ("shoulder", "elbow")):
        return f"{side} arm"
    return f"{side} wrist/hand"


def joint_index_contract(contract: dict[str, Any], default_q: np.ndarray, action_scale: np.ndarray, source: dict[str, np.ndarray]) -> dict[str, Any]:
    names = contract["joint_names"]
    rows = []
    for action_index, name in enumerate(names):
        rows.append(
            {
                "action_index": action_index,
                "asset_joint_index": action_index,
                "joint_name": name,
                "joint_group": joint_group(name),
                "velocity_limit_rad_s": float(np.median(source["joint_velocity_limits"][:, action_index])),
                "position_limit_rad": [float(np.min(source["joint_position_limits"][:, action_index, 0])), float(np.max(source["joint_position_limits"][:, action_index, 1]))],
                "default_q_rad": float(default_q[action_index]),
                "action_scale": float(action_scale[action_index]),
                "asset_order_source": "D25 model_based_teacher_robot_contract.joint_names",
            }
        )
    return {
        "name": "Exp014D26XJointIndexNameContractV1",
        "dimension": 37,
        "runtime_action_contract": "q_cmd = default_q + 0.5 * raw_action; canonical action bound is unbounded",
        "position_limit_source": "D26U captured per-recipe asset limits; no limit changed",
        "velocity_limit_source": "D26U captured per-recipe asset limits; no limit changed",
        "joint_groups": ["left leg", "right leg", "waist", "left arm", "right arm", "left wrist/hand", "right wrist/hand"],
        "joints": rows,
    }


def protected_file_candidates() -> list[Path]:
    tracked = git("ls-files").splitlines()
    result: list[Path] = []
    exp_patterns = tuple(f"exp_{i:03d}_" for i in range(5, 14))
    for relative in tracked:
        lower = relative.lower().replace("\\", "/")
        include = lower.startswith("results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d")
        # D26X is the new output stage and must not enter its own ending hash
        # set as a protected pre-stage artifact.
        if "/phase_2_d26x_" in lower:
            include = False
        if include:
            match = re.search(r"/phase_2_d(\d+)[a-z]?/", lower + "/")
            include = bool(match and 6 <= int(match.group(1)) <= 26)
        include = include or any(pattern in lower for pattern in exp_patterns)
        include = include or any(token in lower for token in ("s_hold", "stage_2q", "w_move", "s_stop_omni"))
        include = include or lower.endswith("/src/g1_explicit_motion_mode/wbik.py")
        include = include or lower.endswith("/src/g1_explicit_motion_mode/wbik_v2.py")
        include = include or lower.endswith("model_based_teacher_robot_contract.json")
        if include:
            path = REPO / relative
            if path.exists():
                result.append(path)
    # The large protected bundles are tracked, but these explicit paths make
    # the protection manifest auditable even if a future path filter changes.
    for path in (SOURCE, NATIVE, D26T / "entry_neighborhood_manifest.json", D26W / "stage_classification.json"):
        if path.exists() and path not in result:
            result.append(path)
    return sorted(set(result))


def protected_snapshot() -> dict[str, Any]:
    paths = protected_file_candidates()
    return {
        "file_count": len(paths),
        "files": {str(path.relative_to(REPO)).replace("\\", "/"): sha256_file(path) for path in paths},
    }


def load_wmove_geometry() -> dict[str, Any]:
    return read_json(D26U / "wmove_transition_geometry_v1.json")


def clearance_value(geometry: dict[str, Any], percentile: int) -> float:
    return float(geometry["aggregate_statistics"]["swing_clearance_m"][f"p{percentile:02d}"])


def parse_plan_meta(plan_id: str) -> dict[str, Any]:
    match = re.fullmatch(r"D26V_R(\d+)_([A-Z]+)_SHIFT([0-9.]+)_SWING([0-9.]+)_C(\d+)", plan_id)
    if not match:
        raise ValueError(f"unrecognized D26 plan id: {plan_id}")
    return {
        "plan_id": plan_id,
        "source_recipe": int(match.group(1)),
        "lead_side": match.group(2),
        "shift_s": float(match.group(3)),
        "swing_multiplier": float(match.group(4)),
        "clearance_percentile": int(match.group(5)),
    }


def phase_lower_bounds(meta: dict[str, Any]) -> dict[str, float]:
    swing = float(meta["swing_multiplier"]) * T_REF
    return {
        "DOUBLE_SUPPORT_SHIFT": float(meta["shift_s"]),
        "FIRST_SWING": swing,
        "LANDING_AND_CAPTURE": max(4 * DT, 0.50 * swing),
        "WMOVE_ACCEPTANCE": max(5 * DT, 0.50 * T_REF),
    }


def target_row_id(native: dict[str, np.ndarray], row: int, side: str) -> str:
    rows = read_json(D26T / "entry_neighborhood_manifest.json")["references"]
    for item in rows:
        if int(item["bundle_row"]) == int(row):
            return str(item["reference_id"])
    return f"{side}_{int(row):05d}"


def target_metadata(native: dict[str, np.ndarray], row: int, side: str) -> dict[str, Any]:
    manifest = read_json(D26T / "entry_neighborhood_manifest.json")
    for item in manifest["references"]:
        if int(item["bundle_row"]) == int(row):
            return {
                "target_id": item["reference_id"],
                "side": item["side"],
                "bundle_row": int(item["bundle_row"]),
                "episode_id": int(item["episode_id"]),
                "control_step": int(item["control_step"]),
            }
    return {"target_id": target_row_id(native, row, side), "side": side, "bundle_row": int(row), "episode_id": None, "control_step": None}


def aligned_target_for_row(source: dict[str, np.ndarray], source_i: int, native: dict[str, np.ndarray], lead: str, target_i: int) -> dict[str, Any]:
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    target_root = np.asarray(native["root_pose"][target_i], dtype=np.float64)
    source_position = source_root[:3]
    source_rotation = d26v.quat_matrix(source_root[3:])
    target_position = target_root[:3]
    target_rotation = d26v.quat_matrix(target_root[3:])
    aligned_root_rotation = d26v.heading_aligned_rotation(target_rotation, source_rotation)
    source_com = np.asarray(source["com_position_w"][source_i], dtype=np.float64)
    target_com = np.asarray(native["com_position"][target_i], dtype=np.float64)
    target_com_rel = target_rotation.T @ (target_com - target_position)
    aligned_com = source_position + source_rotation @ target_com_rel
    source_com_velocity = np.asarray(source["com_velocity_w"][source_i], dtype=np.float64)
    target_com_velocity = source_rotation @ (target_rotation.T @ np.asarray(native["com_velocity"][target_i], dtype=np.float64))
    source_dcm = np.asarray(source["dcm"][source_i], dtype=np.float64)
    target_dcm = np.asarray(native["dcm"][target_i], dtype=np.float64)
    aligned_dcm = source_position[:2] + source_rotation[:2, :2] @ (target_dcm - target_position[:2])
    source_foot_positions = np.asarray(source["left_right_foot_pose"][source_i, :, :3], dtype=np.float64)
    source_foot_rotations = [d26v.quat_matrix(np.asarray(source["body_quat_w"][source_i, body], dtype=np.float64)) for body in (24, 25)]
    target_foot_positions = np.asarray(native["body_pos_w"][target_i, [24, 25]], dtype=np.float64)
    target_foot_rotations = [d26v.quat_matrix(np.asarray(native["body_quat_w"][target_i, body], dtype=np.float64)) for body in (24, 25)]
    aligned_foot_positions: list[np.ndarray] = []
    aligned_foot_rotations: list[np.ndarray] = []
    for position, rotation in zip(target_foot_positions, target_foot_rotations):
        rel_position = target_rotation.T @ (position - target_position)
        rel_rotation = target_rotation.T @ rotation
        aligned_foot_positions.append(source_position + source_rotation @ rel_position)
        aligned_foot_rotations.append(source_rotation @ rel_rotation)
    target_torso_rotation = d26v.quat_matrix(np.asarray(native["body_quat_w"][target_i, 4], dtype=np.float64))
    aligned_torso_rotation = source_rotation @ (target_rotation.T @ target_torso_rotation)
    lead_index = 0 if lead == "LEFT" else 1
    stance_index = 1 - lead_index
    source_dcm_offset = source_dcm - source_foot_positions[stance_index, :2]
    aligned_foot_array = np.asarray(aligned_foot_positions, dtype=np.float64)
    target_dcm_offset = aligned_dcm - aligned_foot_array[lead_index, :2]
    metadata = target_metadata(native, target_i, lead)
    metadata["touchdown_side"] = lead
    return {
        "lead_side": lead,
        "target_family": f"{lead}_POST_TOUCHDOWN",
        "target_id": metadata["target_id"],
        "target_medoid": {"episode_id": metadata["episode_id"], "control_step": metadata["control_step"], "bundle_row": metadata["bundle_row"], "touchdown_side": lead},
        "source_root_pose": source_root.tolist(),
        "source_root_to_com_m": (source_com - source_position).tolist(),
        "target_root_pose_aligned_position": (aligned_com - aligned_root_rotation @ target_com_rel).tolist(),
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
        "target_foot_positions_aligned": aligned_foot_array.tolist(),
        "target_foot_rotations_aligned": [x.tolist() for x in aligned_foot_rotations],
        "target_torso_rotation_aligned": aligned_torso_rotation.tolist(),
        "source_support_configuration": np.asarray(source["support_state"][source_i], dtype=np.int64).tolist(),
        "target_support_configuration": (np.linalg.norm(native["contact_force"][target_i], axis=1) > 5.0).astype(int).tolist(),
        "required_foot_displacement_m": (aligned_foot_array[lead_index] - source_foot_positions[lead_index]).tolist(),
        "required_com_displacement_m": (aligned_com - source_com).tolist(),
        "required_pelvis_translation_m": (aligned_com - aligned_root_rotation @ target_com_rel - source_position).tolist(),
        "required_pelvis_orientation_rad": d26v.so3_log_np(aligned_root_rotation @ source_rotation.T).tolist(),
        "source_target_joint_distance_l2": float(np.linalg.norm(native["joint_pos"][target_i] - source["joint_pos"][source_i])),
        "source_target_action_distance_l2": float(np.linalg.norm(native["next_action"][target_i] - source["current_action"][source_i])),
        "source_target_com_height_difference_m": float(aligned_com[2] - source_com[2]),
        "source_target_dcm_difference_world_m": (aligned_dcm - source_dcm).tolist(),
        "target_native_step_geometry": {"step_length_m": None, "step_width_m": None},
        "target_preserved_without_averaging": True,
    }


def source_support_anchor(source: dict[str, np.ndarray], source_i: int) -> np.ndarray:
    positions = np.asarray(source["left_right_foot_pose"][source_i, :, :3], dtype=np.float64)
    support = np.asarray(source["support_state"][source_i], dtype=np.float64) > 0
    return np.mean(positions[support], axis=0) if np.any(support) else np.mean(positions, axis=0)


def target_support_anchor(target: dict[str, Any]) -> np.ndarray:
    positions = np.asarray(target["target_foot_positions_aligned"], dtype=np.float64)
    support = np.asarray(target["target_support_configuration"], dtype=np.float64) > 0
    return np.mean(positions[support], axis=0) if np.any(support) else np.mean(positions, axis=0)


def q_cmd_from_action(action: np.ndarray, default_q: np.ndarray) -> np.ndarray:
    return default_q + ACTION_SCALE * np.asarray(action, dtype=np.float64)


def compatibility_raw_features(source: dict[str, np.ndarray], source_i: int, native: dict[str, np.ndarray], target: dict[str, Any], target_i: int, default_q: np.ndarray) -> dict[str, float]:
    source_anchor = source_support_anchor(source, source_i)
    target_anchor = target_support_anchor(target)
    source_com_rel = np.asarray(source["com_position_w"][source_i], dtype=np.float64) - source_anchor
    target_com_rel = np.asarray(target["target_com_position_aligned"], dtype=np.float64) - target_anchor
    source_dcm_rel = np.asarray(source["dcm"][source_i], dtype=np.float64) - source_anchor[:2]
    target_dcm_rel = np.asarray(target["target_dcm_xy_aligned"], dtype=np.float64) - target_anchor[:2]
    source_qcmd = q_cmd_from_action(source["current_action"][source_i], default_q)
    target_qcmd = q_cmd_from_action(native["current_action"][target_i], default_q)
    source_rotation = d26v.quat_matrix(source["root_pose"][source_i, 3:])
    target_rotation = np.asarray(target["target_root_rotation_aligned"], dtype=np.float64)
    root_orientation = d26v.so3_log_np(target_rotation @ source_rotation.T)
    return {
        "joint_position_distance": float(np.linalg.norm(native["joint_pos"][target_i] - source["joint_pos"][source_i])),
        "joint_velocity_distance": float(np.linalg.norm(native["joint_vel"][target_i] - source["joint_vel"][source_i])),
        "policy_command_q_cmd_distance": float(np.linalg.norm(target_qcmd - source_qcmd)),
        "previous_action_distance": float(np.linalg.norm(native["previous_action"][target_i] - source["previous_action"][source_i])),
        "com_relative_to_support_foot_distance": float(np.linalg.norm(target_com_rel - source_com_rel)),
        "com_velocity_distance": float(np.linalg.norm(np.asarray(target["target_com_velocity_aligned"]) - source["com_velocity_w"][source_i])),
        "dcm_offset_distance": float(np.linalg.norm(target_dcm_rel - source_dcm_rel)),
        "foot_placement_displacement": float(np.linalg.norm(np.asarray(target["target_foot_positions_aligned"])[0 if target["lead_side"] == "LEFT" else 1] - np.asarray(target["source_foot_positions"])[0 if target["lead_side"] == "LEFT" else 1])),
        "root_orientation_distance": float(np.linalg.norm(root_orientation)),
    }


FEATURES = (
    "joint_position_distance",
    "joint_velocity_distance",
    "policy_command_q_cmd_distance",
    "previous_action_distance",
    "com_relative_to_support_foot_distance",
    "com_velocity_distance",
    "dcm_offset_distance",
    "foot_placement_displacement",
    "root_orientation_distance",
)


def robust_scales(values: dict[str, list[float]]) -> dict[str, Any]:
    result = {}
    for name, array in values.items():
        x = np.asarray(array, dtype=np.float64)
        med = float(np.median(x))
        mad = float(1.4826 * np.median(np.abs(x - med)))
        iqr = float((np.quantile(x, 0.75) - np.quantile(x, 0.25)) / 1.349)
        scale = mad if mad > 1.0e-9 else iqr if iqr > 1.0e-9 else float(np.std(x)) if float(np.std(x)) > 1.0e-9 else 1.0
        result[name] = {"center_median": med, "scale": scale, "method": "MAD_1.4826_then_IQR_1.349_then_STD", "sample_count": int(x.size), "source": "D26U train-only sources plus D26T fresh validated train-only entry references; no replay outcome used"}
    return result


def build_targets_and_compatibility(source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray, geometry: dict[str, Any]) -> tuple[dict[tuple[int, str, str], dict[str, Any]], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(D26T / "entry_neighborhood_manifest.json")
    refs = [item for item in manifest["references"] if item["side"] in SIDES]
    targets: dict[tuple[int, str, str], dict[str, Any]] = {}
    raw_records: list[dict[str, Any]] = []
    scale_values = {name: [] for name in FEATURES}
    for recipe in RECIPES:
        for side in SIDES:
            for item in refs:
                if item["side"] != side:
                    continue
                target_i = int(item["bundle_row"])
                target = aligned_target_for_row(source, recipe, native, side, target_i)
                targets[(recipe, side, item["reference_id"])] = target
                raw = compatibility_raw_features(source, recipe, native, target, target_i, default_q)
                for name in FEATURES:
                    scale_values[name].append(raw[name])
                raw_records.append({"source_recipe": recipe, "lead_side": side, "target_id": item["reference_id"], "target_bundle_row": target_i, "target_episode_id": int(item["episode_id"]), "target_control_step": int(item["control_step"]), "raw_features": raw})
    scales = robust_scales(scale_values)
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        normalized = {name: record["raw_features"][name] / scales[name]["scale"] for name in FEATURES}
        score = float(sum(normalized.values()))
        row = {**record, "normalized_features": normalized, "compatibility_score": score, "scale_contract": "source/validated-target train-only robust scales; result labels excluded"}
        rows.append(row)
        by_key[(record["source_recipe"], record["lead_side"])].append(row)
    for key, items in by_key.items():
        items.sort(key=lambda row: (row["compatibility_score"], row["target_id"]))
        for rank, item in enumerate(items, start=1):
            item["compatibility_rank"] = rank
    medoid_ids = {}
    for side in SIDES:
        medoid_ids[side] = target_row_id(native, MEDOID_ROWS[side], side)
    shortlist: dict[str, Any] = {"name": "WMove03ValidatedEntryTargetShortlistV1", "selection_frozen_before_replay": True, "selection_rule": "five minimum compatibility scores per source/side, plus exact native medoid control if outside those five; no success/failure label or future action used", "max_targets_per_source_side": 6, "entries": {}}
    for recipe in RECIPES:
        for side in SIDES:
            items = by_key[(recipe, side)]
            top = items[:5]
            medoid_id = medoid_ids[side]
            if medoid_id not in {item["target_id"] for item in top}:
                medoid_item = next(item for item in items if item["target_id"] == medoid_id)
                top = top + [medoid_item]
            key = f"R{recipe:02d}_{side}"
            shortlist["entries"][key] = [{"target_id": item["target_id"], "bundle_row": item["target_bundle_row"], "compatibility_rank": item["compatibility_rank"], "compatibility_score": item["compatibility_score"], "medoid_control": item["target_id"] == medoid_id} for item in top]
    compat_summary = {"name": "Exp014SourceTargetCompatibilityV1", "features": list(FEATURES), "scales": scales, "target_count_per_side": {side: sum(1 for item in refs if item["side"] == side) for side in SIDES}, "source_count": 8, "rows": rows}
    return targets, compat_summary, shortlist, rows


def validation_manifest(native: dict[str, np.ndarray]) -> dict[str, Any]:
    manifest = read_json(D26T / "entry_neighborhood_manifest.json")
    validation = read_json(D26T / "entry_reference_validation.json")
    replay = read_json(D26T / "entry_neighborhood_replay.json")
    refs = []
    for item in manifest["references"]:
        replay_item = next((row for row in replay["references"] if row["reference_id"] == item["reference_id"]), None)
        refs.append({
            "target_id": item["reference_id"],
            "side": item["side"],
            "bundle_row": int(item["bundle_row"]),
            "episode_id": int(item["episode_id"]),
            "control_step": int(item["control_step"]),
            "validated_in_d26t": bool(replay_item and replay_item["tracking_window_pass"] and replay_item["identity_match"] and replay_item["phase_retained"] and not replay_item["safety_bad"]),
            "tracking_window_pass": None if replay_item is None else bool(replay_item["tracking_window_pass"]),
            "identity_match": None if replay_item is None else bool(replay_item["identity_match"]),
            "phase_retained": None if replay_item is None else bool(replay_item["phase_retained"]),
        })
    return {
        "name": "WMove03ValidatedEntryTargetSetV1",
        "source": "D26T entry_neighborhood_manifest.json and entry_neighborhood_replay.json read-only",
        "native_bundle_sha256": sha256_file(NATIVE),
        "validation_artifact_sha256": sha256_file(D26T / "entry_reference_validation.json"),
        "fresh_replay_validation": {"status": validation["status"], "references": validation["references"], "identity_replay_rate": validation["identity_replay_rate"], "tracking_window_retention": validation["tracking_window_retention"], "gate": validation["gate"]},
        "references_per_side": {side: sum(1 for row in refs if row["side"] == side and row["validated_in_d26t"]) for side in SIDES},
        "references": refs,
        "new_state_created": False,
        "average_or_interpolated_state_created": False,
    }


def velocity_decomposition(source: dict[str, np.ndarray], contract: dict[str, Any], d26v_rows: list[dict[str, Any]], d26w_ledger: list[dict[str, str]], medoid_ids: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = contract["joint_names"]
    by_plan: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in d26v_rows:
        by_plan[row["plan_id"]].append(row)
    ledger_by_plan = {row["plan_id"]: row for row in d26w_ledger}
    plan_rows: list[dict[str, Any]] = []
    counters = {"side": Counter(), "source": Counter(), "phase": Counter(), "joint": Counter(), "group": Counter(), "duration": Counter(), "clearance": Counter(), "severity": Counter()}
    for plan_id, rows in by_plan.items():
        meta = parse_plan_meta(plan_id)
        source_i = meta["source_recipe"]
        limits = np.asarray(source["joint_velocity_limits"][source_i], dtype=np.float64)
        first = next((row for row in rows if float(row["planned_joint_velocity_ratio_max"]) > VELOCITY_RATIO_LIMIT + 1.0e-9), None)
        if first is None:
            record = {**meta, "first_velocity_violation": False, "first_velocity_violation_step": None, "phase": None, "violating_joints": [], "max_velocity_ratio": max(float(row["planned_joint_velocity_ratio_max"]) for row in rows), "target_medoid": {"target_id": medoid_ids[meta["lead_side"]], "bundle_row": MEDOID_ROWS[meta["lead_side"]], "episode_id": MEDOID_EPISODES[meta["lead_side"]], "control_step": MEDOID_STEPS[meta["lead_side"]]}}
            plan_rows.append(record)
            continue
        dq = np.asarray(first["dq_des"], dtype=np.float64)
        ratio = np.abs(dq) / np.maximum(limits, 1.0e-12)
        indices = [int(index) for index in np.where(ratio > VELOCITY_RATIO_LIMIT + 1.0e-9)[0]]
        max_ratio = float(np.max(ratio))
        severity = "MILD" if max_ratio <= 1.0 else "MODERATE" if max_ratio <= 2.0 else "SEVERE" if max_ratio <= 4.0 else "EXTREME"
        violations = []
        for index in indices:
            violations.append({"joint_index": index, "joint_name": names[index], "joint_group": joint_group(names[index]), "required_dq_rad_s": float(dq[index]), "velocity_limit_rad_s": float(limits[index]), "velocity_ratio": float(ratio[index]), "authorized_ratio": VELOCITY_RATIO_LIMIT})
            counters["joint"][names[index]] += 1
            counters["group"][joint_group(names[index])] += 1
        counters["side"][meta["lead_side"]] += 1
        counters["source"][str(source_i)] += 1
        counters["phase"][first["phase"]] += 1
        counters["duration"][str(meta["swing_multiplier"])] += 1
        counters["clearance"][str(meta["clearance_percentile"])] += 1
        counters["severity"][severity] += 1
        wrow = ledger_by_plan[plan_id]
        summary = json.loads(wrow["summary"])
        plan_rows.append({
            **meta,
            "first_velocity_violation": True,
            "first_velocity_violation_step": int(first["step"]),
            "phase": first["phase"],
            "violating_joints": violations,
            "max_velocity_ratio": max_ratio,
            "severity": severity,
            "task_errors_at_first_violation": json.loads(first.get("task_errors", "{}")) if isinstance(first.get("task_errors"), str) else first.get("task_errors", {}),
            "summary_nonvelocity_gate_snapshot": {"ik_solution_rate_d26w": float(summary["ik_solution_rate"]), "dcm_endpoint_pass": bool(summary["dcm_endpoint_pass"]), "max_errors": summary["max_errors"]},
            "target_medoid": {"target_id": medoid_ids[meta["lead_side"]], "bundle_row": MEDOID_ROWS[meta["lead_side"]], "episode_id": MEDOID_EPISODES[meta["lead_side"]], "control_step": MEDOID_STEPS[meta["lead_side"]]},
        })
    plan_rows.sort(key=lambda row: row["plan_id"])
    aggregate = {
        "name": "Exp014D26XVelocityFailureDecompositionV1",
        "trace_source": "D26W V2A task trace plus protected D26V dq_des rows; all read-only",
        "plan_count": len(plan_rows),
        "plan_count_with_velocity_violation": sum(bool(row["first_velocity_violation"]) for row in plan_rows),
        "per_side": dict(counters["side"]),
        "per_source": dict(counters["source"]),
        "per_phase": dict(counters["phase"]),
        "per_joint": dict(counters["joint"]),
        "per_joint_group": dict(counters["group"]),
        "per_swing_duration_multiplier": dict(counters["duration"]),
        "per_clearance_percentile": dict(counters["clearance"]),
        "severity": dict(counters["severity"]),
        "authorized_velocity_ratio": VELOCITY_RATIO_LIMIT,
        "classification_unchanged": "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE",
        "rows": plan_rows,
    }
    csv_rows = []
    for row in plan_rows:
        violations = row.get("violating_joints", [])
        csv_rows.append({
            "plan_id": row["plan_id"], "source_recipe": row["source_recipe"], "lead_side": row["lead_side"], "shift_s": row["shift_s"], "swing_multiplier": row["swing_multiplier"], "clearance_percentile": row["clearance_percentile"], "first_velocity_violation_step": row["first_velocity_violation_step"], "phase": row["phase"], "violating_joint_indices": [item["joint_index"] for item in violations], "violating_joint_names": [item["joint_name"] for item in violations], "violating_joint_groups": sorted({item["joint_group"] for item in violations}), "max_velocity_ratio": row["max_velocity_ratio"], "required_dq_rad_s": {item["joint_name"]: item["required_dq_rad_s"] for item in violations}, "velocity_limits_rad_s": {item["joint_name"]: item["velocity_limit_rad_s"] for item in violations}, "severity": row.get("severity"), "target_medoid": row["target_medoid"]["target_id"]})
    return csv_rows, aggregate


def task_errors_from_state(next_state: dict[str, Any], refs: dict[str, Any], step: int, lead: str) -> dict[str, float]:
    stance_index = FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"]
    swing_index = FOOT_BODY[lead]
    feet = refs["foot_refs"][step]
    return {
        "stance_position_m": float(np.linalg.norm(next_state["body_position"][stance_index] - np.asarray(feet["stance_position"]))),
        "stance_rotation_rad": float(np.linalg.norm(d26v.so3_log_np(np.asarray(feet["stance_rotation"]) @ next_state["body_rotation"][stance_index].T))),
        "swing_position_m": float(np.linalg.norm(next_state["body_position"][swing_index] - np.asarray(feet["swing_position"]))),
        "swing_rotation_rad": float(np.linalg.norm(d26v.so3_log_np(np.asarray(feet["swing_rotation"]) @ next_state["body_rotation"][swing_index].T))),
        "com_horizontal_m": float(np.linalg.norm(next_state["com_position"][:2] - refs["com_position"][step, :2])),
        "com_xyz_m": float(np.linalg.norm(next_state["com_position"] - refs["com_position"][step])),
        "root_reference_position_m": float(np.linalg.norm(next_state["root_position"] - refs["root_position"][step])),
        "pelvis_roll_pitch_rad": float(np.linalg.norm(d26v.so3_log_np(refs["root_rotation"][step] @ next_state["root_rotation"].T)[:2])),
    }


def mandatory_nonvelocity_errors(max_errors: dict[str, float]) -> tuple[bool, list[str]]:
    failures = []
    if max_errors["stance_position_m"] > 0.005 or max_errors["stance_rotation_rad"] > 0.03:
        failures.append("STANCE_TASK_INFEASIBLE")
    if max_errors["swing_position_m"] > 0.010 or max_errors["swing_rotation_rad"] > 0.03:
        failures.append("SWING_REACH_INFEASIBLE")
    if max_errors["com_horizontal_m"] > 0.010:
        failures.append("COM_TASK_INFEASIBLE")
    if max_errors["root_reference_position_m"] > 0.005:
        failures.append("ROOT_REFERENCE_INCONSISTENT")
    if max_errors["pelvis_roll_pitch_rad"] > 0.03:
        failures.append("PELVIS_ORIENTATION_INFEASIBLE")
    return not failures, failures


def target_dcm_bound(side: str) -> float:
    stats = read_json(D26T / "entry_distribution_statistics.json")["sides"][side]["dcm_offset"]
    return max(abs(float(stats[axis]["p05"])) for axis in ("x", "y")) if False else max(abs(float(stats[axis][percentile])) for axis in ("x", "y") for percentile in ("p05", "p95"))


def custom_references(source: dict[str, np.ndarray], source_i: int, target: dict[str, Any], phase_durations: dict[str, float], geometry: dict[str, Any], clearance: float, base_phase_durations: dict[str, float] | None = None) -> dict[str, Any]:
    """Regenerate dynamics on a fixed geometric path.

    The path coordinate is inherited from the original geometric phase
    partition.  Custom durations only change the time samples and all
    velocities are recomputed from those samples; CoM/DCM/root references are
    therefore not a stretched q trace.
    """
    base = base_phase_durations or phase_durations
    base_total = float(sum(base[p] for p in PHASES))
    base_cumulative = {}
    cursor = 0.0
    for phase in PHASES:
        base_cumulative[phase] = cursor
        cursor += base[phase]
    counts = {phase: max(1, int(math.ceil(float(phase_durations[phase]) / DT - 1.0e-12))) for phase in PHASES}
    actual_durations = {phase: counts[phase] * DT for phase in PHASES}
    phase_names = [phase for phase in PHASES for _ in range(counts[phase])]
    total_steps = len(phase_names)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    source_root_position = source_root[:3]
    source_root_rotation = d26v.quat_matrix(source_root[3:])
    source_com = np.asarray(source["com_position_w"][source_i], dtype=np.float64)
    source_com_velocity = np.asarray(source["com_velocity_w"][source_i], dtype=np.float64)
    target_com = np.asarray(target["target_com_position_aligned"], dtype=np.float64)
    target_com_velocity = np.asarray(target["target_com_velocity_aligned"], dtype=np.float64)
    root_end = np.asarray(target["target_root_pose_aligned_position"], dtype=np.float64)
    root_end_rotation = np.asarray(target["target_root_rotation_aligned"], dtype=np.float64)
    source_foot = np.asarray(target["source_foot_positions"], dtype=np.float64)
    source_foot_rot = [np.asarray(item, dtype=np.float64) for item in target["source_foot_rotations"]]
    target_foot = np.asarray(target["target_foot_positions_aligned"], dtype=np.float64)
    target_foot_rot = [np.asarray(item, dtype=np.float64) for item in target["target_foot_rotations_aligned"]]
    lead_index = 0 if target["lead_side"] == "LEFT" else 1
    stance_index = 1 - lead_index
    omega = math.sqrt(G / max(float(source_com[2]), 1.0e-6))
    positions = []
    rotations = []
    com_positions = []
    foot_refs = []
    global_s = []
    local_s = []
    for step, phase in enumerate(phase_names):
        phase_step = sum(1 for previous in phase_names[:step] if previous == phase) + 1
        local = phase_step / counts[phase]
        u = (base_cumulative[phase] + base[phase] * local) / base_total
        alpha = d26v.minimum_jerk(u)
        com_position, _ = d26v.hermite(source_com, target_com, source_com_velocity, target_com_velocity, u, base_total)
        root_offset = (1.0 - alpha) * (source_com - source_root_position) + alpha * (target_com - root_end)
        root_position = com_position - root_offset
        root_rotation = d26v.rotation_trajectory(source_root_rotation, root_end_rotation, u)
        if phase == "DOUBLE_SUPPORT_SHIFT":
            swing_alpha = 0.0
        elif phase == "FIRST_SWING":
            swing_alpha = d26v.minimum_jerk(local)
        else:
            swing_alpha = 1.0
        swing_position = (1.0 - swing_alpha) * source_foot[lead_index] + swing_alpha * target_foot[lead_index]
        if phase == "FIRST_SWING":
            swing_position = swing_position.copy()
            swing_position[2] = (1.0 - swing_alpha) * source_foot[lead_index, 2] + swing_alpha * target_foot[lead_index, 2] + clearance * math.sin(math.pi * np.clip(local, 0.0, 1.0))
        swing_rotation = d26v.rotation_trajectory(source_foot_rot[lead_index], target_foot_rot[lead_index], swing_alpha)
        stance_position = (1.0 - alpha) * source_foot[stance_index] + alpha * target_foot[stance_index]
        stance_rotation = d26v.rotation_trajectory(source_foot_rot[stance_index], target_foot_rot[stance_index], alpha)
        positions.append(root_position)
        rotations.append(root_rotation)
        com_positions.append(com_position)
        foot_refs.append({"stance_position": stance_position, "stance_rotation": stance_rotation, "swing_position": swing_position, "swing_rotation": swing_rotation})
        global_s.append(u)
        local_s.append(local)
    root_positions = np.asarray(positions, dtype=np.float64)
    root_rotations = np.asarray(rotations, dtype=np.float64)
    com_positions_array = np.asarray(com_positions, dtype=np.float64)
    root_velocities = []
    com_velocities = []
    for index in range(total_steps):
        previous_root_position = source_root_position if index == 0 else root_positions[index - 1]
        previous_root_rotation = source_root_rotation if index == 0 else root_rotations[index - 1]
        previous_com = source_com if index == 0 else com_positions_array[index - 1]
        root_linear = (root_positions[index] - previous_root_position) / DT
        root_angular = d26v.so3_log_np(root_rotations[index] @ previous_root_rotation.T) / DT
        root_velocities.append(np.concatenate((root_linear, root_angular)))
        com_velocities.append((com_positions_array[index] - previous_com) / DT)
    com_velocities_array = np.asarray(com_velocities, dtype=np.float64)
    dcm = com_positions_array[:, :2] + com_velocities_array[:, :2] / omega
    polygon_xy = np.asarray([[-0.101554609, -0.032734622], [0.101554609, -0.032734622], [0.101554609, 0.032734622], [-0.101554609, 0.032734622]], dtype=np.float64)
    zmp_refs = []
    zmp_inside = []
    swing_velocities = []
    for index, phase in enumerate(phase_names):
        feet = foot_refs[index]
        double_support = phase in ("DOUBLE_SUPPORT_SHIFT", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE")
        centers = [feet["stance_position"], feet["swing_position"]] if double_support else [feet["stance_position"]]
        rotations_for_polygon = [feet["stance_rotation"], feet["swing_rotation"]] if double_support else [feet["stance_rotation"]]
        polygons = [d26v.foot_polygon(center, rotation, polygon_xy) for center, rotation in zip(centers, rotations_for_polygon)]
        support_polygon = d26v.convex_hull(np.concatenate(polygons, axis=0))
        zmp = np.mean(np.asarray(centers)[:, :2], axis=0)
        zmp_refs.append(zmp)
        zmp_inside.append(d26v.polygon_contains(zmp, support_polygon))
        previous_swing = source_foot[lead_index] if index == 0 else foot_refs[index - 1]["swing_position"]
        swing_velocities.append((feet["swing_position"] - previous_swing) / DT)
    return {
        "phase_names": phase_names,
        "phase_lengths": counts,
        "phase_durations_requested_s": {phase: float(phase_durations[phase]) for phase in PHASES},
        "phase_durations_actual_s": actual_durations,
        "total_steps": total_steps,
        "total_duration_s": float(sum(actual_durations.values())),
        "global_s": np.asarray(global_s),
        "local_s": np.asarray(local_s),
        "com_position": com_positions_array,
        "com_velocity": com_velocities_array,
        "dcm": dcm,
        "root_position": root_positions,
        "root_rotation": root_rotations,
        "root_velocity": np.asarray(root_velocities),
        "swing_velocity": np.asarray(swing_velocities),
        "foot_refs": foot_refs,
        "zmp": np.asarray(zmp_refs),
        "zmp_inside": zmp_inside,
        "target_dcm": np.asarray(target["target_dcm_xy_aligned"], dtype=np.float64),
        "clearance_m": float(clearance),
        "omega": omega,
        "contract": {"name": "Exp014ModelBasedStartTimingV2", "path_coordinate": "normalized geometric u in [0,1]", "phase_durations": {phase: float(phase_durations[phase]) for phase in PHASES}, "base_geometric_phase_durations": {phase: float(base[phase]) for phase in PHASES}, "cop_zmp_regenerated": True, "lipm_com_regenerated": True, "dcm_regenerated": True, "prescribed_root_regenerated": True, "swing_foot_minimum_jerk_regenerated": True},
    }


def reference_for_timed_step(refs: dict[str, Any], step: int, target: dict[str, Any], source: dict[str, np.ndarray], source_i: int) -> dict[str, torch.Tensor]:
    feet = refs["foot_refs"][step]
    return {
        "root_pose": d26v.torch_reference(np.concatenate((refs["root_position"][step], d26v.matrix_quat(refs["root_rotation"][step])))),
        "root_velocity": d26v.torch_reference(refs["root_velocity"][step]),
        "stance_position": d26v.torch_reference(feet["stance_position"]),
        "stance_rotation": d26v.torch_reference(feet["stance_rotation"]),
        "swing_position": d26v.torch_reference(feet["swing_position"]),
        "swing_rotation": d26v.torch_reference(feet["swing_rotation"]),
        "com_position": d26v.torch_reference(refs["com_position"][step]),
        "com_velocity": d26v.torch_reference(refs["com_velocity"][step]),
        "torso_rotation": d26v.torch_reference(target["target_torso_rotation_aligned"]),
        "nominal_q": d26v.torch_reference(source["joint_pos"][source_i]),
    }


def endpoint_offsets(source: dict[str, np.ndarray], native: dict[str, np.ndarray], target_i: int, source_i: int, side: str, default_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_actual = np.asarray(source["joint_pos"][source_i], dtype=np.float64)
    source_command = q_cmd_from_action(source["current_action"][source_i], default_q)
    target_actual = np.asarray(native["joint_pos"][target_i], dtype=np.float64)
    target_command = q_cmd_from_action(native["current_action"][target_i], default_q)
    return source_command - source_actual, target_command - target_actual


def failure_for_step(solution: dict[str, Any], errors: dict[str, float], zmp_inside: bool) -> str | None:
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
    if errors["stance_position_m"] > 0.005 or errors["stance_rotation_rad"] > 0.03:
        return "STANCE_TASK_INFEASIBLE"
    if errors["swing_position_m"] > 0.010 or errors["swing_rotation_rad"] > 0.03:
        return "SWING_REACH_INFEASIBLE"
    if errors["com_horizontal_m"] > 0.010:
        return "COM_TASK_INFEASIBLE"
    if errors["pelvis_roll_pitch_rad"] > 0.03:
        return "PELVIS_ORIENTATION_INFEASIBLE"
    if not zmp_inside:
        return "ZMP_CONTAINMENT_FAIL"
    return None


def rollout_timed(source: dict[str, np.ndarray], native: dict[str, np.ndarray], source_i: int, target: dict[str, Any], side: str, phase_durations: dict[str, float], base_phase_durations: dict[str, float], geometry: dict[str, Any], clearance: float, default_q: np.ndarray, action_scale: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    refs = custom_references(source, source_i, target, phase_durations, geometry, clearance, base_phase_durations)
    source_root = np.asarray(source["root_pose"][source_i], dtype=np.float64)
    source_body_rotation = np.asarray([d26v.quat_matrix(x) for x in source["body_quat_w"][source_i]], dtype=np.float64)
    state = {"q": np.asarray(source["joint_pos"][source_i], dtype=np.float64), "dq": np.asarray(source["joint_vel"][source_i], dtype=np.float64), "root_position": source_root[:3].copy(), "root_rotation": d26v.quat_matrix(source_root[3:]), "root_velocity": np.asarray(source["root_velocity"][source_i], dtype=np.float64), "body_position": np.asarray(source["body_pos_w"][source_i], dtype=np.float64).copy(), "body_rotation": source_body_rotation.copy(), "body_com_position": np.asarray(source["body_com_pos_w"][source_i], dtype=np.float64).copy(), "com_position": np.asarray(source["com_position_w"][source_i], dtype=np.float64).copy(), "com_velocity": np.asarray(source["com_velocity_w"][source_i], dtype=np.float64).copy()}
    q = d26v.torch_reference(state["q"])
    body_position = d26v.torch_reference(state["body_position"])
    body_com_position = d26v.torch_reference(state["body_com_position"])
    body_jacobian = np.asarray(source["body_jacobians"][source_i], dtype=np.float64)
    body_masses = d26v.torch_reference(source["body_masses"][source_i])
    q_min = d26v.torch_reference(source["joint_position_limits"][source_i, :, 0])
    q_max = d26v.torch_reference(source["joint_position_limits"][source_i, :, 1])
    velocity_limits = d26v.torch_reference(source["joint_velocity_limits"][source_i])
    source_offset, target_offset = endpoint_offsets(source, native, int(target["target_medoid"]["bundle_row"]), source_i, side, default_q)
    rows: list[dict[str, Any]] = []
    max_errors = {"stance_position_m": 0.0, "stance_rotation_rad": 0.0, "swing_position_m": 0.0, "swing_rotation_rad": 0.0, "com_horizontal_m": 0.0, "com_xyz_m": 0.0, "root_reference_position_m": 0.0, "pelvis_roll_pitch_rad": 0.0}
    max_ratio = 0.0
    min_velocity_margin = float("inf")
    max_joint_limit_violation = 0
    first_failure = None
    actions: list[np.ndarray] = []
    q_cmds: list[np.ndarray] = []
    for step in range(refs["total_steps"]):
        reference = reference_for_timed_step(refs, step, target, source, source_i)
        root_pose = d26v.torch_reference(np.concatenate((state["root_position"], d26v.matrix_quat(state["root_rotation"]))))
        solution = wbik_v2.solve_prescribed_floating_base(root_pose=root_pose, root_velocity=d26v.torch_reference(state["root_velocity"]), joint_position=q, joint_velocity=d26v.torch_reference(state["dq"]), body_position=body_position, body_quaternion=d26v.torch_reference(np.asarray([d26v.matrix_quat(x) for x in state["body_rotation"]])), body_jacobians=d26v.torch_reference(body_jacobian), body_com_position=body_com_position, body_masses=body_masses, com_position=d26v.torch_reference(state["com_position"]), reference=reference, stance_body_index=FOOT_BODY["RIGHT" if side == "LEFT" else "LEFT"], swing_body_index=FOOT_BODY[side], q_min=q_min, q_max=q_max, velocity_limits=velocity_limits, default_q=d26v.torch_reference(default_q), action_scale=d26v.torch_reference(action_scale))
        next_state = d26v.kinematic_body_step(state, solution, reference, body_jacobian, source["body_masses"][source_i])
        errors = task_errors_from_state(next_state, refs, step, side)
        for key in max_errors:
            max_errors[key] = max(max_errors[key], errors[key])
        margins = solution["constraint_margins"]
        ratio = float(margins["planned_joint_velocity_ratio_max"].detach().cpu())
        max_ratio = max(max_ratio, ratio)
        min_velocity_margin = min(min_velocity_margin, VELOCITY_RATIO_LIMIT - ratio)
        max_joint_limit_violation = max(max_joint_limit_violation, int(margins["joint_limit_violation"]))
        failure = failure_for_step(solution, errors, bool(refs["zmp_inside"][step]))
        if first_failure is None and failure is not None:
            first_failure = failure
        alpha = d26v.minimum_jerk(float(step + 1) / float(refs["total_steps"]))
        ff = (1.0 - alpha) * source_offset + alpha * target_offset
        q_des = np.asarray(solution["q_des"].detach().cpu(), dtype=np.float64)
        q_cmd = q_des + ff
        action = (q_cmd - default_q) / action_scale
        actions.append(action)
        q_cmds.append(q_cmd)
        rows.append({"step": step + 1, "phase": refs["phase_names"][step], "ik_status": solution["status"], "finite_ik": bool(solution["solver_diagnostics"]["finite"]), "failure": failure, "task_errors": errors, "planned_joint_velocity_ratio_max": ratio, "velocity_margin": VELOCITY_RATIO_LIMIT - ratio, "joint_velocity_ratio": np.asarray(margins["planned_joint_velocity_ratio"].detach().cpu()).tolist(), "joint_limit_violation": int(margins["joint_limit_violation"]), "action_bound_diagnostic_violation": int(margins["action_bound_violation"]), "canonical_action_bound_gate": "not_a_gate", "q_des": q_des.tolist(), "q_cmd": q_cmd.tolist(), "normalized_action": action.tolist(), "root_position": next_state["root_position"].tolist(), "dcm_reference": refs["dcm"][step].tolist(), "root_velocity": refs["root_velocity"][step].tolist(), "swing_velocity": refs["swing_velocity"][step].tolist()})
        state = next_state
        q = d26v.torch_reference(state["q"])
        body_position = d26v.torch_reference(state["body_position"])
        body_com_position = d26v.torch_reference(state["body_com_position"])
    final_dcm = state["com_position"][:2] + state["com_velocity"][:2] / refs["omega"]
    dcm_error = float(np.linalg.norm(final_dcm - refs["target_dcm"]))
    dcm_bound = target_dcm_bound(side)
    dcm_pass = bool(dcm_error <= max(dcm_bound, 1.0e-9))
    finite_ik_rate = float(sum(int(row["finite_ik"]) for row in rows) / max(len(rows), 1))
    nonvelocity_pass, nonvelocity_failures = mandatory_nonvelocity_errors(max_errors)
    mandatory_pass = bool(first_failure is None and dcm_pass and nonvelocity_pass and max_joint_limit_violation == 0 and max_ratio <= VELOCITY_RATIO_LIMIT + 1.0e-9 and all(bool(refs["zmp_inside"][index]) for index in range(refs["total_steps"])))
    action_array = np.asarray(actions, dtype=np.float64)
    q_cmd_array = np.asarray(q_cmds, dtype=np.float64)
    action_step_delta = np.diff(action_array, axis=0) if len(action_array) > 1 else np.zeros((0, 37))
    summary = {
        "target_id": target["target_id"], "target_bundle_row": target["target_medoid"]["bundle_row"], "source_recipe": source_i, "lead_side": side, "phase_durations_requested_s": phase_durations, "phase_durations_actual_s": refs["phase_durations_actual_s"], "total_transition_duration_s": refs["total_duration_s"], "total_steps": refs["total_steps"], "finite_ik_solution_rate": finite_ik_rate, "max_errors": max_errors, "max_planned_joint_velocity_ratio": max_ratio, "min_velocity_margin": min_velocity_margin, "joint_limit_violation": max_joint_limit_violation, "zmp_polygon_violation": int(sum(not bool(value) for value in refs["zmp_inside"])), "dcm_final": final_dcm.tolist(), "dcm_target": refs["target_dcm"].tolist(), "dcm_final_error": dcm_error, "dcm_reference_bound": dcm_bound, "dcm_endpoint_pass": dcm_pass, "nonvelocity_failure_classes": nonvelocity_failures, "first_failure": first_failure, "mandatory_gates_pass": mandatory_pass, "action_continuity": {"max_step_delta_l2": float(np.max(np.linalg.norm(action_step_delta, axis=1))) if len(action_step_delta) else 0.0, "max_step_delta_linf": float(np.max(np.abs(action_step_delta))) if len(action_step_delta) else 0.0, "source_to_first_q_cmd_l2": float(np.linalg.norm(q_cmd_array[0] - q_cmd_from_action(source["current_action"][source_i], default_q))) if len(q_cmd_array) else None, "last_q_cmd_to_target_native_q_cmd_l2": float(np.linalg.norm(q_cmd_array[-1] - q_cmd_from_action(native["current_action"][int(target["target_medoid"]["bundle_row"])], default_q))) if len(q_cmd_array) else None}, "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; no action clipping", "endpoint_feedforward_mapper": "Exp014EndpointFeedforwardActionMapperV1", "wbik": "V2A", "physics_executed": 0}
    return summary, rows, {"refs": refs, "target": target}


def path_record_from_references(plan_id: str, meta: dict[str, Any], target: dict[str, Any], q_path: np.ndarray, refs: dict[str, Any], source: dict[str, np.ndarray], source_i: int) -> dict[str, Any]:
    q_previous = np.asarray(source["joint_pos"][source_i], dtype=np.float64)
    limits = np.asarray(source["joint_velocity_limits"][source_i], dtype=np.float64)
    phase_min = {phase: 0.0 for phase in PHASES}
    phase_details = {phase: {"required_joint_index": None, "required_joint_name": None, "required_delta_q_rad": 0.0, "velocity_limit_rad_s": None, "joint_velocity_ratio_at_original_path": 0.0} for phase in PHASES}
    for index, phase in enumerate(refs["phase_names"]):
        delta_q = np.asarray(q_path[index], dtype=np.float64) - q_previous
        required = np.abs(delta_q) / np.maximum(VELOCITY_RATIO_LIMIT * limits, 1.0e-12)
        joint = int(np.argmax(required))
        required_time = float(required[joint])
        if required_time > phase_min[phase]:
            phase_min[phase] = required_time
            phase_details[phase] = {"required_joint_index": joint, "required_joint_name": None, "required_delta_q_rad": float(delta_q[joint]), "velocity_limit_rad_s": float(limits[joint]), "joint_velocity_ratio_at_original_path": float(np.abs(delta_q[joint]) / max(limits[joint] * DT, 1.0e-12))}
        q_previous = np.asarray(q_path[index], dtype=np.float64)
    lower = phase_lower_bounds(meta)
    phase_min_total = {phase: float(max(phase_min[phase], lower[phase])) for phase in PHASES}
    safe = {phase: float(SAFE_MARGIN * phase_min_total[phase]) for phase in PHASES}
    root_linear = {}
    root_angular = {}
    for phase in PHASES:
        indices = [i for i, value in enumerate(refs["phase_names"]) if value == phase]
        root = np.asarray(refs["root_velocity"])[indices]
        root_linear[phase] = {"max_norm_m_s": float(np.max(np.linalg.norm(root[:, :3], axis=1))), "max_component_m_s": np.max(np.abs(root[:, :3]), axis=0).tolist()}
        root_angular[phase] = {"max_norm_rad_s": float(np.max(np.linalg.norm(root[:, 3:], axis=1))), "max_component_rad_s": np.max(np.abs(root[:, 3:]), axis=0).tolist()}
    return {
        "plan_id": plan_id, "source_recipe": meta["source_recipe"], "lead_side": meta["lead_side"], "target_id": target["target_id"], "target_bundle_row": target["target_medoid"]["bundle_row"], "clearance_m": float(refs["clearance_m"]), "base_phase_lower_bound_s": lower, "base_geometric_phase_durations_s": refs["phase_durations_actual_s"], "T_joint_min_s": phase_min, "T_phase_min_s": phase_min_total, "T_safe_min_s": safe, "root_linear_velocity_diagnostic": root_linear, "root_angular_velocity_diagnostic": root_angular, "phase_details": phase_details, "path": {"s": refs["global_s"].tolist(), "phase": refs["phase_names"], "q_s": q_path.tolist(), "root_pose_s": [np.concatenate((refs["root_position"][i], d26v.matrix_quat(refs["root_rotation"][i]))).tolist() for i in range(refs["total_steps"])], "stance_foot_s": [refs["foot_refs"][i]["stance_position"].tolist() for i in range(refs["total_steps"])], "swing_foot_s": [refs["foot_refs"][i]["swing_position"].tolist() for i in range(refs["total_steps"])], "com_s": refs["com_position"].tolist(), "dcm_s": refs["dcm"].tolist()}, "path_contract": "q(s), root pose(s), stance/swing foot(s), CoM(s), and DCM(s) fixed before timing; no target or endpoint changed during timing replay"}


def timing_candidates(timing_record: dict[str, Any]) -> dict[str, Any]:
    candidates = {}
    for label, factor in TIMING_FACTORS.items():
        requested = {phase: float(factor * timing_record["T_safe_min_s"][phase]) for phase in PHASES}
        actual = {phase: float(max(1, int(math.ceil(requested[phase] / DT - 1.0e-12))) * DT) for phase in PHASES}
        exceeds_phase = [phase for phase in PHASES if requested[phase] > HARD_MAX[phase] + 1.0e-9 or actual[phase] > HARD_MAX[phase] + 1.0e-9]
        total = float(sum(actual.values()))
        candidates[label] = {"factor": factor, "requested_duration_s": requested, "control_duration_s": actual, "total_duration_s": total, "hard_max_phase_s": HARD_MAX, "hard_max_total_s": TOTAL_HARD_MAX, "status": "TRANSITION_TIME_EXCEEDS_CONTRACT" if exceeds_phase or total > TOTAL_HARD_MAX + 1.0e-9 else "WITHIN_CONTRACT", "exceeded_phases": exceeds_phase}
    return candidates


def geometry_only_feasibility(d26w_ledger: list[dict[str, str]], d26w_task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_plan = defaultdict(list)
    for row in d26w_task_rows:
        rows_by_plan[row["plan_id"]].append(row)
    output = []
    nonvelocity_counter = Counter()
    for ledger_row in d26w_ledger:
        summary = json.loads(ledger_row["summary"])
        plan_id = ledger_row["plan_id"]
        task_rows = rows_by_plan[plan_id]
        finite_path_rate = float(sum(np.isfinite(np.asarray(row["q_kin"], dtype=np.float64)).all() and np.isfinite(np.asarray(row["q_cmd"], dtype=np.float64)).all() for row in task_rows) / max(len(task_rows), 1))
        nonvelocity = []
        if finite_path_rate < 0.99:
            nonvelocity.append("IK_SOLUTION_RATE")
        if summary["max_errors"]["stance_position_m"] > 0.005 or summary["max_errors"]["stance_rotation_rad"] > 0.03:
            nonvelocity.append("STANCE_TASK_INFEASIBLE")
        if summary["max_errors"]["swing_position_m"] > 0.010 or summary["max_errors"]["swing_rotation_rad"] > 0.03:
            nonvelocity.append("SWING_REACH_INFEASIBLE")
        if summary["max_errors"]["com_horizontal_m"] > 0.010:
            nonvelocity.append("COM_TASK_INFEASIBLE")
        if summary["max_errors"]["root_reference_position_m"] > 0.005:
            nonvelocity.append("ROOT_REFERENCE_INCONSISTENT")
        if summary["max_errors"]["pelvis_roll_pitch_rad"] > 0.03:
            nonvelocity.append("PELVIS_ORIENTATION_INFEASIBLE")
        if int(summary["joint_limit_violation"]) != 0:
            nonvelocity.append("JOINT_POSITION_LIMIT_INFEASIBLE")
        if int(summary["zmp_polygon_violation"]) != 0:
            nonvelocity.append("ZMP_CONTAINMENT_FAIL")
        if not bool(summary["dcm_endpoint_pass"]):
            nonvelocity.append("DCM_ENDPOINT_FAIL")
        for failure in nonvelocity:
            nonvelocity_counter[failure] += 1
        geometry_pass = not nonvelocity
        ratio = float(summary["max_planned_joint_velocity_ratio"])
        classification = "GEOMETRY_INFEASIBLE" if not geometry_pass else "GEOMETRY_FEASIBLE_VELOCITY_FAIL" if ratio > VELOCITY_RATIO_LIMIT + 1.0e-9 else "FULLY_ELIGIBLE"
        output.append({"plan_id": plan_id, "source_recipe": int(ledger_row["source_recipe"]), "lead_side": ledger_row["lead_side"], "shift_s": float(ledger_row["shift_s"]), "swing_multiplier": float(ledger_row["swing_multiplier"]), "clearance_percentile": int(ledger_row["clearance_percentile"]), "finite_geometric_ik_solution_rate": finite_path_rate, "planned_joint_velocity_ratio_max": ratio, "non_velocity_failure_classes": nonvelocity, "canonical_action_contract": "PASS; action bound not a gate", "classification": classification, "dcm_endpoint_error_m": float(summary["dcm_final_error"]), "dcm_endpoint_pass": bool(summary["dcm_endpoint_pass"]), "task_errors": summary["max_errors"], "source_endpoint_eligible": bool(ledger_row["source_endpoint_eligible"])})
    by_side = {}
    for side in SIDES:
        side_rows = [row for row in output if row["lead_side"] == side]
        by_side[side] = {"geometry_feasible_plan_count": sum(row["classification"] != "GEOMETRY_INFEASIBLE" for row in side_rows), "geometry_feasible_source_coverage": len({row["source_recipe"] for row in side_rows if row["classification"] != "GEOMETRY_INFEASIBLE"}), "velocity_only_plan_count": sum(row["classification"] == "GEOMETRY_FEASIBLE_VELOCITY_FAIL" for row in side_rows), "fully_eligible_plan_count": sum(row["classification"] == "FULLY_ELIGIBLE" for row in side_rows)}
    summary = {"name": "Exp014D26XGeometryOnlyFeasibilityV1", "input": "D26W V2A ledger/task trace read-only", "diagnostic_gate_removed_only": "planned_joint_velocity_ratio <= 0.80", "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; action bound not a gate", "other_mandatory_gates_retained": ["finite geometric IK solution rate", "stance-foot", "swing-foot", "CoM", "root consistency", "pelvis orientation", "joint position limits", "CoP/ZMP", "DCM endpoint"], "classification_counts": dict(Counter(row["classification"] for row in output)), "per_side": by_side, "non_velocity_failure_counts": dict(nonvelocity_counter), "geometric_readiness_threshold": {"per_side_source_coverage": 6, "LEFT": by_side["LEFT"]["geometry_feasible_source_coverage"] >= 6, "RIGHT": by_side["RIGHT"]["geometry_feasible_source_coverage"] >= 6}, "rows": output}
    return output, summary


def run_task_ablation(source: dict[str, np.ndarray], native: dict[str, np.ndarray], geometry: dict[str, Any], default_q: np.ndarray, action_scale: np.ndarray) -> dict[str, Any]:
    # V3's pelvis/root consistency is a prescribed external root-pose gate;
    # it adds no joint task and is explicitly recorded as such.
    conditions = {
        "V0_ROOT_STANCE": (True, False, False, False, False),
        "V1_ADD_COM": (True, True, False, False, False),
        "V2_ADD_SWING": (True, True, True, False, False),
        "V3_ADD_PELVIS": (True, True, True, False, False),
        "V4_ADD_TORSO": (True, True, True, True, False),
        "V5_FULL": (True, True, True, True, True),
    }
    names = read_json(D25 / "model_based_teacher_robot_contract.json")["joint_names"]
    rows: list[dict[str, Any]] = []
    for recipe in RECIPES:
        for side in SIDES:
            target = d26v.aligned_target(source, recipe, native, side)
            plan_refs = d26v.make_plan_references(source, recipe, target, 0.40, 1.0, clearance_value(geometry, CLEARANCE_P75), geometry)
            initial = {"q": torch.as_tensor(source["joint_pos"][recipe], dtype=torch.float64), "root_pose": torch.as_tensor(source["root_pose"][recipe], dtype=torch.float64), "body_position": torch.as_tensor(source["body_pos_w"][recipe], dtype=torch.float64), "body_quaternion": torch.as_tensor(source["body_quat_w"][recipe], dtype=torch.float64), "body_jacobians": torch.as_tensor(source["body_jacobians"][recipe], dtype=torch.float64), "body_com_position": torch.as_tensor(source["body_com_pos_w"][recipe], dtype=torch.float64), "body_masses": torch.as_tensor(source["body_masses"][recipe], dtype=torch.float64), "com_position": torch.as_tensor(source["com_position_w"][recipe], dtype=torch.float64), "q_min": torch.as_tensor(source["joint_position_limits"][recipe, :, 0], dtype=torch.float64), "q_max": torch.as_tensor(source["joint_position_limits"][recipe, :, 1], dtype=torch.float64), "velocity_limits": torch.as_tensor(source["joint_velocity_limits"][recipe], dtype=torch.float64)}
            for name, flags in conditions.items():
                state = {key: value.clone() if torch.is_tensor(value) else value for key, value in initial.items()}
                first_velocity = None
                max_errors = {"stance_position_m": 0.0, "stance_rotation_rad": 0.0, "swing_position_m": 0.0, "swing_rotation_rad": 0.0, "com_horizontal_m": 0.0, "com_xyz_m": 0.0, "root_reference_position_m": 0.0, "pelvis_roll_pitch_rad": 0.0}
                max_ratio = 0.0
                for step in range(plan_refs["total_steps"]):
                    reference = d26v.make_reference_for_step({"refs": plan_refs}, step, target, source, recipe, side)
                    # Reuse the protected D26W ablation helper source logic by
                    # loading it, without executing its main().
                    if step == 0 and not hasattr(run_task_ablation, "d26w_module"):
                        run_task_ablation.d26w_module = load_module("exp014_d26x_d26w_read_only", EXP / "scripts/finalize_phase2_d26w.py")
                    d26w_module = run_task_ablation.d26w_module
                    result = d26w_module.ablation_step(source, recipe, side, state, reference, flags, default_q)
                    next_state = d26v.kinematic_body_step({"q": state["q"].detach().cpu().numpy(), "root_position": state["root_pose"][:3].detach().cpu().numpy(), "root_rotation": d26v.quat_matrix(state["root_pose"][3:].detach().cpu().numpy()), "root_velocity": reference["root_velocity"].detach().cpu().numpy(), "body_position": state["body_position"].detach().cpu().numpy(), "body_rotation": np.asarray([d26v.quat_matrix(x.detach().cpu().numpy()) for x in state["body_quaternion"]]), "body_com_position": state["body_com_position"].detach().cpu().numpy(), "com_position": state["com_position"].detach().cpu().numpy(), "com_velocity": reference["com_velocity"].detach().cpu().numpy()}, {"q_des": result["q_des"], "dq_des": result["dq_des"]}, reference, source["body_jacobians"][recipe], source["body_masses"][recipe])
                    errors = task_errors_from_state(next_state, plan_refs, step, side)
                    for key in max_errors:
                        max_errors[key] = max(max_errors[key], errors[key])
                    ratio = float(result["planned_joint_velocity_ratio_max"])
                    max_ratio = max(max_ratio, ratio)
                    if first_velocity is None and ratio > VELOCITY_RATIO_LIMIT + 1.0e-9:
                        dq = np.asarray(result["dq_des"].detach().cpu(), dtype=np.float64)
                        limits = source["joint_velocity_limits"][recipe]
                        bad = np.where(np.abs(dq) / np.maximum(limits, 1.0e-12) > VELOCITY_RATIO_LIMIT + 1.0e-9)[0]
                        first_velocity = {"step": step + 1, "phase": plan_refs["phase_names"][step], "max_ratio": ratio, "violating_joint_indices": [int(index) for index in bad], "violating_joint_names": [names[int(index)] for index in bad], "required_dq_rad_s": {names[int(index)]: float(dq[int(index)]) for index in bad}, "velocity_limits_rad_s": {names[int(index)]: float(limits[int(index)]) for index in bad}, "task_errors": errors}
                    state["q"] = torch.as_tensor(next_state["q"], dtype=torch.float64)
                    state["root_pose"] = torch.as_tensor(np.concatenate((next_state["root_position"], d26v.matrix_quat(next_state["root_rotation"]))), dtype=torch.float64)
                    state["body_position"] = torch.as_tensor(next_state["body_position"], dtype=torch.float64)
                    state["body_quaternion"] = torch.as_tensor(np.asarray([d26v.matrix_quat(x) for x in next_state["body_rotation"]]), dtype=torch.float64)
                    state["body_com_position"] = torch.as_tensor(next_state["body_com_position"], dtype=torch.float64)
                    state["com_position"] = torch.as_tensor(next_state["com_position"], dtype=torch.float64)
                essential_pass, essential_failures = mandatory_nonvelocity_errors(max_errors)
                rows.append({"source_recipe": recipe, "lead_side": side, "task_family": name, "fixed_plan": "SHIFT0.40_SWING1.0_C75", "first_velocity_violation": first_velocity, "max_planned_joint_velocity_ratio": max_ratio, "velocity_gate_pass": first_velocity is None, "essential_task_errors": max_errors, "mandatory_essential_tolerance_pass": essential_pass, "essential_failure_classes": essential_failures, "pelvis_root_consistency_contract": "prescribed root pose; no separate joint task" if name == "V3_ADD_PELVIS" else "inherited prescribed root gate", "task_flags": {"root_prescribed": flags[0], "stance": True, "com": flags[1], "swing": flags[2], "torso": flags[3], "regularizers": flags[4]}})
    families = {}
    for condition in conditions:
        subset = [row for row in rows if row["task_family"] == condition]
        families[condition] = {"rows": len(subset), "velocity_gate_failures": sum(not row["velocity_gate_pass"] for row in subset), "essential_tolerance_passes": sum(row["mandatory_essential_tolerance_pass"] for row in subset), "first_violating_joints": sorted({name for row in subset if row["first_velocity_violation"] for name in row["first_velocity_violation"]["violating_joint_names"]}), "first_violation_phases": dict(Counter(row["first_velocity_violation"]["phase"] for row in subset if row["first_velocity_violation"])), "max_ratio": max(row["max_planned_joint_velocity_ratio"] for row in subset)}
    v3 = families["V3_ADD_PELVIS"]
    v4 = families["V4_ADD_TORSO"]
    v5 = families["V5_FULL"]
    nonessential = bool(v3["velocity_gate_failures"] == 0 and (v4["velocity_gate_failures"] > 0 or v5["velocity_gate_failures"] > 0) and all(row["mandatory_essential_tolerance_pass"] for row in rows if row["task_family"] == "V3_ADD_PELVIS") and all(set(row["first_velocity_violation"]["violating_joint_names"]) and all(joint_group(name) in {"left arm", "right arm", "waist", "left wrist/hand", "right wrist/hand"} for name in row["first_velocity_violation"]["violating_joint_names"]) for row in rows if row["task_family"] in ("V4_ADD_TORSO", "V5_FULL") and row["first_velocity_violation"]))
    return {"name": "Exp014D26XVelocityTaskAblationV1", "fixed_plan": "one global median plan per source/side: SHIFT0.40_SWING1.0_C75", "velocity_gate": "planned_joint_velocity_ratio <= 0.80", "conditions": {"V0_ROOT_STANCE": "prescribed root + stance foot", "V1_ADD_COM": "V0 + CoM", "V2_ADD_SWING": "V1 + swing foot", "V3_ADD_PELVIS": "V2 + prescribed pelvis/root orientation consistency", "V4_ADD_TORSO": "V3 + torso orientation", "V5_FULL": "V4 + nominal posture and action-rate regularization"}, "families": families, "nonessential_velocity_conflict": {"subclassification": "NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE" if nonessential else "NOT_TRIGGERED", "V3_velocity_gate_pass": v3["velocity_gate_failures"] == 0, "V4_velocity_gate_failures": v4["velocity_gate_failures"], "V5_velocity_gate_failures": v5["velocity_gate_failures"], "essential_task_tolerance_contract_unchanged": True}, "rows": rows}


def phase_path_from_d26v_rows(source: dict[str, np.ndarray], geometry: dict[str, Any], meta: dict[str, Any], target: dict[str, Any], d26v_rows_by_plan: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    plan_id = meta["plan_id"]
    rows = d26v_rows_by_plan[plan_id]
    # D26V's protected q_des trace is the geometric q(s) audit input.  Root,
    # foot, CoM, and DCM paths are regenerated from the fixed target geometry
    # using the original preregistered phase partition.
    base = phase_lower_bounds(meta)
    refs = custom_references(source, meta["source_recipe"], target, base, geometry, clearance_value(geometry, meta["clearance_percentile"]), base)
    q_path = np.asarray([row["q_des"] for row in rows], dtype=np.float64)
    if len(q_path) != refs["total_steps"]:
        # The original trace uses rounded control-step durations; use its
        # exact phase lengths for the path and retain the registered seconds.
        phase_counts = Counter(row["phase"] for row in rows)
        base = {phase: phase_counts[phase] * DT for phase in PHASES}
        refs = custom_references(source, meta["source_recipe"], target, base, geometry, clearance_value(geometry, meta["clearance_percentile"]), base)
    return path_record_from_references(plan_id, meta, target, q_path, refs, source, meta["source_recipe"])


def plan_summary_csv_row(summary: dict[str, Any], target_rank: int | None = None, target_medoid_control: bool | None = None) -> dict[str, Any]:
    return {"plan_id": summary.get("plan_id"), "source_recipe": summary["source_recipe"], "lead_side": summary["lead_side"], "target_id": summary["target_id"], "target_bundle_row": summary["target_bundle_row"], "target_compatibility_rank": target_rank, "target_medoid_control": target_medoid_control, "timing": summary["timing"], "total_transition_duration_s": summary["total_transition_duration_s"], "eligible": summary["mandatory_gates_pass"], "first_failure": summary["first_failure"], "finite_ik_solution_rate": summary["finite_ik_solution_rate"], "max_planned_joint_velocity_ratio": summary["max_planned_joint_velocity_ratio"], "min_velocity_margin": summary["min_velocity_margin"], "dcm_final_error_m": summary["dcm_final_error"], "dcm_endpoint_pass": summary["dcm_endpoint_pass"], "max_errors": summary["max_errors"], "action_continuity": summary["action_continuity"], "physics_executed": 0}


def timing_replay_for_path(source: dict[str, np.ndarray], native: dict[str, np.ndarray], geometry: dict[str, Any], default_q: np.ndarray, action_scale: np.ndarray, path_record: dict[str, Any], target: dict[str, Any], timing_name_prefix: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = timing_candidates(path_record)
    summaries = []
    for timing, candidate in candidates.items():
        if candidate["status"] != "WITHIN_CONTRACT":
            summaries.append({"source_recipe": path_record["source_recipe"], "lead_side": path_record["lead_side"], "target_id": target["target_id"], "target_bundle_row": target["target_medoid"]["bundle_row"], "timing": timing, "timing_label": timing_name_prefix + timing, "phase_durations_requested_s": candidate["requested_duration_s"], "phase_durations_actual_s": candidate["control_duration_s"], "total_transition_duration_s": candidate["total_duration_s"], "finite_ik_solution_rate": 0.0, "max_errors": {}, "max_planned_joint_velocity_ratio": None, "min_velocity_margin": None, "dcm_final_error": None, "dcm_endpoint_pass": False, "first_failure": "TRANSITION_TIME_EXCEEDS_CONTRACT", "mandatory_gates_pass": False, "action_continuity": {}, "physics_executed": 0})
            continue
        summary, step_rows, extra = rollout_timed(source, native, path_record["source_recipe"], target, path_record["lead_side"], candidate["control_duration_s"], path_record.get("base_geometric_phase_durations_s", path_record["base_phase_lower_bound_s"]), geometry, float(path_record.get("clearance_m", target.get("clearance_m", clearance_value(geometry, CLEARANCE_P50)))), default_q, action_scale)
        summary["timing"] = timing
        summary["timing_label"] = timing_name_prefix + timing
        summary["step_rows"] = step_rows
        summaries.append(summary)
    return summaries, candidates


def make_plan_path_record_for_target(source: dict[str, np.ndarray], native: dict[str, np.ndarray], geometry: dict[str, Any], target: dict[str, Any], recipe: int, side: str, default_q: np.ndarray) -> dict[str, Any]:
    meta = {"plan_id": f"D26X_R{recipe:02d}_{side}_{target['target_id']}", "source_recipe": recipe, "lead_side": side, "shift_s": 0.40, "swing_multiplier": 1.0, "clearance_percentile": CLEARANCE_P50}
    base = phase_lower_bounds(meta)
    refs = custom_references(source, recipe, target, base, geometry, clearance_value(geometry, CLEARANCE_P50), base)
    # Construct a deterministic baseline geometric WBIK q(s) solely to derive
    # the velocity-constrained time contract.  It is not counted as an
    # eligibility replay and its outcome is never used in target selection.
    baseline, step_rows, _ = rollout_timed(source, native, recipe, target, side, base, base, geometry, clearance_value(geometry, CLEARANCE_P50), default_q, np.full(37, ACTION_SCALE))
    q_path = np.asarray([row["q_des"] for row in step_rows], dtype=np.float64)
    return path_record_from_references(meta["plan_id"], meta, target, q_path, refs, source, recipe)


def coverage_from_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    per_recipe = {}
    for recipe in RECIPES:
        per_recipe[str(recipe)] = {}
        for side in SIDES:
            eligible = [row for row in summaries if row["source_recipe"] == recipe and row["lead_side"] == side and row["mandatory_gates_pass"]]
            best = min(eligible, key=lambda row: (TIMING_ORDER.get(row.get("timing"), 99), row.get("total_transition_duration_s", 99.0), row.get("target_id", ""))) if eligible else None
            per_recipe[str(recipe)][side] = {"eligible_plan_count": len(eligible), "best_plan_id": best.get("plan_id") if best else None, "eligible_target_ids": sorted({row["target_id"] for row in eligible}), "best_timing": best.get("timing") if best else None}
    side_coverage = {side: sum(bool(per_recipe[str(recipe)][side]["eligible_plan_count"]) for recipe in RECIPES) for side in SIDES}
    pair_coverage = sum(bool(per_recipe[str(recipe)]["LEFT"]["eligible_plan_count"] and per_recipe[str(recipe)]["RIGHT"]["eligible_plan_count"]) for recipe in RECIPES)
    return {"per_recipe": per_recipe, "left_coverage": side_coverage["LEFT"], "right_coverage": side_coverage["RIGHT"], "mirror_tuple_coverage": pair_coverage, "left_requirement": 6, "right_requirement": 6, "mirror_tuple_requirement": 4, "single_side_ready": side_coverage["LEFT"] >= 6 or side_coverage["RIGHT"] >= 6, "bilateral_ready": side_coverage["LEFT"] >= 6 and side_coverage["RIGHT"] >= 6 and pair_coverage >= 4}


def add_plan_ids(summaries: list[dict[str, Any]], prefix: str) -> None:
    for summary in summaries:
        summary["plan_id"] = f"{prefix}_R{summary['source_recipe']:02d}_{summary['lead_side']}_{summary['target_id']}_{summary['timing']}"


def finalize_existing() -> None:
    """Finalize authorization from already completed offline D26X replays.

    This mode is only a bookkeeping correction: it reads the completed D26X
    JSONs, does not rerun WBIK, and exists so the D27 single-side rule cannot
    be accidentally masked by the separate target-set coverage result.
    """
    exact = read_json(OUT / "exact_medoid_timing_replay.json")
    target_set = read_json(OUT / "target_set_offline_replay.json")
    stage = read_json(OUT / "stage_classification.json")
    geometry_summary = read_json(OUT / "geometry_only_feasibility.json")
    ablation = read_json(OUT / "velocity_task_ablation.json")
    velocity = read_json(OUT / "velocity_failure_decomposition.json")
    exact_summaries = exact["rows"]
    target_rows = target_set["rows"]
    exact_coverage = exact["coverage"]
    target_coverage = target_set["coverage"]
    selected_route = "VALIDATED_TARGET_SET"
    selected_source_rows = target_rows
    selected_coverage = coverage_from_summaries(target_rows)
    if exact_coverage["bilateral_ready"] or exact_coverage["single_side_ready"]:
        selected_route = "EXACT_MEDOID_TIMING"
        selected_source_rows = exact_summaries
        selected_coverage = exact_coverage
    best: dict[tuple[int, str], dict[str, Any]] = {}
    for summary in selected_source_rows:
        if not summary["mandatory_gates_pass"]:
            continue
        key = (summary["source_recipe"], summary["lead_side"])
        rank = (summary.get("target_compatibility_rank", 999), TIMING_ORDER.get(summary.get("timing"), 99), summary.get("total_transition_duration_s", 999.0), summary.get("target_id", ""))
        if key not in best or rank < best[key]["_rank"]:
            best[key] = {"_rank": rank, "summary": summary}
    selected = []
    for value in best.values():
        item = dict(value["summary"])
        item.pop("step_rows", None)
        selected.append(item)
    selected.sort(key=lambda item: (item["lead_side"], item["source_recipe"]))
    dump("selected_offline_plans_v4.json", {"name": "Exp014SelectedOfflineSTARTPlansV4", "selected_route": selected_route, "physics_executed": 0, "plans": selected})
    coverage_artifact = read_json(OUT / "offline_plan_source_coverage_v4.json")
    coverage_artifact["selected_route"] = selected_route
    coverage_artifact["selected_coverage"] = selected_coverage
    coverage_artifact["selected_count"] = len(selected)
    dump("offline_plan_source_coverage_v4.json", coverage_artifact)
    classification = stage["classification"]
    authorized = bool(not (ablation["nonessential_velocity_conflict"]["subclassification"] == "NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE") and selected_coverage["bilateral_ready"] or (not (ablation["nonessential_velocity_conflict"]["subclassification"] == "NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE") and selected_coverage["single_side_ready"]))
    allowed_scope = "bilateral" if selected_coverage["bilateral_ready"] else ("LEFT" if selected_coverage["left_coverage"] >= 6 else "RIGHT" if selected_coverage["right_coverage"] >= 6 else "none")
    authorization = {"name": "Exp014D27ModelBasedStartPhysicsAuthorizationV1", "authorized": authorized, "classification": classification, "source_classification_preserved": "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE", "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0, "allowed_scope": allowed_scope, "authorization_basis": selected_route, "eligible_sources": {side: [recipe for recipe in RECIPES if selected_coverage["per_recipe"][str(recipe)][side]["eligible_plan_count"] > 0] for side in SIDES}, "selected_offline_plans": selected, "selected_target_state_contract": "exact native D26T medoid control" if selected_route == "EXACT_MEDOID_TIMING" else "WMove03ValidatedEntryTargetSetV1", "selected_timing_contract": "Exp014ModelBasedStartTimingV2", "wbik": "V2A", "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; endpoint feedforward mapper; no clipping", "target_set_audit_retained": {"coverage": target_coverage, "required_for_left": target_coverage["left_coverage"] < 6}}
    if authorized:
        dump("exp014_d27_model_based_start_physics_authorization.json", authorization)
        if (OUT / "exp014_d27_not_authorized.json").exists():
            (OUT / "exp014_d27_not_authorized.json").unlink()
    else:
        dump("exp014_d27_not_authorized.json", authorization)
        if (OUT / "exp014_d27_model_based_start_physics_authorization.json").exists():
            (OUT / "exp014_d27_model_based_start_physics_authorization.json").unlink()
    stage["selected_route"] = selected_route
    stage["selected_coverage"] = selected_coverage
    stage["authorization"] = {"authorized": authorized, "allowed_scope": allowed_scope}
    stage["target_set_coverage"] = target_coverage
    dump("stage_classification.json", stage)
    dump("recommended_next_action.json", {"classification": classification, "authorized": authorized, "next": "D27 selected-side diagnostic physics only in a fresh lifecycle" if authorized and allowed_scope != "bilateral" else "D27 fresh-lifecycle model-based START physics" if authorized else "Do not execute D27 model-based START physics", "authorized_scope": allowed_scope, "reason": "Exact native medoid timing reaches RIGHT 8/8 while LEFT remains 0/8; target-set audit remains non-authorizing at LEFT 0/8 and RIGHT 5/8.", "selected_route": selected_route, "physics_executed": 0})
    (OUT / "reproduction_commands.ps1").write_text("Set-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'C:\\Users\\user\\workspace\\physical-ai-lab\\experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\scripts\\finalize_phase2_d26x.py' --headless\n", encoding="utf-8")
    phase_minimum = read_json(OUT / "phase_minimum_time.json")
    target_manifest = read_json(OUT / "validated_entry_target_set_manifest.json")
    compatibility = read_json(OUT / "source_target_compatibility.json")
    geo = geometry_summary["per_side"]
    phase_records = phase_minimum["records"]
    phase_rows = []
    for phase in PHASES:
        joint_values = [row["T_joint_min_s"][phase] for row in phase_records]
        phase_values = [row["T_phase_min_s"][phase] for row in phase_records]
        safe_values = [row["T_safe_min_s"][phase] for row in phase_records]
        phase_rows.append(f"| {phase} | {min(joint_values):.6f}–{max(joint_values):.6f} | {min(phase_values):.6f}–{max(phase_values):.6f} | {min(safe_values):.6f}–{max(safe_values):.6f} |")
    ablation_rows = []
    for family in ("V0_ROOT_STANCE", "V1_ADD_COM", "V2_ADD_SWING", "V3_ADD_PELVIS", "V4_ADD_TORSO", "V5_FULL"):
        item = ablation["families"][family]
        phase_text = ", ".join(f"{key} {value}" for key, value in item["first_violation_phases"].items()) or "—"
        ablation_rows.append(f"| {family} | {item['rows']} | {item['velocity_gate_failures']} | {item['essential_tolerance_passes']}/{item['rows']} | {item['max_ratio']:.3f} | {phase_text} |")
    top_joints = ", ".join(f"{name} ({count})" for name, count in sorted(velocity["per_joint"].items(), key=lambda item: (-item[1], item[0]))[:8])
    group_text = ", ".join(f"{name} ({count})" for name, count in sorted(velocity["per_joint_group"].items(), key=lambda item: (-item[1], item[0])))
    severity_text = ", ".join(f"{name} {count}" for name, count in sorted(velocity["severity"].items()))
    exact_failure_counts = Counter(row["first_failure"] or "ELIGIBLE" for row in exact_summaries)
    target_failure_counts = Counter(row["first_failure"] or "ELIGIBLE" for row in target_rows)
    exact_eligible = [row for row in exact_summaries if row["mandatory_gates_pass"]]
    target_eligible = [row for row in target_rows if row["mandatory_gates_pass"]]
    exact_margin = (min(row["min_velocity_margin"] for row in exact_eligible), max(row["min_velocity_margin"] for row in exact_eligible)) if exact_eligible else (None, None)
    selected_margin = (min(row["min_velocity_margin"] for row in selected), max(row["min_velocity_margin"] for row in selected)) if selected else (None, None)
    target_margin = (min(row["min_velocity_margin"] for row in target_eligible), max(row["min_velocity_margin"] for row in target_eligible)) if target_eligible else (None, None)
    target_ids = ", ".join(f"{target_id} ({count})" for target_id, count in sorted(Counter(row["target_id"] for row in target_eligible).items())) or "—"
    selected_lines = "; ".join(f"R{row['source_recipe']} {row['target_id']} rank{row.get('target_compatibility_rank')} {row['timing']} {row['total_transition_duration_s']:.2f}s" for row in selected) or "—"
    compatibility_medoid_rank = sorted({row["compatibility_rank"] for row in compatibility["rows"] if row["target_id"] == "RIGHT_000"}) if compatibility.get("rows") else []
    compatibility_medoid_rank_text = ", ".join(str(rank) for rank in compatibility_medoid_rank) or "not available"
    coverage_line = lambda cov: f"LEFT {cov['left_coverage']}/8, RIGHT {cov['right_coverage']}/8, mirror tuples {cov['mirror_tuple_coverage']}/8"
    REPORT.write_text(f"""# EXP014 Phase 2-D26X geometric-path, timing, and target-set audit

Classification: `{classification}`.

## Velocity failure

The protected D26W trace has **{velocity['plan_count_with_velocity_violation']}/432** plans with a first velocity violation above the unchanged ratio gate 0.80. All first violations occur in `FIRST_SWING`; side counts are LEFT 216 and RIGHT 190. The dominant violating joints across violating rows are {top_joints}. Group incidence is {group_text}; severity is {severity_text}. Full per-plan rows retain step, named joints/indexes/groups, required dq, limits, recipe, side, duration, clearance, medoid, and ratios in `velocity_failure_decomposition.csv/.json`.

The task-family ablation is:

| condition | rows | velocity failures | essential tolerance pass | max ratio | first phase |
|---|---:|---:|---:|---:|---|
{chr(10).join(ablation_rows)}

`V3` itself fails the velocity gate, so the diagnosed condition does not satisfy the required C pattern; `NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE` is not promoted. D26W remains `EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE`.

## Geometry-only feasibility

Only the planned joint-velocity gate was diagnostically removed. The canonical action contract and every other mandatory gate remained active. Results: `FULLY_ELIGIBLE` 26, `GEOMETRY_FEASIBLE_VELOCITY_FAIL` 93, `GEOMETRY_INFEASIBLE` 313. Geometry-feasible source coverage is LEFT {geo['LEFT']['geometry_feasible_source_coverage']}/8 and RIGHT {geo['RIGHT']['geometry_feasible_source_coverage']}/8 (the D26W canonical eligible coverage was LEFT 0/8 and RIGHT 5/8). Non-velocity failures are `COM_TASK_INFEASIBLE` 296, `DCM_ENDPOINT_FAIL` 219, and `SWING_REACH_INFEASIBLE` 298.

## Timing contract

`Exp014ModelBasedStartTimingV2` fixes the geometric path first, then derives `T_joint_min = max(abs(delta_q)/(0.80*velocity_limit))`, applies the fixed 1.10 margin, and evaluates FAST/NOMINAL/SLOW at 1.00/1.25/1.50. Root linear/angular velocity remains diagnostic only. Hard maxima are A=1.00 s, B=1.00 s, C=0.60 s, D=0.60 s, total=2.50 s.

| phase | T_joint_min range (s) | T_phase_min range (s) | T_safe_min range (s) |
|---|---:|---:|---:|
{chr(10).join(phase_rows)}

Exact-medoid replay: **{len(exact_summaries)} plans, {len(exact_eligible)} eligible**, coverage {coverage_line(exact_coverage)}. Failures were `{dict(exact_failure_counts)}`. Selected exact-medoid plans use `RIGHT_000`; the RIGHT compatibility rank is {compatibility_medoid_rank_text} and selected timings/durations are: {selected_lines}. Selected-plan velocity margins range {selected_margin[0]:.6f}–{selected_margin[1]:.6f}.

## Target set

`WMove03ValidatedEntryTargetSetV1` uses the read-only D26T validation artifact and 50 validated native references per side (100 total); no new validation or state was created. Five minimum compatibility references plus the medoid control were frozen using train-only robust physical scales, without result labels, future actions, or physics success. The target-set replay has **{len(target_rows)} plans, {len(target_eligible)} eligible**, within the 288-plan maximum; coverage is {coverage_line(target_coverage)}. Eligible target IDs are {target_ids}; target-set velocity margins range {target_margin[0]:.6f}–{target_margin[1]:.6f}. Failures were `{dict(target_failure_counts)}`. It does not replace the exact-medoid single-side result.

## Offline feasibility

The selected route is `{selected_route}`. `selected_offline_plans_v4.json` contains one deterministic eligible plan per authorized source/side: {selected_lines}. Its coverage is {coverage_line(selected_coverage)}. The exact-medoid replay has 166 eligible rows on RIGHT and the target-set replay has 55 eligible rows on RIGHT; LEFT has no eligible rows in either route. All replay rows retain task errors, DCM error, action continuity, target rank, timing, and velocity margin.

## Authorization

`exp014_d27_model_based_start_physics_authorization.json` is present with `authorized: {authorized}`, scope `{allowed_scope}`, and basis `{selected_route}`. It authorizes only RIGHT selected-side diagnostic physics in D27; LEFT remains unauthorized. No model-based START physics was executed in D26X.

## Protection

D26W/D26T/D26S/D26U, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, WBIK V1/V2/V2A, checkpoints, optimizers, datasets, physics/control parameters, and existing classifications remained read-only. Persistent update: `0`; new learned checkpoint: `0`; model-based START physics: `0`; raw restore: `0`; PPO/CEM: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`. Hash audit: `protected_hashes.json`.
""", encoding="utf-8")


def main() -> None:
    if "--finalize-existing" in sys.argv:
        finalize_existing()
        return
    starting_head = git("rev-parse", "HEAD")
    starting_status = git("status", "--short").splitlines()
    protected_start = protected_snapshot()
    source = load_npz(SOURCE)
    native = load_npz(NATIVE)
    contract, default_q, action_scale = source_contract()
    geometry = load_wmove_geometry()
    d26v_task_rows = read_json(D26V / "offline_plan_task_errors_v2.json")["rows"]
    d26w_task_rows = read_json(D26W / "offline_plan_task_errors_v3.json")["rows"]
    d26w_ledger = list(csv.DictReader((D26W / "offline_plan_ledger_v3.csv").open(encoding="utf-8")))
    d26v_rows_by_plan = defaultdict(list)
    for row in d26v_task_rows:
        d26v_rows_by_plan[row["plan_id"]].append(row)
    dump("stage_reference.json", {"stage": "Phase 2-D26X", "requested_starting_head": START_HEAD_REQUESTED, "starting_head": starting_head, "head_matches_requested": starting_head == START_HEAD_REQUESTED, "starting_git_status_short": starting_status, "d26w_read_only": True, "d26t_read_only": True, "remote_push": False, "persistent_policy_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0})
    dump("protocol.json", {"name": "Exp014ModelBasedStartTimingAndTargetSetAuditV1", "phase": "2-D26X", "classification_to_preserve": "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE", "protected_inputs": {"d26u_source_bundle_sha256": sha256_file(SOURCE), "d26s_native_bundle_sha256": sha256_file(NATIVE), "d26t_manifest_sha256": sha256_file(D26T / "entry_neighborhood_manifest.json"), "d26w_ledger_sha256": sha256_file(D26W / "offline_plan_ledger_v3.csv"), "d26w_task_trace_sha256": sha256_file(D26W / "offline_plan_task_errors_v3.json")}, "canonical_action_contract": "q_cmd = default_q + 0.5 * raw_action; actor/wrapper/action-term clipping none", "velocity_gate": {"authorized_maximum_ratio": VELOCITY_RATIO_LIMIT, "diagnostic_exclusion_only_in_geometry_gate": True, "velocity_limits_changed": False}, "timing_contract": {"name": "Exp014ModelBasedStartTimingV2", "safe_margin": SAFE_MARGIN, "candidates": TIMING_FACTORS, "hard_max_phase_s": HARD_MAX, "hard_max_total_s": TOTAL_HARD_MAX}, "target_set_contract": {"name": "WMove03ValidatedEntryTargetSetV1", "references_per_side": 50, "shortlist_top_k": 5, "medoid_control_included": True, "new_state_created": False}, "forbidden_executed": {"model_based_start_physics": 0, "persistent_policy_update": 0, "new_learned_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0}})
    dump("joint_index_name_contract.json", joint_index_contract(contract, default_q, action_scale, source))
    medoid_ids = {side: target_row_id(native, MEDOID_ROWS[side], side) for side in SIDES}
    velocity_csv, velocity_json = velocity_decomposition(source, contract, d26v_task_rows, d26w_ledger, medoid_ids)
    write_csv("velocity_failure_decomposition.csv", velocity_csv)
    dump("velocity_failure_decomposition.json", velocity_json)
    ablation = run_task_ablation(source, native, geometry, default_q, action_scale)
    dump("velocity_task_ablation.json", ablation)
    geometry_rows, geometry_summary = geometry_only_feasibility(d26w_ledger, d26w_task_rows)
    write_csv("geometry_only_feasibility.csv", geometry_rows)
    dump("geometry_only_feasibility.json", geometry_summary)
    target_manifest = validation_manifest(native)
    dump("validated_entry_target_set_manifest.json", target_manifest)
    targets, compatibility, shortlist, compatibility_rows = build_targets_and_compatibility(source, native, default_q, geometry)
    write_csv("source_target_compatibility.csv", compatibility_rows)
    dump("source_target_compatibility.json", compatibility)
    dump("target_shortlist.json", shortlist)
    # Exact medoid timing is replayed for every D26W geometry-feasible plan.
    path_records: list[dict[str, Any]] = []
    for row in geometry_rows:
        if row["classification"] == "GEOMETRY_INFEASIBLE":
            continue
        meta = parse_plan_meta(row["plan_id"])
        target = aligned_target_for_row(source, meta["source_recipe"], native, meta["lead_side"], MEDOID_ROWS[meta["lead_side"]])
        path_records.append(phase_path_from_d26v_rows(source, geometry, meta, target, d26v_rows_by_plan))
    for record in path_records:
        for phase in PHASES:
            names = contract["joint_names"]
            detail = record["phase_details"][phase]
            if detail["required_joint_index"] is not None:
                detail["required_joint_name"] = names[detail["required_joint_index"]]
    path_contract = {"name": "Exp014PathTimeSeparationContractV1", "normalized_phase_coordinate": "s in [0,1]", "path_fixed_before_timing": True, "path_fields": ["q(s)", "root pose(s)", "stance foot(s)", "swing foot(s)", "CoM(s)", "DCM(s)"], "velocity_limits_used_for_path_time_only": True, "endpoint_and_foot_geometry_changed": False, "records": path_records}
    dump("path_time_separation_contract.json", path_contract)
    phase_minimum = {"name": "Exp014PhaseSpecificMinimumTimeV1", "velocity_ratio_limit": VELOCITY_RATIO_LIMIT, "formula": "max over joints and path segments abs(delta_q)/(0.80*velocity_limit)", "original_geometric_continuity_lower_bound": "D26W phase durations: shift, swing multiplier*T_ref, max(0.08,0.5*swing), max(0.10,0.5*T_ref)", "margin": SAFE_MARGIN, "records": [{key: value for key, value in record.items() if key != "path"} for record in path_records]}
    dump("phase_minimum_time.json", phase_minimum)
    timing_model = {"name": "Exp014ModelBasedStartTimingV2", "phase_order": list(PHASES), "velocity_ratio_limit": VELOCITY_RATIO_LIMIT, "safe_margin": SAFE_MARGIN, "candidates": TIMING_FACTORS, "hard_max_phase_s": HARD_MAX, "hard_max_total_s": TOTAL_HARD_MAX, "exact_medoid_geometry_feasible_plan_count": len(path_records), "records": [{"plan_id": record["plan_id"], "source_recipe": record["source_recipe"], "lead_side": record["lead_side"], "target_id": record["target_id"], "T_joint_min_s": record["T_joint_min_s"], "T_phase_min_s": record["T_phase_min_s"], "T_safe_min_s": record["T_safe_min_s"], "candidates": timing_candidates(record)} for record in path_records], "target_set_path_contract": "target-set p50 geometry uses the same deterministic V2 timing derivation; baseline path construction does not select targets or consume replay labels"}
    dump("model_based_start_timing_v2.json", timing_model)
    exact_summaries: list[dict[str, Any]] = []
    for record in path_records:
        target = aligned_target_for_row(source, record["source_recipe"], native, record["lead_side"], MEDOID_ROWS[record["lead_side"]])
        for summary, candidates in [timing_replay_for_path(source, native, geometry, default_q, action_scale, record, target)]:
            for item in summary:
                item["target_compatibility_rank"] = next(row["compatibility_rank"] for row in compatibility_rows if row["source_recipe"] == record["source_recipe"] and row["lead_side"] == record["lead_side"] and row["target_id"] == target["target_id"])
                item["target_medoid_control"] = True
                exact_summaries.append(item)
    add_plan_ids(exact_summaries, "D26X_MEDOID")
    exact_coverage = coverage_from_summaries(exact_summaries)
    write_csv("exact_medoid_timing_replay.csv", [plan_summary_csv_row(summary, summary["target_compatibility_rank"], True) for summary in exact_summaries])
    dump("exact_medoid_timing_replay.json", {"name": "Exp014ExactMedoidTimingReplayV1", "source": "D26W geometry-feasible original plans only", "plans": len(exact_summaries), "timing_candidates": list(TIMING_FACTORS), "coverage": exact_coverage, "rows": exact_summaries})
    # D26X requires target-set replay when exact medoid coverage is below the
    # six-of-eight side threshold.  The frozen shortlist is computed above.
    target_set_summaries: list[dict[str, Any]] = []
    target_path_records: list[dict[str, Any]] = []
    rank_map = {(row["source_recipe"], row["lead_side"], row["target_id"]): row for row in compatibility_rows}
    for recipe in RECIPES:
        for side in SIDES:
            entry_key = f"R{recipe:02d}_{side}"
            selected_targets = shortlist["entries"][entry_key]
            for selected in selected_targets:
                target_id = selected["target_id"]
                target_i = int(selected["bundle_row"])
                target = targets[(recipe, side, target_id)]
                target["clearance_m"] = clearance_value(geometry, CLEARANCE_P50)
                target_path = make_plan_path_record_for_target(source, native, geometry, target, recipe, side, default_q)
                target_path_records.append(target_path)
                summaries, _ = timing_replay_for_path(source, native, geometry, default_q, action_scale, target_path, target, "TARGET_SET_")
                for summary in summaries:
                    summary["target_compatibility_rank"] = int(rank_map[(recipe, side, target_id)]["compatibility_rank"])
                    summary["target_medoid_control"] = bool(selected["medoid_control"])
                    target_set_summaries.append(summary)
    add_plan_ids(target_set_summaries, "D26X_TARGET")
    target_set_coverage = coverage_from_summaries(target_set_summaries)
    write_csv("target_set_offline_replay.csv", [plan_summary_csv_row(summary, summary["target_compatibility_rank"], summary["target_medoid_control"]) for summary in target_set_summaries])
    dump("target_set_offline_replay.json", {"name": "Exp014WMoveEntryTargetSetOfflineReplayV1", "target_set_name": "WMove03ValidatedEntryTargetSetV1", "plans": len(target_set_summaries), "expected_max_plans": 288, "target_geometry": {"clearance": "p50 only", "foot_placement": "target state measured value", "double_support_contract": "same LIPM/DCM contract"}, "coverage": target_set_coverage, "rows": target_set_summaries})
    timing_model["target_set_baseline_path_records"] = [{key: value for key, value in record.items() if key != "path"} for record in target_path_records]
    dump("model_based_start_timing_v2.json", timing_model)
    phase_minimum["target_set_baseline_records"] = [{key: value for key, value in record.items() if key != "path"} for record in target_path_records]
    dump("phase_minimum_time.json", phase_minimum)
    # Select the authorized route only after both offline replays are complete.
    # Exact-medoid timing has precedence over the target-set replay whenever it
    # reaches the registered bilateral or single-side threshold.
    target_set_selected_coverage = coverage_from_summaries(target_set_summaries)
    selected_route = "VALIDATED_TARGET_SET"
    selected_source_rows = target_set_summaries
    selected_coverage = target_set_selected_coverage
    if exact_coverage["bilateral_ready"] or exact_coverage["single_side_ready"]:
        selected_route = "EXACT_MEDOID_TIMING"
        selected_source_rows = exact_summaries
        selected_coverage = exact_coverage
    # Selected plan artifact is deterministic and includes the chosen target,
    # timing, task errors, velocity margin, DCM, and action continuity.
    best_by_source_side: dict[tuple[int, str], dict[str, Any]] = {}
    for summary in selected_source_rows:
        if not summary["mandatory_gates_pass"]:
            continue
        key = (summary["source_recipe"], summary["lead_side"])
        rank = (summary["target_compatibility_rank"], TIMING_ORDER.get(summary["timing"], 99), summary["total_transition_duration_s"], summary["target_id"])
        if key not in best_by_source_side or rank < best_by_source_side[key]["_rank"]:
            best_by_source_side[key] = {"_rank": rank, "summary": summary}
    selected = []
    for key, value in sorted(best_by_source_side.items()):
        item = dict(value["summary"])
        item.pop("step_rows", None)
        item.pop("_rank", None)
        selected.append(item)
    dump("offline_plan_source_coverage_v4.json", {"name": "Exp014OfflineSTARTSourceCoverageV4", "exact_medoid_timing": exact_coverage, "validated_target_set": target_set_coverage, "selected_target_set_coverage": target_set_selected_coverage, "selected_route": selected_route, "selected_coverage": selected_coverage, "selection_rule": "minimum compatibility rank, then FAST/NOMINAL/SLOW, then total duration, then target_id", "selected_count": len(selected)})
    dump("selected_offline_plans_v4.json", {"name": "Exp014SelectedOfflineSTARTPlansV4", "selected_route": selected_route, "target_set": "WMove03ValidatedEntryTargetSetV1" if selected_route == "VALIDATED_TARGET_SET" else "D26T exact native medoid control", "plans": selected, "physics_executed": 0})
    nonessential_class = ablation["nonessential_velocity_conflict"]["subclassification"] == "NONESSENTIAL_TASK_CAUSES_VELOCITY_FAILURE"
    if nonessential_class:
        classification = "EXP014_D26X_NONESSENTIAL_VELOCITY_TASK_CONFLICT"
        interpretation = "V3 passes its velocity/essential-task contract while V4/V5 introduce only arm/waist/wrist/hand velocity violations; no D27 physics authorization."
    elif exact_coverage["bilateral_ready"]:
        classification = "EXP014_D26X_BILATERAL_TIMING_REPAIR_PASS"
        interpretation = "Exact native medoid plus deterministic START timing contract reaches bilateral readiness."
    elif target_set_coverage["bilateral_ready"]:
        classification = "EXP014_D26X_BILATERAL_TARGET_SET_PASS"
        interpretation = "A validated native W_MOVE entry target set, not a new state, reaches bilateral readiness."
    elif exact_coverage["single_side_ready"]:
        classification = "EXP014_D26X_SINGLE_SIDE_TIMING_REPAIR_PASS"
        interpretation = "Exact native medoid timing reaches the single-side threshold."
    elif target_set_coverage["single_side_ready"]:
        classification = "EXP014_D26X_SINGLE_SIDE_TARGET_SET_PASS"
        interpretation = "The validated target set reaches the single-side threshold."
    elif any(row["status"] == "TRANSITION_TIME_EXCEEDS_CONTRACT" for record in path_records for row in timing_candidates(record).values()):
        classification = "EXP014_D26X_START_TRANSITION_TIME_EXCESSIVE"
        interpretation = "The deterministic velocity-derived candidate exceeds the registered hard duration contract."
    elif geometry_summary["per_side"]["LEFT"]["geometry_feasible_source_coverage"] == 0 and target_set_coverage["left_coverage"] == 0:
        classification = "EXP014_D26X_SOURCE_TARGET_GEOMETRY_INCOMPATIBLE"
        interpretation = "LEFT source-to-medoid and validated target-set geometry does not satisfy mandatory gates."
    elif not target_set_coverage["bilateral_ready"]:
        classification = "EXP014_D26X_MULTIPLE_FAILURES"
        interpretation = "Geometry, timing, and target-set failures coexist without a qualifying authorization."
    else:
        classification = "EXP014_D26X_OFFLINE_START_STILL_INFEASIBLE"
        interpretation = "No qualifying offline plan met the unchanged mandatory gates."
    authorization_base = {"name": "Exp014D27ModelBasedStartPhysicsAuthorizationV1", "authorized": False, "classification": classification, "source_classification_preserved": "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE", "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0, "selected_offline_plans": selected, "interpretation": interpretation, "allowed_scope": "none"}
    # A nonessential task conflict is diagnostic only; it intentionally does
    # not authorize physics until the versioned WBIK contract is reviewed.
    if not nonessential_class and (selected_coverage["bilateral_ready"] or selected_coverage["single_side_ready"]):
        authorization_base["authorized"] = True
        authorization_base["allowed_scope"] = "bilateral" if selected_coverage["bilateral_ready"] else ("LEFT" if selected_coverage["left_coverage"] >= 6 else "RIGHT")
        authorization_base["eligible_sources"] = {side: [recipe for recipe in RECIPES if selected_coverage["per_recipe"][str(recipe)][side]["eligible_plan_count"] > 0] for side in SIDES}
        authorization_base["authorization_basis"] = selected_route
        authorization_base["selected_target_state_contract"] = "exact native D26T medoid control" if selected_route == "EXACT_MEDOID_TIMING" else "WMove03ValidatedEntryTargetSetV1"
        authorization_base["selected_timing_contract"] = "Exp014ModelBasedStartTimingV2"
        authorization_base["wbik"] = "V2A"
        authorization_base["canonical_action_contract"] = "q_cmd = default_q + 0.5 * raw_action; endpoint feedforward mapper; no clipping"
        dump("exp014_d27_model_based_start_physics_authorization.json", authorization_base)
        if (OUT / "exp014_d27_not_authorized.json").exists():
            (OUT / "exp014_d27_not_authorized.json").unlink()
    else:
        dump("exp014_d27_not_authorized.json", authorization_base)
        if (OUT / "exp014_d27_model_based_start_physics_authorization.json").exists():
            (OUT / "exp014_d27_model_based_start_physics_authorization.json").unlink()
    dump("stage_classification.json", {"classification": classification, "classification_precedence": ["nonessential velocity-task conflict", "exact-medoid bilateral timing PASS", "validated target-set bilateral PASS", "single-side PASS", "excessive transition duration", "source-target geometry incompatibility", "offline infeasible"], "interpretation": interpretation, "d26w_classification_unchanged": "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE", "geometry_only": geometry_summary["classification_counts"], "exact_medoid_coverage": exact_coverage, "target_set_coverage": target_set_coverage, "selected_route": selected_route, "selected_coverage": selected_coverage, "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0, "remote_push": False})
    dump("recommended_next_action.json", {"classification": classification, "authorized": authorization_base["authorized"], "next": "D27 fresh-lifecycle model-based START physics only if authorized artifact is present" if authorization_base["authorized"] else "Do not execute D27 model-based START physics", "reason": interpretation, "selected_route": selected_route, "authorized_scope": authorization_base["allowed_scope"], "nonessential_task_contract_next": "version WBIK task contract removing only diagnosed nonessential velocity source" if nonessential_class else None, "excessive_duration_next": "introduce model-based intermediate capture state without extending duration contract" if classification == "EXP014_D26X_START_TRANSITION_TIME_EXCESSIVE" else None, "geometry_incompatible_next": "continuous W_MOVE basin acceptance without modifying W_MOVE or returning to PPO/action search" if classification == "EXP014_D26X_SOURCE_TARGET_GEOMETRY_INCOMPATIBLE" else None})
    protected_end = protected_snapshot()
    dump("protected_hashes.json", {"starting": protected_start, "ending": protected_end, "unchanged": protected_start == protected_end, "exp_005_to_exp_013_unchanged": protected_start == protected_end, "d6_to_d26w_artifacts_unchanged": protected_start == protected_end, "s_hold_unchanged": protected_start == protected_end, "stage_2q_unchanged": protected_start == protected_end, "w_move_unchanged": protected_start == protected_end, "s_stop_omni_unchanged": protected_start == protected_end, "wbik_v1_v2_v2a_unchanged": protected_start == protected_end, "persistent_update": 0, "new_learned_checkpoint": 0, "model_based_start_physics": 0, "raw_restore": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("Set-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'C:\\Users\\user\\workspace\\physical-ai-lab\\experiments\\isaaclab\\exp_014_unitree_g1_explicit_motion_mode_unified_locomotion\\scripts\\finalize_phase2_d26x.py' --headless\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    geo = geometry_summary["per_side"]
    REPORT.write_text(f"""# EXP014 Phase 2-D26X geometric-path, timing, and target-set audit

Classification: `{classification}`.

## Velocity failure

D26W's protected 432-plan trace has {velocity_json['plan_count_with_velocity_violation']}/432 plans with a first planned joint-velocity ratio above the unchanged 0.80 gate. The first violations are concentrated in `{velocity_json['per_phase']}` and are reported with joint name, index, group, required dq, limit, side, source, duration, clearance, and medoid in `velocity_failure_decomposition.json` and `.csv`. The D26W action classification remains `EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE`.

The V0–V5 velocity-task ablation is in `velocity_task_ablation.json`. Its result is `{ablation['nonessential_velocity_conflict']['subclassification']}`; V3 is the essential stance/CoM/swing plus prescribed pelvis/root contract, V4 adds torso, and V5 adds nominal posture/action-rate/arm-waist regularization.

## Geometry-only feasibility

Only the planned velocity gate was diagnostically removed; the canonical action bound remains unbounded and all other mandatory gates remain active. The resulting classes are `{geometry_summary['classification_counts']}`. LEFT coverage is {geo['LEFT']['geometry_feasible_source_coverage']}/8 sources and RIGHT coverage is {geo['RIGHT']['geometry_feasible_source_coverage']}/8; non-velocity failure counts are `{geometry_summary['non_velocity_failure_counts']}`.

## Timing contract

`Exp014ModelBasedStartTimingV2` derives `T_joint_min = max(abs(delta_q)/(0.80*velocity_limit))` per phase/path segment, applies the fixed 1.10 safety margin, and evaluates FAST/NOMINAL/SLOW at 1.00/1.25/1.50. Phase hard maxima are A=1.00 s, B=1.00 s, C=0.60 s, D=0.60 s, total=2.50 s. Root linear/angular velocities are diagnostic only. The exact-medoid replay contains {len(exact_summaries)} plans and has coverage `{exact_coverage}`.

## Target set

`WMove03ValidatedEntryTargetSetV1` contains {target_manifest['references_per_side']['LEFT']}/{target_manifest['references_per_side']['RIGHT']} fresh D26T-validated native references. Five minimum train-only robust-scale compatibility references plus the exact medoid control were frozen before replay for each source/side. The full ranking is in `source_target_compatibility.csv/.json`; the shortlist is in `target_shortlist.json`. The target-set replay contains {len(target_set_summaries)} plans (maximum 288) and uses p50 clearance, measured target foot placement, and the same LIPM/DCM contract. Coverage is `{target_set_coverage}`.

## Offline feasibility

Selected plans are in `selected_offline_plans_v4.json`. Eligible target/timing rows include task errors, velocity margin, DCM endpoint error, and action continuity in `target_set_offline_replay.json`; no physics was executed. The selected target-set coverage is `{target_set_selected_coverage}`.

## Authorization

`exp014_d27_model_based_start_physics_authorization.json` is present only when the unchanged D27 coverage thresholds are met and the nonessential diagnostic conflict is absent; otherwise `exp014_d27_not_authorized.json` is present. Current authorization: **{authorization_base['authorized']}**.

## Protection

D26W, D26T, D26S, D26U, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, WBIK V1/V2/V2A, checkpoints, optimizers, datasets, physics/control parameters, and existing classifications are read-only. Persistent update: `0`; new learned checkpoint: `0`; model-based START physics: `0`; raw restore: `0`; PPO/CEM: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`. The hash audit is in `protected_hashes.json`.
""", encoding="utf-8")
    print(json.dumps({"classification": classification, "exact_coverage": exact_coverage, "target_set_coverage": target_set_coverage, "selected_coverage": target_set_selected_coverage, "authorization": authorization_base["authorized"], "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
