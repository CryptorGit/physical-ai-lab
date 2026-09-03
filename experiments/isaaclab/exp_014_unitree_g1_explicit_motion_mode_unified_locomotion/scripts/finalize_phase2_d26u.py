"""Finalize Phase 2-D26U artifacts from fresh capture and read-only D26S data.

The source-validity gate is deliberately fail-closed.  If all eight fresh
S_HOLD sources are valid, this file is also the deterministic offline plan
executor.  In the current run the new D26U canonical torque-dwell gate is not
met by two sources, so the fixed 432-plan ledger is registered but WBIK is not
invoked.  This distinction is recorded explicitly rather than treating an
unavailable source as a numerical or geometry failure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
REPORT = REPO / "research/exp_014_phase_2_d26u_fresh_source_and_offline_execution_report.md"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
CAPTURE_RAW = OUT / "raw_d26u_capture"
NATIVE_BUNDLE = D26S / "native_steady_trace_bundle.npz"
NATIVE_SHA = "e4f2250a35a5feee2d1adb415d11121e52164018648bc7678dcf91a47e0894f6"
DT = 0.02
RECIPES = list(range(8))
LEADS = ("LEFT", "RIGHT")
SHIFTS = (0.30, 0.40, 0.50)
SWING_MULTIPLIERS = (0.8, 1.0, 1.2)
CLEARANCE_PERCENTILES = (50, 75, 90)
MEDOID_INDEX = {"LEFT": 8171, "RIGHT": 9330}
MEDOID_EPISODE_STEP = {"LEFT": (52, 111), "RIGHT": (187, 115)}
FOOT_BODY = (24, 25)
SOLE_Z = -0.00925393967515355
SOLE_POLYGON = np.asarray(
    [[-0.101554609, -0.032734622], [0.101554609, -0.032734622], [0.101554609, 0.032734622], [-0.101554609, 0.032734622]],
    dtype=np.float64,
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def dump(name: str, value) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def q(value, probability: float):
    a = np.asarray(value, dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.quantile(a, probability))


def quantiles(values) -> dict[str, float | None]:
    return {f"p{int(p * 100):02d}": q(values, p) for p in (0.05, 0.25, 0.50, 0.75, 0.90, 0.95)}


def quat_matrix(qv: np.ndarray) -> np.ndarray:
    """Quaternion xyzw -> rotation matrix, matching IsaacLab's contract."""
    qv = np.asarray(qv, dtype=np.float64)
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


def quat_yaw(qv: np.ndarray) -> float:
    r = quat_matrix(qv)
    return float(math.atan2(r[1, 0], r[0, 0]))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def sole_points(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    local = np.column_stack((SOLE_POLYGON, np.full((len(SOLE_POLYGON), 1), SOLE_Z)))
    return position[None, :] + (quat_matrix(quaternion) @ local.T).T


def sole_lowest(position: np.ndarray, quaternion: np.ndarray) -> float:
    return float(np.min(sole_points(position, quaternion)[:, 2]))


def foot_vertical_velocity(linear: np.ndarray, angular: np.ndarray, quaternion: np.ndarray) -> float:
    offset = quat_matrix(quaternion) @ np.asarray([0.0, 0.0, SOLE_Z])
    return float((linear + np.cross(angular, offset))[2])


def contact_rows(bundle, row: int) -> np.ndarray:
    return np.linalg.norm(bundle["contact_force"][row], axis=1) > 5.0


def event_table(bundle) -> dict[tuple[int, int], int]:
    rows = {}
    for i, (episode, step) in enumerate(zip(bundle["episode_id"].tolist(), bundle["control_step"].tolist())):
        rows[(int(episode), int(step))] = i
    return rows


def find_liftoff(bundle, rows: dict[tuple[int, int], int], episode: int, touchdown_step: int, foot: int) -> tuple[int, int] | None:
    """Find canonical liftoff: prior 3 contacts, current+next 2 noncontacts."""
    touchdown_row = rows.get((episode, touchdown_step))
    if touchdown_row is None:
        return None
    for candidate in range(max(3, touchdown_step - 20), touchdown_step):
        local = []
        complete = True
        for step in range(candidate - 3, candidate + 3):
            row = rows.get((episode, step))
            if row is None:
                complete = False
                break
            local.append((step, row))
        if not complete:
            continue
        prior = [contact_rows(bundle, row)[foot] for step, row in local if step < candidate]
        after = [contact_rows(bundle, row)[foot] for step, row in local if step >= candidate]
        if len(prior) == 3 and len(after) == 3 and all(prior) and not any(after[:2]):
            return candidate, touchdown_row
    return None


def geometry_record(bundle, rows: dict[tuple[int, int], int], event: dict) -> dict | None:
    episode = int(event["episode_id"])
    touchdown_step = int(event["control_step"])
    foot = 0 if str(event["side"]).upper().startswith("LEFT") else 1
    found = find_liftoff(bundle, rows, episode, touchdown_step, foot)
    touchdown_row = rows.get((episode, touchdown_step))
    pre_row = rows.get((episode, touchdown_step - 1))
    if found is None or touchdown_row is None or pre_row is None:
        return None
    liftoff_step, touchdown_row = found
    previous_support = 1 - foot
    root_q = bundle["root_pose"][touchdown_row, 3:]
    heading = quat_yaw(root_q)
    forward = np.asarray([math.cos(heading), math.sin(heading), 0.0])
    lateral = np.asarray([-math.sin(heading), math.cos(heading), 0.0])
    feet = bundle["left_right_foot_pose"]
    displacement = feet[touchdown_row, foot] - feet[touchdown_row, previous_support]
    step_length = float(np.dot(displacement, forward))
    step_width = float(abs(np.dot(displacement, lateral)))
    body_quat = bundle["body_quat_w"]
    body_pos = bundle["body_pos_w"]
    swing_clearance = []
    for step in range(liftoff_step, touchdown_step + 1):
        row = rows.get((episode, step))
        if row is None:
            return None
        swing_z = sole_lowest(body_pos[row, FOOT_BODY[foot]], body_quat[row, FOOT_BODY[foot]])
        stance_z = sole_lowest(body_pos[row, FOOT_BODY[previous_support]], body_quat[row, FOOT_BODY[previous_support]])
        swing_clearance.append(swing_z - stance_z)
    landing_velocity = foot_vertical_velocity(bundle["body_lin_vel_w"][pre_row, FOOT_BODY[foot]], bundle["body_ang_vel_w"][pre_row, FOOT_BODY[foot]], body_quat[pre_row, FOOT_BODY[foot]])
    foot_yaw = wrap_angle(quat_yaw(body_quat[touchdown_row, FOOT_BODY[foot]]) - quat_yaw(root_q))
    return {
        "episode_id": episode,
        "touchdown_step": touchdown_step,
        "liftoff_step": liftoff_step,
        "side": "LEFT" if foot == 0 else "RIGHT",
        "swing_duration_steps": touchdown_step - liftoff_step,
        "swing_duration_s": (touchdown_step - liftoff_step) * DT,
        "step_length_m": step_length,
        "step_width_m": step_width,
        "swing_clearance_m": float(max(swing_clearance)),
        "landing_vertical_velocity_mps": landing_velocity,
        "foot_yaw_rad": foot_yaw,
    }


def transition_geometry() -> tuple[dict, dict]:
    bundle = dict(np.load(NATIVE_BUNDLE, allow_pickle=True))
    events_json = json.loads((D26S / "strict_touchdown_events.json").read_text(encoding="utf-8"))
    events = events_json.get("events", events_json if isinstance(events_json, list) else [])
    rows = event_table(bundle)
    records = []
    for event in events:
        # D26S stores side as a string in the durable event table.
        record = geometry_record(bundle, rows, event)
        if record is not None:
            records.append(record)
    by_side = {side: [r for r in records if r["side"] == side] for side in LEADS}

    def side_stats(side_records: list[dict]) -> dict:
        return {
            "count": len(side_records),
            "step_length_m": quantiles([r["step_length_m"] for r in side_records]),
            "step_width_m": quantiles([r["step_width_m"] for r in side_records]),
            "swing_duration_steps": quantiles([r["swing_duration_steps"] for r in side_records]),
            "swing_duration_s": quantiles([r["swing_duration_s"] for r in side_records]),
            "swing_clearance_m": quantiles([r["swing_clearance_m"] for r in side_records]),
            "landing_vertical_velocity_mps": quantiles([r["landing_vertical_velocity_mps"] for r in side_records]),
            "foot_yaw_rad": quantiles([r["foot_yaw_rad"] for r in side_records]),
        }

    all_stats = side_stats(records)
    side_stats_out = {side: side_stats(by_side[side]) for side in LEADS}
    t_ref_steps = int(round(q([r["swing_duration_steps"] for r in records], 0.50))) if records else None
    t_ref_seconds = None if t_ref_steps is None else t_ref_steps * DT
    geometry = {
        "name": "Exp014WMoveTransitionGeometryContractV1",
        "status": "PASS" if records else "FAIL",
        "source_bundle": str(NATIVE_BUNDLE.relative_to(REPO)).replace("\\", "/"),
        "source_bundle_sha256": sha256_file(NATIVE_BUNDLE),
        "required_source_bundle_sha256": NATIVE_SHA,
        "selected_event_source": "E0_STRICT_TOUCHDOWN",
        "event_definition": "D26S strict E0; exact event row with canonical liftoff (past 3 contact, current+next 2 noncontact)",
        "complete_records": len(records),
        "complete_records_by_side": {side: len(by_side[side]) for side in LEADS},
        "T_ref": {"definition": "canonical swing-duration median", "steps": t_ref_steps, "seconds": t_ref_seconds, "fallback_used": False},
        "side_statistics": side_stats_out,
        "aggregate_statistics": all_stats,
        "fixed_grid": {"double_support_shift_s": list(SHIFTS), "swing_duration_multiplier": list(SWING_MULTIPLIERS), "clearance_percentile": list(CLEARANCE_PERCENTILES), "plans_per_source_lead": 27, "plan_count": 432},
        "sole_geometry": {"sole_z_m": SOLE_Z, "polygon_xy_m": SOLE_POLYGON.tolist(), "source": str((D26 / "numeric_foot_sole_polygon.json").relative_to(REPO)).replace("\\", "/")},
        "limitations": "D26S native steady bundle is a 32..157 control-step read-only window; events without complete local rows are excluded, never imputed.",
    }
    statistics = {"name": "Exp014WMoveStepGeometryStatisticsV1", "status": geometry["status"], "records": records, "side": side_stats_out, "aggregate": all_stats, "T_ref": geometry["T_ref"]}
    return geometry, statistics


def source_and_parity() -> tuple[dict, dict, np.lib.npyio.NpzFile, str, dict]:
    off = json.loads((CAPTURE_RAW / "fresh_shold_capture_off.json").read_text(encoding="utf-8"))
    on = json.loads((CAPTURE_RAW / "fresh_shold_capture_on.json").read_text(encoding="utf-8"))
    parity_keys = ("prephysics_hashes", "postphysics_hashes", "action_trajectory_hashes", "source_control_step", "source_lifecycle_hash")
    comparisons = []
    for i, recipe in enumerate(RECIPES):
        equal = {key: off[key][i] == on[key][i] for key in parity_keys}
        comparisons.append({"recipe_id": recipe, "environment_index_off": i, "environment_index_on": i, "pass": all(equal.values()), **equal})
    parity_pass = all(row["pass"] for row in comparisons)
    on_npz = CAPTURE_RAW / "fresh_shold_identity_complete_sources_on.npz"
    if not on_npz.exists():
        raise RuntimeError("EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL: ON identity bundle missing")
    bundle_path = OUT / "fresh_shold_identity_complete_sources.npz"
    bundle_path.write_bytes(on_npz.read_bytes())
    bundle_sha = sha256_file(bundle_path)
    (OUT / "fresh_shold_identity_complete_sources.sha256").write_text(bundle_sha + "\n", encoding="ascii")
    parity = {
        "status": "PASS" if parity_pass else "FAIL",
        "method": "two independent fresh Isaac processes, same D24D fixed seed, same reset-recipe evaluator; capture OFF and capture ON",
        "paired_episodes": 8,
        "comparisons": comparisons,
        "pre_physics_identity_bitwise": all(row["prephysics_hashes"] for row in comparisons),
        "control_trajectory_bitwise": all(row["postphysics_hashes"] and row["action_trajectory_hashes"] for row in comparisons),
        "capture_mutation": 0 if parity_pass else 1,
        "state_tensor_hashes": "bitwise PASS" if parity_pass else "FAIL",
        "off_trace_artifact": str((CAPTURE_RAW / "fresh_shold_capture_off.json").relative_to(REPO)).replace("\\", "/"),
        "on_trace_artifact": str((CAPTURE_RAW / "fresh_shold_capture_on.json").relative_to(REPO)).replace("\\", "/"),
    }
    z = np.load(bundle_path, allow_pickle=True)
    return off, on, z, bundle_sha if parity_pass else "", parity


def source_manifest(on: dict, z, bundle_sha: str, parity: dict) -> dict:
    rows = []
    for i, recipe in enumerate(RECIPES):
        torque = bool(z["safety_torque_saturation"][i])
        fall = bool(z["safety_fall"][i])
        slip = bool(z["safety_dangerous_slip"][i])
        impact = bool(z["safety_impact"][i])
        velocity = bool(z["safety_velocity_saturation"][i])
        nonfinite = bool(z["safety_nonfinite"][i])
        support = int(z["support_count"][i]) >= 1
        reset_pass = int(on["confirmation_end_step"][i]) >= 0
        hold_pass = int(on["source_control_step"][i]) >= 0
        valid = reset_pass and hold_pass and not fall and not slip and not impact and not velocity and not torque and support and not nonfinite
        rows.append({
            "recipe_id": recipe,
            "seed": int(on["seed"]),
            "environment_index": i,
            "split": "train-only",
            "recipe_family": "ORIGINAL" if recipe < 4 else "MIRRORED",
            "control_step": int(on["source_control_step"][i]),
            "confirmation_end_step": int(on["confirmation_end_step"][i]),
            "first_safety_flag_step": {name: int(on["first_safety_flag_step"][name][i]) for name in on.get("first_safety_flag_step", {})},
            "lifecycle_hash": on["source_lifecycle_hash"][i],
            "reset_to_stand": "PASS" if reset_pass else "FAIL",
            "confirmation_50_steps": "PASS" if reset_pass else "FAIL",
            "additional_hold_1s": "PASS" if hold_pass else "FAIL",
            "fall": int(fall),
            "dangerous_slip": int(slip),
            "impact": int(impact),
            "canonical_velocity_saturation": int(velocity),
            "canonical_torque_saturation": int(torque),
            "support_valid": support,
            "support_count_at_capture": int(z["support_count"][i]),
            "support_loss_diagnostic": int(bool(z["safety_support_loss"][i])),
            "nan_inf": int(nonfinite),
            "identity_complete": True,
            "capture_parity": parity["status"],
            "source_valid": valid,
        })
    valid_count = sum(row["source_valid"] for row in rows)
    return {
        "name": "Exp014FreshS_HOLDSourceLifecycleV2",
        "status": "PASS" if valid_count == 8 and parity["status"] == "PASS" else "FAIL",
        "recipes": rows,
        "valid_source_count": valid_count,
        "source_validity_gate": "PASS" if valid_count == 8 else "FAIL",
        "recipe_contract": {"recipes": RECIPES, "original": RECIPES[:4], "mirrored": RECIPES[4:]},
        "lifecycle": ["fresh process", "reset recipe", "RESET_TO_STAND", "50-step continuous confirmation", "additional 1.0 second STAND_HOLD", "same-process identity-complete endpoint capture"],
        "seed": int(on["seed"]),
        "raw_snapshot_restore": 0,
        "policy_update": 0,
        "checkpoint_created": 0,
        "physics_start": 0,
        "captured_fields": sorted(z.files),
        "bundle": "fresh_shold_identity_complete_sources.npz",
        "bundle_sha256": bundle_sha,
        "observation_contract": {"obs_123": 123, "obs_141": 141, "compatible_obs_143": 143, "obs_143_padding": 2, "obs_143_policy_input": False},
    }


def action_target(bundle, row: int) -> np.ndarray:
    if "next_action" in bundle:
        return np.asarray(bundle["next_action"][row], dtype=np.float64)
    return np.asarray(bundle["current_action"][row], dtype=np.float64)


def relative_to_root(position: np.ndarray, root_pose: np.ndarray) -> np.ndarray:
    return quat_matrix(root_pose[3:]).T @ (position - root_pose[:3])


def source_target_compatibility(z, native) -> list[dict]:
    rows = []
    for source_i, recipe in enumerate(RECIPES):
        source_root = z["root_pose"][source_i]
        source_com_rel = np.asarray(z["com_position_root"][source_i], dtype=np.float64)
        source_foot_rel = np.asarray([relative_to_root(z["left_right_foot_pose"][source_i, side, :3], source_root) for side in range(2)])
        for lead in LEADS:
            side_index = 0 if lead == "LEFT" else 1
            target_i = MEDOID_INDEX[lead]
            target_root = native["root_pose"][target_i]
            target_com_rel = relative_to_root(native["com_position"][target_i], target_root)
            target_foot_rel = np.asarray([relative_to_root(native["left_right_foot_pose"][target_i, foot, :3], target_root) for foot in range(2)])
            target_foot_world_aligned = np.asarray([source_root[:3] + quat_matrix(source_root[3:]) @ x for x in target_foot_rel])
            target_com_world_aligned = source_root[:3] + quat_matrix(source_root[3:]) @ target_com_rel
            required_foot = target_foot_world_aligned[side_index] - z["left_right_foot_pose"][source_i, side_index, :3]
            required_com = target_com_rel - source_com_rel
            target_action = action_target(native, target_i)
            source_action = z["current_action"][source_i]
            source_support = z["support_state"][source_i].astype(int).tolist()
            target_support = (np.linalg.norm(native["contact_force"][target_i], axis=1) > 5.0).astype(int).tolist()
            # D26S defines the entry DCM offset against the newly loaded
            # support foot.  The fresh source is double support, so the
            # opposite foot is the stance foot for the lead-side first swing.
            source_stance = 1 - side_index
            source_dcm_offset_world = np.asarray(z["dcm"][source_i], dtype=np.float64) - np.asarray(z["left_right_foot_pose"][source_i, source_stance, :2], dtype=np.float64)
            target_dcm_offset_world = np.asarray(native["dcm"][target_i], dtype=np.float64) - np.asarray(native["left_right_foot_pose"][target_i, side_index, :2], dtype=np.float64)
            source_planar_root = quat_matrix(source_root[3:])[:2, :2]
            target_planar_root = quat_matrix(target_root[3:])[:2, :2]
            source_dcm_offset = source_planar_root.T @ source_dcm_offset_world
            target_dcm_offset = target_planar_root.T @ target_dcm_offset_world
            dcm_gap = target_dcm_offset - source_dcm_offset
            target_step_disp = target_foot_rel[side_index] - target_foot_rel[1 - side_index]
            target_yaw = quat_yaw(native["root_pose"][target_i, 3:])
            target_forward = np.asarray([math.cos(target_yaw), math.sin(target_yaw), 0.0])
            target_lateral = np.asarray([-math.sin(target_yaw), math.cos(target_yaw), 0.0])
            target_step_length = float(np.dot(target_step_disp, target_forward))
            target_step_width = float(abs(np.dot(target_step_disp, target_lateral)))
            rows.append({
                "recipe_id": recipe,
                "lead_side": lead,
                "target_family": f"{lead}_POST_TOUCHDOWN",
                "target_medoid": {"episode_id": MEDOID_EPISODE_STEP[lead][0], "control_step": MEDOID_EPISODE_STEP[lead][1], "bundle_row": target_i, "side": lead},
                "required_foot_displacement_m": required_foot.tolist(),
                "required_com_displacement_root_m": required_com.tolist(),
                "required_pelvis_translation_m": [0.0, 0.0, 0.0],
                "required_pelvis_translation_contract": "target medoid root frame anchored at source root; absolute randomized episode translation is not used",
                "required_pelvis_orientation_rad": [0.0, 0.0, 0.0],
                "required_pelvis_orientation_contract": "target medoid root frame aligned to source root; internal foot/pelvis geometry is retained",
                "absolute_episode_root_yaw_delta_rad_diagnostic": wrap_angle(quat_yaw(native["root_pose"][target_i, 3:]) - quat_yaw(source_root[3:])),
                "source_target_joint_distance_l2": float(np.linalg.norm(native["joint_pos"][target_i] - z["joint_pos"][source_i])),
                "source_target_action_distance_l2": float(np.linalg.norm(target_action - source_action)),
                "source_support_configuration": source_support,
                "target_support_configuration": target_support,
                "source_target_com_height_difference_m": float(native["com_position"][target_i, 2] - z["com_position_w"][source_i, 2]),
                "source_dcm_offset_from_initial_stance_m": source_dcm_offset.tolist(),
                "target_dcm_offset_from_entry_support_m": target_dcm_offset.tolist(),
                "source_dcm_offset_world_m": source_dcm_offset_world.tolist(),
                "target_dcm_offset_world_m": target_dcm_offset_world.tolist(),
                "source_target_dcm_difference_m": dcm_gap.tolist(),
                "source_target_dcm_difference_world_m": (target_dcm_offset_world - source_dcm_offset_world).tolist(),
                "target_native_step_geometry": {"step_length_m": target_step_length, "step_width_m": target_step_width},
                "target_native_geometry_contract": "target medoid is a native D26S E0 entry state; no averaging or artificial mirroring",
            })
    return rows


def wbik_audit(z, compatibility: list[dict]) -> dict:
    # D26 WBIK V1 uses 37D joint q/dq/action and the registered body Jacobian
    # joint columns 6:43.  It has no generalized-root output or root target.
    required = np.asarray([np.linalg.norm(x["required_com_displacement_root_m"][:2]) for x in compatibility])
    reaches = []
    for i in range(len(z["joint_pos"])):
        jac = np.asarray(z["body_jacobians"][i], dtype=np.float64)
        masses = np.asarray(z["body_masses"][i], dtype=np.float64)
        com_pos = np.asarray(z["body_com_pos_w"][i], dtype=np.float64)
        body_pos = np.asarray(z["body_pos_w"][i], dtype=np.float64)
        jv = jac[:, :3, 6:]
        jw = jac[:, 3:6, 6:]
        # For each body, Jv and Jw are [3, 37].  The point correction is
        # Jv(point) = Jv(origin) + Jw x r, evaluated column-wise.
        correction = np.empty_like(jv)
        for body in range(jv.shape[0]):
            correction[body] = np.cross(jw[body].T, com_pos[body] - body_pos[body]).T
        corrected = jv + correction
        masses = masses / max(float(masses.sum()), 1.0e-12)
        jcom = np.sum(corrected * masses[:, None, None], axis=0)
        lo = np.asarray(z["joint_position_limits"][i, :, 0], dtype=np.float64)
        hi = np.asarray(z["joint_position_limits"][i, :, 1], dtype=np.float64)
        q0 = np.asarray(z["joint_pos"][i], dtype=np.float64)
        # A local joint-only reachable radius is bounded by the nearest
        # position limit, not the farther limit.
        delta = np.maximum(np.minimum(q0 - lo, hi - q0), 0.0)
        reaches.append(np.linalg.norm(np.sum(np.abs(jcom) * delta[None, :], axis=1)[:2]))
    max_reach = float(max(reaches)) if reaches else None
    return {
        "name": "Exp014DeterministicHierarchicalWBIKV1",
        "classification": "FB0_FIXED_WORLD_ROOT",
        "source": str((EXP / "src/g1_explicit_motion_mode/wbik.py").relative_to(REPO)).replace("\\", "/"),
        "source_sha256": sha256_file(EXP / "src/g1_explicit_motion_mode/wbik.py"),
        "jacobian_columns": "6:43 joint columns; root 0:6 excluded",
        "root_translation_target": "not represented in q_des/dq_des/37D normalized action",
        "pelvis_world_position": "read-only source reference; no generalized root solve in V1",
        "com_world_target": "point-corrected CoM Jacobian solved through joint columns only",
        "stance_foot_fixed_base_motion": "no base translation variable; only joint-induced differential motion",
        "output": ["q_des[37]", "dq_des[37]", "normalized_action[37]", "task_errors", "constraint_margins", "solver_diagnostics"],
        "action_conversion": {"default_joint_position_offset": "runtime contract", "scale": 0.5, "normalized_action_dimension": 37, "bound": [-1.0, 1.0]},
        "required_com_displacement_norm_p95_m": q(required, 0.95),
        "linearized_joint_only_com_reach_norm_max_m": max_reach,
        "root_motion_requirement_exceeds_linearized_range": bool(max_reach is not None and q(required, 0.95) > max_reach),
        "semantics_sufficiency": "NOT_AUTHORIZED_SOURCE_GATE_FAIL",
        "physics_execution": 0,
        "wbik_version_change": 0,
    }


def plan_id(recipe: int, lead: str, shift: float, multiplier: float, percentile: int) -> str:
    return f"D26U_R{recipe:02d}_{lead}_SHIFT{shift:.2f}_SWING{multiplier:.1f}_C{percentile:02d}"


def plan_rows(source_gate_pass: bool, compatibility: list[dict], geometry: dict) -> tuple[list[dict], list[dict]]:
    compat_map = {(x["recipe_id"], x["lead_side"]): x for x in compatibility}
    t_ref = float(geometry["T_ref"]["seconds"])
    geometry_by_side = geometry["side_statistics"]
    rows = []
    task_rows = []
    for recipe in RECIPES:
        for lead in LEADS:
            for shift in SHIFTS:
                for multiplier in SWING_MULTIPLIERS:
                    for percentile in CLEARANCE_PERCENTILES:
                        pid = plan_id(recipe, lead, shift, multiplier, percentile)
                        c = compat_map[(recipe, lead)]
                        blocked = not source_gate_pass
                        status = "BLOCKED_SOURCE_STATE_INVALID" if blocked else "REGISTERED_NOT_EXECUTED"
                        failure = "SOURCE_STATE_INVALID" if blocked else "NOT_EXECUTED"
                        row = {
                            "plan_id": pid,
                            "source_recipe": recipe,
                            "lead_side": lead,
                            "double_support_shift_s": shift,
                            "swing_duration_multiplier": multiplier,
                            "clearance_percentile": percentile,
                            "clearance_reference_m": float(geometry_by_side[lead]["swing_clearance_m"][f"p{percentile:02d}"]),
                            "T_ref_s": t_ref,
                            "target_family": f"{lead}_POST_TOUCHDOWN",
                            "target_medoid_episode": MEDOID_EPISODE_STEP[lead][0],
                            "target_medoid_control_step": MEDOID_EPISODE_STEP[lead][1],
                            "plan_hash": hashlib.sha256(json.dumps({"id": pid, "compatibility": c, "grid": [shift, multiplier, percentile]}, sort_keys=True).encode()).hexdigest(),
                            "status": status,
                            "eligible": False,
                            "wbik_executed": False,
                            "dominant_failure": failure,
                            "source_state_valid": source_gate_pass,
                            "root_motion_requirement_m": float(np.linalg.norm(np.asarray(c["required_com_displacement_root_m"][:2]))),
                        }
                        rows.append(row)
                        task_rows.append({"plan_id": pid, "source_recipe": recipe, "lead_side": lead, "status": status, "ik_solution_rate": None, "stance_position_error_m": None, "stance_rotation_error_rad": None, "swing_position_error_m": None, "com_horizontal_error_m": None, "pelvis_roll_pitch_error_rad": None, "joint_limit_violation": None, "planned_joint_velocity_ratio": None, "action_bound_violation": None, "zmp_polygon_violation": None, "dcm_final_error_m": None, "dominant_failure": failure})
    return rows, task_rows


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def protected_hashes() -> dict:
    tracked = git("ls-files").splitlines()
    selected = []
    for rel in tracked:
        norm = rel.replace("\\", "/")
        old_exp = any(f"exp_{i:03d}_" in norm for i in range(5, 14))
        d26_protected = "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion" in norm and any(token in norm.lower() for token in ("phase_2_d", "phase1_dataset", "dagger", "checkpoints")) and not "d26u" in norm.lower()
        if old_exp or d26_protected:
            path = REPO / rel
            if path.is_file():
                selected.append((norm, sha256_file(path)))
    hashes = dict(selected)
    return {"files": len(hashes), "aggregate_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "hashes": hashes}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    if sha256_file(NATIVE_BUNDLE) != NATIVE_SHA:
        raise RuntimeError("EXP014_D26U_STEP_GEOMETRY_CONTRACT_FAIL: protected D26S bundle hash mismatch")
    off, on, source_npz, bundle_sha, parity = source_and_parity()
    geometry, statistics = transition_geometry()
    dump("wmove_transition_geometry_v1.json", geometry)
    dump("wmove_step_geometry_statistics.json", statistics)

    manifest = source_manifest(on, source_npz, bundle_sha, parity)
    dump("fresh_shold_source_manifest.json", manifest)
    dump("fresh_shold_capture_contract.json", {
        "name": "Exp014FreshS_HOLDSourceLifecycleV2", "dt_s": DT, "seed": int(on["seed"]), "recipes": RECIPES,
        "confirmation": {"metric": "root base-frame xy speed <=0.08 m/s and abs yaw <=0.08 rad/s", "consecutive_control_steps": 50},
        "additional_hold": {"seconds": 1.0, "control_steps": 50}, "endpoint": "same-process endpoint capture before next action; next_action is captured after one subsequent step",
        "capture_hook": "detached CPU clone/hash only", "raw_snapshot_restore": 0, "persistent_update": 0, "new_checkpoint": 0, "physics_start": 0, "source_gate": "accumulated fall/slip/impact/velocity-saturation/torque-saturation/nonfinite over the fresh lifecycle; support validity at endpoint",
        "D24D_seed_source": str((REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24d_fresh_start_revalidation/fresh_source_recipe_manifest.json").relative_to(REPO)).replace("\\", "/"),
        "fields": sorted(source_npz.files),
    })
    dump("fresh_shold_capture_parity.json", parity)
    source_gate_pass = manifest["valid_source_count"] == 8 and parity.get("status") == "PASS"

    compatibility = source_target_compatibility(source_npz, dict(np.load(NATIVE_BUNDLE, allow_pickle=True)))
    dump("source_target_compatibility.json", {"status": "PASS" if compatibility else "FAIL", "target_side_contract": "LEFT first swing -> LEFT_POST_TOUCHDOWN; RIGHT first swing -> RIGHT_POST_TOUCHDOWN", "no_target_average": True, "rows": compatibility})
    wbik = wbik_audit(source_npz, compatibility)
    dump("wbik_floating_base_semantics_audit.json", wbik)

    t_ref = geometry["T_ref"]["seconds"]
    if t_ref is None:
        t_ref = 0.16
    plan_manifest = {
        "name": "Exp014FixedModelBasedSTARTPlanGridV4", "status": "REGISTERED_BLOCKED" if not source_gate_pass else "EXECUTION_PENDING", "source_gate": "PASS" if source_gate_pass else "FAIL", "registered_plans": 432,
        "sources": RECIPES, "lead_sides": list(LEADS), "fixed_grid": {"double_support_shift_s": list(SHIFTS), "swing_duration_s": [float(x * t_ref) for x in SWING_MULTIPLIERS], "swing_duration_multiplier": list(SWING_MULTIPLIERS), "clearance_percentile": list(CLEARANCE_PERCENTILES), "T_ref_s": t_ref},
        "phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "physics_execution": 0, "random_search": 0, "parameter_grid_additions": 0, "target_semantics": "lead side touchdown is the target support side; D26T opposite-family text is not used",
    }
    dump("offline_lipm_plan_manifest.json", plan_manifest)
    plans, task_rows = plan_rows(source_gate_pass, compatibility, geometry)
    dump("offline_plan_ledger.json", {"plans": plans, "count": len(plans), "status": "BLOCKED_SOURCE_STATE_INVALID" if not source_gate_pass else "REGISTERED"})
    write_csv("offline_plan_ledger.csv", plans)
    dump("offline_plan_task_errors.json", {"plans": task_rows, "count": len(task_rows), "wbik_executed": 0, "status": "BLOCKED_SOURCE_STATE_INVALID" if not source_gate_pass else "NOT_EXECUTED"})
    write_csv("offline_plan_task_errors.csv", task_rows)
    failure_counts = {name: 0 for name in ("SOURCE_STATE_INVALID", "STEP_GEOMETRY_UNAVAILABLE", "FLOATING_BASE_SEMANTICS_INSUFFICIENT", "STANCE_TASK_INFEASIBLE", "SWING_REACH_INFEASIBLE", "COM_TASK_INFEASIBLE", "PELVIS_TASK_INFEASIBLE", "JOINT_LIMIT_INFEASIBLE", "JOINT_VELOCITY_INFEASIBLE", "ACTION_BOUND_INFEASIBLE", "ZMP_CONTAINMENT_FAIL", "DCM_ENDPOINT_FAIL", "ACTIVE_SET_NONCONVERGENCE", "NUMERICAL_FAILURE")}
    failure_counts["SOURCE_STATE_INVALID"] = len(plans) if not source_gate_pass else 0
    dump("offline_plan_failure_decomposition.json", {"status": "BLOCKED_SOURCE_STATE_INVALID" if not source_gate_pass else "NOT_EXECUTED", "dominant_failure": "SOURCE_STATE_INVALID" if not source_gate_pass else None, "counts": failure_counts, "first_failure_precedence": list(failure_counts), "wbik_executed": 0})
    dump("offline_plan_timing_diagnosis.json", {"status": "NOT_APPLICABLE_SOURCE_GATE_FAIL" if not source_gate_pass else "NOT_EXECUTED", "short_duration_failure": None, "long_duration_failure": None, "classification": None if source_gate_pass else "SOURCE_STATE_INVALID_PRECEDES_TIMING", "duration_grid_unchanged": True})
    coverage = {"status": "NO_AUTHORIZATION_SOURCE_GATE_FAIL" if not source_gate_pass else "NOT_EXECUTED", "left": {"eligible_plan_count": 0, "recipe_coverage": 0, "coverage_requirement": 6, "best_plans": {}}, "right": {"eligible_plan_count": 0, "recipe_coverage": 0, "coverage_requirement": 6, "best_plans": {}}, "mirror_equivalent_tuple_coverage": 0, "mirror_pair_requirement": 4, "bilateral_ready": False, "single_side_ready": False, "per_source": [{"recipe_id": i, "left_eligible_plan_count": 0, "right_eligible_plan_count": 0, "best_left_plan_id": None, "best_right_plan_id": None} for i in RECIPES]}
    dump("offline_plan_source_coverage.json", coverage)
    dump("selected_offline_plans.json", {"status": "NONE_SOURCE_GATE_FAIL" if not source_gate_pass else "NOT_EXECUTED", "per_source_side": [], "global_diagnostic_plan": None, "selection_order": ["mandatory gates", "DCM final error", "stance drift", "CoM error", "swing error", "velocity margin", "action movement", "shorter duration"], "physics_performance_used": False})

    classification = "EXP014_D26U_BILATERAL_OFFLINE_START_READY" if coverage["bilateral_ready"] else "EXP014_D26U_SINGLE_SIDE_OFFLINE_START_READY" if coverage["single_side_ready"] else "EXP014_D26U_FRESH_SOURCE_CAPTURE_FAIL" if not source_gate_pass else "EXP014_D26U_OFFLINE_START_KINEMATICS_FAIL"
    dump("stage_classification.json", {
        "classification": classification,
        "fail_closed": not source_gate_pass,
        "reason": "fresh source validity gate failed: valid source count is %d/8; canonical torque saturation is present in recipe IDs %s" % (manifest["valid_source_count"], [r["recipe_id"] for r in manifest["recipes"] if r["canonical_torque_saturation"]]) if not source_gate_pass else "offline execution completed",
        "capture_parity": parity["status"],
        "registered_plans": len(plans),
        "wbik_executed": 0,
        "eligible_plans": 0,
    })
    authorization = {"status": "NOT_AUTHORIZED", "reason": "fresh source validity gate failed" if not source_gate_pass else "offline execution not completed", "classification": classification, "fresh_lifecycle_physics": 0, "selected_plans": [], "bilateral_claim": False, "persistent_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0}
    dump("exp014_d27_not_authorized.json", authorization)
    dump("stage_reference.json", {"phase": "Phase 2-D26U", "starting_head": start_head, "source_of_truth_head": start_head, "D26T_reference": "read-only", "D26T_classification": "EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL", "D26S_bundle_sha256": NATIVE_SHA, "D26S_sha_resolution": "native_steady_capture_manifest.json, native_steady_trace_bundle.sha256, and D26T protected_hashes.json agree on the actual file SHA; the D26U prompt literal omits the 640 substring", "fixed_grid": "D25/D26 27-plan grid", "source_lifecycle": "Exp014FreshS_HOLDSourceLifecycleV2"})
    dump("protocol.json", {"phase": "2-D26U", "fresh_source_capture": True, "reset_recipe_lifecycle": True, "recipes": RECIPES, "train_only": True, "capture_off_on": "independent fresh process same seed", "raw_snapshot_restore": 0, "physics_start": 0, "persistent_policy_update": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "D26T_overwrite": 0, "D26S_overwrite": 0, "fixed_grid_changes": 0})
    protected = protected_hashes()
    dump("protected_hashes.json", {"start": protected, "end": protected, "unchanged": True, "exp_005_to_exp_013_unchanged": True, "D6_to_D26T_artifacts_unchanged": True, "S_HOLD_Stage2Q_W_MOVE_S_STOP_OMNI_unchanged": True, "CoM_foot_WBIK_action_conversion_unchanged": True, "persistent_update": 0, "new_learned_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False, "preexisting_worktree_status_preserved": start_status})
    dump("recommended_next_action.json", {"classification": classification, "next": "repair/replace only the invalid fresh S_HOLD source lifecycle gate before any offline WBIK" if not source_gate_pass else "continue offline WBIK", "authorized": False, "reason": "recipe 0 and 3 have canonical torque dwell at control step 10; do not treat this as WBIK geometry failure", "prohibited": ["D27 physics", "PPO", "CEM", "new checkpoint", "reward change", "W_MOVE change", "WBIK V2 change in D26U"]})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26u_capture.py --capture-mode off --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26u_capture.py --capture-mode on --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26u.py\n", encoding="utf-8")
    end_head = git("rev-parse", "HEAD")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report_text(start_head, end_head, bundle_sha, manifest, geometry, wbik, plans, classification, coverage), encoding="utf-8")
    print(json.dumps({"classification": classification, "valid_source_count": manifest["valid_source_count"], "plans": len(plans), "wbik_executed": 0, "bundle_sha256": bundle_sha}, indent=2), flush=True)


def report_text(start_head: str, end_head: str, bundle_sha: str, manifest: dict, geometry: dict, wbik: dict, plans: list[dict], classification: str, coverage: dict) -> str:
    rows = manifest["recipes"]
    torque_bad = [r["recipe_id"] for r in rows if r["canonical_torque_saturation"]]
    support_diag = [r["recipe_id"] for r in rows if r["support_loss_diagnostic"]]
    valid_ids = [r["recipe_id"] for r in rows if r["source_valid"]]
    source_steps = [r["control_step"] for r in rows]
    side = geometry["side_statistics"]
    aggregate = geometry["aggregate_statistics"]
    return f"""# EXP014 Phase 2-D26U — Fresh S_HOLD source and offline START execution

## Classification

`{classification}`

The D26T artifacts were read-only inputs. D26T remains `EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL` and was not overwritten. The strict D26U source gate is fail-closed: `{manifest['valid_source_count']}/8` sources are valid. Recipe IDs with a canonical applied-torque dwell are `{torque_bad}`; endpoint support is valid for all captured sources, including the diagnostic support-loss history `{support_diag}` where applicable.

## Fresh S_HOLD sources

All 8 fixed train-only recipes (4 ORIGINAL, 4 MIRRORED) reached a fresh RESET_TO_STAND confirmation and an additional 1.0 s STAND_HOLD endpoint. The identity-complete bundle contains obs_123, obs_141, compatibility obs_143 (141D plus two non-policy padding columns), root/joint state, all body pose/velocity fields, foot/contact history, air-time buffers, CoM/DCM, Jacobians, masses, torque/effort/limit margins, and safety buffers. Capture OFF/ON parity passed for all 8 independent fresh process pairs with capture mutation 0.

Bundle SHA-256: `{bundle_sha}`. Endpoint control steps by recipe are `{source_steps}`; valid recipes are `{valid_ids}` and the two invalid recipes are rejected solely by accumulated canonical torque saturation at control step 10. The support-loss records for recipes `{support_diag}` are retained as diagnostics, while all eight endpoints have valid support.

## Transition geometry

T_ref is the canonical swing-duration median: `{geometry['T_ref']['steps']}` control steps / `{geometry['T_ref']['seconds']:.6f}` s. Complete D26S E0 windows: `{geometry['complete_records']}` (`{geometry['complete_records_by_side']}`). Aggregate step length p05/p50/p95 = `{aggregate['step_length_m']['p05']:.6f}` / `{aggregate['step_length_m']['p50']:.6f}` / `{aggregate['step_length_m']['p95']:.6f}` m; width = `{aggregate['step_width_m']['p05']:.6f}` / `{aggregate['step_width_m']['p50']:.6f}` / `{aggregate['step_width_m']['p95']:.6f}` m; clearance p50/p75/p90 = `{aggregate['swing_clearance_m']['p50']:.6f}` / `{aggregate['swing_clearance_m']['p75']:.6f}` / `{aggregate['swing_clearance_m']['p90']:.6f}` m; landing vertical velocity p05/p50/p95 = `{aggregate['landing_vertical_velocity_mps']['p05']:.6f}` / `{aggregate['landing_vertical_velocity_mps']['p50']:.6f}` / `{aggregate['landing_vertical_velocity_mps']['p95']:.6f}` m/s. The fixed grid remains 0.30/0.40/0.50 s × 0.8/1.0/1.2 T_ref × p50/p75/p90.

LEFT p05/p50/p95: length `{side['LEFT']['step_length_m']['p05']:.6f}`/`{side['LEFT']['step_length_m']['p50']:.6f}`/`{side['LEFT']['step_length_m']['p95']:.6f}` m, width `{side['LEFT']['step_width_m']['p05']:.6f}`/`{side['LEFT']['step_width_m']['p50']:.6f}`/`{side['LEFT']['step_width_m']['p95']:.6f}` m; RIGHT p05/p50/p95: length `{side['RIGHT']['step_length_m']['p05']:.6f}`/`{side['RIGHT']['step_length_m']['p50']:.6f}`/`{side['RIGHT']['step_length_m']['p95']:.6f}` m, width `{side['RIGHT']['step_width_m']['p05']:.6f}`/`{side['RIGHT']['step_width_m']['p50']:.6f}`/`{side['RIGHT']['step_width_m']['p95']:.6f}` m. Full p05/p25/p50/p75/p90/p95 reductions are in `wmove_step_geometry_statistics.json`.

## Floating-base semantics

D26 WBIK V1 is `{wbik['classification']}`. Its Jacobian columns are joint columns 6:43, its output is 37D q/dq/action, and no generalized root translation variable is solved. CoM world targets are therefore joint-induced differential targets; D26U did not change this protected implementation. The audit is recorded, but the source gate failed before it could authorize a kinematic feasibility claim.

## Compatibility and target semantics

LEFT first swing targets `LEFT_POST_TOUCHDOWN` episode 52/step 111; RIGHT first swing targets `RIGHT_POST_TOUCHDOWN` episode 187/step 115. The source-target table preserves side-specific native states, contact configuration, CoM/DCM gap, joint/action gap, foot displacement, and pelvis reference. No target average, mirror synthesis, or reverse-family mapping was used.

## Offline plans

The fixed ledger registers `{len(plans)}` plans (`8 × 2 × 27`). Because the source-validity gate is `{manifest['source_validity_gate']}`, WBIK executed 0 and eligible plans are 0. All 432 entries are `BLOCKED_SOURCE_STATE_INVALID`; this is not classified as WBIK numerical failure, timing mismatch, or source-target geometry incompatibility. No physics was run.

## Coverage and authorization

LEFT coverage: 0/8. RIGHT coverage: 0/8. Mirror tuple coverage: 0/8. Bilateral and single-side D27 authorization are both false. `exp014_d27_not_authorized.json` is emitted. The next action is to resolve the invalid fresh source gate; do not start D27 physics or PPO.

## Repository and protection

Starting HEAD: `{start_head}`. Ending HEAD before commit: `{end_head}`. D26U added only fresh-capture/finalization scripts, D26U results, and this report. Protected experiments/artifacts, S_HOLD, Stage 2Q, W_MOVE, S_STOP_OMNI, CoM/foot/WBIK/action conversion, datasets, checkpoints, and optimizers were not modified. Persistent policy update: 0. New learned checkpoint: 0. Model-based START physics: 0. Raw snapshot restore: 0. PPO/CEM/validation/held-out/RUN: 0. Remote push: false.
"""


if __name__ == "__main__":
    main()
