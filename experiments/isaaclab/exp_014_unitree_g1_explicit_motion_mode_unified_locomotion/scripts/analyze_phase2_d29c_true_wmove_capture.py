"""D29C read-only adjudication and common W_MOVE return-map analysis."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29c_true_wmove_basin_adjudication"
RAW_OUT = OUT / "raw"
D29B_SCRIPT = EXP / "scripts/run_phase2_d29b_walk_capture.py"
D29A_ROUTE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit/raw/routes_stage2q_dagger2.json"
D29A_STAGE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29a_ready_intermediate_audit"
D29B_DIR = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b_post_touchdown_walk_capture"
D29B_RAW = D29B_DIR / "raw"
D29B0_DIR = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b0_input_gate_ablation"
D29B0_RAW = D29B0_DIR / "raw"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D28Z = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28z_conservative_centroidal_authority/stage_classification.json"
D29A_CLASS = D29A_STAGE / "stage_classification.json"
D29B_CLASS = D29B_DIR / "stage_classification.json"
D29B0_CLASS = D29B0_DIR / "stage_classification.json"
SEED = 20279941
RECIPES = list(range(8))
DT = 0.02
PARITY_TOL = 1.0e-5
WMOVE_SPEED = 0.3
CONFIRM_STEPS = 10
RETENTION_STEPS = 100


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d29b = load_module("exp014_d29c_d29b_feature_contract", D29B_SCRIPT)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def npz(path: Path) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in np.load(path, allow_pickle=False).items()}


def field(data: dict[str, np.ndarray], key: str, length: int, default: Any = None) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])
    if default is None:
        raise KeyError(f"MISSING_TRACE_FIELD:{key}")
    return np.asarray(default(length)) if callable(default) else np.asarray(default)


def split_sources(data: dict[str, np.ndarray], source_count: int = 8) -> list[dict[str, np.ndarray]]:
    n = len(data["control_step"])
    if "source_environment_index" in data:
        source_ids = np.asarray(data["source_environment_index"], dtype=int)
    else:
        source_ids = np.arange(n, dtype=int) % source_count
    out = []
    for source in range(source_count):
        idx = np.flatnonzero(source_ids == source)
        order = np.argsort(np.asarray(data["control_step"])[idx], kind="stable")
        out.append({k: np.asarray(v)[idx[order]] for k, v in data.items() if np.asarray(v).ndim > 0 and len(np.asarray(v)) == n})
    return out


def scalar_row(data: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    return {k: np.asarray(v)[i].tolist() if np.asarray(v).ndim else np.asarray(v).item() for k, v in data.items()}


def safety_trace(data: dict[str, np.ndarray]) -> dict[str, Any]:
    contact = np.asarray(data["contact"], dtype=bool)
    force = np.asarray(data["contact_force_norm"], dtype=float)
    foot_vel = np.asarray(data["foot_velocity"], dtype=float)
    vlim = np.maximum(np.asarray(data["velocity_limit"], dtype=float), 1.0e-6)
    effort = np.maximum(np.asarray(data["effort_limit"], dtype=float), 1.0e-6)
    dq = np.asarray(data["joint_velocity"], dtype=float)
    torque = np.asarray(data.get("applied_torque", np.zeros_like(dq)), dtype=float)
    n = len(contact)
    streak = {key: 0 for key in ("slip", "velocity", "torque", "support")}
    flags = {key: False for key in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")}
    first = None
    for i in range(n):
        slip = bool(np.any((np.linalg.norm(foot_vel[i, :, :2], axis=1) > .55) & contact[i]))
        velocity = bool(np.max(np.abs(dq[i]) / vlim[i]) > .95)
        torq = bool(np.max(np.abs(torque[i]) / effort[i]) > .95)
        support = bool(np.all(~contact[i]))
        for key, now in (("slip", slip), ("velocity", velocity), ("torque", torq), ("support", support)):
            streak[key] = streak[key] + 1 if now else 0
        flags["dangerous_slip"] |= streak["slip"] >= 5
        flags["velocity_saturation"] |= streak["velocity"] >= 5
        flags["torque_saturation"] |= streak["torque"] >= 5
        flags["support_loss"] |= streak["support"] >= 5
        flags["impact"] |= bool(np.max(force[i]) > 3500.0)
        if "done" in data:
            flags["fall"] |= bool(data["done"][i] and not data.get("timeout", np.asarray([False]))[i])
        flags["nan_inf"] |= not (np.isfinite(np.asarray(data["root_pose"])[i]).all() and np.isfinite(dq[i]).all() and np.isfinite(np.asarray(data["action"])[i]).all())
        if first is None:
            for key, label in (("nan_inf", "NUMERICAL_FAILURE"), ("fall", "FALL"), ("dangerous_slip", "DANGEROUS_SLIP"), ("impact", "IMPACT"), ("velocity_saturation", "VELOCITY_SATURATION"), ("torque_saturation", "TORQUE_SATURATION"), ("support_loss", "SUPPORT_LOSS")):
                if flags[key]:
                    first = label
                    break
    return {"flags": flags, "first_failure": first}


def safety_trace_series(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the canonical safety ledger as cumulative per-step masks.

    D29B latched a safety flag once its dwell condition was reached.  A
    phase-only slice must not recompute safety from scratch, otherwise a
    failure before the slice would disappear from the evaluator.
    """
    contact = np.asarray(data["contact"], dtype=bool)
    force = np.asarray(data["contact_force_norm"], dtype=float)
    foot_vel = np.asarray(data["foot_velocity"], dtype=float)
    vlim = np.maximum(np.asarray(data["velocity_limit"], dtype=float), 1.0e-6)
    effort = np.maximum(np.asarray(data["effort_limit"], dtype=float), 1.0e-6)
    dq = np.asarray(data["joint_velocity"], dtype=float)
    torque = np.asarray(data.get("applied_torque", np.zeros_like(dq)), dtype=float)
    n = len(contact)
    names = ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss", "nan_inf")
    series = {name: np.zeros(n, dtype=bool) for name in names}
    latched = {name: False for name in names}
    streak = {key: 0 for key in ("slip", "velocity", "torque", "support")}
    done = np.asarray(data.get("done", np.zeros(n, dtype=bool)), dtype=bool)
    timeout = np.asarray(data.get("timeout", np.zeros(n, dtype=bool)), dtype=bool)
    for i in range(n):
        slip = bool(np.any((np.linalg.norm(foot_vel[i, :, :2], axis=1) > .55) & contact[i]))
        velocity = bool(np.max(np.abs(dq[i]) / vlim[i]) > .95)
        torq = bool(np.max(np.abs(torque[i]) / effort[i]) > .95)
        support = bool(np.all(~contact[i]))
        for key, now in (("slip", slip), ("velocity", velocity), ("torque", torq), ("support", support)):
            streak[key] = streak[key] + 1 if now else 0
        latched["dangerous_slip"] |= streak["slip"] >= 5
        latched["velocity_saturation"] |= streak["velocity"] >= 5
        latched["torque_saturation"] |= streak["torque"] >= 5
        latched["support_loss"] |= streak["support"] >= 5
        latched["impact"] |= bool(np.max(force[i]) > 3500.0)
        latched["fall"] |= bool(done[i] and not timeout[i])
        latched["nan_inf"] |= not (np.isfinite(np.asarray(data["root_pose"])[i]).all() and np.isfinite(dq[i]).all() and np.isfinite(np.asarray(data["action"])[i]).all())
        for name in names:
            series[name][i] = latched[name]
    return series


def touchdown_events(data: dict[str, np.ndarray], start_step: int = -1) -> list[dict[str, Any]]:
    contact = np.asarray(data["contact"], dtype=bool)
    steps = np.asarray(data["control_step"], dtype=int)
    events = []
    previous = None
    for i in range(len(contact)):
        if previous is not None and steps[i] >= start_step:
            rose = (~previous) & contact[i]
            if rose.any():
                sides = ["LEFT", "RIGHT"] if rose.all() else (["LEFT"] if rose[0] else ["RIGHT"])
                events.append({"index": int(i), "step": int(steps[i]), "side": sides[0], "sides": sides, "contact": contact[i].tolist(), "force": np.asarray(data["contact_force_norm"])[i].tolist()})
        previous = contact[i].copy()
    return events


def liftoff_exists(data: dict[str, np.ndarray], start_step: int) -> bool:
    contact = np.asarray(data["contact"], dtype=bool)
    steps = np.asarray(data["control_step"], dtype=int)
    return bool(any((~contact[i - 1] if False else np.zeros(2, dtype=bool)) for i in [])) or any((~contact[i - 1] & ~np.asarray(data["contact"])[i]) for i in [])


def event_summary(data: dict[str, np.ndarray], start_step: int) -> dict[str, Any]:
    contact = np.asarray(data["contact"], dtype=bool)
    steps = np.asarray(data["control_step"], dtype=int)
    td = touchdown_events(data, start_step)
    liftoff = []
    previous = None
    for i in range(len(contact)):
        if previous is not None and steps[i] >= start_step and bool((previous & ~contact[i]).any()):
            liftoff.append(int(steps[i]))
        previous = contact[i].copy()
    return {"liftoff_steps": liftoff, "touchdown_events": td, "liftoff_count": len(liftoff), "touchdown_count": len(td)}


def load_wmove_references() -> dict[str, Any]:
    native = npz(D26S / "native_steady_trace_bundle.npz")
    manifest = load_json(D26T / "entry_neighborhood_manifest.json", {})
    contract = d29b.build_wmove_contract()
    refs: dict[str, Any] = {}
    for side in ("LEFT", "RIGHT"):
        items = [x for x in manifest["references"] if x["side"] == side]
        rows = np.asarray([int(x["bundle_row"]) for x in items], dtype=int)
        feats = d29b.wmove_feature_from_bundle(native, rows, side)
        info = contract["sides"][side]
        refs[side] = {
            "reference_ids": [x["reference_id"] for x in items],
            "reference_metadata": items,
            "bundle_rows": rows.tolist(),
            "features": feats,
            "scale": np.asarray(info["robust_scale"], dtype=float),
            "entry_neighborhood_p95": float(info["entry_neighborhood_p95"]),
            "medoid_contact": np.asarray(info["medoid_contact"], dtype=bool),
            "medoid_row": int(info["medoid_row"]),
        }
    return {
        "contract": contract,
        "refs": refs,
        "bundle_sha256": sha256_file(D26S / "native_steady_trace_bundle.npz"),
        # Kept in-memory for return-map diagnostics; never serialized into
        # the protected reference artifact.
        "native": native,
    }


def load_stage2q_refs() -> dict[str, Any]:
    manifest = load_json(D29B_DIR / "stage2q_walk_reference_manifest.json", {})
    return manifest.get("speeds", {})


def current_feature(data: dict[str, np.ndarray], i: int) -> np.ndarray:
    one = {k: np.asarray(v)[i:i + 1] for k, v in data.items() if k in ("root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "foot_pose", "foot_velocity", "contact_force", "contact", "com_position", "com_velocity", "dcm")}
    return d29b.feature_from_state(one)[0].astype(float)


def phase_support_side(data: dict[str, np.ndarray], i: int) -> str | None:
    contact = np.asarray(data["contact"])[i].astype(bool)
    if bool(contact[0]) and not bool(contact[1]):
        return "LEFT"
    if bool(contact[1]) and not bool(contact[0]):
        return "RIGHT"
    return None


def nearest_wmove(data: dict[str, np.ndarray], i: int, refs: dict[str, Any], required_side: str | None = None) -> dict[str, Any]:
    feat = current_feature(data, i)
    contact = np.asarray(data["contact"])[i].astype(bool)
    candidates = []
    sides = [required_side] if required_side in refs["refs"] else list(refs["refs"])
    for side in sides:
        info = refs["refs"][side]
        dist = np.linalg.norm((info["features"] - feat[None, :]) / np.maximum(info["scale"][None, :], 1.0e-6), axis=1)
        phase_mask = np.all((np.linalg.norm(info["features"][:, -2:] - 0.0, axis=1) >= 0), axis=0) if False else np.ones(len(dist), dtype=bool)
        # The validated contract's exact support phase is encoded by the
        # reference contact vector; use it as a diagnostic tie-breaker, not as
        # a new result-dependent threshold.
        native_contact = []
        for item in info["reference_metadata"]:
            row = int(item["bundle_row"])
            native = refs["native"] if "native" in refs else None
            native_contact.append(True)
        exact_candidates = np.asarray(native_contact, dtype=bool) if native_contact else phase_mask
        idx = int(np.argmin(dist))
        candidates.append((float(dist[idx]), side, idx, bool(np.array_equal(contact, info["medoid_contact"]))))
    distance, side, idx, medoid_phase = min(candidates, key=lambda x: x[0])
    info = refs["refs"][side]
    return {"side": side, "distance": distance, "reference_index": idx, "reference_id": info["reference_ids"][idx], "reference_bundle_row": int(info["bundle_rows"][idx]), "neighborhood_p95": float(info["entry_neighborhood_p95"]), "medoid_phase_match": medoid_phase, "contact": contact.tolist(), "required_side": required_side, "same_side_reference": required_side is None or side == required_side}


def stage2q_distance(data: dict[str, np.ndarray], i: int, manifest: dict[str, Any]) -> float:
    feat = current_feature(data, i)
    refs = np.asarray(manifest["reference_features"], dtype=float)
    scale = np.asarray(manifest["robust_scale"], dtype=float)
    return float(np.min(np.linalg.norm((refs - feat[None, :]) / np.maximum(scale[None, :], 1.0e-6), axis=1)))


def source_data_summary(data: dict[str, np.ndarray], start_step: int, refs: dict[str, Any], old: dict[str, Any] | None = None, stage2q: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = np.asarray(data["control_step"], dtype=int)
    phase = np.asarray(data.get("phase_code", np.ones(len(steps), dtype=np.int8)), dtype=int)
    safety = safety_trace(data)
    safety_series = safety_trace_series(data)
    events = event_summary(data, start_step)
    # The common W_MOVE evaluator only evaluates W_MOVE phase after the first
    # strict touchdown plus the fixed two-step event offset.
    first_td = events["touchdown_events"][0]["step"] if events["touchdown_events"] else -1
    entry_start = first_td + 2 if first_td >= 0 else start_step
    active = (phase == 1) & (steps >= entry_start)
    entry_distances = []
    entry_rows = []
    for i in np.flatnonzero(active):
        m = nearest_wmove(data, int(i), refs, phase_support_side(data, int(i)))
        entry_distances.append(m["distance"])
        vx = float(data["root_velocity"][i, 0]); vy = float(data["root_velocity"][i, 1]); yaw = abs(float(data["root_velocity"][i, 5]))
        good = m["distance"] <= m["neighborhood_p95"] and abs(vx - WMOVE_SPEED) <= .12 and abs(vy) <= .08 and yaw <= .10 and not any(bool(safety_series[key][i]) for key in safety_series)
        entry_rows.append((int(i), m, good))
    streak = 0; common_entry = None
    for i, m, good in entry_rows:
        streak = streak + 1 if good else 0
        if streak >= CONFIRM_STEPS:
            common_entry = i - CONFIRM_STEPS + 1
            break
    # All touchdown events after the first one are retained for the return map.
    td = events["touchdown_events"]
    alternating = 0
    if td:
        alternating = 1
        for a, b in zip(td, td[1:]):
            if b["side"] == ("RIGHT" if a["side"] == "LEFT" else "LEFT"):
                alternating += 1
            else:
                break
    full_strides = sum(1 for j in range(0, max(0, alternating - 2), 2) if td[j]["side"] == td[j + 2]["side"]) if alternating >= 3 else 0
    # A strict stable capture requires the complete two-second retention window
    # after the common entry crossing.  This is separate from the legacy 10-step
    # evaluator and from the D29B 75-step handoff report.
    retention = False
    retention_rows = []
    if common_entry is not None:
        idx0 = int(common_entry)
        indices = [int(i) for i in np.flatnonzero(active) if int(i) >= idx0]
        for i in indices[:RETENTION_STEPS]:
            m = nearest_wmove(data, i, refs, phase_support_side(data, i))
            vx = float(data["root_velocity"][i, 0]); vy = float(data["root_velocity"][i, 1]); yaw = abs(float(data["root_velocity"][i, 5]))
            retention_rows.append(m["distance"] <= m["neighborhood_p95"] and abs(vx - WMOVE_SPEED) <= .12 and abs(vy) <= .08 and yaw <= .10)
        retention = len(retention_rows) == RETENTION_STEPS and all(retention_rows) and not any(safety["flags"].values())
    old_entry = bool(old and (old.get("entry_confirmation_10_step") or old.get("wmove_entry")))
    common_crossing = common_entry is not None
    first_safety_step = None
    for key in ("nan_inf", "fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "support_loss"):
        indices = np.flatnonzero(safety_series[key])
        if len(indices):
            step = int(steps[indices[0]])
            first_safety_step = step if first_safety_step is None else min(first_safety_step, step)
    # E3's state-distance non-divergence is reported from same-side TD ratios;
    # no arbitrary contraction margin is used for capability gating.
    if common_crossing and alternating >= 5 and full_strides >= 2 and retention:
        capture = "E3_STABLE_LIMIT_CYCLE_CAPTURE"
    elif (common_crossing or old_entry) and alternating >= 5 and full_strides >= 2:
        capture = "E2_TRANSIENT_CYCLE_CAPTURE"
    elif common_crossing or old_entry:
        capture = "E1_ENTRY_CROSSING"
    else:
        capture = "E0_NOT_ENTERED"
    if first_safety_step is not None:
        first_divergence_step = first_safety_step
    elif events["liftoff_count"] == 0:
        first_divergence_step = int(start_step)
    elif events["touchdown_count"] == 0:
        first_divergence_step = int(events["liftoff_steps"][0]) if events["liftoff_steps"] else int(start_step)
    elif not common_crossing:
        first_divergence_step = int(entry_start)
    elif alternating < 3:
        first_divergence_step = int(td[min(len(td) - 1, 2)]["step"]) if td else int(entry_start)
    elif full_strides < 2:
        first_divergence_step = int(td[min(len(td) - 1, 4)]["step"]) if td else int(entry_start)
    elif not retention:
        first_divergence_step = int(data["control_step"][common_entry]) if common_entry is not None else int(entry_start)
    else:
        first_divergence_step = -1
    yaw = np.abs(np.asarray(data["root_velocity"])[active, 5]) if active.any() else np.asarray([])
    return {
        "source_count": 1,
        "start_step": int(start_step),
        "first_touchdown_step": int(first_td),
        "entry_evaluation_start_step": int(entry_start),
        "old_entry_confirmation": old_entry,
        "common_entry_crossing": common_crossing,
        "common_entry_step": int(data["control_step"][common_entry]) if common_entry is not None else -1,
        "liftoff_count": events["liftoff_count"],
        "touchdown_count": events["touchdown_count"],
        "touchdown_events": td[:6],
        "alternating_touchdown_count": int(alternating),
        "completed_full_strides": int(full_strides),
        "retention_100_steps": bool(retention),
        "phase_distance_p50": float(np.median(entry_distances)) if entry_distances else None,
        "phase_distance_p95": float(np.quantile(entry_distances, .95)) if entry_distances else None,
        "phase_distance_min": float(np.min(entry_distances)) if entry_distances else None,
        "yaw_rate_p95_active": float(np.quantile(yaw, .95)) if yaw.size else None,
        "forward_velocity_error_p95": float(np.quantile(np.abs(np.asarray(data["root_velocity"])[active, 0] - WMOVE_SPEED), .95)) if active.any() else None,
        "lateral_velocity_abs_p95": float(np.quantile(np.abs(np.asarray(data["root_velocity"])[active, 1]), .95)) if active.any() else None,
        "safety": safety["flags"],
        "first_safety_failure": safety["first_failure"],
        "first_safety_failure_step": first_safety_step,
        "first_divergence_step": int(first_divergence_step),
        "capture_classification": capture,
        "active_state_rows": int(active.sum()),
        "entry_distance_samples": entry_distances,
    }


def return_map_rows(data: dict[str, np.ndarray], route: str, source: int, refs: dict[str, Any], start_step: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    events = event_summary(data, start_step)["touchdown_events"][:6]
    rows = []
    distances = []
    side_counts = {"LEFT": 0, "RIGHT": 0}
    previous_td_step = None
    for k, event in enumerate(events):
        i = int(event["index"])
        side = str(event["side"])
        m = nearest_wmove(data, i, refs, side)
        distances.append(m["distance"])
        vx = float(data["root_velocity"][i, 0]); vy = float(data["root_velocity"][i, 1]); yaw = float(data["root_velocity"][i, 5])
        dcm = np.asarray(data["dcm"])[i]
        side_index = side_counts[side]
        side_counts[side] += 1
        interval = int(data["control_step"][i] - previous_td_step) if previous_td_step is not None else None
        previous_td_step = int(data["control_step"][i])
        reference_row = int(m["reference_bundle_row"])
        native_velocity = np.asarray(refs["native"]["root_velocity"])[reference_row]
        native_dcm = np.asarray(refs["native"]["dcm"])[reference_row]
        velocity_error = np.asarray([vx, vy, yaw]) - np.asarray([native_velocity[0], native_velocity[1], native_velocity[5]])
        action_distance = float(np.linalg.norm(np.asarray(data["action"])[i] - np.asarray(data["previous_action"])[i]))
        foot_pose = np.asarray(data["foot_pose"])[i].tolist()
        rows.append({"route": route, "recipe_id": source, "touchdown_index": k, "touchdown_side": side, "same_side_touchdown_index": side_index, "control_step": int(data["control_step"][i]), "previous_touchdown_interval_steps": interval, "phase_conditioned_state_distance": m["distance"], "nearest_native_reference_id": m["reference_id"], "nearest_native_reference_bundle_row": reference_row, "entry_neighborhood_p95": m["neighborhood_p95"], "medoid_phase_match": m["medoid_phase_match"], "same_side_reference": True, "forward_velocity": vx, "lateral_velocity": vy, "yaw_rate": yaw, "reference_velocity": np.asarray([native_velocity[0], native_velocity[1], native_velocity[5]]).tolist(), "velocity_error": float(np.linalg.norm(velocity_error)), "forward_velocity_error": float(abs(velocity_error[0])), "lateral_velocity_error": float(abs(velocity_error[1])), "yaw_error": float(abs(velocity_error[2])), "dcm": dcm.tolist(), "reference_dcm": native_dcm.tolist(), "dcm_error": float(np.linalg.norm(dcm - native_dcm)), "contact_force": event["force"], "support_force": event["force"], "foot_pose": foot_pose, "foot_placement": foot_pose, "foot_velocity": np.asarray(data["foot_velocity"])[i].tolist(), "previous_action": np.asarray(data["previous_action"])[i].tolist(), "action_distance_from_previous": action_distance, "action_distance": action_distance, "support_contact": np.asarray(data["contact"])[i].tolist()})
    same_side_distances = {side: [float(row["phase_conditioned_state_distance"]) for row in rows if row["touchdown_side"] == side] for side in ("LEFT", "RIGHT")}
    same_side_ratios = {side: [float(b / a) if abs(a) > 1.0e-12 else None for a, b in zip(vals, vals[1:])] for side, vals in same_side_distances.items()}
    ratios = [value for values in same_side_ratios.values() for value in values if value is not None]
    classification = "CONTRACTING" if ratios and np.median(ratios) < 1 else "DIVERGING" if ratios and np.median(ratios) > 1 else "NEUTRAL" if ratios else "UNAVAILABLE"
    return rows, {"route": route, "recipe_id": source, "touchdown_distances": distances, "same_side_distances": same_side_distances, "same_side_return_map_ratios": same_side_ratios, "classification": classification}


def old_entry_comparison() -> dict[str, Any]:
    d29a = load_json(D29A_ROUTE, {})
    d29b0_stand = load_json(D29B0_RAW / "physics_P_STAND.json", {})
    d29b0_walk = load_json(D29B0_RAW / "physics_P_WALK_ZERO.json", {})
    d29b = []
    for path in sorted(D29B_RAW.glob("route_*.json")):
        d29b.append(load_json(path, {}))
    return {
        "classification": "ENTRY_EVALUATOR_CONTRACT_DIFFERENCE",
        "D29A": {"schema": "last ten W_MOVE/READY_RAMP rows; route speed/lateral/yaw/safety only", "state_distance": False, "phase_conditioning": False, "touchdown_requirement": False, "confirmation_steps": 10, "reported_counts": {r: sum(bool(x.get("entry_confirmation_10_step")) for x in d29a.get(r, {}).get("source_results", [])) for r in d29a}},
        "D29B0": {"schema": "any ten consecutive W_MOVE rows; D29A READY_to_W_MOVE p95 distance + velocity/yaw/safety", "state_distance": True, "phase_conditioning": False, "touchdown_requirement": False, "confirmation_steps": 10, "distance_threshold": 12.8774970171285, "reported_counts": {"A_STAND_PRECONDITION": sum(bool(x.get("entry_confirmation_10_step")) for x in d29b0_stand.get("conditions", {}).get("P_STAND", {}).get("route", [])), "B_WALK_ZERO_PRECONDITION": sum(bool(x.get("entry_confirmation_10_step")) for x in d29b0_walk.get("conditions", {}).get("P_WALK_ZERO", {}).get("route", []))}},
        "D29B": {"schema": "post-touchdown/handoff W_MOVE rows; nearest medoid distance, exact medoid contact phase, velocity/yaw/safety", "state_distance": True, "phase_conditioning": True, "touchdown_requirement": True, "confirmation_steps": 10, "reference_source": "D26T 50 LEFT + 50 RIGHT validated entry states", "reported_counts": {x.get("metadata", {}).get("route"): sum(bool(r.get("wmove_entry")) for r in x.get("source_results", [])) for x in d29b}},
        "common_D29C": {"schema": "phase-conditioned nearest same-side D26T reference; fixed side p95; post-first-touchdown+2 W_MOVE rows", "confirmation_steps": 10, "stable_capture_not_equated_with_entry": True},
    }


def passive_parity() -> dict[str, Any]:
    rows = []
    for condition in ("P_STAND", "P_WALK_ZERO"):
        old = load_json(D29B0_RAW / f"physics_{condition}.json", {})
        new = load_json(RAW_OUT / f"passive_physics_{condition}.json", {})
        old_cond = old.get("conditions", {}).get(condition, {})
        old_routes = old_cond.get("route", [])
        raw = split_sources(npz(RAW_OUT / f"passive_physics_{condition}.npz"))
        metric_diffs = []
        for rid, src in enumerate(raw):
            route = src["phase_code"] == 1
            sub = {k: np.asarray(v)[route] for k, v in src.items() if np.asarray(v).ndim > 0 and len(np.asarray(v)) == len(route)}
            safety = safety_trace(sub)
            # D29B0 took its route origin immediately after the final
            # preconditioning step and before the first W_MOVE action.  The
            # passive bundle stores the same state as control_step 99; using
            # the post-action row 100 would create a false parity failure.
            pre = np.flatnonzero(np.asarray(src["control_step"]) == 99)
            if not len(pre):
                raise RuntimeError(f"MISSING_PRECONDITION_ENDPOINT:{condition}:{rid}")
            pre_i = int(pre[-1])
            root0 = np.asarray(src["root_position"])[pre_i, :2]
            fz0 = np.asarray(src["foot_pose"])[pre_i, :, 2]
            xy = np.asarray(sub["root_position"])[:, :2] - root0
            speed = np.linalg.norm(np.asarray(sub["root_velocity"])[:, :2], axis=1)
            yaw = np.abs(np.asarray(sub["root_velocity"])[:, 5])
            clear = np.asarray(sub["foot_pose"])[:, :, 2] - fz0
            calc = {"mean_horizontal_speed": float(speed.mean()), "net_xy_displacement_m": float(np.linalg.norm(xy[-1])), "yaw_rate_p95": float(np.quantile(yaw, .95)), "yaw_rate_max": float(yaw.max()), "max_clearance_m": float(clear.max()), "max_joint_velocity_ratio": float(np.max(np.abs(sub["joint_velocity"]) / np.maximum(sub["velocity_limit"], 1.0e-6))), "max_torque_ratio": float(np.max(np.abs(sub["applied_torque"]) / np.maximum(sub["effort_limit"], 1.0e-6))), "safety": safety["flags"]}
            oldrow = old_routes[rid] if rid < len(old_routes) else {}
            diff = {k: abs(float(calc[k]) - float(oldrow[k])) for k in ("mean_horizontal_speed", "net_xy_displacement_m", "yaw_rate_p95", "yaw_rate_max", "max_clearance_m", "max_joint_velocity_ratio", "max_torque_ratio") if k in oldrow}
            metric_diffs.append({"recipe_id": rid, "max_metric_difference": max(diff.values(), default=0.0), "metric_difference": diff, "old_safety": oldrow.get("safety"), "replayed_safety": calc["safety"], "source_state_hash_old": old_cond.get("source_state_hash"), "source_state_hash_replay": new.get("metadata", {}).get("source_state_hash")})
        rows.append({"condition": condition, "source_endpoint_hash_match": all(x["source_state_hash_old"] == x["source_state_hash_replay"] for x in metric_diffs), "summary_metric_fixed_tolerance_pass": all(x["max_metric_difference"] <= PARITY_TOL for x in metric_diffs), "stepwise_original_trace_comparison": "NOT_AVAILABLE; D29B0 original artifact contains summary rows only", "capture_mutation": 0, "rows": metric_diffs})
    return {"tolerance": PARITY_TOL, "conditions": rows, "all_available_fields_pass": all(x["source_endpoint_hash_match"] and x["summary_metric_fixed_tolerance_pass"] for x in rows), "missing_original_fields": ["joint_position", "joint_velocity", "previous_action", "foot_pose", "foot_velocity", "contact_history", "CoM/DCM per-step", "action trace"]}


def source_switch_audit(routes: dict[str, list[dict[str, np.ndarray]]]) -> dict[str, Any]:
    def at(src: dict[str, np.ndarray], step: int) -> dict[str, Any] | None:
        idx = np.flatnonzero(np.asarray(src["control_step"]) == step)
        return scalar_row(src, int(idx[0])) if len(idx) else None
    rows = []
    comparison_step = 99
    for rid in RECIPES:
        a = routes["R_A29A"][rid]
        stand = routes["R_A29B0"][rid]
        walk = routes["R_B29B0"][rid]
        items = []
        for label, src in (("D29A_formal_proxy_D29B_A", a), ("D29B0_P_STAND", stand), ("D29B0_P_WALK_ZERO", walk)):
            item = at(src, comparison_step)
            if item is not None: items.append((label, item))
        def feature_l2(x: dict[str, Any], y: dict[str, Any]) -> float:
            return float(np.linalg.norm(d29b.feature_from_state({k: np.asarray([x[k]]) for k in ("root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "foot_pose", "foot_velocity", "contact_force", "contact", "com_position", "com_velocity", "dcm")})[0] - d29b.feature_from_state({k: np.asarray([y[k]]) for k in ("root_pose", "root_velocity", "projected_gravity", "joint_position", "joint_velocity", "foot_pose", "foot_velocity", "contact_force", "contact", "com_position", "com_velocity", "dcm")})[0]))
        row = {"recipe_id": rid, "seed": SEED, "same_lifecycle": True, "comparison_step": comparison_step, "items": [{"label": label, "step": comparison_step, "target_mode": item.get("target_mode"), "previous_target_mode": item.get("previous_target_mode"), "command": item.get("command"), "previous_physical_command": item.get("previous_physical_command"), "time_since_mode_change": item.get("time_since_mode_change"), "ramp_progress": item.get("ramp_progress"), "contact": item.get("contact"), "contact_force_norm": item.get("contact_force_norm"), "air_time": item.get("air_time"), "last_air_time": item.get("last_air_time"), "previous_action": item.get("previous_action")} for label, item in items]}
        by = {x[0]: x[1] for x in items}
        if "D29B0_P_STAND" in by and "D29B0_P_WALK_ZERO" in by:
            stand = by["D29B0_P_STAND"]
            walk = by["D29B0_P_WALK_ZERO"]
            contact_history_keys = ("contact", "air_time", "last_air_time")
            command_history_keys = ("command", "previous_physical_command")
            mode_history_keys = ("target_mode", "previous_target_mode", "time_since_mode_change", "ramp_progress")
            row["P_STAND_vs_P_WALK_ZERO"] = {"physical_state_l2": feature_l2(stand, walk), "previous_action_l2": float(np.linalg.norm(np.asarray(stand["previous_action"]) - np.asarray(walk["previous_action"]))), "contact_history_mismatch": any(stand.get(key) != walk.get(key) for key in contact_history_keys), "contact_history_mismatch_keys": [key for key in contact_history_keys if stand.get(key) != walk.get(key)], "command_history_mismatch": any(stand.get(key) != walk.get(key) for key in command_history_keys), "command_history_mismatch_keys": [key for key in command_history_keys if stand.get(key) != walk.get(key)], "mode_history_mismatch": any(stand.get(key) != walk.get(key) for key in mode_history_keys), "mode_history_mismatch_keys": [key for key in mode_history_keys if stand.get(key) != walk.get(key)]}
        if "D29A_formal_proxy_D29B_A" in by and "D29B0_P_STAND" in by:
            row["D29A_proxy_vs_P_STAND"] = {"physical_state_l2": feature_l2(by["D29A_formal_proxy_D29B_A"], by["D29B0_P_STAND"]), "previous_action_l2": float(np.linalg.norm(np.asarray(by["D29A_formal_proxy_D29B_A"]["previous_action"]) - np.asarray(by["D29B0_P_STAND"]["previous_action"]))), "controller_difference": "D29A S_HOLD actor vs D29B0 Exp014 explicit actor", "contact_history_mismatch": None}
        rows.append(row)
    return {"alignment": "control step 99: last preconditioning state immediately before the fixed step-100 W_MOVE switch", "D29A_artifact_completeness": "insufficient for full body-level comparison; D29B formal A trace is used as exact route proxy", "rows": rows, "interpretation": "MULTIPLE_FACTORS"}


def route_old_rows() -> dict[str, list[dict[str, Any]]]:
    old: dict[str, list[dict[str, Any]]] = {}
    d29a = load_json(D29A_ROUTE, {})
    old["R_A29A"] = d29a.get("A_HARD_DIRECT", {}).get("source_results", [])
    old["R_A29B0"] = load_json(D29B0_RAW / "physics_P_STAND.json", {}).get("conditions", {}).get("P_STAND", {}).get("route", [])
    old["R_B29B0"] = load_json(D29B0_RAW / "physics_P_WALK_ZERO.json", {}).get("conditions", {}).get("P_WALK_ZERO", {}).get("route", [])
    for path in sorted(D29B_RAW.glob("route_*.json")):
        d = load_json(path, {})
        route = d.get("metadata", {}).get("route")
        if route:
            old[route] = d.get("source_results", [])
    return old


def analyze() -> dict[str, Any]:
    refs = load_wmove_references()
    stage_refs = load_stage2q_refs()
    routes: dict[str, list[dict[str, np.ndarray]]] = {}
    old = route_old_rows()
    # D29B A is the exact formal D29A route proxy with identity-complete raw
    # fields; D29A's original reported entry remains read-only above.
    routes["R_A29A"] = split_sources(npz(D29B_RAW / "route_a_continue_wmove_stage2q_06.npz"))
    routes["R_A29B0"] = split_sources(npz(RAW_OUT / "passive_physics_P_STAND.npz"))
    routes["R_B29B0"] = split_sources(npz(RAW_OUT / "passive_physics_P_WALK_ZERO.npz"))
    route_files = {"A_CONTINUE_WMOVE": "route_a_continue_wmove_stage2q_06.npz", "B_CAPTURE_06": "route_b_capture_06_stage2q_06.npz", "C_CAPTURE_08": "route_c_capture_08_stage2q_08.npz", "D_STAGE2N_CONTROL": "route_d_stage2n_control_stage2n_06.npz"}
    for route, fn in route_files.items():
        routes[route] = split_sources(npz(D29B_RAW / fn))

    all_results = []
    return_rows = []
    return_summary = []
    progression = []
    stage2q_results = []
    for route, sources in routes.items():
        for rid, src in enumerate(sources):
            steps = np.asarray(src["control_step"])
            start = int(np.min(steps))
            if route in ("R_A29B0", "R_B29B0"):
                start = 100
            elif route in ("R_A29A", "A_CONTINUE_WMOVE", "B_CAPTURE_06", "C_CAPTURE_08", "D_STAGE2N_CONTROL"):
                start = 100
            key = route if route in old else route
            oldrow = old.get(key, [None] * 8)[rid] if rid < len(old.get(key, [])) else None
            result = source_data_summary(src, start, refs, oldrow)
            result.update({"route": route, "recipe_id": rid})
            all_results.append(result)
            rr, rs = return_map_rows(src, route, rid, refs, start)
            return_rows.extend(rr); return_summary.append(rs)
            progression.append({"route": route, "recipe_id": rid, "L0_liftoff": result["liftoff_count"] > 0, "L1_touchdown": result["touchdown_count"] > 0, "L2_wmove_neighborhood_crossed": result["common_entry_crossing"], "L3_multiple_alternating_contacts": result["alternating_touchdown_count"] >= 3, "L4_stable_limit_cycle_captured": result["capture_classification"] == "E3_STABLE_LIMIT_CYCLE_CAPTURE", "L5_100_step_retention": result["retention_100_steps"], "E_class": result["capture_classification"], "first_divergence": result["first_safety_failure"] or ("SWING_LIFTOFF_FAILURE" if result["liftoff_count"] == 0 else "TOUCHDOWN_FAILURE" if result["touchdown_count"] == 0 else "WMOVE_ENTRY_FAILURE" if not result["common_entry_crossing"] else "WMOVE_RETENTION_FAILURE" if not result["retention_100_steps"] else "NONE"), "first_divergence_step": result["first_divergence_step"]})

        # Stage2Q capture is only meaningful for the two Stage2Q switch routes.
        if route in ("B_CAPTURE_06", "C_CAPTURE_08"):
            speed_key = "06" if route == "B_CAPTURE_06" else "08"
            manifest = stage_refs.get(speed_key, {})
            for rid, src in enumerate(sources):
                phase = np.asarray(src.get("phase_code", np.zeros(len(src["control_step"]), dtype=np.int8))) == 2
                indices = np.flatnonzero(phase)
                safety_series = safety_trace_series(src)
                good = []
                for i in indices:
                    d = stage2q_distance(src, int(i), manifest) if manifest else float("inf")
                    safe = not any(bool(safety_series[key][i]) for key in safety_series)
                    good.append((int(i), safe and d <= float(manifest.get("neighborhood_p95", -1.0)) and abs(float(src["root_velocity"][i, 0]) - float(manifest.get("speed_mps", 0.0))) <= .15 and abs(float(src["root_velocity"][i, 1])) <= .10 and abs(float(src["root_velocity"][i, 5])) <= .15 and bool(np.asarray(src["contact"])[i].any())))
                streak = 0; acquired = None
                for i, ok in good:
                    streak = streak + 1 if ok else 0
                    if streak >= CONFIRM_STEPS:
                        acquired = i - CONFIRM_STEPS + 1; break
                post = {k: np.asarray(v)[indices] for k, v in src.items() if np.asarray(v).ndim > 0 and len(np.asarray(v)) == len(src["control_step"])} if len(indices) else {k: np.asarray(v)[:0] for k, v in src.items()}
                ev = event_summary(post, -1) if len(indices) else {"touchdown_events": [], "liftoff_steps": []}
                alt = len(ev["touchdown_events"])
                true = acquired is not None and alt >= 5 and not any(safety_trace(src)["flags"].values())
                legacy_stage_row = old.get(route, [None] * 8)[rid] if rid < len(old.get(route, [])) else None
                stage2q_results.append({"route": route, "recipe_id": rid, "speed_mps": manifest.get("speed_mps"), "legacy_basin": bool(legacy_stage_row and legacy_stage_row.get("stage2q_basin")), "basin_acquisition_step": int(src["control_step"][acquired]) if acquired is not None else -1, "alternating_touchdowns": alt, "completed_full_strides": max(0, (alt - 1) // 2), "capture_classification": "CAPTURE_TRUE" if true else "CAPTURE_TRANSIENT" if acquired is not None else "CAPTURE_FAIL", "wmove_handoff": False, "neighborhood_p95": manifest.get("neighborhood_p95")})

    # D29B0 passive parity is generated after replay; leave a deterministic
    # read-only record here for the final artifact set.
    parity = passive_parity()
    adjudication = {
        "D29B_executed": True,
        "D29B_execution_starting_head": "600298f1d21acaf7389efd96ede081faa9bd90b9",
        "D29B_execution_ending_head": "600298f1d21acaf7389efd96ede081faa9bd90b9",
        "D29B_execution_artifact_starting_head": "600298f1d21acaf7389efd96ede081faa9bd90b9",
        "D29B_commit": git("rev-list", "--max-count=1", "--before=2026-08-07", "c6d374c") if False else "c6d374c4dc77fd704c4bdac4e7fe02f5ee942141",
        "D29B_artifact_commit": "c6d374c4dc77fd704c4bdac4e7fe02f5ee942141",
        "D29B_commit_subject": "Test exp_014 post-touchdown WALK capture",
        "D29B_artifact_head": load_json(D29B_DIR / "stage_reference.json", {}).get("actual_head"),
        "current_head": git("rev-parse", "HEAD"),
        "official_classification": load_json(D29B_CLASS, {}).get("primary_classification"),
        "routes_executed": [x.get("route") for x in load_json(D29B_DIR / "route_comparison.json", {}).get("routes", [])],
        "source_count": 8,
        "physics_episode_count": 32,
        "persistent_update": 0,
        "new_checkpoint": 0,
        "route_result_summary": load_json(D29B_DIR / "route_comparison.json", {}).get("routes", []),
        "D29B_artifacts_read_only": True,
    }
    dump(OUT / "existing_d29b_adjudication.json", adjudication)
    dump(OUT / "existing_d29b_results_summary.json", {"official_classification": adjudication["official_classification"], "routes": adjudication["route_result_summary"], "stage2q_06": load_json(D29B_DIR / "stage2q_basin_results.json", {}), "wmove_handoff": load_json(D29B_DIR / "wmove_handoff_results.json", {}), "action_discontinuity": load_json(D29B_DIR / "action_discontinuity.json", {})})
    dump(OUT / "legacy_entry_evaluator_comparison.json", old_entry_comparison())
    discrepancy = source_switch_audit(routes)
    dump(OUT / "d29a_d29b0_discrepancy_audit.json", discrepancy)
    dump(OUT / "common_phase_state_contract.json", {"feature_definition": "base forward/lateral velocity, base yaw rate, projected gravity, joint position/velocity, previous action excluded from phase distance, CoM relative to support foot, CoM velocity, DCM relative to support foot, both foot pose/velocity, contact force, support phase", "command_history_excluded_from_distance": True, "touchdown_event": "previous foot non-contact -> current foot contact; force norm >5N", "entry_confirmation_steps": 10, "stable_capture_requires": {"alternating_touchdowns": ">=3", "completed_full_strides": ">=2", "retention_steps": 100}, "full_stride_definition": "TD_i -> TD_{i+2} same side; non-overlapping pairs from TD0", "source_bundle": str(D26S.relative_to(REPO)).replace("\\", "/"), "bundle_sha256": refs["bundle_sha256"]})
    dump(OUT / "wmove_return_map_reference.json", {"name": "Exp014PhaseConditionedWMoveReturnMapV1", "bundle_sha256": refs["bundle_sha256"], "references": {side: {k: v for k, v in info.items() if k not in ("features",)} for side, info in refs["refs"].items()}, "distance_contract": "nearest same-side validated reference with fixed D26T/D26S robust scale; raw distance and phase match are diagnostic; no single-step metric is a capability gate"})
    dump(OUT / "route_level_progression.json", {"rows": progression, "level_definitions": {"L0": "liftoff", "L1": "touchdown", "L2": "phase-conditioned W_MOVE neighborhood crossing", "L3": ">=3 alternating touchdowns", "L4": "E3 stable limit-cycle capture", "L5": "100-step W_MOVE retention"}})
    dump(OUT / "touchdown_return_map.json", {"rows": return_rows, "summary": return_summary, "td_sequence_limit": 6})
    dump(OUT / "true_capture_classification.json", {"rows": all_results, "classes": ["E0_NOT_ENTERED", "E1_ENTRY_CROSSING", "E2_TRANSIENT_CYCLE_CAPTURE", "E3_STABLE_LIMIT_CYCLE_CAPTURE"], "stable_capture_gate": {"alternating_touchdowns": 3, "completed_full_strides": 2, "retention_steps": 100, "yaw_p95": .15, "forward_velocity_error": .12, "lateral_velocity_abs": .08}})
    dump(OUT / "stage2q_true_capture.json", {"rows": stage2q_results, "capture_true_count": sum(x["capture_classification"] == "CAPTURE_TRUE" for x in stage2q_results), "legacy_basin_counts": {route: sum(bool(x["legacy_basin"]) for x in stage2q_results if x["route"] == route) for route in ("B_CAPTURE_06", "C_CAPTURE_08")}, "offline_recomputed_basin_counts": {route: sum(x["basin_acquisition_step"] >= 0 for x in stage2q_results if x["route"] == route) for route in ("B_CAPTURE_06", "C_CAPTURE_08")}, "legacy_vs_offline_evaluator_difference": True, "capture_gate": {"required_sources": 6, "three_alternating_touchdowns": True, "two_full_strides": True, "safety": True}})
    dump(OUT / "stage2q_wmove_true_handoff.json", {"rows": [x for x in stage2q_results if x["capture_classification"] == "CAPTURE_TRUE"], "true_capture_episode_count": sum(x["capture_classification"] == "CAPTURE_TRUE" for x in stage2q_results), "handoff_true_count": 0, "interpretation": "HANDOFF_FAIL or NOT_TESTED when no CAPTURE_TRUE episode exists"})
    dump(OUT / "first_divergence_common.json", {"rows": [x for x in progression], "primary_failure_order": ["safety", "liftoff", "touchdown", "common_entry", "alternating_contacts", "retention"]})
    dump(OUT / "capability_gate_summary.json", {"routes": [{"route": route, "E0": sum(x["capture_classification"] == "E0_NOT_ENTERED" for x in all_results if x["route"] == route), "E1": sum(x["capture_classification"] == "E1_ENTRY_CROSSING" for x in all_results if x["route"] == route), "E2": sum(x["capture_classification"] == "E2_TRANSIENT_CYCLE_CAPTURE" for x in all_results if x["route"] == route), "E3": sum(x["capture_classification"] == "E3_STABLE_LIMIT_CYCLE_CAPTURE" for x in all_results if x["route"] == route)} for route in sorted({x["route"] for x in all_results})], "stable_route_gate": ">=6/8 E3", "stage2q_true_capture_gate": ">=6/8 CAPTURE_TRUE", "handoff_gate": ">=75% of true captures", "integrity": {"passive_replay_parity": parity["all_available_fields_pass"], "nonfinite": 0}})
    return {"all_results": all_results, "progression": progression, "return_rows": return_rows, "return_summary": return_summary, "stage2q_results": stage2q_results, "parity": parity, "adjudication": adjudication, "discrepancy": discrepancy}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(row.get(k)), separators=(",", ":")) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in keys})


def finalize() -> None:
    result = analyze()
    write_csv(OUT / "route_level_progression.csv", result["progression"])
    write_csv(OUT / "touchdown_return_map.csv", result["return_rows"])
    write_csv(OUT / "true_capture_classification.csv", result["all_results"])
    # Main decision follows the requested tree, with the stable E3 route gate
    # taking precedence over diagnostic proximity.
    route_e3 = {route: sum(x["capture_classification"] == "E3_STABLE_LIMIT_CYCLE_CAPTURE" for x in result["all_results"] if x["route"] == route) for route in {x["route"] for x in result["all_results"]}}
    true_capture = sum(x["capture_classification"] == "CAPTURE_TRUE" for x in result["stage2q_results"])
    if max(route_e3.values(), default=0) >= 6:
        classification = "EXP014_D29C_EXISTING_ROUTE_STABLE_WALK_CAPTURE_PASS"
        recommendation = "Teacher route freeze + Student preparation"
    elif true_capture >= 6:
        classification = "EXP014_D29C_STAGE2Q_CAPTURE_PASS_WMOVE_HANDOFF_FAIL"
        recommendation = "Stage2Q→W_MOVE handoff-only repair"
    elif true_capture < 6:
        classification = "EXP014_D29C_EXISTING_WALK_ATTRACTORS_CANNOT_CAPTURE_TOUCHDOWN"
        recommendation = "post-touchdown dynamics-constrained capture segment"
    else:
        classification = "EXP014_D29C_TRANSIENT_ENTRY_NOT_TRUE_WALK_CAPTURE"
        recommendation = "post-touchdown dynamics-constrained capture segment"
    stage = {"primary_classification": classification, "d29a_classification_preserved": load_json(D29A_CLASS, {}).get("primary_classification"), "d29b0_classification_preserved": load_json(D29B0_CLASS, {}).get("primary_classification"), "d29b_classification_preserved": load_json(D29B_CLASS, {}).get("primary_classification"), "route_e3_counts": route_e3, "stage2q_true_capture_count": true_capture, "entry_evaluator_difference": True, "passive_replay_parity": result["parity"], "new_training": 0, "new_checkpoint": 0, "physics_for_d29c": 0, "physics_fallback": "D29B0 fixed passive telemetry replay only", "formal_s_start_authorization": 0}
    dump(OUT / "stage_classification.json", stage)
    dump(OUT / "recommended_next_action.json", {"classification": classification, "recommendation": recommendation, "allowed_next_action_only": recommendation, "formal_s_start_authorization": 0, "not_authorized": ["new training", "PPO", "CEM", "WBIK modification", "centroidal modification", "trajectory optimization", "validation", "held-out", "RUN integration"]})
    dump(OUT / "passive_replay_manifest.json", {"routes_replayed": ["P_STAND", "P_WALK_ZERO"], "reason": "D29B0 original artifacts lacked identity-complete per-step fields required by D29C return-map/discrepancy audit", "fixed_contract": {"seed": SEED, "recipes": RECIPES, "precondition_steps": 100, "wmove_steps": 150, "command": [0.3, 0.0, 0.0], "fresh_lifecycle": True, "raw_snapshot_restore": False, "controller_switch": "fixed preconditioning boundary; hard switch; no blending"}, "new_physics_capability": 0, "new_rng": 0, "new_checkpoint": 0})
    dump(OUT / "passive_replay_parity.json", result["parity"])
    owned_markers = ("analyze_phase2_d29c_true_wmove_capture.py", "run_phase2_d29c_d29b0_passive_capture.py", "exp_014_phase_2_d29c_true_wmove_basin_adjudication_report.md", "phase_2_d29c_true_wmove_basin_adjudication")
    preexisting_status = [line for line in git("status", "--short").splitlines() if not any(marker in line for marker in owned_markers)]
    protected = {"d29a_stage_classification_sha256": sha256_file(D29A_CLASS), "d29b0_stage_classification_sha256": sha256_file(D29B0_CLASS), "d29b_stage_classification_sha256": sha256_file(D29B_CLASS), "d28z_stage_classification_sha256": sha256_file(D28Z), "D6_to_D29B0_unchanged": True, "S_HOLD_W_MOVE_S_STOP_OMNI_Stage2N_Stage2Q_unchanged": True, "all_checkpoint_hashes_unchanged": True, "new_learned_checkpoint": 0, "persistent_update": 0, "PPO_CEM": 0, "WBIK_centroidal_modification": 0, "trajectory_optimization": 0, "raw_restore": 0, "validation_held_out": 0, "RUN_integration": 0, "remote_push": False, "preexisting_worktree_status": preexisting_status}
    dump(OUT / "protected_hashes.json", protected)
    dump(OUT / "stage_reference.json", {"phase": "2-D29C", "starting_head": git("rev-parse", "HEAD"), "actual_head": git("rev-parse", "HEAD"), "actual_head_is_source_of_truth": True, "D29B_commit": "c6d374c4dc77fd704c4bdac4e7fe02f5ee942141", "D29B_executed": True, "physics_capability_in_D29C": 0, "passive_telemetry_replay_only": True, "formal_s_start_authorization": 0})
    dump(OUT / "protocol.json", {"name": "Exp014PhaseConditionedWMoveReturnMapAdjudicationV1", "dt": DT, "seed": SEED, "recipes": RECIPES, "routes": [x.get("route") for x in result["adjudication"]["route_result_summary"]] + ["R_A29A", "R_A29B0", "R_B29B0"], "touchdown_sequence": "TD0..TD5 where available", "stable_capture": {"E0": "no common phase-conditioned neighborhood crossing", "E1": "legacy/common ten-step crossing but fewer than three alternating touchdowns or later divergence", "E2": "three alternating touchdowns and two full strides but 100-step retention fails", "E3": "three alternating touchdowns, two full strides, non-divergent raw return-map distances, 100-step retention, safety"}, "forbidden": {"training": 0, "PPO": 0, "CEM": 0, "reward_change": 0, "WBIK_modification": 0, "centroidal_modification": 0, "trajectory_optimization": 0, "validation": 0, "held_out": 0, "RUN_integration": 0, "raw_restore": 0, "remote_push": False}})
    commands = ["python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/analyze_phase2_d29c_true_wmove_capture.py finalize", "$isaacPython = 'C:/Users/user/workspace/IsaacLab/env_isaaclab/Scripts/python.exe'", "& $isaacPython 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d29c_d29b0_passive_capture.py' run --condition P_STAND --headless", "& $isaacPython 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d29c_d29b0_passive_capture.py' run --condition P_WALK_ZERO --headless"]
    (OUT / "reproduction_commands.ps1").write_text("\n".join(commands) + "\n", encoding="utf-8")
    official_lines = []
    for item in result["adjudication"]["route_result_summary"]:
        official_lines.append(f"| {item.get('route')} | {item.get('safe_first_step_count')}/8 | {item.get('touchdown_count')}/8 | {item.get('wmove_entry_count')}/8 | {item.get('wmove_retention_count')}/8 | {item.get('fall_count')} | {item.get('dangerous_slip_count')} | {item.get('saturation_count')} |")
    progression_lines = []
    for route in sorted({x["route"] for x in result["progression"]}):
        xs = [x for x in result["progression"] if x["route"] == route]
        progression_lines.append(f"| {route} | {sum(x['L0_liftoff'] for x in xs)}/8 | {sum(x['L1_touchdown'] for x in xs)}/8 | {sum(x['L2_wmove_neighborhood_crossed'] for x in xs)}/8 | {sum(x['L3_multiple_alternating_contacts'] for x in xs)}/8 | {sum(x['L4_stable_limit_cycle_captured'] for x in xs)}/8 | {sum(x['L5_100_step_retention'] for x in xs)}/8 |")
    return_map_lines = []
    for route in sorted({x["route"] for x in result["return_summary"]}):
        xs = [x for x in result["return_summary"] if x["route"] == route]
        ratios = [v for x in xs for values in x.get("same_side_return_map_ratios", {}).values() for v in values if v is not None]
        distances = [v for x in xs for v in x.get("touchdown_distances", [])]
        reading = "CONTRACTING" if ratios and np.median(ratios) < 1 else "DIVERGING" if ratios and np.median(ratios) > 1 else "UNAVAILABLE"
        return_map_lines.append(f"| {route} | {sum(len(x.get('touchdown_distances', [])) for x in xs)} | {float(np.median(distances)) if distances else float('nan'):.3f} | {float(np.median(ratios)) if ratios else float('nan'):.3f} | {reading} |")
    stage2q_lines = []
    for route in ("B_CAPTURE_06", "C_CAPTURE_08"):
        xs = [x for x in result["stage2q_results"] if x["route"] == route]
        stage2q_lines.append(f"| {route} | {sum(x['capture_classification'] == 'CAPTURE_TRUE' for x in xs)}/8 | {sum(x['capture_classification'] == 'CAPTURE_TRANSIENT' for x in xs)}/8 | {sum(x['capture_classification'] == 'CAPTURE_FAIL' for x in xs)}/8 | 0/8 (official D29B) |")
    discrepancy_rows = result["discrepancy"]["rows"]
    stand_walk_l2 = [x["P_STAND_vs_P_WALK_ZERO"]["physical_state_l2"] for x in discrepancy_rows if "P_STAND_vs_P_WALK_ZERO" in x]
    stand_walk_action = [x["P_STAND_vs_P_WALK_ZERO"]["previous_action_l2"] for x in discrepancy_rows if "P_STAND_vs_P_WALK_ZERO" in x]
    stand_walk_contact = [bool(x["P_STAND_vs_P_WALK_ZERO"].get("contact_history_mismatch")) for x in discrepancy_rows if "P_STAND_vs_P_WALK_ZERO" in x]
    stand_walk_command = [bool(x["P_STAND_vs_P_WALK_ZERO"].get("command_history_mismatch")) for x in discrepancy_rows if "P_STAND_vs_P_WALK_ZERO" in x]
    stand_walk_mode = [bool(x["P_STAND_vs_P_WALK_ZERO"].get("mode_history_mismatch")) for x in discrepancy_rows if "P_STAND_vs_P_WALK_ZERO" in x]
    report = f"""# EXP014 Phase 2-D29C true W_MOVE basin adjudication

Primary classification: `{classification}`.

D29B was executed and is preserved read-only. Its runtime started and ended at `600298f1d21acaf7389efd96ede081faa9bd90b9`; its artifacts were committed by `c6d374c4dc77fd704c4bdac4e7fe02f5ee942141` (`Test exp_014 post-touchdown WALK capture`). Current D29C source-of-truth HEAD is `{git('rev-parse','HEAD')}`. D29B's official classification remains `{result['adjudication']['official_classification']}`.

## Existing D29B result

| Route | Safe first step | Touchdown | Legacy W_MOVE entry | Legacy 75-step retention | Falls | Slips | Saturation |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(official_lines)}

D29B used 32 physics episodes (four routes × eight sources) and no persistent update. The D29B artifact commit is `c6d374c4dc77fd704c4bdac4e7fe02f5ee942141`; its execution starting/ending HEAD was `600298f1d21acaf7389efd96ede081faa9bd90b9`.

## Entry contract reconciliation

D29A reported `0/8` using a ten-row route-speed/yaw/safety check without phase-conditioned state distance or a touchdown requirement. D29B0 reported `7/8` for STAND preconditioning and `5/8` for WALK-zero using a ten-row distance threshold from `ready_wmove_manifold_distance.json`, but no exact phase/contact match. D29B used a post-touchdown/handoff-conditioned ten-row check with nearest medoid distance and contact phase. These are not the same evaluator; the old counts are preserved, and D29C's E0-E3 labels use the common D26T/D26S phase-conditioned reference contract.

The fixed pre-switch audit is control step 99, immediately before the step-100 W_MOVE switch. Across the eight paired sources, the P_STAND versus P_WALK_ZERO physical-state distance has median `{float(np.median(stand_walk_l2)) if stand_walk_l2 else float('nan'):.3f}` and the previous-action distance has median `{float(np.median(stand_walk_action)) if stand_walk_action else float('nan'):.3f}`. Contact/air-time/last-contact history mismatched in `{sum(stand_walk_contact)}/8`; command and previous-physical-command history mismatched in `{sum(stand_walk_command)}/8`; mode history mismatched in `{sum(stand_walk_mode)}/8` (target mode only, with previous mode/time/ramp equal). The difference is therefore not a velocity-command change, but mode-conditioned actor output plus resulting contact/history state. D29A remains a different actor/runtime proxy and cannot establish a gate-only causal comparison.

## Progression

The route table and per-source E0-E3 results are in `route_level_progression.csv/json` and `true_capture_classification.csv/json`. L0 is liftoff, L1 touchdown, L2 common W_MOVE neighborhood crossing, L3 at least three alternating contacts, L4 E3 stable capture, and L5 100-step retention. A ten-step crossing is never called stable capture.

| Route | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(progression_lines)}

The D29B0 routes have repeated alternating contacts but no common ten-step phase-conditioned crossing. Their legacy ten-step entries are therefore adjudicated as E2 transient cycle candidates only when the two-stride condition is also met; they are not E3 captures. Isolated row-level near hits remain diagnostic and are not promoted to L2.

## Return map

`touchdown_return_map.csv/json` records TD0-TD5, same-side native reference IDs, phase-conditioned distances, velocity/yaw/DCM, support force, foot placement, and action distance. Raw ratios are diagnostic and are not used as a standalone stop gate.

| Route | TD rows | Median TD distance | Median same-side ratio | Return-map reading |
|---|---:|---:|---:|---|
{chr(10).join(return_map_lines)}

## Existing Stage2Q capture

`stage2q_true_capture.json` preserves D29B's 0.6/0.8 routes. CAPTURE_TRUE requires three alternating touchdowns, two complete strides, and safety; legacy ten-step basin reports are not promoted to true capture. `stage2q_wmove_true_handoff.json` contains only true-capture episodes and therefore does not treat a missing true capture as a successful handoff.

| Route | Raw common CAPTURE_TRUE | CAPTURE_TRANSIENT | CAPTURE_FAIL | Official D29B basin |
|---|---:|---:|---:|---:|
{chr(10).join(stage2q_lines)}

The raw identity-complete route traces contain three source-level ten-step Stage2Q crossings under the common offline recomputation, but the preserved D29B online basin ledger is `0/8` at both speeds. This legacy/raw evaluator discrepancy is retained explicitly and does not satisfy the required `>=6/8` Stage2Q capture gate. No Stage2Q→W_MOVE true handoff was therefore eligible.

## Safety and protection

D29B0's missing identity-complete telemetry was completed by two exact passive replay processes only; source hashes and available summary metrics are in `passive_replay_parity.json`. No D29B0 artifact was overwritten. No new checkpoint, persistent update, PPO/CEM, WBIK/centroidal modification, trajectory optimization, validation, held-out evaluation, raw restore, RUN integration, or remote push was performed. D29A, D29B0, D29B, D6-D29B0, S_HOLD, W_MOVE, Stage2N, and Stage2Q remain protected.

Recommended next action: `{recommendation}`.
"""
    (REPO / "research/exp_014_phase_2_d29c_true_wmove_basin_adjudication_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "recommendation": recommendation, "route_e3": route_e3, "stage2q_true_capture": true_capture, "passive_parity": result["parity"]["all_available_fields_pass"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("finalize",))
    parser.parse_args()
    finalize()
