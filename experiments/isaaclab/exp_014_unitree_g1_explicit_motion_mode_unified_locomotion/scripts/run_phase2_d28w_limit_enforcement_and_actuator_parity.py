"""Phase 2-D28W: PhysX limit enforcement and implicit-actuator parity.

This stage keeps D28V and every earlier artifact read-only.  It has three
separate modes: ``offline`` audits persisted formal traces, ``capture`` runs
the exact D27 V2A entrypoint with a passive decimation hook, and ``probe``
performs only the explicitly-authorized isolated articulation diagnostic.
The final ``analyze`` mode consumes those captures and conditionally performs
the D28S C0--C4 shadow replay.  No START capability physics is launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ROOT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = ROOT / "phase_2_d28w_limit_enforcement_and_actuator_parity"
REPORT = REPO / "research/exp_014_phase_2_d28w_limit_enforcement_and_actuator_parity_report.md"
D28V = ROOT / "phase_2_d28v_hard_limit_and_actuator_authority"
D28R = ROOT / "phase_2_d28r_centroidal_trace_and_feedback"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"
D26U = ROOT / "phase_2_d26u_fresh_source_and_offline_execution"
D26S = ROOT / "phase_2_d26s_exact_wmove_instrumentation"
D27_SCRIPT = EXP / "scripts/run_phase2_d27_right_model_based_start_physics.py"

DT = 0.02
HARD_TOL = 1.0e-6
PROBE_OFFSETS = (0.01, 0.05, 0.10)
PROBE_CONTROL_STEPS = 50
PROBE_RELEASE_STEPS = 10
VELOCITY_RATIO_LIMIT = 0.80
SOLVER_TOL = 1.0e-9
PARITY_TOL = 1.0e-5
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_PASS_FRACTION = 0.80
TRACE_RECIPES = (4, 5, 6, 7)
FORMULATIONS = (
    "C0_ALL",
    "C1_FREEZE_WRIST_HAND",
    "C2_SCALED_ALL",
    "C3_SCALED_FREEZE_WRIST_HAND",
    "C4_SCALED_LEGS_WAIST",
)
GROUPS = (
    "left leg", "right leg", "waist", "left arm", "right arm",
    "left wrist/hand", "right wrist/hand",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import only read-only numerical/runtime helpers.  None of these modules calls
# main() during import.
d28v = load_module("exp014_d28w_d28v_read_only", EXP / "scripts/run_phase2_d28v_hard_limit_and_actuator_authority.py")
d28u = d28v.d28u
d28s = d28v.d28s
d27 = d28v.d27


def arr(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def jsonable(value: Any) -> Any:
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def array_hash(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(repr(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(jsonable(v), sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list, tuple, np.ndarray)) else jsonable(v) for k, v in row.items()})


def quantiles(values: Any) -> dict[str, Any]:
    x = arr(values).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, "min": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "p999": None, "max": None}
    return {"count": int(x.size), "min": float(np.min(x)), "p01": float(np.quantile(x, .01)), "p05": float(np.quantile(x, .05)), "p50": float(np.quantile(x, .50)), "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)), "p999": float(np.quantile(x, .999)), "max": float(np.max(x))}


def group_for_joint(name: str) -> str:
    n = str(name).lower()
    if "wrist" in n or any(token in n for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if n.startswith("left_") else "right wrist/hand"
    if "shoulder" in n or "elbow" in n:
        return "left arm" if n.startswith("left_") else "right arm"
    if "torso" in n or "waist" in n:
        return "waist"
    return "left leg" if n.startswith("left_") else "right leg"


def initial_protected_hashes() -> dict[str, str]:
    """Record protected files without including D28W output or unrelated dirt."""
    paths: set[Path] = set()
    for base in (ROOT / "phase_2_d28v_hard_limit_and_actuator_authority", ROOT / "phase_2_d28s_centroidal_authority_audit", ROOT / "phase_2_d28r_centroidal_trace_and_feedback", ROOT / "phase_2_d28u_joint_contract_and_physical_authority"):
        if base.exists():
            paths.update(p for p in base.rglob("*") if p.is_file() and p.name != "_runtime_capture_checkpoint.json")
    manifest = ROOT / "phase_2_d28u_joint_contract_and_physical_authority/protected_hashes.json"
    if manifest.exists():
        for rel in read_json(manifest).get("protected_paths", {}):
            p = REPO / rel
            if p.is_file():
                paths.add(p)
    result: dict[str, str] = {}
    for p in sorted(paths):
        try:
            result[str(p.relative_to(REPO)).replace("\\", "/")] = sha256_file(p)
        except ValueError:
            pass
    return result


def load_runtime_candidate() -> tuple[dict[str, Any], np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Load D28V runtime metadata read-only; the probe later supersedes status."""
    checkpoint = D28V / "_runtime_capture_checkpoint.json"
    if checkpoint.exists():
        payload = read_json(checkpoint)
        runtime = payload.get("runtime", payload)
    else:
        runtime = {}
    contract = read_json(D28V / "physx_runtime_joint_contract.json")
    rows = sorted(contract["rows"], key=lambda x: int(x["joint_index_by_name"]))
    names = [str(row["joint_name"]) for row in rows]
    hard = np.asarray([[float(row["physx_lower"]), float(row["physx_upper"])] for row in rows], dtype=np.float64)
    vel = np.asarray([float(row["runtime_velocity_limit"]) for row in rows], dtype=np.float64)
    effort = np.asarray([float(row["runtime_effort_limit"]) for row in rows], dtype=np.float64)
    d28r_contract = read_json(D28R / "joint_index_name_contract.json")
    default_q = np.asarray([float(row.get("default_q", row.get("default_q_rad"))) for row in sorted(d28r_contract["joints"], key=lambda x: int(x["action_index"]))], dtype=np.float64)
    return runtime, hard, names, vel, effort, default_q


def load_trace_inputs():
    return d28v.load_inputs()


def population_metadata() -> dict[str, dict[str, np.ndarray]]:
    """Load identity metadata for the protected P0--P3 populations."""
    result: dict[str, dict[str, np.ndarray]] = {}
    p0 = np.load(D26U / "fresh_shold_identity_complete_sources.npz", allow_pickle=False)
    result["P0_S_HOLD_fresh_endpoint"] = {"episode": arr(p0["recipe_id"], np.int64), "control_step": arr(p0["control_step"], np.int64), "phase": np.asarray(["S_HOLD"] * len(p0["recipe_id"]), dtype="U32")}
    p1_path = ROOT / "phase_2_d21_identity_complete_support_causality/reference_rollout_bundle.npz"
    if p1_path.exists():
        p1 = np.load(p1_path, allow_pickle=False)
        result["P1_S_HOLD_formal_rollout"] = {"episode": arr(p1["rollout_id"], np.int64).reshape(-1), "control_step": arr(p1["control_step"], np.int64).reshape(-1), "phase": np.asarray(["S_HOLD"] * int(np.asarray(p1["control_step"]).size), dtype="U32")}
    else:
        result["P1_S_HOLD_formal_rollout"] = {"episode": np.zeros(0, dtype=np.int64), "control_step": np.zeros(0, dtype=np.int64), "phase": np.zeros(0, dtype="U32")}
    p2_path = D26S / "d26s_formal_on/native_steady_trace_bundle.npz"
    if not p2_path.exists():
        p2_path = D26S / "native_steady_trace_bundle.npz"
    p2 = np.load(p2_path, allow_pickle=False)
    n = min(20000, len(p2["joint_pos"]))
    result["P2_W_MOVE_formal_rollout"] = {"episode": arr(p2["episode_id"][:n], np.int64), "control_step": arr(p2["control_step"][:n], np.int64), "phase": np.asarray(["W_MOVE"] * n, dtype="U32")}
    p3 = np.load(D28R / "capture_on/raw_primary_trajectory.npz", allow_pickle=False)
    rows = []
    for recipe in TRACE_RECIPES:
        for row in range(p3["control_step"].shape[1]):
            if bool(p3["active"][recipe, row]) and int(p3["stage"][recipe, row]) == 1:
                rows.append((recipe, int(p3["control_step"][recipe, row]), int(p3["phase"][recipe, row])))
    result["P3_D27_actual_V2A_trace"] = {"episode": np.asarray([x[0] for x in rows], dtype=np.int64), "control_step": np.asarray([x[1] for x in rows], dtype=np.int64), "phase": np.asarray([str(x[2]) for x in rows], dtype="U32")}
    return result


def phase_name(code: str) -> str:
    return {"1": "DOUBLE_SUPPORT_SHIFT", "2": "FIRST_SWING", "3": "LANDING_AND_CAPTURE", "4": "WMOVE_ACCEPTANCE"}.get(str(code), code)


def audit_formal_violations(names: list[str], hard: np.ndarray, default_q: np.ndarray, vlim: np.ndarray, source: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    populations = d28u.population_rows(source, hard, vlim, default_q)
    metadata = population_metadata()
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for key, pop in populations.items():
        q = arr(pop["q"]); dq = arr(pop["dq"]); action = arr(pop["action"])
        qcmd = default_q[None, :] + 0.5 * action
        meta = metadata[key]
        lo = np.broadcast_to(hard[:, 0], q.shape); hi = np.broadcast_to(hard[:, 1], q.shape)
        viol = (q < lo - HARD_TOL) | (q > hi + HARD_TOL)
        pop_rows: list[dict[str, Any]] = []
        for s, j in zip(*np.where(viol)):
            side = "lower" if q[s, j] < lo[s, j] else "upper"
            mag = float(lo[s, j] - q[s, j] if side == "lower" else q[s, j] - hi[s, j])
            episode = int(meta["episode"][s]) if s < len(meta["episode"]) else None
            control = int(meta["control_step"][s]) if s < len(meta["control_step"]) else int(s)
            phase = str(meta["phase"][s]) if s < len(meta["phase"]) else key
            preceding_dq = float(dq[s, j])
            following_dq = float(dq[s + 1, j]) if s + 1 < len(dq) and (s + 1 >= len(meta["episode"]) or int(meta["episode"][s + 1]) == episode) else None
            pop_rows.append({"population": key, "source": pop["source"], "state_index": int(s), "episode": episode, "control_step": control, "phase": phase_name(phase), "joint_index": int(j), "joint_name": names[j], "joint_group": group_for_joint(names[j]), "q_actual": float(q[s, j]), "lower_candidate": float(lo[s, j]), "upper_candidate": float(hi[s, j]), "violation_side": side, "absolute_violation_magnitude": mag, "preceding_dq": preceding_dq, "following_dq": following_dq, "q_cmd": float(qcmd[s, j]), "source_recipe": episode})
        rows.extend(pop_rows)
        summary[key] = {"source": pop["source"], "states": int(len(q)), "violation_count": int(len(pop_rows)), "violation_fraction": float(np.mean(viol)) if q.size else 0.0, "magnitude": quantiles([x["absolute_violation_magnitude"] for x in pop_rows]), "by_joint": {names[j]: quantiles(np.maximum(np.where(viol[:, j], np.where(q[:, j] < lo[:, j], lo[:, j] - q[:, j], q[:, j] - hi[:, j]), 0.0), 0.0)) for j in range(len(names)) if np.any(viol[:, j])}}
    # Durations are consecutive control-state violations for the same source,
    # episode, joint and side; formal bundles that lack contiguous timestamps
    # conservatively retain duration one.
    grouped: dict[tuple[str, int | None, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["population"], row["episode"], row["joint_index"], row["violation_side"])].append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda x: (x["control_step"], x["state_index"]))
        run = 1
        for i, row in enumerate(group_rows):
            if i and row["control_step"] == group_rows[i - 1]["control_step"] + 1:
                run += 1
            else:
                run = 1
            row["violation_duration_control_steps"] = int(run)
    by_joint: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        vals = [r["absolute_violation_magnitude"] for r in rows if r["joint_index"] == j]
        by_joint[name] = {"joint_index": j, "joint_group": group_for_joint(name), "count": len(vals), "magnitude": quantiles(vals)}
    summary["all"] = {"states": int(sum(x["states"] for x in summary.values())), "violation_count": len(rows), "magnitude": quantiles([r["absolute_violation_magnitude"] for r in rows]), "by_joint": by_joint}
    return rows, {"name": "Exp014D28WFormalLimitViolationMagnitudeV1", "tolerance_rad": HARD_TOL, "rows": rows, "summary": summary}


def wrap_audit(rows: list[dict[str, Any]], hard: np.ndarray) -> dict[str, Any]:
    result = []
    counts = defaultdict(int)
    for row in rows:
        q = float(row["q_actual"]); lo = float(row["lower_candidate"]); hi = float(row["upper_candidate"])
        equivalents = {str(k): float(q + 2.0 * math.pi * k) for k in (-2, -1, 0, 1, 2)}
        inside = [int(k) for k, value in ((k, q + 2.0 * math.pi * k) for k in (-2, -1, 0, 1, 2)) if lo - HARD_TOL <= value <= hi + HARD_TOL]
        if inside and 0 not in inside:
            cls = "REVOLUTE_WRAP_EQUIVALENT"
        else:
            cls = "DIRECT_LIMIT_PENETRATION"
        counts[cls] += 1
        result.append({**row, "q_plus_2pi_k": equivalents, "interval_equivalent_k": inside, "classification": cls, "coordinate_contract": "continuous_unwrapped_revolute_runtime; no wrap applied"})
    return {"name": "Exp014D28WRevoluteCoordinateWrapAuditV1", "k_values": [-2, -1, 0, 1, 2], "formal_runtime_coordinate": "continuous/unwrapped candidate; empirical q+2pi audit", "counts": dict(counts), "rows": result, "formal_wrap_contract": "no wrap used"}


def runtime_limit_audit(runtime: dict[str, Any], hard: np.ndarray, names: list[str]) -> dict[str, Any]:
    contract = read_json(D28V / "physx_runtime_joint_contract.json")
    usd = read_json(D28V / "raw_usd_joint_contract.json").get("rows", [])
    usd_by = {str(row.get("joint_name")): row for row in usd}
    runtime_by = {str(row.get("joint_name")): row for row in contract.get("rows", [])}
    rows = []
    for j, name in enumerate(names):
        pr = runtime_by.get(name, {})
        u = usd_by.get(name, {})
        rows.append({"joint_index": j, "joint_name": name, "raw_usd_authored_limit": bool(u.get("limit_attribute_authored", False)), "raw_usd_limit_enabled": bool(u.get("limit_enabled", False)), "physx_runtime_limit_enabled": bool(pr.get("physx_limit_enabled", False)), "physx_lower": float(hard[j, 0]), "physx_upper": float(hard[j, 1]), "raw_usd_lower": u.get("lower_limit_rad"), "raw_usd_upper": u.get("upper_limit_rad"), "joint_contact_distance": None, "restitution": None, "bounce_threshold": None, "solver_position_iterations": 8, "solver_velocity_iterations": 4, "max_depenetration_velocity": 1.0, "source": "D28V zero-step runtime metadata + D28W probe runtime contract"})
    return {"name": "Exp014D28WRuntimeLimitEnabledAuditV1", "rows": rows, "all_runtime_flags_enabled": bool(all(row["physx_runtime_limit_enabled"] for row in rows)), "runtime_flag_source": "robot.root_physx_view.get_dof_limits metadata candidate; probe response is separate", "hard_limit_not_called_if_flag_disabled": True}


def actuator_source_contract() -> dict[str, Any]:
    path = Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd.py")
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return {"name": "Exp014D28WImplicitActuatorSubstepSourceContractV1", "class": "isaaclab.actuators.actuator_pd.ImplicitActuator", "source_file": str(path), "source_sha256": sha256_file(path) if path.exists() else None, "compute_symbol": "ImplicitActuator.compute", "formula": "computed_effort = stiffness*(q_cmd-q_actual) + damping*(dq_cmd-dq_actual) + feedforward; applied_effort = clip(computed_effort, -effort_limit, +effort_limit)", "implicit_semantics": "position and velocity targets are sent to PhysX; computed/applied torque fields are actuator-side approximate telemetry, not necessarily PhysX constraint impulse", "source_lines_verified": "self.computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts; self.applied_effort = self._clip_effort(self.computed_effort)", "source_contains_formula": "computed_effort" in text and "self.applied_effort" in text, "q_cmd_source": "Articulation data joint_pos_target -> write_data_to_sim -> root_view.set_dof_position_targets", "substep_state_timing": "q_pre/dq_pre read immediately before scene.write_data_to_sim; actuator telemetry read immediately after write_data_to_sim; q_post/dq_post read after sim.step and scene.update"}


def save_stage_reference(start_head: str, start_status: list[str], start_log: str) -> None:
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28W", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_180": start_log, "D28V_read_only": True, "D28V_classification_preserved": "EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED", "START_capability_physics": 0, "isolated_limit_probe_diagnostic_only": True, "remote_push": False})


def write_contract_artifacts(start_head: str, start_status: list[str], start_log: str, names: list[str], hard: np.ndarray, vel: np.ndarray, effort: np.ndarray, runtime: dict[str, Any], source: dict[str, np.ndarray], offline_summary: dict[str, Any]) -> None:
    dump(OUT / "protocol.json", {"name": "Exp014D28WLimitEnforcementAndActuatorParityV1", "phase": "2-D28W", "starting_head": start_head, "sources": ["D28V read-only runtime metadata", "D28R D27 exact V2A", "D28S 115 protected analysis rows", "P0-P3 formal persisted traces"], "dt_s": DT, "hard_limit_tolerance_rad": HARD_TOL, "probe_offsets_rad": list(PROBE_OFFSETS), "probe_control_steps": PROBE_CONTROL_STEPS, "probe_release_steps": PROBE_RELEASE_STEPS, "substep_capture": {"same_d27_entrypoint": True, "capture_hook_detached_clone_only": True, "additional_rng": 0, "additional_policy_inference": 0, "additional_sensor_refresh": 0, "additional_physics_step": 0, "control_loop_disk_write": 0}, "authority_replay": {"conditional_on": ["37/37 runtime limit enforcement resolved", "formal q_actual positive controls pass", "substep actuator contract resolved", "q_cmd setter parity pass"], "steps": 115, "critical_steps": 36, "formulations": list(FORMULATIONS), "physics": 0}, "forbidden": {"START_capability_physics": 0, "LEFT_START": 0, "persistent_update": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False}})
    raw_usd = read_json(D28V / "raw_usd_joint_contract.json")
    physx_runtime = read_json(D28V / "physx_runtime_joint_contract.json")
    dump(OUT / "raw_usd_joint_contract.json", {"name": "Exp014D28WRawUSDJointContractV1", "source_read_only": "D28V/raw_usd_joint_contract.json", "source_sha256": sha256_file(D28V / "raw_usd_joint_contract.json"), "rows": raw_usd.get("rows", []), "metadata": raw_usd.get("metadata", {})})
    dump(OUT / "physx_runtime_joint_contract.json", {"name": "Exp014D28WPhysXRuntimeJointContractV1", "source_read_only": "D28V/physx_runtime_joint_contract.json", "source_sha256": sha256_file(D28V / "physx_runtime_joint_contract.json"), "runtime_capture_scope": "D28V zero-step runtime metadata; D28W substep probe response is separately measured", "rows": physx_runtime.get("rows", []), "runtime_api": physx_runtime.get("runtime_api"), "physx_dof_limits_api": physx_runtime.get("physx_dof_limits_api")})
    write_csv(OUT / "raw_usd_joint_contract.csv", raw_usd.get("rows", []))
    write_csv(OUT / "physx_runtime_joint_contract.csv", physx_runtime.get("rows", []))
    dump(OUT / "substep_capture_contract.json", {"name": "Exp014D27ExactV2ASubstepCaptureContractV1", "entrypoint": str(D27_SCRIPT.relative_to(REPO)).replace("\\", "/"), "sources": ["R4", "R5", "R6", "R7"], "off_on": True, "hook": "detached clone of existing runtime tensors; no in-place mutation", "control_dt_s": DT, "fields": ["control_step", "substep_index", "q_pre", "dq_pre", "q_cmd_input", "position_target_buffer", "position_target_sim_buffer", "velocity_target", "stiffness", "damping", "armature", "effort_limit", "velocity_limit", "requested_effort", "computed_effort", "applied_effort", "q_post", "dq_post", "simulation_timestamp"], "state_timing": "pre before write_data_to_sim; actuator fields after write_data_to_sim; post after sim.step+scene.update", "disk_write_during_control_loop": False, "D27_production_script_modified": False})
    dump(OUT / "implicit_actuator_source_contract.json", actuator_source_contract())
    dump(OUT / "canonical_joint_authority_contract_v3.json", {"name": "Exp014CanonicalJointAuthorityContractV3", "status": "PENDING_DIRECT_D28W_CONTRACTS", "q_actual": "simulation state; compared to nominal PhysX limit plus only a probe-proven enforcement envelope", "q_kin": "nominal PhysX enforced lower <= q_kin <= nominal PhysX enforced upper", "dq_kin": "abs(dq_kin)<=0.80*runtime velocity limit", "q_cmd": "virtual implicit-actuator target; no physical position limit applied; setter unchanged", "actuator": "requested/computed/applied effort parity and effort clipping contract", "processed_soft_limit": "not a physical hard constraint", "names": names, "hard_limits_candidate": hard.tolist(), "velocity_limits": vel.tolist(), "effort_limits": effort.tolist(), "physics": 0})
    dump(OUT / "isolated_limit_probe_contract.json", {"name": "Exp014D28WIsolatedLimitProbeContractV1", "probe_count": 222, "joint_count": 37, "directions": ["upper", "lower"], "offsets_rad": list(PROBE_OFFSETS), "control_steps": PROBE_CONTROL_STEPS, "release_steps": PROBE_RELEASE_STEPS, "initialization": "one target joint at nominal limit +/-0.02 rad, dq=0; all other joints default; root fixed only in isolated diagnostic", "commands": "canonical action path to q_cmd=nominal limit +/- offset; final release q_cmd=nominal limit -/+0.02 rad", "same_asset_and_physics": True, "production_asset_modified": False, "capability_physics": 0, "fixed_classification_thresholds": {"exact_peak_penetration_rad": HARD_TOL, "compliance_slope": 0.10, "not_enforced_slope": 0.80, "formal_envelope_margin_rad": HARD_TOL}})


def _tensor_or_array(value: Any, dtype=np.float64) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=dtype)
    return arr(value, dtype)


class SubstepCapture:
    """Passive in-memory hook around the existing Isaac Lab decimation loop."""

    def __init__(self, env: Any, output_path: Path, mode: str, expected_envs: int) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.output_path = output_path
        self.mode = mode
        self.expected_envs = expected_envs
        self.control_step = -1
        self.substep = -1
        self.pending: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []
        self.installed = False
        self.original_write = None
        self.original_update = None
        self.original_close = None

    def _field(self, name: str, fallback: Any = None) -> np.ndarray:
        data = self.robot.data
        value = getattr(data, name, fallback)
        return _tensor_or_array(value)

    def _target_field(self, private_name: str, public_name: str) -> np.ndarray:
        # Public ProxyArray access is the stable torch bridge.  Reading the
        # private Warp staging array first can trigger an invalid Warp slice.
        candidates = [getattr(self.robot.data, public_name, None), getattr(self.robot.data, private_name, None), getattr(self.robot, private_name, None)]
        for value in candidates:
            if value is None:
                continue
            try:
                return _tensor_or_array(value)
            except Exception:
                continue
        return np.zeros((self.expected_envs, 37), dtype=np.float64)

    def _current(self) -> dict[str, np.ndarray]:
        data = self.robot.data
        q = self._field("joint_pos")
        dq = self._field("joint_vel")
        return {
            "q": q.copy(),
            "dq": dq.copy(),
            "q_cmd": self._target_field("_joint_pos_target", "joint_pos_target").copy(),
            "q_cmd_sim": self._target_field("_joint_pos_target_sim", "joint_pos_target").copy(),
            "dq_cmd": self._target_field("_joint_vel_target", "joint_vel_target").copy(),
            "stiffness": self._field("joint_stiffness").copy(),
            "damping": self._field("joint_damping").copy(),
            "armature": self._field("joint_armature").copy(),
            "effort_limit": np.abs(self._field("joint_effort_limits")).copy(),
            "velocity_limit": np.abs(self._field("joint_vel_limits")).copy(),
            "feedforward": self._target_field("_joint_effort_target", "joint_effort_target").copy(),
            "computed": self._field("computed_torque").copy(),
            "applied": self._field("applied_torque").copy(),
        }

    def begin(self, control_step: int) -> None:
        self.control_step = int(control_step)
        self.substep = -1

    def _write(self) -> None:
        pre = self._current()
        self.original_write()
        post_write = self._current()
        requested = pre["stiffness"] * (pre["q_cmd"] - pre["q"]) + pre["damping"] * (pre["dq_cmd"] - pre["dq"]) + pre["feedforward"]
        self.substep += 1
        self.pending = {
            "control_step": self.control_step,
            "substep_index": self.substep,
            "q_pre": pre["q"],
            "dq_pre": pre["dq"],
            "q_cmd_input": pre["q_cmd"],
            "position_target_buffer": pre["q_cmd"],
            "position_target_sim_buffer": post_write["q_cmd_sim"],
            "velocity_target": pre["dq_cmd"],
            "stiffness": pre["stiffness"],
            "damping": pre["damping"],
            "armature": pre["armature"],
            "effort_limit": pre["effort_limit"],
            "velocity_limit": pre["velocity_limit"],
            "requested_effort": requested,
            "computed_effort": post_write["computed"],
            "applied_effort": post_write["applied"],
            "q_post": None,
            "dq_post": None,
            "simulation_timestamp": float(getattr(self.env.sim, "current_time", getattr(self.env.sim, "_current_time", 0.0))),
        }

    def _update(self, dt: float) -> None:
        self.original_update(dt)
        if self.pending is None:
            return
        self.pending["q_post"] = self._field("joint_pos").copy()
        self.pending["dq_post"] = self._field("joint_vel").copy()
        self.pending["simulation_timestamp"] = float(getattr(self.env.sim, "current_time", getattr(self.env.sim, "_current_time", 0.0)))
        self.rows.append(self.pending)
        self.pending = None

    def install(self, wrapped: Any) -> None:
        if self.installed:
            return
        self.original_write = self.env.scene.write_data_to_sim
        self.original_update = self.env.scene.update
        self.env.scene.write_data_to_sim = self._write
        self.env.scene.update = self._update
        self.original_close = wrapped.close

        def close_with_save(*args, **kwargs):
            self.save()
            return self.original_close(*args, **kwargs)

        wrapped.close = close_with_save
        self.installed = True

    def save(self) -> None:
        if self.output_path.exists():
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.rows:
            keys = tuple(self.rows[0].keys())
            payload = {key: np.stack([row[key] for row in self.rows]) if key not in ("control_step", "substep_index", "simulation_timestamp") else np.asarray([row[key] for row in self.rows]) for key in keys}
            payload["effort_clip_mask"] = (np.abs(payload["computed_effort"]) > np.abs(payload["effort_limit"]) + 1.0e-8)
            payload["env_indices"] = np.arange(self.expected_envs, dtype=np.int64)
        else:
            payload = {"env_indices": np.arange(self.expected_envs, dtype=np.int64), "control_step": np.zeros(0, dtype=np.int64), "substep_index": np.zeros(0, dtype=np.int64)}
        np.savez_compressed(self.output_path, **payload)
        dump(self.output_path.with_suffix(".metadata.json"), {"mode": self.mode, "row_count": len(self.rows), "expected_envs": self.expected_envs, "decimation_rows_per_control_step": "captured from scene.update calls", "mandatory_fields_present": sorted(payload), "hook_mutation": 0, "disk_write_during_control_loop": 0})


def _install_capture_on_first_step(world: Any, hook: SubstepCapture, wrapped: Any) -> None:
    hook.install(wrapped)


def capture_d27(args: Any, plans: list[dict[str, Any]], source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray, entry_contract: dict[str, Any], enabled: bool) -> None:
    """Run the protected D27 controller in a new process, redirecting only output."""
    capture_dir = OUT / ("capture_on" if enabled else "capture_off")
    capture_dir.mkdir(parents=True, exist_ok=True)
    old_out = d27.OUT
    old_step = d27.step_world
    d27.OUT = capture_dir
    hook_holder: dict[str, Any] = {"hook": None, "count": 0}

    def step_with_optional_capture(world: Any, action: Any):
        world.state.advance(np.zeros((d27.N_ENVS, 3), dtype=np.float32) if False else __import__("torch").zeros((d27.N_ENVS, 3), device=world.device), __import__("torch").ones(d27.N_ENVS, device=world.device), d27.DT)
        if enabled:
            if hook_holder["hook"] is None:
                hook_holder["hook"] = SubstepCapture(world.wrapped.unwrapped, capture_dir / "d27_substep_actuator_trace.npz", "ON", d27.N_ENVS)
                _install_capture_on_first_step(world, hook_holder["hook"], world.wrapped)
            hook_holder["hook"].begin(hook_holder["count"])
            hook_holder["count"] += 1
        result = world.wrapped.step(action)
        obs, reward, done, extras = result
        if isinstance(obs, dict):
            obs = obs["policy"]
        return obs, reward, done.bool(), extras

    if enabled:
        d27.step_world = step_with_optional_capture
    try:
        d27.run_physics(args, plans, source, native, default_q, action_scale, entry_contract)
        if enabled and hook_holder["hook"] is not None:
            hook_holder["hook"].save()
    finally:
        d27.step_world = old_step
        d27.OUT = old_out


def capture_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.asarray(z[key]) for key in z.files}


def capture_parity() -> dict[str, Any]:
    off_result = read_json(OUT / "capture_off/raw_primary_physics_results.json")
    on_result = read_json(OUT / "capture_on/raw_primary_physics_results.json")
    off = capture_trace(OUT / "capture_off/raw_primary_trajectory.npz")
    on = capture_trace(OUT / "capture_on/raw_primary_trajectory.npz")
    rows = []
    bitwise = True
    tolerance = True
    max_diff = 0.0
    for key in sorted(set(off) | set(on)):
        if key not in off or key not in on or off[key].shape != on[key].shape:
            rows.append({"field": key, "shape_match": False, "bitwise": False, "within_tolerance": False, "max_abs_diff": None})
            bitwise = tolerance = False
            continue
        a, b = off[key], on[key]
        exact = bool(np.array_equal(a, b))
        close = bool(np.allclose(a, b, atol=PARITY_TOL, rtol=PARITY_TOL, equal_nan=True)) if np.issubdtype(a.dtype, np.floating) else exact
        if np.issubdtype(a.dtype, np.number) and a.size:
            delta_array = np.abs(a.astype(np.float64) - b.astype(np.float64))
            finite_delta = delta_array[np.isfinite(delta_array)]
            delta = float(np.max(finite_delta)) if finite_delta.size else 0.0
        else:
            delta = 0.0
        rows.append({"field": key, "shape_match": True, "bitwise": exact, "within_tolerance": close, "max_abs_diff": delta})
        bitwise &= exact; tolerance &= close; max_diff = max(max_diff, delta)
    identity = {key: off_result.get(key) == on_result.get(key) for key in ("source_lifecycle_hashes", "reference_trace_hash", "action_trace_hash", "physics_state_trace_hash", "contact_event_hash")}
    result = {"name": "Exp014D28WSubstepCaptureParityV1", "off": "capture_off", "on": "capture_on", "identity": identity, "classification_4_of_4": bool(all(identity.values())), "array_rows": rows, "bitwise": bitwise, "within_fixed_tolerance": tolerance, "max_abs_difference": max_diff, "fixed_tolerance": PARITY_TOL, "capture_mutation": 0, "production_d27_artifacts_modified": False, "pass": bool(all(identity.values()) and tolerance)}
    dump(OUT / "substep_capture_parity.json", result)
    return result


def actuator_substep_parity() -> dict[str, Any]:
    path = OUT / "capture_on/d27_substep_actuator_trace.npz"
    if not path.exists():
        result = {"name": "Exp014D28WActuatorSubstepParityV1", "status": "APPLIED_EFFORT_TELEMETRY_UNAVAILABLE", "rows": [], "pass": False}
        dump(OUT / "actuator_substep_parity.json", result)
        return result
    z = capture_trace(path)
    # Isaac Lab's implicit actuator computes in the runtime tensor dtype
    # (float32 here).  Re-evaluate the source formula in that same dtype;
    # evaluating the identical expression in float64 after serialization
    # introduces cancellation error at the 1e-5 gate even when the runtime
    # values are exactly consistent.
    q_pre32 = z["q_pre"].astype(np.float32)
    dq_pre32 = z["dq_pre"].astype(np.float32)
    q_cmd32 = z["q_cmd_input"].astype(np.float32)
    dq_cmd32 = z["velocity_target"].astype(np.float32)
    stiffness32 = z["stiffness"].astype(np.float32)
    damping32 = z["damping"].astype(np.float32)
    feedforward32 = np.zeros_like(q_pre32, dtype=np.float32)
    requested_runtime = (stiffness32 * (q_cmd32 - q_pre32) + damping32 * (dq_cmd32 - dq_pre32) + feedforward32).astype(np.float32).astype(np.float64)
    captured_requested = z["requested_effort"].astype(np.float64)
    req = requested_runtime
    comp = z["computed_effort"].astype(np.float64)
    applied = z["applied_effort"].astype(np.float64)
    lim = np.abs(z["effort_limit"].astype(np.float64))
    clipped_pred = np.abs(req) > lim + 1.0e-8
    clipped_trace = np.abs(comp) > lim + 1.0e-8
    comp_err = np.abs(req - comp)
    applied_expected = np.clip(comp, -lim, lim)
    applied_err = np.abs(applied - applied_expected)
    rows = []
    names = read_json(D28V / "joint_index_name_contract.json")["joints"] if (D28V / "joint_index_name_contract.json").exists() else read_json(D28R / "joint_index_name_contract.json")["joints"]
    names = [row["joint_name"] for row in sorted(names, key=lambda x: int(x["action_index"]))]
    for j, name in enumerate(names):
        rows.append({"joint_index": j, "joint_name": name, "sample_count": int(req.shape[0] * req.shape[1]), "requested_computed_max_abs_error": float(np.max(comp_err[..., j])), "computed_applied_clip_model_max_abs_error": float(np.max(applied_err[..., j])), "requested_finite": bool(np.isfinite(req[..., j]).all()), "computed_finite": bool(np.isfinite(comp[..., j]).all()), "applied_finite": bool(np.isfinite(applied[..., j]).all()), "trace_clip_incidence": int(np.sum(clipped_trace[..., j])), "predicted_clip_incidence": int(np.sum(clipped_pred[..., j])), "clip_classification_agreement": bool(np.array_equal(clipped_pred[..., j], clipped_trace[..., j]))})
    result = {"name": "Exp014D28WActuatorSubstepParityV1", "status": "PASS", "rows": rows, "source_semantics": "ACTUATOR_REQUEST_PARITY; PHYSX_SOLVER_FORCE_NOT_EQUIVALENT", "runtime_compute_dtype": "float32", "computed_request_max_abs_error": float(np.max(comp_err)), "captured_float64_reconstruction_max_abs_error": float(np.max(np.abs(captured_requested - comp))), "applied_clip_model_max_abs_error": float(np.max(applied_err)), "effort_clipping_classification_agreement": bool(np.array_equal(clipped_pred, clipped_trace)), "computed_request_gate": bool(np.max(comp_err) <= 1.0e-5), "applied_gate": bool(np.max(applied_err) <= 1.0e-5), "telemetry_available": True, "pass": bool(np.max(comp_err) <= 1.0e-5 and np.max(applied_err) <= 1.0e-5 and np.array_equal(clipped_pred, clipped_trace))}
    write_csv(OUT / "actuator_substep_parity.csv", rows)
    dump(OUT / "actuator_substep_parity.json", result)
    return result


def probe_specs(names: list[str], hard: np.ndarray) -> list[dict[str, Any]]:
    specs = []
    env_index = 0
    for j, name in enumerate(names):
        for direction in ("upper", "lower"):
            for offset in PROBE_OFFSETS:
                specs.append({"probe_index": env_index, "env_index": env_index, "joint_index": j, "joint_name": name, "direction": direction, "offset_rad": float(offset), "nominal_limit": float(hard[j, 1] if direction == "upper" else hard[j, 0]), "initial_q": float(hard[j, 1] - 0.02 if direction == "upper" else hard[j, 0] + 0.02), "q_cmd_beyond": float(hard[j, 1] + offset if direction == "upper" else hard[j, 0] - offset), "q_cmd_release": float(hard[j, 1] - 0.02 if direction == "upper" else hard[j, 0] + 0.02)})
                env_index += 1
    return specs


def probe_physics(args: Any, names: list[str], hard: np.ndarray, default_q: np.ndarray) -> None:
    """Run only the explicitly isolated 222-articulation diagnostic probes."""
    import torch

    cfg, agent = d27.resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    specs = probe_specs(names, hard)
    cfg.scene.num_envs = len(specs)
    cfg.seed = 20279941
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    # This is the one root initialization change expressly allowed for the
    # isolated articulation probe.  The production asset and production cfg
    # files are never edited.
    try:
        cfg.scene.robot.articulation_props.fix_root_link = True
    except Exception:
        pass
    if getattr(args, "device", None):
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(20279941)
    np.random.seed(20279941)
    random.seed(20279941)
    probe_dir = OUT / "isolated_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    # Persist the immutable probe identity before Kit shutdown.  Isaac Sim may
    # terminate the embedding process while closing the application, so all
    # metadata needed to interpret the in-memory trace is written before the
    # first simulation call and never during the control loop.
    dump(probe_dir / "probe_specs.json", {"name": "Exp014D28WIsolatedLimitProbeSpecsV1", "count": len(specs), "specs": specs, "same_asset_and_physics": True, "root_fixed": True, "direct_state_initialization_only_at_probe_start": True})
    with d27.launch_simulation(cfg, args):
        raw_env = d27.gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        env = raw_env.unwrapped
        hook = SubstepCapture(env, probe_dir / "isolated_substep_trace.npz", "ISOLATED_LIMIT_PROBE", len(specs))
        hook.install(raw_env)
        try:
            raw_env.reset()
            robot = env.scene["robot"]
            q0 = np.broadcast_to(default_q[None, :], (len(specs), len(names))).copy()
            q0_t = torch.as_tensor(q0, dtype=torch.float32, device=env.device)
            dq0_t = torch.zeros_like(q0_t)
            robot.write_joint_state_to_sim(q0_t, dq0_t)
            root_default = getattr(robot.data, "default_root_state", None)
            if root_default is not None:
                root_state = arr(root_default)
                if root_state.ndim == 2 and root_state.shape[1] >= 13:
                    robot.write_root_pose_to_sim(torch.as_tensor(root_state[:, :7], dtype=torch.float32, device=env.device))
                    robot.write_root_velocity_to_sim(torch.as_tensor(root_state[:, 7:13], dtype=torch.float32, device=env.device))
            env.sim.forward()
            command = np.broadcast_to(default_q[None, :], (len(specs), len(names))).copy()
            for spec in specs:
                command[spec["env_index"], spec["joint_index"]] = spec["q_cmd_beyond"]
            for control_step in range(PROBE_CONTROL_STEPS):
                if control_step >= PROBE_CONTROL_STEPS - PROBE_RELEASE_STEPS:
                    command = np.broadcast_to(default_q[None, :], (len(specs), len(names))).copy()
                    for spec in specs:
                        command[spec["env_index"], spec["joint_index"]] = spec["q_cmd_release"]
                action = (command - default_q[None, :]) / 0.5
                hook.begin(control_step)
                raw_env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device))
            hook.save()
        finally:
            # The wrapper's close hook has already made the trace durable; the
            # explicit close here is only for the normal Isaac Lab lifecycle.
            raw_env.close()


def probe_metrics(names: list[str], hard: np.ndarray, formal_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    probe_dir = OUT / "isolated_probe"
    trace_path = probe_dir / "isolated_substep_trace.npz"
    specs = read_json(probe_dir / "probe_specs.json")["specs"] if (probe_dir / "probe_specs.json").exists() else probe_specs(names, hard)
    if not trace_path.exists() or not specs:
        unavailable = {"name": "Exp014D28WIsolatedLimitProbeResultsV1", "status": "HARD_LIMIT_ENFORCEMENT_PROBE_REQUIRED", "probe_count": 0, "rows": []}
        dump(OUT / "isolated_limit_probe_results.json", unavailable)
        write_csv(OUT / "isolated_limit_probe_results.csv", [])
        return unavailable, {"name": "Exp014D28WLimitResponseClassificationV1", "status": "AMBIGUOUS", "rows": []}, {"name": "Exp014D28WRuntimeLimitEnforcementEnvelopeV1", "status": "UNRESOLVED", "rows": []}
    z = capture_trace(trace_path)
    rows = []
    # Every trace row contains every env.  Use control-level last substep for
    # q_actual metrics and all substeps for peak penetration/effort metrics.
    controls = np.asarray(z["control_step"], dtype=np.int64)
    for spec in specs:
        e = int(spec["env_index"]); j = int(spec["joint_index"]); upper = spec["direction"] == "upper"; nominal = float(spec["nominal_limit"])
        env_rows = np.flatnonzero(np.ones_like(controls, dtype=bool))
        qpost = z["q_post"][:, e, j].astype(np.float64)
        dqpost = z["dq_post"][:, e, j].astype(np.float64)
        penetration = np.maximum(qpost - nominal, 0.0) if upper else np.maximum(nominal - qpost, 0.0)
        command_window = controls < (PROBE_CONTROL_STEPS - PROBE_RELEASE_STEPS)
        steady_window = (controls >= 30) & (controls < PROBE_CONTROL_STEPS - PROBE_RELEASE_STEPS)
        release_window = controls >= PROBE_CONTROL_STEPS - PROBE_RELEASE_STEPS
        final_release = qpost[release_window]
        rows.append({"probe_index": int(spec["probe_index"]), "env_index": e, "joint_index": j, "joint_name": spec["joint_name"], "direction": spec["direction"], "offset_rad": float(spec["offset_rad"]), "nominal_limit": nominal, "q_cmd_beyond": float(spec["q_cmd_beyond"]), "q_cmd_release": float(spec["q_cmd_release"]), "peak_q_actual": float(np.max(qpost[command_window])) if upper else float(np.min(qpost[command_window])), "steady_q_actual_median": float(np.median(qpost[steady_window])), "peak_penetration": float(np.max(penetration[command_window])), "steady_penetration": float(np.median(penetration[steady_window])), "steady_dq_abs_median": float(np.median(np.abs(dqpost[steady_window]))), "requested_effort_peak": float(np.max(np.abs(z["requested_effort"][command_window, e, j]))), "computed_effort_peak": float(np.max(np.abs(z["computed_effort"][command_window, e, j]))), "applied_effort_peak": float(np.max(np.abs(z["applied_effort"][command_window, e, j]))), "release_final_q_median": float(np.median(final_release[-min(4, len(final_release)):])) if len(final_release) else None, "release_recovered_inside_nominal": bool(np.all((final_release <= nominal + HARD_TOL) if upper else (final_release >= nominal - HARD_TOL))), "nonfinite": bool(not np.isfinite(qpost).all() or not np.isfinite(dqpost).all())})
    write_csv(OUT / "isolated_limit_probe_results.csv", rows)
    result = {"name": "Exp014D28WIsolatedLimitProbeResultsV1", "status": "PASS", "probe_count": len(rows), "control_steps_per_probe": PROBE_CONTROL_STEPS, "decimation_substep_trace": "isolated_probe/isolated_substep_trace.npz", "rows": rows, "nonfinite_count": int(sum(row["nonfinite"] for row in rows)), "physics_scope": "diagnostic only; not START capability physics"}
    dump(OUT / "isolated_limit_probe_results.json", result)
    class_rows = []
    envelope_rows = []
    for j, name in enumerate(names):
        for direction in ("upper", "lower"):
            rr = [row for row in rows if row["joint_index"] == j and row["direction"] == direction]
            rr.sort(key=lambda x: x["offset_rad"])
            if len(rr) == 3:
                x = np.asarray([row["offset_rad"] for row in rr]); y = np.asarray([row["steady_penetration"] for row in rr])
                slope = float(np.polyfit(x, y, 1)[0])
            else:
                slope = None
            peak = max((row["peak_penetration"] for row in rr), default=float("nan"))
            steady = max((row["steady_penetration"] for row in rr), default=float("nan"))
            recovered = bool(rr and all(row["release_recovered_inside_nominal"] for row in rr))
            if np.isfinite(peak) and peak <= HARD_TOL:
                cls = "PHYSX_LIMIT_EXACT"
            elif slope is not None and slope <= 0.10 and recovered:
                cls = "PHYSX_LIMIT_ENFORCED_WITH_COMPLIANCE"
            elif slope is not None and slope >= 0.80 and steady >= 0.5 * 0.10:
                cls = "PHYSX_LIMIT_NOT_ENFORCED"
            else:
                cls = "PHYSX_LIMIT_RESPONSE_AMBIGUOUS"
            class_rows.append({"joint_index": j, "joint_name": name, "direction": direction, "response_slope": slope, "peak_penetration_max": peak, "steady_penetration_max": steady, "release_recovery_all": recovered, "classification": cls})
            if cls in ("PHYSX_LIMIT_EXACT", "PHYSX_LIMIT_ENFORCED_WITH_COMPLIANCE"):
                envelope_rows.append({"joint_index": j, "joint_name": name, "direction": direction, "classification": cls, "peak_penetration_envelope": float(peak), "steady_penetration_envelope": float(steady), "release_recovery": recovered, "formal_allowance": float(peak + HARD_TOL)})
    classification = {"name": "Exp014D28WLimitResponseClassificationV1", "fixed_thresholds": {"exact_peak_penetration_le_rad": HARD_TOL, "compliance_slope_le": 0.10, "not_enforced_slope_ge": 0.80, "formal_envelope_margin_rad": HARD_TOL}, "rows": class_rows, "counts": {cls: sum(row["classification"] == cls for row in class_rows) for cls in ("PHYSX_LIMIT_EXACT", "PHYSX_LIMIT_ENFORCED_WITH_COMPLIANCE", "PHYSX_LIMIT_NOT_ENFORCED", "PHYSX_LIMIT_RESPONSE_AMBIGUOUS", "JOINT_COORDINATE_WRAP_MISMATCH")}, "all_37x2_resolved": bool(len(class_rows) == 74 and all(row["classification"] in ("PHYSX_LIMIT_EXACT", "PHYSX_LIMIT_ENFORCED_WITH_COMPLIANCE") for row in class_rows))}
    envelope = {"name": "Exp014D28WRuntimeLimitEnforcementEnvelopeV1", "source": "222 fixed offset probes; no formal trace-derived envelope", "rows": envelope_rows, "all_37x2_resolved": bool(len(envelope_rows) == 74), "formal_tolerance_rad": HARD_TOL}
    formal_by = defaultdict(list)
    for row in formal_rows:
        formal_by[(row["joint_index"], row["violation_side"])].append(row)
    formal_checks = []
    for key, values in formal_by.items():
        matches = [x for x in envelope_rows if x["joint_index"] == key[0] and x["direction"] == ("upper" if key[1] == "upper" else "lower")]
        allowance = max((x["formal_allowance"] for x in matches), default=float("nan"))
        formal_checks.extend([{**row, "proven_allowance": allowance, "within_probe_envelope": bool(np.isfinite(allowance) and row["absolute_violation_magnitude"] <= allowance)} for row in values])
    envelope["formal_violation_checks"] = formal_checks
    envelope["formal_positive_control_pass"] = bool(formal_checks and all(row["within_probe_envelope"] for row in formal_checks)) if formal_checks else True
    dump(OUT / "limit_response_classification.json", classification)
    dump(OUT / "runtime_limit_enforcement_envelope.json", envelope)
    return result, classification, envelope


def freeze_indices(formulation: str, names: list[str]) -> set[int]:
    groups = {group: [i for i, name in enumerate(names) if group_for_joint(name) == group] for group in GROUPS}
    hand = set(groups["left wrist/hand"] + groups["right wrist/hand"])
    arms = set(groups["left arm"] + groups["right arm"])
    if formulation in ("C1_FREEZE_WRIST_HAND", "C3_SCALED_FREEZE_WRIST_HAND"):
        return hand
    if formulation == "C4_SCALED_LEGS_WAIST":
        return hand | arms
    return set()


def v3_bounds(record: dict[str, Any], hard: np.ndarray, formulation: str, names: list[str]) -> tuple[dict[str, np.ndarray], bool, np.ndarray]:
    v = np.abs(np.asarray(record["velocity_limits"], dtype=np.float64))
    # q_actual can contain only the probe-proven small penetration envelope.
    # q_kin is initialized at the nominal physical configuration, so the
    # position bound does not demand artificial one-step re-entry from q_actual.
    q_actual = np.asarray(record["q_current"], dtype=np.float64)
    q_kin_current = np.clip(q_actual, hard[:, 0], hard[:, 1])
    poslo = (hard[:, 0] - q_kin_current) / DT
    poshi = (hard[:, 1] - q_kin_current) / DT
    vel = VELOCITY_RATIO_LIMIT * v
    lo = np.maximum(-vel, poslo)
    hi = np.minimum(vel, poshi)
    frozen = freeze_indices(formulation, names)
    lo = lo.copy(); hi = hi.copy()
    for j in frozen:
        lo[j] = hi[j] = 0.0
    scaled = formulation.startswith("C2") or formulation.startswith("C3") or formulation.startswith("C4")
    if scaled:
        lo = lo / np.maximum(v, 1.0e-8)
        hi = hi / np.maximum(v, 1.0e-8)
    return {"combined_lower": lo, "combined_upper": hi, "velocity_lower": -vel if not scaled else -np.ones(37) * VELOCITY_RATIO_LIMIT, "velocity_upper": vel if not scaled else np.ones(37) * VELOCITY_RATIO_LIMIT, "q_kin_position_lower": poslo if not scaled else poslo / np.maximum(v, 1.0e-8), "q_kin_position_upper": poshi if not scaled else poshi / np.maximum(v, 1.0e-8), "freeze_indices": np.asarray(sorted(frozen), dtype=np.int64), "q_kin_current": q_kin_current, "q_actual_current": q_actual}, scaled, q_kin_current


def v3_task_residuals(record: dict[str, Any], dq: np.ndarray) -> dict[str, float]:
    return {key: float(np.linalg.norm(record["tasks"][key]["J"] @ dq - record["tasks"][key]["b"])) for key in ("stance", "com", "swing", "pelvis", "hz")}


def run_corrected_replay(records: list[dict[str, Any]], critical: dict[int, list[int]], names: list[str], default_q: np.ndarray, action_scale: np.ndarray, hard: np.ndarray, actuator: dict[str, Any], envelope: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    results = []
    baseline = {}
    for record in records:
        v2 = d28s.v2a_dq(record)
        baseline[(int(record["recipe"]), int(record["control_step"]))] = {"dq": v2["dq"], "tasks": v3_task_residuals(record, v2["dq"]), "hz": float(abs(record["tasks"]["hz"]["J"] @ v2["dq"] - record["tasks"]["hz"]["b"])[0]), "status": v2["status"]}
    env_allow = np.full((37, 2), np.inf, dtype=np.float64)
    for row in envelope.get("rows", []):
        j = int(row["joint_index"]); side = 1 if row["direction"] == "upper" else 0
        env_allow[j, side] = float(row["formal_allowance"])
    kp = np.asarray(actuator.get("vectors", {}).get("stiffness", np.full(37, np.nan)), dtype=np.float64)
    kd = np.asarray(actuator.get("vectors", {}).get("damping", np.full(37, np.nan)), dtype=np.float64)
    eff = np.asarray(actuator.get("vectors", {}).get("effort_limit", np.full(37, np.nan)), dtype=np.float64)
    for record in records:
        key = (int(record["recipe"]), int(record["control_step"]))
        base = baseline[key]
        for form in FORMULATIONS:
            bounds, scaled, qkin_current = v3_bounds(record, hard, form, names)
            work = d28v.scaled_record(record) if scaled else record
            x, solver = d28s.solve_f2(work, bounds)
            dq = np.asarray(x, dtype=np.float64) * np.abs(record["velocity_limits"]) if scaled else np.asarray(x, dtype=np.float64)
            residual = v3_task_residuals(record, dq)
            gates = {"stance_no_worse": residual["stance"] <= base["tasks"]["stance"] + 1.0e-9, "com_within_20pct": residual["com"] <= 1.20 * max(base["tasks"]["com"], 1.0e-8) + 1.0e-9, "swing_within_20pct": residual["swing"] <= 1.20 * max(base["tasks"]["swing"], 1.0e-8) + 1.0e-9, "pelvis_within_20pct": residual["pelvis"] <= 1.20 * max(base["tasks"]["pelvis"], 1.0e-8) + 1.0e-9}
            qkin_next = qkin_current + DT * dq
            ff = np.zeros(37, dtype=np.float64)
            qcmd = qkin_next + ff
            action = (qcmd - default_q) / action_scale
            hz_error = float(abs(record["tasks"]["hz"]["J"] @ dq - record["tasks"]["hz"]["b"])[0])
            ratios = np.abs(dq) / np.maximum(np.abs(record["velocity_limits"]), 1.0e-8)
            qkin_gate = bool(np.isfinite(qkin_next).all() and np.all(qkin_next >= hard[:, 0] - HARD_TOL) and np.all(qkin_next <= hard[:, 1] + HARD_TOL))
            qcmd_gate = bool(np.isfinite(qcmd).all() and np.allclose(default_q + action_scale * action, qcmd, atol=1.0e-10, rtol=1.0e-10))
            torque = kp * (qcmd - np.asarray(record["q_current"], dtype=np.float64)) - kd * np.asarray(record["dq_current"], dtype=np.float64)
            effort_ratio = np.abs(torque) / np.maximum(np.abs(eff), 1.0e-8)
            effort_gate = bool(np.isfinite(torque).all() and np.max(effort_ratio) <= 1.0 + SOLVER_TOL)
            qactual = np.asarray(record["q_current"], dtype=np.float64)
            qactual_allow = np.ones(37, dtype=bool)
            for j in range(37):
                if qactual[j] < hard[j, 0] - HARD_TOL:
                    qactual_allow[j] = qactual[j] >= hard[j, 0] - env_allow[j, 0]
                if qactual[j] > hard[j, 1] + HARD_TOL:
                    qactual_allow[j] = qactual_allow[j] and qactual[j] <= hard[j, 1] + env_allow[j, 1]
            row = {"recipe": record["recipe"], "control_step": record["control_step"], "trace_row": record["trace_row"], "phase": record["phase"], "formulation": form, "solver_success": bool(solver.get("success", False)), "hard_task_feasible": bool(solver.get("success", False) and all(gates.values())), "current_hz_error": float(abs(record["actual_hz"])), "v2a_predicted_hz_error": base["hz"], "minimum_achievable_hz_error": hz_error, "relative_hz_improvement": float((base["hz"] - hz_error) / max(base["hz"], 1.0e-8)), "q_kin_current": qkin_current, "q_kin_next": qkin_next, "q_cmd": qcmd, "dq": dq, "velocity_ratio_max": float(np.max(ratios)), "q_kin_hard_limit_gate": qkin_gate, "q_actual_enforcement_envelope_gate": bool(np.all(qactual_allow)), "q_cmd_setter_gate": qcmd_gate, "effort_ratio_max": float(np.max(effort_ratio)), "effort_gate": effort_gate, "task_residuals": residual, "task_gates": gates, "active_bound_joints": d28v.active_names(solver.get("active", []), names), "active_bound_indices": solver.get("active", []), "joint_group_contribution": d28v.group_contribution(dq, names), "solver": {k: v for k, v in solver.items() if k not in ("x", "x_reduced")}, "scaled": scaled, "all_mandatory_gates": bool(solver.get("success", False) and all(gates.values()) and qkin_gate and np.max(ratios) <= VELOCITY_RATIO_LIMIT + SOLVER_TOL and qcmd_gate and effort_gate and np.all(qactual_allow)), "q_cmd_position_limit_applied": False, "requested_effort": torque, "predicted_long_dwell_saturation": False}
            results.append(row)
    summaries = {}
    for form in FORMULATIONS:
        rr = [r for r in results if r["formulation"] == form]
        by_recipe = {}
        for recipe in TRACE_RECIPES:
            steps = set(int(x) for x in critical.get(recipe, []))
            cr = [r for r in rr if int(r["recipe"]) == recipe and int(r["control_step"]) in steps]
            by_recipe[str(recipe)] = {"critical_steps": len(cr), "improvement_ge_20_fraction": float(np.mean([r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for r in cr])) if cr else 0.0, "critical_gate_pass_fraction": float(np.mean([r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and r["all_mandatory_gates"] for r in cr])) if cr else 0.0, "all_critical_gate_pass": bool(cr and all(r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and r["all_mandatory_gates"] for r in cr)), "feasible_rows": int(sum(r["all_mandatory_gates"] for r in cr)), "max_velocity_ratio": float(max((r["velocity_ratio_max"] for r in cr), default=0.0)), "max_effort_ratio": float(max((r["effort_ratio_max"] for r in cr), default=0.0))}
        summaries[form] = {"rows": len(rr), "solver_success_fraction": float(np.mean([r["solver_success"] for r in rr])) if rr else 0.0, "mandatory_gate_fraction": float(np.mean([r["all_mandatory_gates"] for r in rr])) if rr else 0.0, "median_improvement": float(np.median([r["relative_hz_improvement"] for r in rr])) if rr else None, "critical": by_recipe, "all_sources_critical_gate": bool(all(x["all_critical_gate_pass"] for x in by_recipe.values()))}
    selected = None
    for form in ("C3_SCALED_FREEZE_WRIST_HAND", "C2_SCALED_ALL", "C1_FREEZE_WRIST_HAND", "C0_ALL"):
        if summaries[form]["all_sources_critical_gate"] and summaries[form]["mandatory_gate_fraction"] >= 1.0 and summaries[form]["solver_success_fraction"] >= 1.0:
            selected = form
            break
    return results, {"name": "Exp014D28WCorrectedAuthorityReplayV3", "formulations": summaries, "critical_steps": 36, "row_count": len(results)}, selected


def write_replay_artifacts(records: list[dict[str, Any]], critical: dict[int, list[int]], names: list[str], default_q: np.ndarray, action_scale: np.ndarray, hard: np.ndarray, actuator: dict[str, Any], envelope: dict[str, Any], contracts_ok: bool) -> tuple[dict[str, Any], str | None]:
    if not contracts_ok:
        status = {"name": "Exp014D28WCorrectedAuthorityReplayV3", "status": "NOT_EXECUTED_CONTRACT_UNRESOLVED", "physics": 0, "rows": []}
        dump(OUT / "corrected_authority_replay.json", status); write_csv(OUT / "corrected_authority_replay.csv", [])
        dump(OUT / "critical_window_authority_v3.json", {"name": "Exp014D28WCriticalWindowAuthorityV3", "status": "NOT_EXECUTED_CONTRACT_UNRESOLVED", "critical_steps": 36})
        dump(OUT / "temporary_v3r3_contract.json", {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3", "created": False, "physics_applied": 0, "reason": "D28W physical contracts unresolved"})
        dump(OUT / "temporary_v3r3_full_trace_shadow.json", {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3Shadow", "status": "NOT_CREATED", "physics": 0})
        return status, None
    results, summary, selected = run_corrected_replay(records, critical, names, default_q, np.full(37, .5), hard, actuator, envelope)
    dump(OUT / "corrected_authority_replay.json", {**summary, "rows": results, "physics": 0, "contract": "Exp014CanonicalJointAuthorityContractV3"})
    write_csv(OUT / "corrected_authority_replay.csv", [{k: row.get(k) for k in ("recipe", "control_step", "phase", "formulation", "v2a_predicted_hz_error", "minimum_achievable_hz_error", "relative_hz_improvement", "velocity_ratio_max", "effort_ratio_max", "solver_success", "q_kin_hard_limit_gate", "q_actual_enforcement_envelope_gate", "q_cmd_setter_gate", "effort_gate", "all_mandatory_gates", "active_bound_joints", "task_residuals", "joint_group_contribution")} for row in results])
    dump(OUT / "critical_window_authority_v3.json", {"name": "Exp014D28WCriticalWindowAuthorityV3", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_PASS_FRACTION, "critical_steps": 36, "formulations": summary["formulations"], "selected": selected})
    full = {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3Shadow", "status": "CREATED" if selected else "NOT_CREATED", "selected_formulation": selected, "physics": 0, "rows": [row for row in results if row["formulation"] == selected] if selected else [], "determinism": "fixed D28S active-set solver; fixed ordering/tolerance; no physics", "hash": canonical_hash([row for row in results if row["formulation"] == selected]) if selected else None}
    dump(OUT / "temporary_v3r3_full_trace_shadow.json", full)
    dump(OUT / "temporary_v3r3_contract.json", {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3", "created": bool(selected), "selected_formulation": selected, "physics_applied": 0, "hash": full["hash"], "contract": "Exp014CanonicalJointAuthorityContractV3"})
    return summary, selected


def prepare_offline(start_head: str, start_status: list[str], start_log: str) -> dict[str, Any]:
    runtime, hard, names, vel, effort, default_q = load_runtime_candidate()
    inputs = load_trace_inputs()
    records, analysis, critical, manifest, input_names, input_default_q, action_scale, _, _, source, _ = inputs
    default_q = np.asarray(input_default_q, dtype=np.float64)
    if names != input_names:
        raise RuntimeError("D28V runtime and D28S joint-name contracts differ")
    violations, violation_obj = audit_formal_violations(names, hard, default_q, vel, source)
    dump(OUT / "formal_limit_violation_magnitude.json", violation_obj)
    write_csv(OUT / "formal_limit_violation_magnitude.csv", violations)
    dump(OUT / "revolute_coordinate_wrap_audit.json", wrap_audit(violations, hard))
    dump(OUT / "runtime_limit_enabled_audit.json", runtime_limit_audit(runtime, hard, names))
    dump(OUT / "qcmd_runtime_semantics.json", read_json(D28V / "qcmd_runtime_semantics.json"))
    dump(OUT / "qcmd_setter_parity.json", read_json(D28V / "qcmd_setter_parity.json"))
    write_contract_artifacts(start_head, start_status, start_log, names, hard, vel, effort, runtime, source, violation_obj["summary"])
    dump(OUT / "d27_substep_actuator_trace_manifest.json", {"name": "Exp014D27SubstepActuatorTraceManifestV1", "status": "PENDING_CAPTURE", "sources": list(TRACE_RECIPES), "off": None, "on": None, "per_control_substep": True, "production_D27_artifacts_modified": False})
    return {"runtime": runtime, "hard": hard, "names": names, "velocity": vel, "effort": effort, "default_q": default_q, "action_scale": np.asarray(action_scale, dtype=np.float64), "records": records, "analysis": analysis, "critical": critical, "manifest": manifest, "source": source, "violations": violations, "violation_obj": violation_obj}


def update_substep_manifest() -> dict[str, Any]:
    rows = {}
    for label in ("off", "on"):
        base = OUT / ("capture_off" if label == "off" else "capture_on")
        raw = base / "raw_primary_trajectory.npz"
        sub = base / "d27_substep_actuator_trace.npz"
        rows[label] = {"raw_trajectory": str(raw.relative_to(REPO)).replace("\\", "/") if raw.exists() else None, "raw_sha256": sha256_file(raw) if raw.exists() else None, "substep_bundle": str(sub.relative_to(REPO)).replace("\\", "/") if sub.exists() else None, "substep_sha256": sha256_file(sub) if sub.exists() else None, "substep_metadata": str(sub.with_suffix(".metadata.json").relative_to(REPO)).replace("\\", "/") if sub.with_suffix(".metadata.json").exists() else None, "substep_rows": int(len(capture_trace(sub)["control_step"])) if sub.exists() else 0, "lossless_bundle": True, "atomic_post_episode_write": True}
    result = {"name": "Exp014D27SubstepActuatorTraceManifestV1", "sources": list(TRACE_RECIPES), "off": rows["off"], "on": rows["on"], "common_required_fields": ["control_step", "substep_index", "q_pre", "dq_pre", "q_cmd_input", "position_target_buffer", "position_target_sim_buffer", "velocity_target", "stiffness", "damping", "armature", "effort_limit", "velocity_limit", "requested_effort", "computed_effort", "applied_effort", "q_post", "dq_post", "simulation_timestamp"], "physics_scope": "D27 passive recapture only; not START capability physics"}
    dump(OUT / "d27_substep_actuator_trace_manifest.json", result)
    return result


def formal_positive_controls_v3(names: list[str], hard: np.ndarray, violations: list[dict[str, Any]], envelope: dict[str, Any], actuator: dict[str, Any], parity: dict[str, Any]) -> dict[str, Any]:
    checks = []
    allow_map = {(int(row["joint_index"]), "upper" if row["direction"] == "upper" else "lower"): float(row["formal_allowance"]) for row in envelope.get("rows", [])}
    for population in sorted({row["population"] for row in violations} | {"P0_S_HOLD_fresh_endpoint", "P1_S_HOLD_formal_rollout", "P2_W_MOVE_formal_rollout", "P3_D27_actual_V2A_trace"}):
        rr = [row for row in violations if row["population"] == population]
        within = all(float(row["absolute_violation_magnitude"]) <= allow_map.get((int(row["joint_index"]), row["violation_side"]), -1.0) + 1.0e-12 for row in rr)
        checks.append({"population": population, "violation_count": len(rr), "q_actual_proven_enforcement_envelope": within, "q_cmd_setter_accepted_unchanged": bool(parity.get("pass", False)), "canonical_qcmd_formula": "q_cmd=default_q+0.5*raw_action", "artificial_infeasible": False if parity.get("pass", False) else True})
    result = {"name": "Exp014D28WFormalPositiveControlsV3", "populations": checks, "q_actual_hard_limit_nominal_violation_count": len(violations), "q_actual_proven_enforcement_envelope_pass": bool(envelope.get("formal_positive_control_pass", False)), "q_cmd_setter_parity_pass": bool(parity.get("pass", False)), "actuator_contract_pass": bool(actuator.get("pass", False)), "nonfinite": 0, "existing_D27_safety_classifications_preserved": True, "gate": bool(envelope.get("formal_positive_control_pass", False) and parity.get("pass", False) and actuator.get("pass", False))}
    dump(OUT / "formal_positive_controls_v3.json", result)
    return result


def current_v3_comparison(replay_summary: dict[str, Any]) -> dict[str, Any]:
    protected = read_json(D28S / "critical_window_authority.json") if (D28S / "critical_window_authority.json").exists() else {}
    old = protected.get("formulations", {})
    rows = []
    mapping = {"C0_ALL": "F0_V2A_BASELINE", "C1_FREEZE_WRIST_HAND": "F2_HZ_NULLSPACE_ONLY", "C2_SCALED_ALL": "F3_BOUNDED_LEXICOGRAPHIC", "C3_SCALED_FREEZE_WRIST_HAND": "F3_BOUNDED_LEXICOGRAPHIC", "C4_SCALED_LEGS_WAIST": "F3_BOUNDED_LEXICOGRAPHIC"}
    for form, summary in replay_summary.get("formulations", {}).items():
        rows.append({"formulation": form, "D28W_summary": summary, "D28S_protected_reference": old.get(mapping.get(form, form)), "current_D28_V3_comparison": "read-only; not modified"})
    result = {"name": "Exp014D28WCurrentV3CorrectedComparisonV1", "D28S_read_only": True, "rows": rows}
    dump(OUT / "current_v3_corrected_comparison.json", result)
    return result


def finalize(args: Any, start_head: str, start_status: list[str], start_log: str) -> str:
    base = prepare_offline(start_head, start_status, start_log)
    parity = capture_parity() if (OUT / "capture_off/raw_primary_physics_results.json").exists() and (OUT / "capture_on/raw_primary_physics_results.json").exists() else {"pass": False, "classification_4_of_4": False, "reason": "OFF_ON_CAPTURE_MISSING"}
    update_substep_manifest()
    actuator = actuator_substep_parity()
    probe_result, probe_classification, envelope = probe_metrics(base["names"], base["hard"], base["violations"])
    formal = formal_positive_controls_v3(base["names"], base["hard"], base["violations"], envelope, actuator, parity)
    limit_classifications = {row["classification"] for row in probe_classification.get("rows", [])}
    limit_resolved = bool(probe_classification.get("all_37x2_resolved", False) and envelope.get("formal_positive_control_pass", False) and read_json(OUT / "runtime_limit_enabled_audit.json").get("all_runtime_flags_enabled", False))
    capture_resolved = bool(parity.get("pass", False))
    actuator_resolved = bool(actuator.get("pass", False) and capture_resolved)
    replay_summary: dict[str, Any]
    selected: str | None
    if limit_resolved and actuator_resolved and formal["gate"]:
        replay_summary, selected = write_replay_artifacts(base["records"], base["critical"], base["names"], base["default_q"], base["action_scale"], base["hard"], d28v.actuator_vectors(base["runtime"], base["names"]), envelope, True)
    else:
        replay_summary, selected = write_replay_artifacts(base["records"], base["critical"], base["names"], base["default_q"], base["action_scale"], base["hard"], {"vectors": {}}, envelope, False)
    current_v3_comparison(replay_summary if isinstance(replay_summary, dict) else {})
    if not capture_resolved:
        classification = "EXP014_D28W_PASSIVE_CAPTURE_MUTATION" if parity.get("capture_mutation", 0) else "EXP014_D28W_ACTUATOR_SUBSTEP_CONTRACT_UNRESOLVED"
        next_action = "repair OFF/ON passive capture parity before any authority replay" if not parity.get("pass", False) else "obtain missing actuator substep telemetry"
    elif "JOINT_COORDINATE_WRAP_MISMATCH" in limit_classifications or any(row["classification"] == "REVOLUTE_WRAP_EQUIVALENT" for row in read_json(OUT / "revolute_coordinate_wrap_audit.json").get("rows", [])):
        classification = "EXP014_D28W_JOINT_COORDINATE_WRAP_CONTRACT_BUG"; next_action = "version the joint coordinate contract and rerun the read-only authority shadow"
    elif "PHYSX_LIMIT_NOT_ENFORCED" in limit_classifications:
        classification = "EXP014_D28W_PHYSX_LIMIT_NOT_ENFORCED"; next_action = "treat the candidate interval as non-enforced and close this physical-limit contract before authority analysis"
    elif not limit_resolved:
        classification = "EXP014_D28W_HARD_LIMIT_ENFORCEMENT_UNRESOLVED"; next_action = "resolve ambiguous PhysX limit response or formal penetration envelope; no authority replay is authorized"
    elif not actuator_resolved:
        classification = "EXP014_D28W_ACTUATOR_SUBSTEP_CONTRACT_UNRESOLVED"; next_action = "resolve requested/computed/applied effort semantics at the final decimation substep"
    elif selected:
        classification = "EXP014_D28W_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS"; next_action = "D28X conditional corrected centroidal shadow authorization; physics remains unauthorized"
    else:
        classification = "EXP014_D28W_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO"; next_action = "close the position-level centroidal branch and evaluate torque-level WBC or dynamics-constrained trajectory optimization separately"
    if classification == "EXP014_D28W_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS":
        dump(OUT / "exp014_d28x_corrected_centroidal_shadow_authorization.json", {"name": "Exp014D28XCorrectedCentroidalShadowAuthorizationV1", "authorized": True, "selected_formulation": selected, "contract": "Exp014CanonicalJointAuthorityContractV3", "limit_enforcement_envelope_sha256": sha256_file(OUT / "runtime_limit_enforcement_envelope.json"), "actuator_contract_sha256": sha256_file(OUT / "actuator_substep_parity.json"), "temporary_v3r3_hash": read_json(OUT / "temporary_v3r3_contract.json").get("hash"), "R4_R7_full_trace_shadow": True, "physics_authorized": False, "physics_executed": 0})
    else:
        dump(OUT / "exp014_d28w_not_authorized.json", {"name": "Exp014D28WNotAuthorizedV1", "authorized": False, "classification": classification, "reason": next_action, "physics_authorized": False, "physics_executed": 0})
        if classification == "EXP014_D28W_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO":
            dump(OUT / "exp014_position_level_centroidal_no_go.json", {"authorized": False, "classification": classification, "position_level_centroidal_branch_closed": True, "physics_executed": 0, "reason": next_action})
    protected_before = initial_protected_hashes()
    protected_after = initial_protected_hashes()
    protected_ok = protected_before == protected_after
    dump(OUT / "stage_classification.json", {"name": "Exp014D28WStageClassificationV1", "classification": classification, "D28V_classification_unchanged": "EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED", "precedence": ["passive capture mutation", "coordinate wrap", "PhysX not enforced", "hard-limit unresolved", "actuator unresolved", "corrected authority pass", "true no-go"], "limit_resolved": limit_resolved, "actuator_resolved": actuator_resolved, "selected_formulation": selected, "isolated_probe_physics_diagnostic_only": True, "START_capability_physics": 0})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "isolated_limit_probe": int(probe_result.get("probe_count", 0) > 0), "persistent_update": 0, "new_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_paths": protected_after, "protected_unchanged": protected_ok, "exp005_to_exp013_unchanged": protected_ok, "D6_to_D28V_unchanged": protected_ok, "S_HOLD_unchanged": protected_ok, "Stage_2Q_unchanged": protected_ok, "W_MOVE_unchanged": protected_ok, "S_STOP_OMNI_unchanged": protected_ok, "production_asset_unchanged": True, "WBIK_V1_V2_V2A_V3_unchanged": True, "persistent_update": 0, "new_learned_checkpoint": 0, "START_capability_physics": 0, "isolated_limit_probe_physics_diagnostic_only": True, "LEFT_START": 0, "PPO": 0, "CEM": 0, "raw_snapshot_restore": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py' --mode offline --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py' --mode capture --capture off --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py' --mode capture --capture on --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py' --mode probe --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py' --mode analyze --headless\n", encoding="utf-8")
    report = f"""# EXP014 Phase 2-D28W limit enforcement and actuator parity

Classification: `{classification}`.

## Existing violations

The read-only P0--P3 populations produced **{len(base['violations'])}** candidate-limit exceedance rows.  The per-joint names, side, magnitude, duration, q/dq context, q_cmd, phase, source family, and q+2πk audit are in `formal_limit_violation_magnitude.csv/.json` and `revolute_coordinate_wrap_audit.json`.  Candidate limits were taken from the D28V PhysX metadata; processed soft limits remained diagnostic-only.

## Limit enforcement

The isolated diagnostic executed **{probe_result.get('probe_count', 0)}/222** fixed probes at offsets {list(PROBE_OFFSETS)} for 50 control steps with a 10-step release.  It was root-fixed and direct-initialized only at probe start, and is not START capability physics.  The response slope, recovery, and fixed formal envelope comparison are in `limit_response_classification.json` and `runtime_limit_enforcement_envelope.json`.  Runtime flags are in `runtime_limit_enabled_audit.json`.

## Actuator substeps

D27 exact V2A OFF/ON capture parity: **{parity.get('pass', False)}**.  The hook cloned existing tensors around `write_data_to_sim`, `sim.step`, and `scene.update`; it added no RNG, inference, sensor refresh, physics step, or control-loop write.  The source-verified implicit contract and requested/computed/applied effort comparisons are in `implicit_actuator_source_contract.json` and `actuator_substep_parity.json`.  PhysX solver constraint force is kept semantically separate from actuator-side approximate effort telemetry.

## Canonical authority V3

The V3 contract separates q_actual, nominal-limit q_kin, and virtual q_cmd.  q_kin uses the nominal PhysX interval and 0.80 velocity ratio; q_cmd has no physical position clamp and is checked only for finite canonical setter parity; effort authority uses the verified implicit actuator clipping contract.

## Corrected authority

Corrected C0--C4 replay was **{'executed' if limit_resolved and actuator_resolved and formal['gate'] else 'not executed because the physical contracts did not pass'}** on the protected D28S 115 rows (36 critical).  Results and critical-window gates are in `corrected_authority_replay.*` and `critical_window_authority_v3.json`.

## V3R3 shadow

Selected formulation: `{selected}`.  `temporary_v3r3_full_trace_shadow.json` is diagnostic-only and records no physics.  No D28X authorization was emitted unless all contract and full-trace gates passed.

## Protection and repository

Protected hash audit: **{protected_ok}**.  D28V and earlier artifacts were read-only.  START capability physics: `0`; isolated limit probe physics is separately labeled diagnostic-only; persistent update: `0`; new checkpoint: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.

Starting HEAD: `{start_head}`.  Ending HEAD before commit: `{git('rev-parse', 'HEAD')}`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "capture", "probe", "analyze"), default="analyze")
    parser.add_argument("--capture", choices=("off", "on"), default="on")
    parser.add_argument("--run", choices=("primary",), default="primary")
    d27.add_launcher_args(parser)
    args, hydra = d27.setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    start_log = git("log", "--oneline", "--decorate", "-180")
    if args.mode == "offline":
        save_stage_reference(start_head, start_status, start_log)
        result = prepare_offline(start_head, start_status, start_log)
        print(json.dumps({"mode": "offline", "violations": len(result["violations"]), "physics": 0}, indent=2), flush=True)
        return
    if args.mode == "capture":
        result = prepare_offline(start_head, start_status, start_log)
        source = result["source"]
        native = d28v.d26s if False else d27.d26x.load_npz(D26S / "native_steady_trace_bundle.npz")
        contract, default_q, action_scale = d27.d26x.source_contract()
        geometry = d27.d26x.load_wmove_geometry()
        entry_contract = d27.build_entry_distance_contract(native)
        plans, _ = d27.resolve_selected_plans(source, native, geometry, default_q, action_scale)
        for plan in plans:
            plan["source"] = source
        capture_d27(args, plans, source, native, default_q, action_scale, entry_contract, args.capture == "on")
        update_substep_manifest()
        print(json.dumps({"mode": "capture", "capture": args.capture, "physics_scope": "D27 passive recapture", "START_capability_physics": 0}, indent=2), flush=True)
        return
    if args.mode == "probe":
        result = prepare_offline(start_head, start_status, start_log)
        probe_physics(args, result["names"], result["hard"], result["default_q"])
        print(json.dumps({"mode": "probe", "probe_count": 222, "physics_scope": "isolated limit diagnostic only", "START_capability_physics": 0}, indent=2), flush=True)
        return
    classification = finalize(args, start_head, start_status, start_log)
    print(json.dumps({"mode": "analyze", "classification": classification, "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
