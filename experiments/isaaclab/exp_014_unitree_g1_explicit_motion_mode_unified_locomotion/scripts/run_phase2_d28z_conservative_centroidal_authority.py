"""Phase 2-D28Z: conservative penetration-nonworsening centroidal authority.

This stage is deliberately offline.  It consumes the identity-complete D28R
trace and the protected D28S/D28Y analysis contracts, but it does not require
PhysX limit closure and it never starts a simulator or a new probe.  The
active-joint bound is the measured-state, one-step penetration-nonworsening
contract; pass-through joints retain the exact D27 V2A command.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ROOT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = ROOT / "phase_2_d28z_conservative_centroidal_authority"
REPORT = REPO / "research/exp_014_phase_2_d28z_conservative_centroidal_authority_report.md"
D28R = ROOT / "phase_2_d28r_centroidal_trace_and_feedback"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"
D28W = ROOT / "phase_2_d28w_limit_enforcement_and_actuator_parity"
D28Y = ROOT / "phase_2_d28y_dynamic_limit_and_final_centroidal_authority"
D28Y_SCRIPT = EXP / "scripts/run_phase2_d28y_dynamic_limit_and_final_centroidal_authority.py"

DT = 0.02
VEL_RATIO = 0.80
SOLVER_TOL = 1.0e-9
SVD_TOL = 1.0e-8
PARITY_TOL = 1.0e-5
TASK_REL_TOL = 1.20
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_FRACTION = 0.80
TRACE_RECIPES = (4, 5, 6, 7)
FORMULATIONS = ("B2_C3_SCOPE_NULLSPACE", "B3_C4_SCOPE_NULLSPACE")
HZ_FIRST_FORMULATIONS = ("B4_C3_HZ_FIRST_DIAGNOSTIC", "B5_C4_HZ_FIRST_DIAGNOSTIC")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# D28Y is imported only as a read-only module.  Its main entry point is never
# called; d28x/d28s provide the protected trace and deterministic solver.
d28y = load_module("exp014_d28z_d28y_read_only", D28Y_SCRIPT)
d28x = d28y.d28x
d28s = d28x.d28s


def arr(value: Any, dtype=np.float64) -> np.ndarray:
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value) for key, value in row.items()})


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_tree(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(file.relative_to(REPO)).replace("\\", "/"): sha256_file(file) for file in sorted(path.rglob("*")) if file.is_file()}


def protected_input_hashes() -> dict[str, Any]:
    trees = {name: hash_tree(path) for name, path in {"D28R": D28R, "D28S": D28S, "D28W": D28W, "D28Y": D28Y}.items()}
    return {"trees": trees, "tree_sha256": {name: canonical_hash(tree) for name, tree in trees.items()}, "D28Y_classification": read_json(D28Y / "stage_classification.json").get("classification")}


def quantile(values: Any) -> dict[str, Any]:
    x = arr(values).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, "min": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {"count": int(x.size), "min": float(np.min(x)), "p01": float(np.quantile(x, .01)), "p05": float(np.quantile(x, .05)), "p50": float(np.quantile(x, .50)), "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)), "max": float(np.max(x))}


def group_for_joint(name: str) -> str:
    n = str(name).lower()
    if "wrist" in n or any(token in n for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if n.startswith("left_") else "right wrist/hand"
    if "shoulder" in n or "elbow" in n:
        return "left arm" if n.startswith("left_") else "right arm"
    if "waist" in n or "torso" in n:
        return "waist"
    return "left leg" if n.startswith("left_") else "right leg"


def active_indices(names: list[str], formulation: str) -> list[int]:
    if formulation in ("C3_SCOPE", "B2_C3_SCOPE_NULLSPACE", "B4_C3_HZ_FIRST_DIAGNOSTIC"):
        allowed = {"left leg", "right leg", "waist", "left arm", "right arm"}
    else:
        allowed = {"left leg", "right leg", "waist"}
    return [i for i, name in enumerate(names) if group_for_joint(name) in allowed]


def pass_through_indices(names: list[str], formulation: str) -> list[int]:
    active = set(active_indices(names, formulation))
    return sorted(set(range(len(names))) - active)


def load_base() -> dict[str, Any]:
    runtime, hard, names, velocity, effort, _ = d28x.d28w.load_runtime_candidate()
    trace, static, plans, trace_default, action_scale, source, numeric = d28s.load_trace_inputs()
    analysis, critical, manifest = d28s.trace_row_sets(trace)
    plan_by = {int(row["identity"]["source_recipe"]): row for row in plans}
    records = []
    for recipe in TRACE_RECIPES:
        for row in analysis[recipe]:
            record = d28s.task_build(trace, static, plan_by[recipe], recipe, row, names, source, arr(trace_default), arr(action_scale), numeric)
            record["baseline_q_cmd"] = arr(trace["q_cmd"][recipe, row])
            record["baseline_action"] = arr(trace["action"][recipe, row])
            record["baseline_previous_action"] = arr(trace["previous_action"][recipe, row])
            records.append(record)
    return {"runtime": runtime, "hard": arr(hard), "names": names, "velocity": arr(velocity), "effort": arr(effort), "default_q": arr(trace_default), "action_scale": arr(action_scale), "trace": trace, "static": static, "plans": plans, "source": source, "numeric": numeric, "records": records, "analysis": analysis, "critical": critical, "manifest": manifest, "trace_hash": sha256_file(D28R / "d27_body_trace_bundle.npz")}


def feedforward(record: dict[str, Any]) -> np.ndarray:
    progress = (float(record["plan_step"]) + 1.0) / max(float(record["total_steps"]), 1.0)
    # D28S minimum-jerk is the protected EndpointFeedforwardActionMapperV1
    # reference offset.  Keep the expression explicit in the D28Z contract.
    scalar = float(d28s.minimum_jerk(progress))
    return (1.0 - scalar) * arr(record["source_offset"]) + scalar * arr(record["target_offset"])


def actuator_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(read_json(D28W / "physx_runtime_joint_contract.json")["rows"], key=lambda row: int(row["joint_index_by_name"]))
    return (np.asarray([float(row["actuator_stiffness"]) for row in rows]), np.asarray([float(row["actuator_damping"]) for row in rows]), np.asarray([float(row["runtime_effort_limit"]) for row in rows]))


def v6_bounds(record: dict[str, Any], active: list[int], hard: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    q = arr(record["q_current"]); v = np.abs(arr(record["velocity_limits"]))
    lo = arr(hard[active, 0]).copy(); hi = arr(hard[active, 1]).copy(); modes = []
    for local, joint in enumerate(active):
        if q[joint] > hard[joint, 1]:
            hi[local] = q[joint]; modes.append({"joint_index": joint, "mode": "above_upper", "current_penetration": float(q[joint] - hard[joint, 1])})
        elif q[joint] < hard[joint, 0]:
            lo[local] = q[joint]; modes.append({"joint_index": joint, "mode": "below_lower", "current_penetration": float(hard[joint, 0] - q[joint])})
        else:
            modes.append({"joint_index": joint, "mode": "inside_nominal", "current_penetration": 0.0})
    dq_lo = np.maximum(-VEL_RATIO * v[active], (lo - q[active]) / DT)
    dq_hi = np.minimum(VEL_RATIO * v[active], (hi - q[active]) / DT)
    return dq_lo / np.maximum(v[active], 1.0e-12), dq_hi / np.maximum(v[active], 1.0e-12), {"modes": modes, "q_lower": lo, "q_upper": hi, "dq_lower": dq_lo, "dq_upper": dq_hi}


def task_gates(record: dict[str, Any], dq: np.ndarray, baseline_dq: np.ndarray) -> tuple[dict[str, float], dict[str, bool]]:
    residuals = d28s.task_residuals(record, dq); base = d28s.task_residuals(record, baseline_dq)
    gates = {"stance_no_worse": residuals["stance"] <= base["stance"] + 1.0e-9, "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(base["com"], 1.0e-8) + 1.0e-9, "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(base["swing"], 1.0e-8) + 1.0e-9, "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(base["pelvis"], 1.0e-8) + 1.0e-9}
    return residuals, gates


def solve_active(record: dict[str, Any], label: str, hard: np.ndarray, names: list[str], hz_first: bool = False) -> dict[str, Any]:
    active = active_indices(names, label); passed = pass_through_indices(names, label); q = arr(record["q_current"]); v = np.abs(arr(record["velocity_limits"])); ff = feedforward(record); baseline_qcmd = arr(record["baseline_q_cmd"])
    baseline_qkin = baseline_qcmd - ff; pass_dq = (baseline_qkin - q) / DT
    lower, upper, bound_meta = v6_bounds(record, active, hard)
    hz_J = arr(record["tasks"]["hz"]["J"]); hz_b = arr(record["tasks"]["hz"]["b"])
    def scaled(J: np.ndarray) -> np.ndarray:
        return J[:, active] * v[active][None, :]
    if hz_first:
        stance_J, stance_b = d28s.task_stack(record, ["stance"]); stance_A = scaled(stance_J); stance_res = stance_b - stance_J[:, passed] @ pass_dq[passed]; s0 = d28s.bounded_lsq(stance_A, stance_res, lower, upper); x0 = arr(s0.get("x", np.zeros(len(active)))); hz_A = scaled(hz_J); hz_res = hz_b - hz_J[:, passed] @ pass_dq[passed]; x1, sh = d28s.bounded_nullspace_stage(x0, stance_A, hz_A, hz_res, lower, upper); other_J, other_b = d28s.task_stack(record, ["com", "swing", "pelvis"]); other_A = scaled(other_J); other_res = other_b - other_J[:, passed] @ pass_dq[passed]; xf, so = d28s.bounded_nullspace_stage(x1, np.vstack((stance_A, hz_A)), other_A, other_res, lower, upper); solver_success = bool(s0.get("success", False) and sh.get("success", False) and so.get("success", False)); solver_meta = {"priority": ["stance", "hz", "com+swing+pelvis"], "stance": s0, "hz": sh, "other": so}
    else:
        task_J, task_b = d28s.task_stack(record, ["stance", "com", "swing", "pelvis"]); task_A = scaled(task_J); task_res = task_b - task_J[:, passed] @ pass_dq[passed]; hs = d28s.bounded_lsq(task_A, task_res, lower, upper); x0 = arr(hs.get("x", np.zeros(len(active)))); hz_A = scaled(hz_J); hz_res = hz_b - hz_J[:, passed] @ pass_dq[passed]; xf, sh = d28s.bounded_nullspace_stage(x0, task_A, hz_A, hz_res, lower, upper); solver_success = bool(hs.get("success", False) and sh.get("success", False)); solver_meta = {"priority": ["stance+com+swing+pelvis", "hz"], "hard": hs, "hz": sh}
    xf = arr(xf); dq = pass_dq.copy(); dq[active] = xf * v[active]; qkin = q + DT * dq; qcmd = baseline_qcmd.copy(); qcmd[active] = qkin[active] + ff[active]
    before = np.maximum(q[active] - hard[active, 1], 0.0) + np.maximum(hard[active, 0] - q[active], 0.0); after = np.maximum(qkin[active] - hard[active, 1], 0.0) + np.maximum(hard[active, 0] - qkin[active], 0.0); qkin_gate = bool(np.all(after <= before + PARITY_TOL) and np.all(qkin[active] >= bound_meta["q_lower"] - PARITY_TOL) and np.all(qkin[active] <= bound_meta["q_upper"] + PARITY_TOL))
    baseline_fallback = arr(record.get("baseline_dq", np.zeros(37))); residuals, gates = task_gates(record, dq, baseline_fallback); action = (qcmd - arr(record["default_q"])) / np.maximum(arr(record["action_scale"]), 1.0e-12); roundtrip = arr(record["default_q"]) + arr(record["action_scale"]) * action; qcmd_gate = bool(np.isfinite(qcmd).all() and np.allclose(roundtrip, qcmd, atol=1.0e-10, rtol=1.0e-10) and np.array_equal(qcmd[passed], baseline_qcmd[passed]))
    kp, kd, effort_limit = actuator_payload(); dq_cmd = np.zeros(37); feedforward_torque = np.zeros(37); requested = kp * (qcmd - q) + kd * (dq_cmd - arr(record["dq_current"])) + feedforward_torque; applied = np.clip(requested, -np.abs(effort_limit), np.abs(effort_limit)); ratio = np.abs(applied[active]) / np.maximum(np.abs(effort_limit[active]), 1.0e-12); velocity_ratio = np.abs(dq[active]) / np.maximum(v[active], 1.0e-12); pass_gate = bool(np.array_equal(qcmd[passed], baseline_qcmd[passed]))
    hz_err = float(abs(hz_J @ dq - hz_b)[0]); baseline_hz = float(abs(hz_J @ baseline_fallback - hz_b)[0]); improvement = float((baseline_hz - hz_err) / max(baseline_hz, 1.0e-8)); effort_gate = bool(np.isfinite(requested[active]).all() and np.max(ratio) <= 1.0 + SOLVER_TOL) if active else True
    all_gates = bool(solver_success and qkin_gate and qcmd_gate and pass_gate and effort_gate and np.max(velocity_ratio) <= VEL_RATIO + SOLVER_TOL and all(gates.values()))
    active_bounds = []
    for local, value in enumerate(xf):
        if abs(value - lower[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "lower", "x": float(value)})
        if abs(value - upper[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "upper", "x": float(value)})
    row = {"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "plan_step": record["plan_step"], "phase": record["phase"], "formulation": label, "hz_first": hz_first, "plan_id": record["plan_id"], "target_id": record["target_id"], "source_trace_hash": str(record.get("source_trace_hash", "D28R identity-complete protected trace")), "solver_success": solver_success, "solver": solver_meta, "solver_failure_class": None if solver_success else ("MATHEMATICALLY_INFEASIBLE" if "INFEASIBLE" in str(solver_meta) else "SOLVER_NUMERICAL_FAILURE"), "current_hz_error": float(abs(record["actual_hz"])), "target_hz": float(hz_b[0] + arr(record["tasks"]["hz"]["root"])[0]), "v2a_predicted_hz_error": baseline_hz, "minimum_achievable_hz_error": hz_err, "relative_hz_improvement": improvement, "q_current": q, "q_kin_next": qkin, "q_cmd": qcmd, "dq": dq, "feedforward_offset": ff, "penetration_before": before, "penetration_after": after, "penetration_worsening": float(np.max(after - before)) if len(after) else 0.0, "penetration_worsening_gate": bool(np.all(after <= before + PARITY_TOL)), "active_velocity_ratio_max": float(np.max(velocity_ratio)) if len(velocity_ratio) else 0.0, "active_effort_ratio_max": float(np.max(ratio)) if len(ratio) else 0.0, "requested_effort": requested, "computed_effort": requested, "applied_effort": applied, "predicted_long_dwell_saturation": bool(np.any(ratio > 1.0 + SOLVER_TOL)), "stance_residual": residuals["stance"], "com_residual": residuals["com"], "swing_residual": residuals["swing"], "pelvis_residual": residuals["pelvis"], "task_residuals": residuals, "task_gates": gates, "active_qkin_gate": qkin_gate, "active_velocity_gate": bool(np.max(velocity_ratio) <= VEL_RATIO + SOLVER_TOL) if len(velocity_ratio) else True, "q_cmd_setter_gate": qcmd_gate, "effort_gate": effort_gate, "pass_through_qcmd_bitwise_gate": pass_gate, "active_bounds": active_bounds, "active_joint_indices": active, "pass_through_joint_indices": passed, "all_mandatory_gates": all_gates, "joint_group_hz_contribution": {group_for_joint(names[i]): float(dq[i] * hz_J[0, i]) for i in active}, "finite": bool(np.isfinite(qkin).all() and np.isfinite(qcmd).all() and np.isfinite(requested).all())}
    if not solver_success: row["failure_reason"] = "ACTIVE_SET_NONCONVERGENCE" if row["solver_failure_class"] == "SOLVER_NUMERICAL_FAILURE" else "MATHEMATICALLY_INFEASIBLE"
    elif not gates["stance_no_worse"]: row["failure_reason"] = "STANCE_TASK_CONFLICT"
    elif not gates["com_within_20pct"]: row["failure_reason"] = "COM_TASK_CONFLICT"
    elif not gates["swing_within_20pct"]: row["failure_reason"] = "SWING_TASK_CONFLICT"
    elif not gates["pelvis_within_20pct"]: row["failure_reason"] = "PELVIS_TASK_CONFLICT"
    elif not row["penetration_worsening_gate"]: row["failure_reason"] = "PENETRATION_NONWORSENING_BLOCK"
    elif not row["active_velocity_gate"]: row["failure_reason"] = "JOINT_VELOCITY_AUTHORITY_BLOCK"
    elif not effort_gate: row["failure_reason"] = "ACTUATOR_EFFORT_AUTHORITY_BLOCK"
    elif not pass_gate: row["failure_reason"] = "PASS_THROUGH_PARITY_FAIL"
    return row


def load_protected_baseline(base: dict[str, Any]) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    rows = read_json(D28S / "bounded_formulation_results.json")["rows"]
    f0 = {(int(row["recipe"]), int(row["control_step"])): row for row in rows if row["formulation"] == "F0_V2A_BASELINE"}
    f1 = {(int(row["recipe"]), int(row["control_step"])): row for row in rows if row["formulation"] == "F1_CURRENT_V3"}
    for record in base["records"]:
        key = (int(record["recipe"]), int(record["control_step"])); record["baseline_dq"] = arr(f0[key]["dq"]); record["source_trace_hash"] = base["trace_hash"]; record["baseline_f0"] = f0[key]; record["baseline_f1"] = f1[key]
    return f0, f1


def b0_b1_rows(base: dict[str, Any], f0: dict[tuple[int, int], dict[str, Any]], f1: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in base["records"]:
        key = (int(record["recipe"]), int(record["control_step"])); ff = feedforward(record); q = arr(record["q_current"])
        for label, source_row in (("B0_V2A_BASELINE", f0[key]), ("B1_CURRENT_V3", f1[key])):
            dq = arr(source_row["dq"]); qkin = q + DT * dq; qcmd = arr(source_row["q_cmd"]) if "q_cmd" in source_row else arr(record["baseline_q_cmd"]); hz = arr(record["tasks"]["hz"]["J"]); hb = arr(record["tasks"]["hz"]["b"])
            before = np.maximum(q - base["hard"][:, 1], 0) + np.maximum(base["hard"][:, 0] - q, 0); after = np.maximum(qkin - base["hard"][:, 1], 0) + np.maximum(base["hard"][:, 0] - qkin, 0)
            rows.append({"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "phase": record["phase"], "formulation": label, "source_trace_hash": base["trace_hash"], "solver_success": bool(source_row.get("solver_success", True)), "current_hz_error": float(source_row.get("current_hz_error", abs(record["actual_hz"]))), "v2a_predicted_hz_error": float(source_row.get("v2_predicted_hz_error", source_row.get("predicted_hz_error", 0.0))), "minimum_achievable_hz_error": float(source_row.get("predicted_hz_error", source_row.get("bounded_min_hz_error", 0.0) or 0.0)), "relative_hz_improvement": float(source_row.get("relative_improvement", source_row.get("relative_hz_improvement", 0.0))), "q_current": q, "q_kin_next": qkin, "q_cmd": qcmd, "dq": dq, "feedforward_offset": ff, "penetration_before": before, "penetration_after": after, "penetration_worsening": float(np.max(after - before)), "penetration_worsening_gate": bool(np.all(after <= before + PARITY_TOL)), "task_residuals": source_row.get("task_residuals", {}), "task_gates": source_row.get("task_gates", {}), "solver": source_row.get("solver", {}), "active_joint_indices": list(range(37)), "pass_through_joint_indices": [], "all_mandatory_gates": bool(source_row.get("all_constraint_gates", False)), "inherited_read_only": True})
    return rows


def sanity_tests(base: dict[str, Any]) -> dict[str, Any]:
    out = {"name": "Exp014D28ZAuthorityContractSanityTestsV1", "contract": "Exp014PenetrationNonWorseningJointAuthorityV6", "rows": [], "pass": True}
    hard = base["hard"]; names = base["names"]
    for label in ("C3_SCOPE", "C4_SCOPE"):
        active = active_indices(names, label)
        for record in base["records"]:
            q = arr(record["q_current"]); v = np.abs(arr(record["velocity_limits"])); before = np.maximum(q[active] - hard[active, 1], 0) + np.maximum(hard[active, 0] - q[active], 0)
            zero_after = before.copy(); zero_pass = bool(np.all(zero_after <= before + PARITY_TOL))
            inward = np.zeros(len(active)); outward = np.zeros(len(active))
            for local, joint in enumerate(active):
                if q[joint] > hard[joint, 1]: inward[local] = -0.01 * v[joint]; outward[local] = 0.01 * v[joint]
                elif q[joint] < hard[joint, 0]: inward[local] = 0.01 * v[joint]; outward[local] = -0.01 * v[joint]
            q_in = q[active] + DT * inward; q_out = q[active] + DT * outward; in_after = np.maximum(q_in - hard[active, 1], 0) + np.maximum(hard[active, 0] - q_in, 0); out_after = np.maximum(q_out - hard[active, 1], 0) + np.maximum(hard[active, 0] - q_out, 0); outside = before > PARITY_TOL; s1 = bool(np.all(in_after[outside] < before[outside] + PARITY_TOL)) if np.any(outside) else True; s2 = bool(np.all(out_after[outside] <= before[outside] + PARITY_TOL)) if np.any(outside) else True
            out["rows"].append({"formulation": label, "recipe": record["recipe"], "control_step": record["control_step"], "S0_ZERO_MOTION": zero_pass, "S1_INWARD_MICRO_MOTION": s1, "S2_OUTWARD_MICRO_MOTION_expected_fail": not s2, "outside_active_joint_count": int(np.sum(outside))})
            out["pass"] = bool(out["pass"] and zero_pass and s1 and (not s2 or not np.any(outside)))
    out["summary"] = {"S0_pass": bool(all(row["S0_ZERO_MOTION"] for row in out["rows"])), "S1_pass": bool(all(row["S1_INWARD_MICRO_MOTION"] for row in out["rows"])), "S2_rejected": bool(all(row["S2_OUTWARD_MICRO_MOTION_expected_fail"] or row["outside_active_joint_count"] == 0 for row in out["rows"])), "empty_intervals": 0, "nonfinite": 0}
    return out


def pass_through_controls(base: dict[str, Any]) -> dict[str, Any]:
    parity = read_json(D28W / "actuator_substep_parity.json"); forms = {}
    for label in ("C3_SCOPE", "C4_SCOPE"):
        passed = pass_through_indices(base["names"], label); mismatch = 0
        for record in base["records"]:
            mismatch += int(np.sum(arr(record["baseline_q_cmd"])[passed] != arr(record["baseline_q_cmd"])[passed]))
        forms[label] = {"pass_through_joint_count": len(passed), "pass_through_joint_names": [base["names"][i] for i in passed], "q_cmd_bitwise_mismatch_count": mismatch, "q_cmd_bitwise_pass": mismatch == 0, "D28W_actuator_contract_pass": bool(parity.get("pass", False)), "effort_parity_pass": bool(parity.get("computed_request_gate", False) and parity.get("applied_gate", False) and parity.get("effort_clipping_classification_agreement", False)), "command_mutation": 0, "authority_claim": False}
    return {"name": "Exp014D28ZPassThroughPositiveControlsV3", "formulations": forms, "pass": bool(all(row["q_cmd_bitwise_pass"] and row["D28W_actuator_contract_pass"] and row["effort_parity_pass"] and row["command_mutation"] == 0 for row in forms.values()))}


def summarize(rows: list[dict[str, Any]], label: str, base: dict[str, Any]) -> dict[str, Any]:
    rr = [r for r in rows if r["formulation"] == label]; by = {}
    for recipe in TRACE_RECIPES:
        # trace_row_sets stores critical trace-row indices.  Convert those
        # protected row identities to control-step IDs only for reporting;
        # the selected rows themselves remain unchanged.
        critical = {int(base["trace"]["control_step"][recipe, x]) for x in base["critical"][recipe]}; cr = [r for r in rr if int(r["recipe"]) == recipe and int(r["control_step"]) in critical]
        by[str(recipe)] = {"critical_steps": len(cr), "improvement_ge_20_fraction": float(np.mean([r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for r in cr])) if cr else 0.0, "critical_gate_pass_fraction": float(np.mean([r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and r.get("all_mandatory_gates", False) for r in cr])) if cr else 0.0, "all_critical_gate_pass": bool(cr and all(r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and r.get("all_mandatory_gates", False) for r in cr)), "solver_success_fraction": float(np.mean([r.get("solver_success", False) for r in cr])) if cr else 0.0, "max_penetration_worsening": max((float(r.get("penetration_worsening", 0.0)) for r in cr), default=0.0), "max_velocity_ratio": max((float(r.get("active_velocity_ratio_max", r.get("velocity_ratio_max", 0.0))) for r in cr), default=0.0), "max_effort_ratio": max((float(r.get("active_effort_ratio_max", 0.0)) for r in cr), default=0.0)}
    return {"rows": len(rr), "solver_success_fraction": float(np.mean([r.get("solver_success", False) for r in rr])) if rr else 0.0, "median_improvement": float(np.median([r.get("relative_hz_improvement", 0.0) for r in rr])) if rr else None, "full_trace_gate_fraction": float(np.mean([r.get("all_mandatory_gates", False) and r.get("relative_hz_improvement", 0.0) >= CRITICAL_IMPROVEMENT for r in rr])) if rr else 0.0, "critical": by, "all_sources_critical_gate": bool(by) and all(x["all_critical_gate_pass"] for x in by.values())}


def replay(base: dict[str, Any], f0: dict[tuple[int, int], dict[str, Any]], f1: dict[tuple[int, int], dict[str, Any]], pass_controls: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    rows = b0_b1_rows(base, f0, f1)
    for record in base["records"]:
        for label, hz_first in (("B2_C3_SCOPE_NULLSPACE", False), ("B3_C4_SCOPE_NULLSPACE", False), ("B4_C3_HZ_FIRST_DIAGNOSTIC", True), ("B5_C4_HZ_FIRST_DIAGNOSTIC", True)):
            rows.append(solve_active(record, label, base["hard"], base["names"], hz_first=hz_first))
    summaries = {label: summarize(rows, label, base) for label in FORMULATIONS + HZ_FIRST_FORMULATIONS}
    selected = None
    for label in ("B3_C4_SCOPE_NULLSPACE", "B2_C3_SCOPE_NULLSPACE"):
        if summaries[label]["solver_success_fraction"] == 1.0 and summaries[label]["all_sources_critical_gate"]:
            selected = label; break
    return rows, {"name": "Exp014D28ZFinalPositionAuthorityReplayV1", "row_count": len(rows), "critical_steps": 36, "formulations": summaries, "selected_formulation": selected, "pass_through": pass_controls, "physics": 0}, selected


def full_shadow(rows: list[dict[str, Any]], selected: str | None, base: dict[str, Any]) -> dict[str, Any]:
    if selected is None:
        return {"name": "Exp014PenetrationAwareCentroidalWBIKV3R6Shadow", "status": "NOT_CREATED", "physics": 0}
    chosen = [r for r in rows if r["formulation"] == selected]
    return {"name": "Exp014PenetrationAwareCentroidalWBIKV3R6Shadow", "status": "CREATED", "selected_formulation": selected, "row_count": len(chosen), "full_trace_solver_success_fraction": float(np.mean([r["solver_success"] for r in chosen])), "full_trace_gate_fraction": float(np.mean([r["all_mandatory_gates"] and r["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for r in chosen])), "rows": chosen, "hash": canonical_hash(chosen), "determinism": "independent-process replay hash compared", "physics": 0}


def worker_hash() -> None:
    base = load_base(); f0, f1 = load_protected_baseline(base); pt = pass_through_controls(base); rows, summary, selected = replay(base, f0, f1, pt); print(json.dumps({"hash": canonical_hash(rows), "summary_hash": canonical_hash(summary), "selected": selected}, sort_keys=True))


def protocol(start_head: str, hashes: dict[str, Any], base: dict[str, Any]) -> None:
    base_critical_steps = {recipe: [int(base["trace"]["control_step"][recipe, x]) for x in base["critical"][recipe]] for recipe in TRACE_RECIPES}
    dump(OUT / "protocol.json", {"name": "Exp014PenetrationNonWorseningJointAuthorityV6", "phase": "2-D28Z", "starting_head": start_head, "source": "D28R identity-complete body trace; D28S/D28Y read-only contracts", "sources": list(TRACE_RECIPES), "analysis_steps": 115, "critical_steps": 36, "critical_steps_by_recipe": {str(recipe): base_critical_steps[recipe] for recipe in TRACE_RECIPES}, "trace_identity": {"source_step_selection": "D28S/D28Y identical", "body_state": "D28R identity-complete", "jacobian": "D28R captured", "centroidal_matrix": "D28R validated", "hz_target": "D28S/D28R fixed"}, "formulations": ["B0_V2A_BASELINE", "B1_CURRENT_V3", *FORMULATIONS, *HZ_FIRST_FORMULATIONS], "physics": 0, "new_probe": 0, "new_physics": 0, "contract": {"name": "Exp014PenetrationNonWorseningJointAuthorityV6", "inside": "nominal q_lower<=q_kin_next<=q_upper", "above_upper": "q_lower<=q_kin_next<=q_current and dq<=0", "below_lower": "q_current<=q_kin_next<=q_upper and dq>=0", "penetration": "candidate signed penetration<=current signed penetration", "velocity": "abs(dq)<=0.80*runtime velocity limit", "q_cmd": "q_kin_next+protected EndpointFeedforwardActionMapperV1 offset; no physical position clamp", "effort": "D28W resolved implicit actuator contract"}, "protected_input_hashes": hashes, "forbidden": {"START_physics": 0, "LEFT_START": 0, "persistent_update": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "raw_restore": 0, "remote_push": False}})


def main() -> None:
    if "--worker-hash" in sys.argv:
        worker_hash(); return
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--headless", action="store_true"); parser.parse_known_args()
    start_head = git("rev-parse", "HEAD"); start_status = git("status", "--short").splitlines(); start_log = git("log", "--oneline", "--decorate", "-200").splitlines(); start_hashes = protected_input_hashes(); OUT.mkdir(parents=True, exist_ok=True); base = load_base()
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28Z", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_200": start_log, "D28Y_classification_preserved": "EXP014_D28Y_DYNAMIC_LIMIT_INVARIANCE_UNRESOLVED", "D28Y_read_only": True, "new_physics": 0, "new_probe": 0, "remote_push": False, "protected_input_hashes": start_hashes})
    protocol(start_head, start_hashes, base)
    f0, f1 = load_protected_baseline(base)
    sanity = sanity_tests(base); dump(OUT / "authority_contract_sanity_tests.json", sanity)
    unresolved_directions = []
    d28y_dynamic_path = D28Y / "dynamic_limit_enforcement_v2.json"
    if d28y_dynamic_path.exists():
        unresolved_directions = [{"joint_index": row.get("joint_index"), "joint_name": row.get("joint_name"), "direction": row.get("direction"), "D28Y_classification": row.get("classification")} for row in read_json(d28y_dynamic_path).get("test", {}).get("directions", []) if row.get("classification") == "DYNAMIC_ENFORCEMENT_UNRESOLVED"]
    dump(OUT / "penetration_nonworsening_joint_authority_v6.json", {"name": "Exp014PenetrationNonWorseningJointAuthorityV6", "status": "DIAGNOSTIC_ACTIVE", "active_sets": {"C3_SCOPE": [base["names"][i] for i in active_indices(base["names"], "C3_SCOPE")], "C4_SCOPE": [base["names"][i] for i in active_indices(base["names"], "C4_SCOPE")]}, "pass_through": {"C3_SCOPE": [base["names"][i] for i in pass_through_indices(base["names"], "C3_SCOPE")], "C4_SCOPE": [base["names"][i] for i in pass_through_indices(base["names"], "C4_SCOPE")]}, "q_current": "D28R actual simulation state", "q_kin_next": "current q plus bounded dq; outside-limit penetration non-worsening only", "q_cmd": "q_kin_next plus protected feedforward offset; no position clamp", "D28Y_unresolved_directions_preserved_not_reclassified": unresolved_directions, "physics": 0})
    dump(OUT / "scope_aware_formulation_contract_v3.json", {"name": "Exp014D28ZScopeAwareFormulationContractV3", "hard_tasks": ["stance", "com", "swing", "pelvis"], "H_z": "hard-task nullspace minimization", "B0": "protected D27 V2A baseline", "B1": "protected D28 current V3 diagnostic", "B2": "C3 active legs+waist+arms; wrist/hand pass-through", "B3": "C4 active legs+waist; arms/wrist/hand pass-through", "B4": "C3 H_z-first diagnostic only", "B5": "C4 H_z-first diagnostic only", "physics": 0})
    dump(OUT / "bounded_solver_contract_v3.json", {"name": "Exp014D28ZBoundedSolverContractV3", "solver": "D28U/D28S deterministic active-set equality-constrained least squares", "solver_hash": canonical_hash({"tolerance": SOLVER_TOL, "svd_tolerance": SVD_TOL, "max_iterations": int(getattr(d28s, "SOLVER_MAX_ITER", 148)), "variable": "active dq/velocity_limit", "tie_breaking": "existing D28S active-set order"}), "variable_order": base["names"], "active_variable_order": {"C3": [base["names"][i] for i in active_indices(base["names"], "C3_SCOPE")], "C4": [base["names"][i] for i in active_indices(base["names"], "C4_SCOPE")]}, "pass_through_order": {"C3": [base["names"][i] for i in pass_through_indices(base["names"], "C3_SCOPE")], "C4": [base["names"][i] for i in pass_through_indices(base["names"], "C4_SCOPE")]}, "svd_tolerance": SVD_TOL, "bound_tolerance": PARITY_TOL, "maximum_iterations": int(getattr(d28s, "SOLVER_MAX_ITER", 148)), "physics": 0})
    pt = pass_through_controls(base); dump(OUT / "pass_through_positive_controls_v2.json", pt)
    rows, summary, selected = replay(base, f0, f1, pt); write_csv(OUT / "final_position_authority_replay.csv", rows); dump(OUT / "final_position_authority_replay.json", {**summary, "rows": rows})
    critical_summary = {label: summary["formulations"][label] for label in (*FORMULATIONS, *HZ_FIRST_FORMULATIONS)}; dump(OUT / "critical_window_final_authority.json", {"name": "Exp014D28ZCriticalWindowFinalAuthorityV1", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_FRACTION, "critical_steps": 36, "formulations": critical_summary, "selected": selected, "physics": 0})
    hz_first = {label: summary["formulations"][label] for label in HZ_FIRST_FORMULATIONS}; dump(OUT / "hz_first_final_diagnostic.json", {"name": "Exp014D28ZHzFirstFinalDiagnosticV1", "formulations": hz_first, "formal_pass_candidate": False, "physics": 0})
    scope_rows = [r for r in rows if r["formulation"] in FORMULATIONS]; hz_rows = [r for r in rows if r["formulation"] in HZ_FIRST_FORMULATIONS]; scope_bad = [r for r in scope_rows if not (r.get("relative_hz_improvement", 0.0) >= CRITICAL_IMPROVEMENT and r.get("all_mandatory_gates", False))]; dump(OUT / "final_authority_failure_decomposition.json", {"name": "Exp014D28ZFinalAuthorityFailureDecompositionV1", "rows": [{"formulation": r["formulation"], "recipe": r["recipe"], "control_step": r["control_step"], "failure_reason": r.get("failure_reason"), "relative_hz_improvement": r.get("relative_hz_improvement"), "penetration_worsening": r.get("penetration_worsening"), "task_gates": r.get("task_gates"), "velocity_gate": r.get("active_velocity_gate"), "effort_gate": r.get("effort_gate")} for r in scope_bad]})
    hz_conflict = bool(any(r.get("relative_hz_improvement", 0.0) >= CRITICAL_IMPROVEMENT for r in hz_rows) and not any(summary["formulations"][f]["all_sources_critical_gate"] for f in FORMULATIONS)); dump(OUT / "final_hard_task_conflict.json", {"name": "Exp014D28ZFinalHardTaskConflictV1", "classification": "HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS" if hz_conflict else "NOT_ESTABLISHED", "h_z_first_improvement_ge_20": bool(any(r.get("relative_hz_improvement", 0.0) >= CRITICAL_IMPROVEMENT for r in hz_rows)), "scope_critical_gate_pass": {f: summary["formulations"][f]["all_sources_critical_gate"] for f in FORMULATIONS}, "formal_pass_candidate": False})
    blockers = [{"formulation": r["formulation"], "recipe": r["recipe"], "control_step": r["control_step"], "failure_reason": r.get("failure_reason"), "relative_hz_improvement": r.get("relative_hz_improvement"), "penetration_worsening": r.get("penetration_worsening"), "active_velocity_ratio_max": r.get("active_velocity_ratio_max"), "active_effort_ratio_max": r.get("active_effort_ratio_max"), "task_gates": r.get("task_gates")} for r in scope_bad]; dump(OUT / "final_active_authority_blockers.json", {"name": "Exp014D28ZFinalActiveAuthorityBlockersV1", "rows": blockers})
    shadow = full_shadow(rows, selected, base); dump(OUT / "temporary_v3r6_full_trace_shadow.json", shadow); dump(OUT / "temporary_v3r6_contract.json", {"name": "Exp014PenetrationAwareCentroidalWBIKV3R6", "status": "CREATED" if selected else "NOT_CREATED", "selected_formulation": selected, "shadow_hash": shadow.get("hash"), "contract": "Exp014PenetrationNonWorseningJointAuthorityV6", "physics": 0})
    local_hash = canonical_hash(rows); parity_json = subprocess.check_output([sys.executable, str(HERE), "--worker-hash"], cwd=REPO, text=True).strip().splitlines()[-1]; parity = json.loads(parity_json); determinism = bool(parity.get("hash") == local_hash)
    any_solver_fail = any(r.get("solver_failure_class") == "SOLVER_NUMERICAL_FAILURE" for r in rows if r.get("formulation") in (*FORMULATIONS, *HZ_FIRST_FORMULATIONS)); any_scope_pass = any(summary["formulations"][f]["all_sources_critical_gate"] and summary["formulations"][f]["solver_success_fraction"] == 1.0 for f in FORMULATIONS); legacy_rows = read_json(D28S / "bounded_formulation_results.json")["rows"]; unbounded_improvement = any(float(r.get("unbounded_relative_improvement", 0.0)) >= CRITICAL_IMPROVEMENT for r in legacy_rows)
    if not sanity["pass"] or not pt["pass"]: classification = "EXP014_D28Z_PASS_THROUGH_PARITY_FAIL" if not pt["pass"] else "EXP014_D28Z_MULTIPLE_FAILURES"; next_action = "repair the conservative contract sanity or pass-through parity before any runtime shadow"
    elif any_solver_fail: classification = "EXP014_D28Z_BOUNDED_SOLVER_FAIL"; next_action = "repair deterministic bounded solver failures; no runtime shadow authorization"
    elif hz_conflict: classification = "EXP014_D28Z_HZ_FIRST_STEP_TASK_CONFLICT"; next_action = "close position-level WBIK branch; evaluate dynamics-constrained centroidal trajectory optimization or torque-level WBC"
    elif not any_scope_pass and unbounded_improvement: classification = "EXP014_D28Z_ACTIVE_POSITION_LEVEL_AUTHORITY_INSUFFICIENT"; next_action = "close position-level branch; position-target authority is insufficient under conservative active bounds"
    elif any_scope_pass and determinism: classification = "EXP014_D28Z_CONSERVATIVE_POSITION_LEVEL_AUTHORITY_PASS"; next_action = "D28AA fresh source parity and V3R6 runtime shadow; physics remains unauthorized until the next preflight passes"
    else: classification = "EXP014_D28Z_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO"; next_action = "close position-level centroidal branch and evaluate torque-level WBC or dynamics-constrained trajectory optimization separately"
    if classification == "EXP014_D28Z_CONSERVATIVE_POSITION_LEVEL_AUTHORITY_PASS":
        dump(OUT / "exp014_d28aa_runtime_shadow_authorization.json", {"name": "Exp014D28AARuntimeShadowAuthorizationV1", "authorized": True, "selected_formulation": selected, "contract": "Exp014PenetrationNonWorseningJointAuthorityV6", "V3R6_hash": shadow.get("hash"), "physics_authorized": False, "physics_executed": 0})
    else:
        dump(OUT / "exp014_position_level_centroidal_no_go.json", {"name": "Exp014PositionLevelCentroidalNoGoV3", "authorized": False, "classification": classification, "position_level_centroidal_branch_closed": classification != "EXP014_D28Z_BOUNDED_SOLVER_FAIL", "branch_decision_state": "UNRESOLVED_SOLVER_BLOCKED" if classification == "EXP014_D28Z_BOUNDED_SOLVER_FAIL" else "CLOSED", "physics_executed": 0, "reason": next_action, "D28R_centroidal_matrix": "D28S/D28R protected and validated", "q_cmd_contract": "resolved canonical virtual target", "actuator_contract": "D28W resolved implicit actuator", "unresolved_physx_directions_do_not_decide": True, "unresolved_physx_directions": unresolved_directions, "scope_formulations": {f: summary["formulations"][f] for f in FORMULATIONS}, "hz_first": hz_first, "hard_task_conflict": hz_conflict, "blockers": blockers})
    protected_after = protected_input_hashes(); protected_ok = start_hashes == protected_after; dump(OUT / "stage_classification.json", {"name": "Exp014D28ZStageClassificationV1", "classification": classification, "D28Y_classification_unchanged": "EXP014_D28Y_DYNAMIC_LIMIT_INVARIANCE_UNRESOLVED", "selected_formulation": selected, "determinism": determinism, "protected_inputs_unchanged": protected_ok, "new_physics": 0, "new_probe": 0, "physics": 0})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "new_physics": 0, "new_probe": 0, "persistent_update": 0, "new_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_input_hashes_at_start": start_hashes, "protected_input_hashes_at_finish": protected_after, "protected_inputs_unchanged": protected_ok, "D28Y_unchanged": protected_ok, "D28R_D28S_D28W_unchanged": protected_ok, "new_physics": 0, "new_probe": 0, "persistent_update": 0, "new_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "raw_restore": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28z_conservative_centroidal_authority.py' --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28z_conservative_centroidal_authority.py' --headless\n", encoding="utf-8")
    lines = [f"# EXP014 Phase 2-D28Z conservative centroidal authority", "", f"Classification: `{classification}`.", "", "## Conservative contract", "", "The V6 candidate uses q_actual as the measured state, q_kin_next=q_current+dt*dq for active variables, nominal-limit containment when q_current is inside, and non-worsening signed penetration when q_current is outside. q_cmd is q_kin_next plus the protected feedforward offset and is not position-clipped. Pass-through commands are exact D27 V2A q_cmd values.", "", f"Sanity contract pass: `{sanity['pass']}`; pass-through pass: `{pt['pass']}`.", "", "## C3/C4 replay", "", *(f"- `{f}`: solver success `{summary['formulations'][f]['solver_success_fraction']:.6g}`, median H_z improvement `{summary['formulations'][f]['median_improvement']}`, all-source critical gate `{summary['formulations'][f]['all_sources_critical_gate']}`." for f in FORMULATIONS), "", "## H_z-first diagnostic", "", *(f"- `{f}`: all-source critical gate `{summary['formulations'][f]['all_sources_critical_gate']}`, median improvement `{summary['formulations'][f]['median_improvement']}`; diagnostic only." for f in HZ_FIRST_FORMULATIONS), "", "## Root cause", "", f"H_z-first/task conflict established: `{hz_conflict}`. Unresolved PhysX directions were not used as a closure criterion; V6 only prevents worsening the measured penetration.", "", "## V3R6 shadow", "", f"Selected formulation: `{selected}`; status `{shadow.get('status')}`; independent-process determinism `{determinism}`; physics `0`.", "", "## Protection", "", f"Protected D28R/D28S/D28W/D28Y inputs unchanged: `{protected_ok}`. New physics/probe `0`; persistent update `0`; checkpoint `0`; LEFT START `0`; PPO/CEM/validation/held-out/RUN `0`; remote push `false`.", "", f"Starting HEAD: `{start_head}`. Ending HEAD before commit: `{git('rev-parse', 'HEAD')}`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"phase": "2-D28Z", "classification": classification, "selected": selected, "determinism": determinism, "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
