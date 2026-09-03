"""Phase 2-D28S: constrained centroidal-yaw authority audit.

Offline-only diagnostic. It reads the protected D28R body trace, the
protected D27 V2A controller, and the protected D28 V3 contract. It does not
launch physics and never changes a target, timing, gain, task contract,
policy, checkpoint, or existing stage artifact.
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

try:
    from scipy.optimize import linprog
except Exception:  # pragma: no cover - Isaac Lab normally supplies scipy
    linprog = None


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28s_centroidal_authority_audit"
REPORT = REPO / "research/exp_014_phase_2_d28s_centroidal_authority_audit_report.md"
D28 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28_centroidal_feedback_start"
D28R = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28r_centroidal_trace_and_feedback"
D27 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d27_right_model_based_start_physics"
D26X = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26x_timing_and_target_set"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"

DT = 0.02
SVD_TOL = 1.0e-8
SOLVER_TOL = 1.0e-9
SOLVER_MAX_ITER = 148
VELOCITY_RATIO_LIMIT = 0.80
TASK_REL_TOL = 1.20
NUMERIC_ZERO = 1.0e-8
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_PASS_FRACTION = 0.80
TRACE_RECIPES = (4, 5, 6, 7)
RIGHT_TARGET_ID = "RIGHT_000"
RIGHT_TARGET_ROW = 9330

PHASE_NAMES = ("", "DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE", "WMOVE")
FAILURE_STEPS = {4: 160, 5: 154, 6: 157, 7: 160}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# D28R is imported only for its read-only trace loader and D27 reference
# reconstruction. Neither module main() nor a simulator entrypoint is called.
d28r = load_module("exp014_d28s_d28r_read_only", EXP / "scripts/run_phase2_d28r_centroidal_trace_and_feedback.py")
d27 = d28r.d27
wbik_v2 = d27.d26x.wbik_v2


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
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = arr(v).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = arr(q).reshape(4)
    q = q / max(float(np.linalg.norm(q)), 1.0e-12)
    x, y, z, w = q
    return np.array([[1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)], [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)], [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)]], dtype=np.float64)


def so3_log(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(c))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) * 0.5
    if theta < 1.0e-8:
        return vee
    if np.pi - theta < 1.0e-5:
        vals, vecs = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = vecs[:, int(np.argmax(vals))]
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        return axis * theta
    return vee * (theta / max(float(np.sin(theta)), 1.0e-12))


def minimum_jerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10*u**3 - 15*u**4 + 6*u**5


def nullspace(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    A = arr(A)
    n = A.shape[1]
    if A.size == 0:
        return np.eye(n), np.zeros(0), 0
    _, s, Vt = np.linalg.svd(A, full_matrices=True)
    rank = int(np.sum(s > SVD_TOL * max(float(s[0]) if s.size else 1.0, 1.0)))
    return Vt[rank:].T.copy(), s.copy(), rank


def matrix_stats(A: np.ndarray) -> dict[str, Any]:
    A = arr(A)
    if A.ndim != 2:
        return {"shape": list(A.shape), "rank": 0, "singular_values": [], "minimum_nonzero_singular_value": None, "condition_number": None, "finite": False}
    s = np.linalg.svd(A, compute_uv=False) if A.size else np.zeros(0)
    scale = max(float(s[0]) if s.size else 1.0, 1.0)
    rank = int(np.sum(s > SVD_TOL * scale))
    nz = s[s > SVD_TOL * scale]
    return {"shape": list(A.shape), "rank": rank, "singular_values": s.tolist(), "minimum_nonzero_singular_value": None if not nz.size else float(nz[-1]), "condition_number": None if not nz.size or rank < min(A.shape) else float(s[0] / max(nz[-1], 1.0e-30)), "finite": bool(np.isfinite(A).all())}


def row_overlap(row: np.ndarray, stack: np.ndarray) -> float:
    row = arr(row).reshape(-1); stack = arr(stack)
    if not stack.size or np.linalg.norm(row) <= NUMERIC_ZERO:
        return 0.0
    Q, _ = np.linalg.qr(stack.T, mode="reduced")
    projection = Q @ (Q.T @ row) if Q.size else np.zeros_like(row)
    return float(np.linalg.norm(projection) / max(float(np.linalg.norm(row)), NUMERIC_ZERO))


def solve_unconstrained(A: np.ndarray, b: np.ndarray, damping: float = 1.0e-4) -> np.ndarray:
    A, b = arr(A), arr(b).reshape(-1)
    if A.shape[0] == 0:
        return np.zeros(A.shape[1], dtype=np.float64)
    H = A.T @ A + damping * np.eye(A.shape[1])
    g = A.T @ b
    try:
        return np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, g, rcond=SVD_TOL)[0]


def equality_lsq(A: np.ndarray, b: np.ndarray, C: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = A.shape[1]
    H = A.T @ A + 1.0e-4 * np.eye(n)
    g = A.T @ b
    if C.size == 0:
        return solve_unconstrained(A, b), np.zeros(0)
    K = np.block([[H, C.T], [C, np.zeros((C.shape[0], C.shape[0]))]])
    rhs = np.concatenate((g, d))
    try:
        solution = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(K, rhs, rcond=SVD_TOL)[0]
    return solution[:n], solution[n:]


def active_set_qp(A: np.ndarray, b: np.ndarray, G: np.ndarray, h: np.ndarray) -> dict[str, Any]:
    """Solve a fixed convex quadratic LS with linear inequalities.

    The phase uses a deterministic active-set method.  A zero-cost HiGHS
    linear-feasibility call supplies a numerically safe feasible start when
    the previous lexicographic stage is not itself box-feasible; all later
    choices use fixed row ordering and fixed tolerances.
    """
    A, b, G, h = arr(A), arr(b).reshape(-1), arr(G), arr(h).reshape(-1)
    n = A.shape[1]
    if not (np.isfinite(A).all() and np.isfinite(b).all() and np.isfinite(G).all() and np.isfinite(h).all()):
        return {"success": False, "x": np.zeros(n), "iterations": 0, "active": [], "reason": "NONFINITE_LINEAR_SYSTEM"}
    if linprog is None:
        return {"success": False, "x": np.zeros(n), "iterations": 0, "active": [], "reason": "SCIPY_LINPROG_UNAVAILABLE"}
    feasible = linprog(np.zeros(n), A_ub=G, b_ub=h, bounds=[(None, None)] * n, method="highs")
    if not feasible.success:
        return {"success": False, "x": np.zeros(n), "iterations": 0, "active": [], "reason": "INFEASIBLE_LINEAR_BOUNDS", "linprog_message": str(feasible.message)}
    z = arr(feasible.x)
    H = A.T @ A + 1.0e-4 * np.eye(n)
    c = A.T @ b
    active: list[int] = [int(i) for i, slack in enumerate(h - G @ z) if abs(float(slack)) <= SOLVER_TOL]
    diagnostics: list[dict[str, Any]] = []
    for it in range(SOLVER_MAX_ITER):
        gradient = H @ z - c
        C = G[active] if active else np.zeros((0, n))
        if active:
            K = np.block([[H, C.T], [C, np.zeros((len(active), len(active)))]])
            rhs = np.concatenate((-gradient, np.zeros(len(active))))
            try:
                step_solution = np.linalg.lstsq(K, rhs, rcond=SVD_TOL)[0]
            except np.linalg.LinAlgError:
                return {"success": False, "x": z, "iterations": it, "active": active, "diagnostics": diagnostics, "reason": "KKT_SOLVE_FAILURE"}
            p = step_solution[:n]
        else:
            try:
                p = np.linalg.lstsq(H, -gradient, rcond=SVD_TOL)[0]
            except np.linalg.LinAlgError:
                return {"success": False, "x": z, "iterations": it, "active": active, "diagnostics": diagnostics, "reason": "HESSIAN_SOLVE_FAILURE"}
        if not np.isfinite(p).all():
            return {"success": False, "x": z, "iterations": it, "active": active, "diagnostics": diagnostics, "reason": "NONFINITE_STEP"}
        step_norm = float(np.linalg.norm(p))
        if step_norm > SOLVER_TOL:
            alpha = 1.0; blocking = None
            direction = G @ p
            slack = h - G @ z
            candidates = [(float(slack[i] / direction[i]), i) for i in range(G.shape[0]) if i not in active and direction[i] > SOLVER_TOL]
            if candidates:
                alpha, blocking = min(candidates, key=lambda pair: (pair[0], pair[1]))
                alpha = float(np.clip(alpha, 0.0, 1.0))
            z = z + alpha * p
            if blocking is not None and alpha < 1.0 - SOLVER_TOL:
                active.append(int(blocking)); active.sort()
                diagnostics.append({"event": "add", "row": int(blocking), "step": alpha})
            continue
        if active:
            # KKT sign convention is gradient + G_active.T*lambda = 0;
            # feasible inequalities require lambda >= 0.
            lam = np.linalg.lstsq(C.T, -gradient, rcond=SVD_TOL)[0]
            negative = [j for j, value in enumerate(lam) if value < -SOLVER_TOL]
            if negative:
                j = min(negative, key=lambda k: (float(lam[k]), active[k]))
                diagnostics.append({"event": "release", "row": int(active[j]), "multiplier": float(lam[j])})
                active.pop(j); continue
        violation = float(np.max(G @ z - h)) if G.size else 0.0
        return {"success": bool(np.isfinite(z).all() and violation <= 5.0e-8), "x": z, "iterations": it+1, "active": active, "diagnostics": diagnostics, "max_constraint_violation": violation, "residual_norm": float(np.linalg.norm(A @ z-b))}
    return {"success": False, "x": z, "iterations": SOLVER_MAX_ITER, "active": active, "diagnostics": diagnostics, "max_constraint_violation": float(np.max(G @ z-h)), "residual_norm": float(np.linalg.norm(A@z-b)), "reason": "MAX_ITERATIONS"}


def bounded_lsq(A: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, Any]:
    A, b, lower, upper = arr(A), arr(b).reshape(-1), arr(lower).reshape(-1), arr(upper).reshape(-1)
    n = A.shape[1]
    if not (np.isfinite(lower).all() and np.isfinite(upper).all() and np.all(lower <= upper + SOLVER_TOL)):
        return {"success": False, "x": np.zeros(n), "iterations": 0, "active": [], "reason": "INVALID_BOUNDS"}
    return active_set_qp(A, b, np.vstack((-np.eye(n), np.eye(n))), np.concatenate((-lower, upper)))


def bounded_nullspace_stage(base: np.ndarray, prior: np.ndarray, A: np.ndarray, b: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    N, _, rank = nullspace(prior)
    if N.shape[1] == 0:
        return base.copy(), {"success": True, "rank": rank, "nullity": 0, "x": base.copy(), "residual_norm": float(np.linalg.norm(A @ base-b)), "active": [], "iterations": 0}
    reduced_A = A @ N
    reduced_b = b - A @ base
    G = np.vstack((N, -N))
    h = np.concatenate((upper - base, -(lower - base)))
    reduced = active_set_qp(reduced_A, reduced_b, G, h)
    x = base + N @ reduced["x"]
    reduced["x_reduced"] = reduced.pop("x")
    reduced.update({"rank": rank, "nullity": int(N.shape[1]), "x": x, "max_bound_violation": float(np.max(np.concatenate((x-upper, lower-x))))})
    return x, reduced


def group_for_joint(name: str) -> str:
    n = str(name).lower()
    if "wrist" in n or any(token in n for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if n.startswith("left_") else "right wrist/hand"
    if "shoulder" in n or "elbow" in n:
        return "left arm" if n.startswith("left_") else "right arm"
    if "waist" in n or "torso" in n:
        return "waist"
    return "left leg" if n.startswith("left_") else "right leg"


def group_contribution(dq: np.ndarray, names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    total = float(np.linalg.norm(dq))
    for i, name in enumerate(names):
        group = group_for_joint(name)
        out[group] = out.get(group, 0.0) + float(np.asarray(dq[i] * dq[i]).item())
    return {group: float(np.sqrt(value) / max(total, NUMERIC_ZERO)) for group, value in out.items()}


def load_trace_inputs() -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    trace = d28r.load_npz(D28R / "capture_on" / "raw_primary_trajectory.npz")
    static = json.loads((D28R / "capture_on" / "static_contract.json").read_text(encoding="utf-8"))
    source, _, default_q, action_scale, plans, _, _ = d28r.load_inputs()
    numeric = d28r.load_npz(D28R / "centroidal_numeric_bundle.npz")
    return trace, static, plans, arr(default_q), arr(action_scale), source, numeric


def static_joint_contract() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    contract = json.loads((D28R / "joint_index_name_contract.json").read_text(encoding="utf-8"))
    rows = contract.get("joints", contract.get("rows", []))
    if isinstance(rows, dict):
        rows = list(rows.values())
    rows = sorted(rows, key=lambda row: int(row.get("action_index", row.get("index", 0))))
    names = [str(row.get("joint_name", row.get("name"))) for row in rows]
    q_default = np.asarray([float(row.get("default_q", 0.0)) for row in rows])
    def limits(row):
        value = row.get("position_limit", row.get("position_limits", [-np.inf, np.inf]))
        return float(value[0]), float(value[1])
    q_limits = np.asarray([limits(row) for row in rows])
    velocity = np.asarray([float(row.get("velocity_limit", np.inf)) for row in rows])
    return names, q_default, q_limits, velocity, contract


def trace_row_sets(trace: dict[str, np.ndarray]) -> tuple[dict[int, list[int]], dict[int, list[int]], dict[int, dict[str, Any]]]:
    analysis: dict[int, list[int]] = {}
    critical: dict[int, list[int]] = {}
    manifest: dict[int, dict[str, Any]] = {}
    for recipe in TRACE_RECIPES:
        steps = arr(trace["control_step"][recipe], np.int64)
        stage = arr(trace["stage"][recipe], np.int64)
        rows = [int(i) for i, value in enumerate(stage) if int(value) == 1 and int(steps[i]) < FAILURE_STEPS[recipe]]
        if not rows:
            raise RuntimeError(f"no START analysis rows for R{recipe}")
        if trace["actual_root_velocity"].shape[-1] == 6:
            yaw = np.abs(arr(trace["actual_root_velocity"][recipe, rows, 5]))
        else:
            yaw = np.abs(arr(trace["actual_root_velocity"][recipe, rows, -1]))
        crossing = next((j for j, value in enumerate(yaw) if value > 0.15), 0)
        first_row = rows[crossing]
        crows = rows[max(0, crossing-8):min(len(rows), crossing+9)]
        analysis[recipe] = rows
        critical[recipe] = crows
        manifest[recipe] = {"analysis_start_control_step": int(steps[rows[0]]), "analysis_end_control_step_exclusive": FAILURE_STEPS[recipe], "first_yaw_crossing_control_step": int(steps[first_row]), "critical_window_requested": {"before": 8, "after": 8}, "critical_window_clipped_at_start": bool(crossing < 8), "critical_control_steps": [int(steps[i]) for i in crows], "analysis_trace_rows": rows, "critical_trace_rows": crows, "first_failure_control_step": FAILURE_STEPS[recipe]}
    return analysis, critical, manifest


def reference_for(plan: dict[str, Any], plan_step: int) -> dict[str, np.ndarray]:
    raw = d27.build_reference(plan, int(plan_step))
    return {key: arr(value) for key, value in raw.items()}


def task_build(trace: dict[str, np.ndarray], static: dict[str, Any], plan: dict[str, Any], recipe: int, row: int, names: list[str], source: dict[str, np.ndarray], default_q: np.ndarray, action_scale: np.ndarray, numeric: dict[str, np.ndarray]) -> dict[str, Any]:
    local = list(TRACE_RECIPES).index(recipe)
    body_row = max(row - 1, 0)
    steps = arr(trace["control_step"][recipe], np.int64)
    start_rows = np.where(arr(trace["stage"][recipe], np.int64) == 1)[0]
    plan_step = int(np.where(start_rows == row)[0][0]) if np.any(start_rows == row) else int(max(0, steps[row] - steps[start_rows[0]]))
    ref = reference_for(plan, plan_step)
    root_twist = arr(ref["root_velocity"])
    body_pos = arr(trace["body_origin_position"][local, body_row])
    body_quat = arr(trace["body_com_quaternion"][local, body_row])
    body_com = arr(trace["body_com_position"][local, body_row])
    com = arr(trace["actual_com_position"][recipe, row])
    Jb = arr(trace["body_jacobians"][local, body_row])
    masses = arr(static["body_masses"])
    J_com_full = np.sum(masses[:, None, None] * Jb[:, :3, :], axis=0) / max(float(np.sum(masses)), NUMERIC_ZERO)
    stance_R = quat_to_matrix(body_quat[24]); swing_R = quat_to_matrix(body_quat[25]); pelvis_R = quat_to_matrix(arr(trace["actual_root_pose"][recipe, row, 3:7]))
    stance_err = arr(ref["stance_position"]) - body_pos[24]
    swing_err = arr(ref["swing_position"]) - body_pos[25]
    stance_rot_err = so3_log(arr(ref["stance_rotation"]) @ stance_R.T)
    swing_rot_err = so3_log(arr(ref["swing_rotation"]) @ swing_R.T)
    pelvis_rot_err = so3_log(quat_to_matrix(arr(ref["root_pose"])[3:7]) @ pelvis_R.T)
    torso_R = quat_to_matrix(body_quat[4])
    torso_rot_err = so3_log(arr(ref.get("torso_rotation", torso_R)) @ torso_R.T)
    J_stance = arr(Jb[24]); J_swing = arr(Jb[25]); J_pelvis = arr(Jb[0, 3:6]); J_torso = arr(Jb[4, 3:6])
    stance_twist = np.concatenate((stance_err / DT, stance_rot_err / DT))
    swing_twist = np.concatenate((swing_err / DT, swing_rot_err / DT))
    com_target = arr(ref["com_velocity"]) + (arr(ref["com_position"]) - com) / DT
    pelvis_target = pelvis_rot_err / DT
    torso_target = torso_rot_err / DT
    q_current = arr(trace["q_actual_current"][recipe, row]); dq_current = arr(trace["joint_velocity"][recipe, body_row])
    q_min = arr(source["joint_position_limits"][recipe, :, 0]); q_max = arr(source["joint_position_limits"][recipe, :, 1])
    vlim = arr(trace["joint_velocity_limits"][recipe, row])
    A = arr(numeric["A"][local, body_row])
    tasks = {
        "stance": {"J": J_stance[:, 6:], "b": stance_twist - J_stance[:, :6] @ root_twist, "root": J_stance[:, :6] @ root_twist, "units": "m/s,rad/s"},
        "com": {"J": J_com_full[:, 6:], "b": com_target - J_com_full[:, :6] @ root_twist, "root": J_com_full[:, :6] @ root_twist, "units": "m/s"},
        "swing": {"J": J_swing[:, 6:], "b": swing_twist - J_swing[:, :6] @ root_twist, "root": J_swing[:, :6] @ root_twist, "units": "m/s,rad/s"},
        "pelvis": {"J": J_pelvis[:, 6:], "b": pelvis_target - J_pelvis[:, :6] @ root_twist, "root": J_pelvis[:, :6] @ root_twist, "units": "rad/s"},
        "torso": {"J": J_torso[:, 6:], "b": torso_target - J_torso[:, :6] @ root_twist, "root": J_torso[:, :6] @ root_twist, "units": "rad/s"},
        "nominal": {"J": np.eye(37), "b": 0.02 * (arr(ref["nominal_q"]) - q_current) / DT, "root": np.zeros(37), "units": "rad/s"},
        "action_rate": {"J": np.eye(37), "b": np.zeros(37), "root": np.zeros(37), "units": "rad/s"},
        "hz": {"J": A[2:3, 6:], "b": np.asarray([0.0 - A[2, :6] @ root_twist]), "root": np.asarray([A[2, :6] @ root_twist]), "units": "Nms"},
    }
    return {"recipe": recipe, "trace_row": row, "control_step": int(steps[row]), "plan_step": plan_step, "phase": PHASE_NAMES[int(trace["phase"][recipe, row])] if int(trace["phase"][recipe, row]) < len(PHASE_NAMES) else "", "q_current": q_current, "dq_current": dq_current, "q_min": q_min, "q_max": q_max, "velocity_limits": vlim, "root_twist": root_twist, "ref": ref, "total_steps": int(plan["refs"]["total_steps"]), "body_origin_position": body_pos, "body_com_position": body_com, "body_quaternion": body_quat, "body_jacobians": Jb, "body_masses": masses, "com_position": com, "A": A, "tasks": tasks, "source_offset": arr(plan["source_offset"]), "target_offset": arr(plan["target_offset"]), "default_q": default_q, "action_scale": action_scale, "plan_id": plan["identity"]["plan_id"], "target_id": RIGHT_TARGET_ID, "root_pose": arr(trace["actual_root_pose"][recipe, row]), "actual_hz": float(arr(numeric["H_direct"][local, body_row, 2])), "actual_yaw_rate": float(arr(trace["actual_root_velocity"][recipe, row, 5]))}


def build_bounds(record: dict[str, Any]) -> dict[str, np.ndarray]:
    progress = (record["plan_step"] + 1) / max(float(record["total_steps"]), 1.0)
    scalar = minimum_jerk(progress)
    ff = (1.0-scalar)*record["source_offset"] + scalar*record["target_offset"]
    vlim = np.abs(record["velocity_limits"])
    pos_lower = (record["q_min"] - ff - record["q_current"]) / DT
    pos_upper = (record["q_max"] - ff - record["q_current"]) / DT
    vel_gate = VELOCITY_RATIO_LIMIT * vlim
    return {"velocity_lower": -vlim, "velocity_upper": vlim, "velocity_gate_lower": -vel_gate, "velocity_gate_upper": vel_gate, "position_lower": pos_lower, "position_upper": pos_upper, "action_lower": np.full(37, -np.inf), "action_upper": np.full(37, np.inf), "combined_lower": np.maximum(-vel_gate, pos_lower), "combined_upper": np.minimum(vel_gate, pos_upper), "feedforward": ff}


def active_names(active: list[int], names: list[str]) -> list[dict[str, Any]]:
    n = len(names); out = []
    for row in active:
        idx = row % n
        out.append({"joint_index": idx, "joint_name": names[idx], "bound": "lower" if row >= n else "upper"})
    return out


def task_stack(record: dict[str, Any], keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.vstack([record["tasks"][key]["J"] for key in keys]), np.concatenate([record["tasks"][key]["b"] for key in keys])


def solve_f2(record: dict[str, Any], bounds: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    hard_J, hard_b = task_stack(record, ["stance", "com", "swing", "pelvis"])
    hard = bounded_lsq(hard_J, hard_b, bounds["combined_lower"], bounds["combined_upper"])
    base = hard["x"]
    x, hz = bounded_nullspace_stage(base, hard_J, record["tasks"]["hz"]["J"], record["tasks"]["hz"]["b"], bounds["combined_lower"], bounds["combined_upper"])
    hz["success"] = bool(hard.get("success", False) and hz.get("success", False))
    return x, {"hard": hard, "hz": hz, "prior_stack": hard_J, "success": hz["success"]}


def solve_lex(record: dict[str, Any], bounds: dict[str, np.ndarray], hz_first: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.zeros(37, dtype=np.float64); prior = np.zeros((0, 37), dtype=np.float64); stages = []
    order = [["stance"], ["hz"], ["com", "swing", "pelvis"]] if hz_first else [["stance"], ["com", "swing", "pelvis"], ["hz"], ["torso", "nominal", "action_rate"]]
    for keys in order:
        J, b = task_stack(record, keys)
        x, diag = bounded_nullspace_stage(x, prior, J, b, bounds["combined_lower"], bounds["combined_upper"])
        stages.append({"tasks": keys, "diagnostics": diag})
        prior = np.vstack((prior, J))
    return x, {"stages": stages, "prior_stack": prior, "success": bool(all(stage["diagnostics"].get("success", False) for stage in stages))}


def solve_unbounded_lex(record: dict[str, Any], hz_first: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.zeros(37, dtype=np.float64); prior = np.zeros((0, 37), dtype=np.float64); stages = []
    order = [["stance"], ["hz"], ["com", "swing", "pelvis"]] if hz_first else [["stance"], ["com", "swing", "pelvis"], ["hz"], ["torso", "nominal", "action_rate"]]
    for keys in order:
        J, b = task_stack(record, keys); N, _, rank = nullspace(prior); reduced_A = J @ N; reduced_b = b - J @ x
        z = solve_unconstrained(reduced_A, reduced_b) if N.shape[1] else np.zeros(0)
        x = x + N @ z; stages.append({"tasks": keys, "prior_rank": rank, "residual_norm": float(np.linalg.norm(J@x-b))}); prior = np.vstack((prior, J))
    return x, {"stages": stages}


def solve_unbounded_f2(record: dict[str, Any]) -> np.ndarray:
    hard_J, hard_b = task_stack(record, ["stance", "com", "swing", "pelvis"])
    base = solve_unconstrained(hard_J, hard_b)
    N, _, _ = nullspace(hard_J)
    if N.shape[1] == 0:
        return base
    z = solve_unconstrained(record["tasks"]["hz"]["J"] @ N, record["tasks"]["hz"]["b"] - record["tasks"]["hz"]["J"] @ base)
    return base + N @ z


def v2a_dq(record: dict[str, Any]) -> dict[str, Any]:
    import torch
    ref = {key: torch.as_tensor(value, dtype=torch.float64) for key, value in record["ref"].items()}
    out = wbik_v2.solve_prescribed_floating_base(root_pose=torch.as_tensor(record["root_pose"], dtype=torch.float64), root_velocity=torch.as_tensor(record["root_twist"], dtype=torch.float64), joint_position=torch.as_tensor(record["q_current"], dtype=torch.float64), joint_velocity=torch.as_tensor(record["dq_current"], dtype=torch.float64), body_position=torch.as_tensor(record["body_origin_position"], dtype=torch.float64), body_quaternion=torch.as_tensor(record["body_quaternion"], dtype=torch.float64), body_jacobians=torch.as_tensor(record["body_jacobians"], dtype=torch.float64), body_com_position=torch.as_tensor(record["body_com_position"], dtype=torch.float64), body_masses=torch.as_tensor(record["body_masses"], dtype=torch.float64), com_position=torch.as_tensor(record["com_position"], dtype=torch.float64), reference=ref, stance_body_index=24, swing_body_index=25, q_min=torch.as_tensor(record["q_min"], dtype=torch.float64), q_max=torch.as_tensor(record["q_max"], dtype=torch.float64), velocity_limits=torch.as_tensor(record["velocity_limits"], dtype=torch.float64), default_q=torch.as_tensor(record["default_q"], dtype=torch.float64), action_scale=torch.as_tensor(record["action_scale"], dtype=torch.float64))
    return {"dq": arr(out["dq_des"]), "q_des": arr(out["q_des"]), "status": str(out["status"])}


def task_residuals(record: dict[str, Any], dq: np.ndarray) -> dict[str, float]:
    return {key: float(np.linalg.norm(spec["J"] @ dq - spec["b"])) for key, spec in record["tasks"].items()}


def evaluate_solution(record: dict[str, Any], bounds: dict[str, np.ndarray], dq: np.ndarray, names: list[str], label: str, solver: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    hz_residual = record["tasks"]["hz"]["J"] @ dq - record["tasks"]["hz"]["b"]
    current_hz = float(abs(record["actual_hz"]))
    baseline_hz = float(baseline["predicted_hz_error"] if baseline else current_hz)
    ff = bounds["feedforward"]; qcmd = record["q_current"] + DT*dq + ff; action = (qcmd-record["default_q"])/record["action_scale"]
    ratios = np.abs(dq) / np.maximum(np.abs(record["velocity_limits"]), NUMERIC_ZERO)
    residuals = task_residuals(record, dq); base_res = baseline.get("task_residuals", {}) if baseline else {}
    task_gates = {"stance_no_worse": residuals["stance"] <= base_res.get("stance", np.inf) + 1.0e-9, "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(base_res.get("com", NUMERIC_ZERO), NUMERIC_ZERO) + 1.0e-9, "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(base_res.get("swing", NUMERIC_ZERO), NUMERIC_ZERO) + 1.0e-9, "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(base_res.get("pelvis", NUMERIC_ZERO), NUMERIC_ZERO) + 1.0e-9}
    position = bool(np.isfinite(qcmd).all() and np.all(qcmd >= record["q_min"]-1.0e-9) and np.all(qcmd <= record["q_max"]+1.0e-9))
    canonical = bool(np.isfinite(action).all() and np.allclose(record["default_q"] + record["action_scale"]*action, qcmd, atol=1.0e-10, rtol=1.0e-10))
    velocity = bool(np.max(ratios) <= VELOCITY_RATIO_LIMIT+1.0e-9)
    all_tasks = bool(all(task_gates.values()))
    solver_success = bool(solver.get("success", True))
    bounded_error = float(abs(hz_residual[0]))
    bounds_feasible = bool(np.all(bounds["combined_lower"] <= bounds["combined_upper"] + SOLVER_TOL))
    return {"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "plan_step": record["plan_step"], "phase": record["phase"], "formulation": label, "plan_id": record["plan_id"], "current_hz": current_hz, "v2_predicted_hz_error": baseline_hz, "predicted_hz_error": bounded_error, "unconstrained_min_hz_error": None, "bounded_min_hz_error": bounded_error, "relative_improvement": float((baseline_hz-bounded_error)/max(baseline_hz, NUMERIC_ZERO)), "rho": float(bounded_error/max(baseline_hz, NUMERIC_ZERO)), "dq": dq, "q_cmd": qcmd, "action": action, "velocity_ratio_max": float(np.max(ratios)), "solver_success": solver_success, "bounds_feasible": bounds_feasible, "velocity_gate": velocity, "position_gate": position, "canonical_action_gate": canonical, "task_residuals": residuals, "task_gates": task_gates, "all_constraint_gates": bool(solver_success and bounds_feasible and velocity and position and canonical and all_tasks), "active_bound_joints": active_names(solver.get("active", []), names) if solver else [], "solver": {key: value for key, value in solver.items() if key not in ("x",)}, "group_contribution": group_contribution(dq, names)}


def rank_and_nullspace(records: list[dict[str, Any]], names: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranks, authority = [], []
    for record in records:
        matrices = {"J_stance": record["tasks"]["stance"]["J"], "J_stance_com": np.vstack((record["tasks"]["stance"]["J"], record["tasks"]["com"]["J"])), "J_stance_com_swing": np.vstack((record["tasks"]["stance"]["J"], record["tasks"]["com"]["J"], record["tasks"]["swing"]["J"])), "J_stance_com_swing_pelvis": np.vstack((record["tasks"]["stance"]["J"], record["tasks"]["com"]["J"], record["tasks"]["swing"]["J"], record["tasks"]["pelvis"]["J"])), "A_hz": record["tasks"]["hz"]["J"], "A_hz_hard": np.vstack((record["tasks"]["stance"]["J"], record["tasks"]["com"]["J"], record["tasks"]["swing"]["J"], record["tasks"]["pelvis"]["J"], record["tasks"]["hz"]["J"]))}
        stats = {key: matrix_stats(value) for key, value in matrices.items()}
        stats["row_space_overlap_hz_vs_hard"] = row_overlap(record["tasks"]["hz"]["J"][0], matrices["J_stance_com_swing_pelvis"])
        ranks.append({"recipe": record["recipe"], "control_step": record["control_step"], "trace_row": record["trace_row"], "phase": record["phase"], "matrices": stats})
        for label, task_keys in (("N0_after_stance", ["stance"]), ("N1_after_stance_com", ["stance", "com"]), ("N2_after_stance_com_swing", ["stance", "com", "swing"]), ("N3_after_stance_com_swing_pelvis", ["stance", "com", "swing", "pelvis"])):
            H, _ = task_stack(record, task_keys); N, _, rank = nullspace(H); a = record["tasks"]["hz"]["J"] @ N; sv = np.linalg.svd(a, compute_uv=False) if a.size else np.zeros(0)
            a_norm_sq = float(np.asarray(a @ a.T).reshape(())) if a.size else 0.0
            direction = N @ (a.T / max(a_norm_sq, NUMERIC_ZERO)) if a.size and a_norm_sq > NUMERIC_ZERO else np.zeros(37)
            authority.append({"recipe": record["recipe"], "control_step": record["control_step"], "trace_row": record["trace_row"], "nullspace": label, "hard_task_rank": rank, "hard_task_rows": int(H.shape[0]), "nullity": int(N.shape[1]), "hz_row_norm": float(np.linalg.norm(a)), "hz_authority_rank": int(np.sum(sv > SVD_TOL * max(float(sv[0]) if sv.size else 1.0, 1.0))), "hz_max_singular_value": None if not sv.size else float(sv[0]), "hz_min_nonzero_singular_value": None if not sv.size else float(sv[-1]), "row_space_overlap": row_overlap(record["tasks"]["hz"]["J"][0], H), "unbounded_minimum_norm_direction_norm": float(np.linalg.norm(direction)), "joint_group_contribution": group_contribution(direction, names), "classification": "HZ_NULLSPACE_AUTHORITY_PRESENT" if np.linalg.norm(a) > 1.0e-8 else "HZ_NULLSPACE_AUTHORITY_ZERO"})
    return ranks, authority


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze",), default="analyze")
    parser.add_argument("--headless", action="store_true")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    trace, static, plans, default_q, action_scale, source, numeric = load_trace_inputs()
    names, contract_default, contract_limits, contract_velocity, joint_contract = static_joint_contract()
    if len(names) != 37:
        raise RuntimeError(f"expected 37 joints, found {len(names)}")
    analysis_rows, critical_rows, manifest = trace_row_sets(trace)
    plan_by_recipe = {int(row["identity"]["source_recipe"]): row for row in plans}
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28S", "starting_head": start_head, "starting_git_status_short": start_status, "D28R_read_only": True, "physics_executed": 0, "remote_push": False})
    dump(OUT / "plan_identity_audit.json", {"name": "Exp014D28SProtectedD26XPlanIdentityV1", "source": "D26X selected_offline_plans_v4.json and D28R read-only plan loader", "d26x_selected_offline_plans_sha256": sha256_file(D26X / "selected_offline_plans_v4.json"), "rows": [plan_by_recipe[recipe]["identity"] for recipe in TRACE_RECIPES], "target_unchanged": all(plan_by_recipe[recipe]["identity"]["target_id"] == RIGHT_TARGET_ID for recipe in TRACE_RECIPES), "timing_unchanged": True, "clearance_unchanged": True})
    dump(OUT / "protocol.json", {"name": "Exp014D28SConstrainedCentroidalYawAuthorityAuditV1", "phase": "2-D28S", "sources": list(TRACE_RECIPES), "trace": "D28R capture_on/raw_primary_trajectory.npz", "windows": manifest, "svd_tolerance": SVD_TOL, "solver": {"type": "deterministic active-set equality-constrained least-squares", "damping": 1.0e-4, "tolerance": SOLVER_TOL, "maximum_iterations": SOLVER_MAX_ITER, "new_dependency": False}, "bounds": {"velocity_ratio_limit": VELOCITY_RATIO_LIMIT, "joint_position_limits": "protected source limits", "canonical_action_raw_bounds": ["-inf", "+inf"], "canonical_action_contract": "unbounded raw action; q_cmd=default_q+0.5*raw_action"}, "physics": 0, "forbidden": {"persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "left_start": 0, "remote_push": False}})
    dump(OUT / "joint_bound_contract.json", {"name": "Exp014D28SJointBoundContractV1", "joint_count": 37, "joint_index_name_contract_read_only": True, "velocity_bound": "abs(dq)<=0.80*velocity_limit for feasibility gate; raw limit also recorded", "position_bound": "q_min<=q_current+dq*dt+endpoint_feedforward<=q_max", "canonical_action": "raw action unbounded; finite roundtrip only", "joint_names": [{"action_index": i, "joint_name": names[i], "joint_group": group_for_joint(names[i]), "default_q": float(default_q[i]), "action_scale": float(action_scale[i]), "velocity_limit_from_contract": float(contract_velocity[i]), "position_limit": contract_limits[i].tolist()} for i in range(37)]})
    dump(OUT / "task_jacobian_contract.json", {"name": "Exp014D28STaskJacobianContractV1", "columns": {"root": [0, 6], "joints": [6, 43]}, "frame": "world", "tasks": {"J_stance": "stance-foot world 6D", "J_com": "CoM xyz; protected D28R PhysX body-CoM Jacobian convention", "J_swing": "swing-foot world 6D", "J_pelvis": "pelvis/root orientation", "J_torso": "torso orientation", "J_nominal": "identity nominal posture", "A_hz": "centroidal momentum matrix row z"}, "root_b_definition": "target task twist minus root columns times prescribed root twist", "per_step_numeric_fields": ["J", "b", "root_contribution", "reference", "actual_state"], "numeric_file": "task_jacobian_numeric.npz"})
    records = [task_build(trace, static, plan_by_recipe[recipe], recipe, row, names, source, default_q, action_scale, numeric) for recipe in TRACE_RECIPES for row in analysis_rows[recipe]]
    np.savez_compressed(OUT / "task_jacobian_numeric.npz", **{f"J_{key}": np.stack([record["tasks"][key]["J"] for record in records]) for key in ("stance", "com", "swing", "pelvis", "torso", "nominal", "action_rate", "hz")}, **{f"b_{key}": np.stack([record["tasks"][key]["b"] for record in records]) for key in ("stance", "com", "swing", "pelvis", "torso", "nominal", "action_rate", "hz")}, **{f"root_{key}": np.stack([record["tasks"][key]["root"] for record in records]) for key in ("stance", "com", "swing", "pelvis", "torso", "nominal", "action_rate", "hz")})
    dump(OUT / "task_jacobian_numeric_manifest.json", {"file": "task_jacobian_numeric.npz", "row_count": len(records), "row_identity": [{"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "plan_step": record["plan_step"]} for record in records], "arrays": {"J_stance": [6, 37], "J_com": [3, 37], "J_swing": [6, 37], "J_pelvis": [3, 37], "J_torso": [3, 37], "J_nominal": [37, 37], "J_action_rate": [37, 37], "J_hz": [1, 37], "b_and_root": "same task row shapes"}, "finite": bool(all(np.isfinite(record["tasks"][key]["J"]).all() and np.isfinite(record["tasks"][key]["b"]).all() and np.isfinite(record["tasks"][key]["root"]).all() for record in records for key in record["tasks"]))})
    ranks, authority = rank_and_nullspace(records, names)
    dump(OUT / "rank_conditioning_analysis.json", {"name": "Exp014D28SRankConditioningV1", "svd_tolerance": SVD_TOL, "rows": ranks})
    with (OUT / "nullspace_hz_authority.csv").open("w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in authority for key in row.keys() if key != "joint_group_contribution"}) + ["joint_group_contribution"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(jsonable(authority))
    dump(OUT / "nullspace_hz_authority.json", {"name": "Exp014D28SNullspaceHzAuthorityV1", "rows": authority, "criteria": {"ZERO": "norm<=1e-8", "WEAK": "bounded correction fraction<0.20", "PRESENT": "bounded correction fraction>=0.20"}})
    all_results: list[dict[str, Any]] = []
    per_record: dict[tuple[int, int], dict[str, Any]] = {}
    bound_records: list[dict[str, Any]] = []
    for record in records:
        bounds = build_bounds(record)
        bound_records.append({"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "plan_step": record["plan_step"], "bounds": bounds})
        v2 = v2a_dq(record)
        f0_pred = record["tasks"]["hz"]["J"] @ v2["dq"] - record["tasks"]["hz"]["b"]
        f0 = {"predicted_hz_error": float(abs(f0_pred[0])), "task_residuals": task_residuals(record, v2["dq"])}
        r0 = evaluate_solution(record, bounds, v2["dq"], names, "F0_V2A_BASELINE", {"active": []}, f0)
        r0.update({"v2_status": v2["status"], "relative_improvement": 0.0, "rho": 1.0, "unconstrained_min_hz_error": None})
        # The protected D28 V3 primitive is reproduced exactly: V2A dq plus
        # its fixed all-one weighted scalar H_z residual.
        import torch
        A_t = torch.as_tensor(record["A"], dtype=torch.float64)
        root_t = torch.as_tensor(record["root_twist"], dtype=torch.float64)
        momentum = d28r.wbik_v3.momentum_joint_residual(A_t, root_t, 0.0, torch.ones(37, dtype=torch.float64))
        f1dq = v2["dq"] + arr(momentum["joint_delta"])
        r1 = evaluate_solution(record, bounds, f1dq, names, "F1_CURRENT_V3", {"active": [], "frozen_v3_metric": "all joint weights 1.0 per protected D28R"}, f0)
        f2dq, f2diag = solve_f2(record, bounds)
        r2 = evaluate_solution(record, bounds, f2dq, names, "F2_HZ_NULLSPACE_ONLY", f2diag["hz"], f0)
        unconstrained = solve_unconstrained(record["tasks"]["hz"]["J"], record["tasks"]["hz"]["b"])
        r2["unconstrained_min_hz_error"] = float(abs((record["tasks"]["hz"]["J"] @ unconstrained - record["tasks"]["hz"]["b"])[0]))
        f3dq, f3diag = solve_lex(record, bounds, False)
        r3 = evaluate_solution(record, bounds, f3dq, names, "F3_BOUNDED_LEXICOGRAPHIC", f3diag, f0)
        f4dq, f4diag = solve_lex(record, bounds, True)
        r4 = evaluate_solution(record, bounds, f4dq, names, "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC", f4diag, f0)
        unbound_f2 = solve_unbounded_f2(record)
        unbound_f3, _ = solve_unbounded_lex(record, False)
        unbound_f4, _ = solve_unbounded_lex(record, True)
        for result, dq_unbound in ((r2, unbound_f2), (r3, unbound_f3), (r4, unbound_f4)):
            result["unbounded_min_hz_error"] = float(abs((record["tasks"]["hz"]["J"] @ dq_unbound - record["tasks"]["hz"]["b"])[0]))
            result["unbounded_relative_improvement"] = float((f0["predicted_hz_error"] - result["unbounded_min_hz_error"]) / max(f0["predicted_hz_error"], NUMERIC_ZERO))
            result["unbounded_task_residuals"] = task_residuals(record, dq_unbound)
            result["unbounded_velocity_ratio_max"] = float(np.max(np.abs(dq_unbound) / np.maximum(np.abs(record["velocity_limits"]), NUMERIC_ZERO)))
        rows = (r0, r1, r2, r3, r4)
        for result in rows:
            result["feedforward_hash"] = canonical_hash(bounds["feedforward"])
            result["current_hz_error"] = abs(record["tasks"]["hz"]["b"][0])
            all_results.append(result)
        per_record[(record["recipe"], record["control_step"])] = {"record": record, "bounds": bounds, "results": {r["formulation"]: r for r in rows}}
    np.savez_compressed(OUT / "joint_bounds_numeric.npz", velocity_lower=np.stack([row["bounds"]["velocity_lower"] for row in bound_records]), velocity_upper=np.stack([row["bounds"]["velocity_upper"] for row in bound_records]), velocity_gate_lower=np.stack([row["bounds"]["velocity_gate_lower"] for row in bound_records]), velocity_gate_upper=np.stack([row["bounds"]["velocity_gate_upper"] for row in bound_records]), position_lower=np.stack([row["bounds"]["position_lower"] for row in bound_records]), position_upper=np.stack([row["bounds"]["position_upper"] for row in bound_records]), combined_lower=np.stack([row["bounds"]["combined_lower"] for row in bound_records]), combined_upper=np.stack([row["bounds"]["combined_upper"] for row in bound_records]), feedforward=np.stack([row["bounds"]["feedforward"] for row in bound_records]))
    dump(OUT / "joint_bound_contract.json", {"name": "Exp014D28SJointBoundContractV1", "joint_count": 37, "joint_index_name_contract_read_only": True, "velocity_bound": "abs(dq)<=0.80*velocity_limit for feasibility gate; raw limit also recorded", "position_bound": "q_min<=q_current+dq*dt+endpoint_feedforward<=q_max", "canonical_action": "raw action unbounded; finite roundtrip only", "numeric_file": "joint_bounds_numeric.npz", "step_count": len(bound_records), "combined_interval_infeasible_steps": int(sum(np.any(row["bounds"]["combined_lower"] > row["bounds"]["combined_upper"] + SOLVER_TOL) for row in bound_records)), "joint_names": [{"action_index": i, "joint_name": names[i], "joint_group": group_for_joint(names[i]), "default_q": float(default_q[i]), "action_scale": float(action_scale[i]), "velocity_limit_from_contract": float(contract_velocity[i]), "position_limit": contract_limits[i].tolist()} for i in range(37)]})
    dump(OUT / "bounded_formulation_results.json", {"name": "Exp014D28SBoundedFormulationResultsV1", "rows": all_results, "formulations": {"F0": "protected D27 V2A", "F1": "protected D28 V3", "F2": "hard stance/CoM/swing/pelvis then H_z nullspace", "F3": "stance > CoM+swing+pelvis > H_z > regularizers", "F4": "stance > H_z > CoM+swing+pelvis diagnostic"}})
    with (OUT / "bounded_formulation_results.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["recipe", "control_step", "phase", "formulation", "current_hz", "predicted_hz_error", "relative_improvement", "rho", "velocity_ratio_max", "velocity_gate", "position_gate", "canonical_action_gate", "all_constraint_gates"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows([{key: row.get(key) for key in fields} for row in all_results])
    for row in authority:
        key = (int(row["recipe"]), int(row["control_step"]))
        entry = per_record[key]["results"]["F2_HZ_NULLSPACE_ONLY"]
        row["required_hz_correction"] = float(entry["v2_predicted_hz_error"])
        row["bounded_hz_correction"] = float(entry["v2_predicted_hz_error"] - entry["predicted_hz_error"]) if entry["solver_success"] else 0.0
        row["bounded_correction_fraction"] = float(max(row["bounded_hz_correction"], 0.0) / max(row["required_hz_correction"], NUMERIC_ZERO))
        row["minimum_joint_velocity_for_required_hz_norm"] = float(row["unbounded_minimum_norm_direction_norm"] * row["required_hz_correction"])
        row["classification"] = "HZ_NULLSPACE_AUTHORITY_PRESENT" if row["bounded_correction_fraction"] >= 0.20 else ("HZ_NULLSPACE_AUTHORITY_WEAK" if row["hz_authority_rank"] > 0 else "HZ_NULLSPACE_AUTHORITY_ZERO")
    dump(OUT / "nullspace_hz_authority.json", {"name": "Exp014D28SNullspaceHzAuthorityV1", "rows": authority, "criteria": {"ZERO": "norm<=1e-8", "WEAK": "bounded correction fraction<0.20", "PRESENT": "bounded correction fraction>=0.20"}})
    def summary_form(form: str) -> dict[str, Any]:
        summary = {}
        for recipe in TRACE_RECIPES:
            rows = [per_record[(recipe, int(trace["control_step"][recipe, row]))]["results"][form] for row in critical_rows[recipe]]
            improvements = np.asarray([x["relative_improvement"] for x in rows])
            gates = [bool(x["relative_improvement"] >= CRITICAL_IMPROVEMENT and x["all_constraint_gates"]) for x in rows]
            solved_improvements = [bool(x["solver_success"] and x["relative_improvement"] >= CRITICAL_IMPROVEMENT) for x in rows]
            unbounded_improvements = [bool(x.get("unbounded_relative_improvement", -1.0) >= CRITICAL_IMPROVEMENT) for x in rows]
            summary[str(recipe)] = {"critical_steps": len(rows), "improvement_ge_20_fraction": float(np.mean(improvements >= CRITICAL_IMPROVEMENT)) if rows else 0.0, "solved_improvement_ge_20_fraction": float(np.mean(solved_improvements)) if rows else 0.0, "unbounded_improvement_ge_20_fraction": float(np.mean(unbounded_improvements)) if rows else 0.0, "critical_gate_pass_fraction": float(np.mean(gates)) if rows else 0.0, "all_critical_gate_pass": bool(rows and all(gates)), "min_improvement": float(np.min(improvements)) if rows else None, "median_improvement": float(np.median(improvements)) if rows else None, "max_velocity_ratio": float(max(x["velocity_ratio_max"] for x in rows)) if rows else None, "position_gate_all": bool(rows and all(x["position_gate"] for x in rows)), "task_gate_all": bool(rows and all(x["all_constraint_gates"] for x in rows))}
        return summary
    critical_summary = {form: summary_form(form) for form in ("F0_V2A_BASELINE", "F1_CURRENT_V3", "F2_HZ_NULLSPACE_ONLY", "F3_BOUNDED_LEXICOGRAPHIC", "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC")}
    dump(OUT / "minimum_achievable_hz.json", {"name": "Exp014D28SMinimumAchievableHzV1", "target": 0.0, "rho_categories": {"STRONG_AUTHORITY": "rho<=0.5", "USEFUL_AUTHORITY": "0.5<rho<=0.8", "WEAK_AUTHORITY": "0.8<rho<1", "NO_AUTHORITY": "rho>=1"}, "rows": [{"recipe": row["recipe"], "control_step": row["control_step"], "formulation": row["formulation"], "current_actual_hz": row["current_hz"], "v2_predicted_hz_error": row["v2_predicted_hz_error"], "bounded_minimum_absolute_hz": row["bounded_min_hz_error"] if row["solver_success"] else None, "rho_vs_v2_prediction": row["rho"] if row["solver_success"] else None, "solver_success": row["solver_success"], "solver_reason": row["solver"].get("reason")} for row in all_results if row["formulation"] in ("F2_HZ_NULLSPACE_ONLY", "F3_BOUNDED_LEXICOGRAPHIC", "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC")]})
    dump(OUT / "critical_window_authority.json", {"name": "Exp014D28SCriticalWindowAuthorityV1", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_PASS_FRACTION, "window_manifest": manifest, "formulations": critical_summary})
    group_rows = []
    for form in ("F2_HZ_NULLSPACE_ONLY", "F3_BOUNDED_LEXICOGRAPHIC", "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"):
        for row in all_results:
            if row["formulation"] == form:
                group_rows.append({"recipe": row["recipe"], "control_step": row["control_step"], "formulation": form, "group_contribution": row["group_contribution"], "active_bound_joints": row["active_bound_joints"], "required_hz_correction_vs_v2": row["v2_predicted_hz_error"], "bounded_hz_correction_vs_v2": row["v2_predicted_hz_error"]-row["predicted_hz_error"] if row["solver_success"] else 0.0, "solver_success": row["solver_success"], "velocity_ratio_max": row["velocity_ratio_max"]})
    group_rows.extend({"recipe": row["recipe"], "control_step": row["control_step"], "formulation": "N3_NULLSPACE_UNIT_HZ_DIRECTION", "group_contribution": row["joint_group_contribution"], "hz_row_norm": row["hz_row_norm"], "bounded_correction_fraction": row.get("bounded_correction_fraction"), "classification": row["classification"]} for row in authority if row["nullspace"] == "N3_after_stance_com_swing_pelvis")
    dump(OUT / "joint_group_authority.json", {"name": "Exp014D28SJointGroupAuthorityV1", "rows": group_rows, "interpretation": "diagnostic only; D28R joint participation metric is not modified"})
    conflict = {}
    for recipe in TRACE_RECIPES:
        f2 = critical_summary["F2_HZ_NULLSPACE_ONLY"][str(recipe)]
        f3 = critical_summary["F3_BOUNDED_LEXICOGRAPHIC"][str(recipe)]
        f4 = critical_summary["F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"][str(recipe)]
        details = []
        for step in manifest[recipe]["critical_control_steps"]:
            item = per_record[(recipe, int(step))]["results"]["F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"]
            details.append({"control_step": int(step), "unbounded_hz_improvement": item.get("unbounded_relative_improvement"), "unbounded_velocity_ratio_max": item.get("unbounded_velocity_ratio_max"), "unbounded_task_residuals": item.get("unbounded_task_residuals"), "bounded_solver_success": item.get("solver_success"), "bounded_solver_reason": item.get("solver", {}).get("reason")})
        conflict[str(recipe)] = {"F2": f2, "F3": f3, "F4": f4, "unbounded_f4_critical_rows": details, "classification": "HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS" if f3["critical_gate_pass_fraction"] < CRITICAL_PASS_FRACTION and f4["solved_improvement_ge_20_fraction"] >= CRITICAL_PASS_FRACTION and not f4["task_gate_all"] else "NOT_ESTABLISHED"}
    dump(OUT / "hz_task_conflict.json", {"name": "Exp014D28SHzTaskConflictV1", "rows": conflict, "formal_task_contract_changed": False})
    blockers = []
    for row in all_results:
        if row["formulation"] in ("F2_HZ_NULLSPACE_ONLY", "F3_BOUNDED_LEXICOGRAPHIC") and (not row["solver_success"] or row["relative_improvement"] < CRITICAL_IMPROVEMENT):
            record_item = per_record[(int(row["recipe"]), int(row["control_step"]))]
            bounds = record_item["bounds"]
            invalid = np.where(bounds["combined_lower"] > bounds["combined_upper"] + SOLVER_TOL)[0].tolist()
            unbounded_dq, _ = solve_unbounded_lex(record_item["record"], True)
            blockers.append({"recipe": row["recipe"], "control_step": row["control_step"], "formulation": row["formulation"], "required_velocity_ratio": row["velocity_ratio_max"], "unbounded_required_velocity_ratio": float(np.max(np.abs(unbounded_dq) / np.maximum(np.abs(record_item["record"]["velocity_limits"]), NUMERIC_ZERO))), "active_bound_joints": row["active_bound_joints"], "position_gate": row["position_gate"], "canonical_action_gate": row["canonical_action_gate"], "solver_success": row["solver_success"], "solver_reason": row["solver"].get("reason"), "invalid_combined_bound_joints": [{"joint_index": int(index), "joint_name": names[index], "joint_group": group_for_joint(names[index]), "lower": float(bounds["combined_lower"][index]), "upper": float(bounds["combined_upper"][index])} for index in invalid], "position_zero_dq_margin": {"lower": (record_item["record"]["q_current"] + bounds["feedforward"] - record_item["record"]["q_min"]).tolist(), "upper": (record_item["record"]["q_max"] - record_item["record"]["q_current"] - bounds["feedforward"]).tolist()}})
    dump(OUT / "joint_authority_blockers.json", {"name": "Exp014D28SJointAuthorityBlockersV1", "rows": blockers, "classification": "HZ_CONTROL_BLOCKED_BY_JOINT_AUTHORITY" if blockers else "NOT_ESTABLISHED"})
    # Multi-step authority is diagnostic only and is conditional on a
    # nonzero N3 row with one-step useful authority absent. It uses recorded
    # time-varying Jacobians/bounds, never future physics state.
    n3 = [row for row in authority if row["nullspace"] == "N3_after_stance_com_swing_pelvis"]
    n3_present = any(row["hz_authority_rank"] > 0 for row in n3)
    f3_improvements = [bool(row["solver_success"] and row["relative_improvement"] >= CRITICAL_IMPROVEMENT) for row in all_results if row["formulation"] == "F3_BOUNDED_LEXICOGRAPHIC"]
    useful_fraction = float(np.mean(f3_improvements)) if f3_improvements else 0.0
    multi: dict[str, Any] = {"name": "Exp014D28SMultiStepControllabilityV1", "executed": bool(n3_present and useful_fraction < 1.0), "reason": "N3 rank exists but one-step useful authority was not universal" if n3_present and useful_fraction < 1.0 else "not required: no N3 rank or one-step useful authority condition not met", "horizons": {str(h): {"status": "NOT_EXECUTED"} for h in (2, 4, 8)}, "future_state_runtime_input": False}
    if multi["executed"]:
        for horizon in (2, 4, 8):
            rows = []
            for recipe in TRACE_RECIPES:
                for start in critical_rows[recipe]:
                    end = min(start + horizon, analysis_rows[recipe][-1] + 1)
                    seq = [per_record[(recipe, int(trace["control_step"][recipe, j]))]["results"]["F3_BOUNDED_LEXICOGRAPHIC"] for j in range(start, end)]
                    if seq:
                        feasible = bool(all(item["solver_success"] and item["all_constraint_gates"] for item in seq))
                        rows.append({"recipe": recipe, "start_control_step": int(trace["control_step"][recipe, start]), "steps_used": len(seq), "status": "FEASIBLE_SEQUENCE" if feasible else "NO_FEASIBLE_SEQUENCE", "terminal_rho": float(seq[-1]["predicted_hz_error"] / max(seq[0]["v2_predicted_hz_error"], NUMERIC_ZERO)) if feasible else None, "terminal_improvement": float((seq[0]["v2_predicted_hz_error"] - seq[-1]["predicted_hz_error"]) / max(seq[0]["v2_predicted_hz_error"], NUMERIC_ZERO)) if feasible else None, "peak_velocity_ratio": float(max(item["velocity_ratio_max"] for item in seq)), "task_constraints_pass": feasible})
            vals = np.asarray([row["terminal_improvement"] for row in rows if row["terminal_improvement"] is not None])
            multi["horizons"][str(horizon)] = {"status": "DIAGNOSTIC_COMPLETE", "rows": rows, "terminal_improvement_median": None if not vals.size else float(np.median(vals)), "terminal_improvement_max": None if not vals.size else float(np.max(vals))}
    dump(OUT / "multi_step_controllability.json", multi)
    f2pass = all(critical_summary["F2_HZ_NULLSPACE_ONLY"][str(recipe)]["critical_gate_pass_fraction"] >= CRITICAL_PASS_FRACTION for recipe in TRACE_RECIPES)
    f3pass = all(critical_summary["F3_BOUNDED_LEXICOGRAPHIC"][str(recipe)]["critical_gate_pass_fraction"] >= CRITICAL_PASS_FRACTION for recipe in TRACE_RECIPES)
    n3_rank_deficient = bool(n3 and all(row["hz_authority_rank"] == 0 for row in n3))
    any_conflict = any(row["classification"] == "HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS" for row in conflict.values())
    f4_useful = all(critical_summary["F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"][str(recipe)]["solved_improvement_ge_20_fraction"] >= CRITICAL_PASS_FRACTION for recipe in TRACE_RECIPES)
    f4_unbounded_useful = all(critical_summary["F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"][str(recipe)]["unbounded_improvement_ge_20_fraction"] >= CRITICAL_PASS_FRACTION for recipe in TRACE_RECIPES)
    if n3_rank_deficient:
        classification = "EXP014_D28S_HZ_NULLSPACE_RANK_DEFICIENT"; next_action = "position-level H_z route is rank deficient after hard first-step tasks; evaluate dynamics-constrained trajectory optimization or torque-level WBC"
    elif any_conflict:
        classification = "EXP014_D28S_HZ_FIRST_STEP_TASK_CONFLICT"; next_action = "H_z control conflicts with mandatory stance/CoM/swing/pelvis tasks; keep the D28 contract unchanged"
    elif f4_unbounded_useful and not f4_useful:
        classification = "EXP014_D28S_JOINT_AUTHORITY_INSUFFICIENT"; next_action = "unbounded/H_z-first correction is available but bounded joint velocity/position/action authority is insufficient"
    elif f2pass or f3pass:
        classification = "EXP014_D28S_BOUNDED_CENTROIDAL_AUTHORITY_PASS"; next_action = "D28T bounded centroidal physics preflight with the fixed selected formulation; physics was not run in D28S"
    else:
        classification = "EXP014_D28S_POSITION_LEVEL_CENTROIDAL_CONTROL_NO_GO"; next_action = "end the position-level centroidal-feedback branch; evaluate dynamics-constrained trajectory optimization or torque-level WBC separately"
    selected = "F2_HZ_NULLSPACE_ONLY" if f2pass else ("F3_BOUNDED_LEXICOGRAPHIC" if f3pass else None)
    dump(OUT / "temporary_v3r_contract.json", {"name": "Exp014BoundedCentroidalWBIKV3R", "created": bool(selected), "physics_applied": False, "selected_formulation": selected, "reason": "critical-window gates passed" if selected else "not created: per-source bounded critical-window gates did not pass", "D28_V3_protected": True})
    dump(OUT / "temporary_v3r_shadow.json", {"name": "Exp014BoundedCentroidalWBIKV3RShadow", "status": "NOT_CREATED" if selected is None else "DIAGNOSTIC_ONLY", "physics_executed": 0, "reason": "D28S does not authorize physics" if selected is None else "shadow artifact remains non-physics diagnostic"})
    if selected is not None:
        dump(OUT / "exp014_d28t_bounded_centroidal_physics_preflight_authorization.json", {"authorized": True, "classification": classification, "selected_formulation": selected, "solver_hash": canonical_hash({"tolerance": SOLVER_TOL, "max_iterations": SOLVER_MAX_ITER, "damping": 1.0e-4}), "physics_executed": 0, "target": RIGHT_TARGET_ID, "timing_changed": False, "gain_changed": False})
    else:
        dump(OUT / "exp014_d28t_not_authorized.json", {"authorized": False, "classification": classification, "physics_executed": 0, "reason": next_action})
    dump(OUT / "stage_classification.json", {"name": "Exp014D28SStageClassificationV1", "classification": classification, "precedence_applied": ["solver/interface", "nullspace rank", "hard-task conflict", "joint authority", "current V3 defect", "bounded pass", "position-level no-go"], "physics_executed": 0, "starting_head": start_head})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "persistent_update": 0, "new_checkpoint": 0, "left_start": 0, "remote_push": False})
    protected = {}
    for relative in git("ls-files").splitlines():
        normalized = relative.replace("\\", "/")
        if any(token in normalized for token in ("/phase_2_d26", "/phase_2_d27", "/phase_2_d28", "exp_005", "exp_006", "exp_007", "exp_008", "exp_009", "exp_010", "exp_011", "exp_012", "exp_013")):
            path = REPO / relative
            if path.is_file(): protected[normalized] = sha256_file(path)
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "starting_status_short": start_status, "protected_file_count": len(protected), "protected_files_sha256": protected, "protected_aggregate_sha256": canonical_hash(protected), "d28r_read_only": True, "persistent_update": 0, "new_learned_checkpoint": 0, "physics": 0, "left_start": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "raw_restore": 0, "run_integration": 0, "remote_push": False})
    dump(OUT / "reproduction_commands.ps1", "Set-Location '" + str(REPO) + "'\n# Offline only; no Isaac physics is launched.\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p '" + str(HERE) + "' --mode analyze --headless\n")
    print(json.dumps({"classification": classification, "physics_executed": 0, "records": len(records), "critical_rows": {str(recipe): len(critical_rows[recipe]) for recipe in TRACE_RECIPES}, "f2_pass": f2pass, "f3_pass": f3pass, "n3_rank_deficient": n3_rank_deficient}, indent=2))


if __name__ == "__main__":
    main()
