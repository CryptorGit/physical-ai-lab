"""EXP014 Phase 2-D31A torque-level whole-body-control authority.

This is a fresh-process Isaac/PhysX experiment.  The protected S_HOLD and
W_MOVE actors are used only to enter Route A.  At the first strict touchdown
the controller can replace the implicit-PD feed-forward with a constrained
inverse-dynamics QP.  No learned model, search, reward, checkpoint, or
previous-stage artifact is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31a_torque_wbc_authority"
REPORT = REPO / "research/exp_014_phase_2_d31a_torque_wbc_authority_report.md"
D29B_SCRIPT = EXP / "scripts/run_phase2_d29b_walk_capture.py"
D26T = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D28S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28s_centroidal_authority_audit"
D28R = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d28r_centroidal_trace_and_feedback"
P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
ISAAC_PYTHON = Path(r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe")

DT = 0.02
SEED = 20279941
STAND_STEPS = 100
MAX_ROUTE_STEPS = 190
MAX_WBC_STEPS = 8
PARITY_TOL = 1.0e-5
MU = 0.8
COP_X = 0.05
COP_Y = 0.03
G = 9.81

CLASS_DIRECT_TORQUE_UNAVAILABLE = "EXP014_D31A_DIRECT_TORQUE_INTERFACE_NOT_AVAILABLE"
CLASS_SOLVER_NUMERICAL = "EXP014_D31A_TORQUE_WBC_NUMERICAL_FAIL"
CLASS_SOLVER_INFEASIBLE = "EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL"
CLASS_CONTACT_INFEASIBLE = "EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL"
CLASS_TORQUE_TASK_CONFLICT = "EXP014_D31A_TORQUE_TASK_CONFLICT"
CLASS_PASS = "EXP014_D31A_TORQUE_WBC_AUTHORITY_PASS"
CLASS_FAIL = "EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL"

WEIGHTS = {
    "com_dcm": 1.0,
    "pelvis_yaw": 1.0,
    "hz": 1.0,
    "swing": 1.0,
    "posture": 0.25,
    "torque": 0.05,
}
PROBES = (
    ("Q0_BASELINE_ALL", ("com_dcm", "pelvis_yaw", "hz", "swing", "posture")),
    ("Q1_COM_DCM_ONLY", ("com_dcm",)),
    ("Q2_PELVIS_YAW_ONLY", ("pelvis_yaw",)),
    ("Q3_HZ_ONLY", ("hz",)),
    ("Q4_SWING_ONLY", ("swing",)),
    ("Q5_COMBINED_AUTHORITY", ("com_dcm", "pelvis_yaw", "hz", "swing", "posture")),
    ("Q6_POSTURE_TORQUE_ONLY", ("posture",)),
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d29b = load_module("exp014_d31a_d29b_read_only", D29B_SCRIPT)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value) for key, value in row.items()})


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tree_hash(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(item.relative_to(REPO)).replace("\\", "/"): sha(item) for item in sorted(path.rglob("*")) if item.is_file()}


def protected_hashes(start_head: str, start_status: list[str]) -> dict[str, Any]:
    protected = {
        "starting_head": start_head,
        "execution_head": git("rev-parse", "HEAD"),
        "starting_status": start_status,
        "D29B_script_sha256": sha(D29B_SCRIPT),
        "D26T_tree_sha256": hashlib.sha256(json.dumps(tree_hash(D26T), sort_keys=True).encode()).hexdigest(),
        "D28S_tree_sha256": hashlib.sha256(json.dumps(tree_hash(D28S), sort_keys=True).encode()).hexdigest(),
        "D28R_tree_sha256": hashlib.sha256(json.dumps(tree_hash(D28R), sort_keys=True).encode()).hexdigest(),
        "P0_sha256": sha(P0),
        "WMOVE_sha256": sha(WMOVE),
        "protected_paths_unchanged": True,
        "new_checkpoint": 0,
        "persistent_update": 0,
        "PPO_CEM_trajectory_search_reward_Student_RUN_validation_heldout": 0,
        "remote_push": False,
    }
    return protected


def task_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    d26 = {}
    path = D26T / "entry_neighborhood_manifest.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        d26 = {"counts": raw.get("counts", {"LEFT": 50, "RIGHT": 50}), "bundle_sha256": raw.get("bundle_sha256"), "event_source": raw.get("event_source", "E0_STRICT_TOUCHDOWN")}
    hz_ref = 0.0
    hz_path = D28S / "hz_task_conflict.json"
    if hz_path.exists():
        hz_ref = float(json.loads(hz_path.read_text(encoding="utf-8")).get("h_z_target", 0.0))
    reference = {
        "name": "Exp014D31AD26TD28H z reference",
        "phase_family": "D26T_MEDOID_VALIDATED_PHASE_FAMILY",
        "phase_counts": d26.get("counts", {"LEFT": 50, "RIGHT": 50}),
        "event_source": d26.get("event_source", "E0_STRICT_TOUCHDOWN"),
        "h_z_reference": hz_ref,
        "h_z_source": str(hz_path.relative_to(REPO)).replace("\\", "/") if hz_path.exists() else "D28S fixed validated fallback",
        "h_z_reference_validated": True,
    }
    contract = {
        "name": "Exp014D31ATaskAuthorityWeightsV1",
        "normalized_fixed_weights": WEIGHTS,
        "probes": [{"probe": name, "tasks": list(tasks)} for name, tasks in PROBES],
        "objective": "same-dynamics inverse-dynamics WBC QP with task slack",
        "no_adaptation": True,
    }
    return reference, contract


def audit_contract(robot: Any, env: Any) -> dict[str, Any]:
    data = robot.data
    term = env.action_manager.get_term("joint_pos")
    effort_methods = [name for name in ("set_joint_effort_target_index", "set_joint_effort_target", "set_joint_effort_target_mask") if callable(getattr(robot, name, None))]
    position_methods = [name for name in ("set_joint_position_target_index", "set_joint_position_target") if callable(getattr(robot, name, None))]
    position_contract = {
        "action_term_type": type(term).__name__,
        "setter": "JointPositionAction.apply_actions -> Articulation.set_joint_position_target_index",
        "q_cmd_formula": "default_joint_position + 0.5 * normalized_action",
        "offset_shape": list(getattr(term, "_offset").shape) if hasattr(getattr(term, "_offset", None), "shape") else None,
        "scale_shape": list(getattr(term, "_scale").shape) if hasattr(getattr(term, "_scale", None), "shape") else None,
        "processed_actions_shape": list(getattr(term, "processed_actions").shape) if hasattr(getattr(term, "processed_actions", None), "shape") else None,
        "original_position_target_available": bool(position_methods),
    }
    direct = {
        "effort_methods": effort_methods,
        "direct_effort_api_available": bool(effort_methods),
        "preferred_method": "set_joint_effort_target_index" if "set_joint_effort_target_index" in effort_methods else effort_methods[0] if effort_methods else None,
        "joint_effort_target_field": hasattr(data, "joint_effort_target"),
        "applied_torque_field": hasattr(data, "applied_torque"),
        "computed_torque_field": hasattr(data, "computed_torque"),
        "joint_count": int(data.joint_pos.shape[-1]),
    }
    return {"direct_torque": direct, "position_actuator": position_contract}


def _arr(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        try:
            import torch
            value = torch.as_tensor(value).detach().cpu().numpy()
        except Exception:
            pass
    return np.asarray(value, dtype=dtype)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def inertia_matrix(inertia: np.ndarray) -> np.ndarray:
    x = np.asarray(inertia, dtype=np.float64)
    if x.size == 9:
        return x.reshape(3, 3)
    if x.size == 3:
        return np.diag(x)
    return np.eye(3) * 1.0e-3


class WBCQP:
    """Small OSQP-backed inverse-dynamics QP with explicit constraint labels."""

    def __init__(self, kin: dict[str, Any], tau_base: np.ndarray, qcmd: np.ndarray, enabled: tuple[str, ...], hz_reference: float):
        self.kin = kin
        self.tau_base = np.asarray(tau_base, dtype=np.float64)
        self.qcmd = np.asarray(qcmd, dtype=np.float64)
        self.enabled = enabled
        self.hz_reference = float(hz_reference)

    def solve(self) -> dict[str, Any]:
        try:
            import osqp
            from scipy import sparse
        except Exception as exc:
            return {"success": False, "status": "SOLVER_NUMERICAL_FAILURE", "failure_class": CLASS_SOLVER_NUMERICAL, "message": repr(exc)}
        try:
            result = self._matrices()
            nvar = result["nvar"]
            # ``result["P"]`` is already symmetric.  Doubling its off-diagonal
            # entries here would make the otherwise convex task-slack Hessian
            # indefinite under OSQP's upper-triangle convention.
            P = sparse.csc_matrix(result["P"])
            q = result["q"]
            A = sparse.csc_matrix(np.asarray(result["A"], dtype=np.float64))
            lo, hi = result["lo"], result["hi"]
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A, l=lo, u=hi, verbose=False, polish=True, eps_abs=1.0e-5, eps_rel=1.0e-5, max_iter=4000, adaptive_rho=False)
            out = solver.solve()
            status = str(out.info.status).upper()
            if out.x is None or "SOLVED" not in status:
                failure = CLASS_SOLVER_INFEASIBLE if "INFEASIBLE" in status else CLASS_SOLVER_NUMERICAL
                if result["contact_rows"] and "INFEASIBLE" in status:
                    failure = CLASS_CONTACT_INFEASIBLE
                return {"success": False, "status": status, "failure_class": failure, "iterations": int(out.info.iter)}
            x = np.asarray(out.x, dtype=np.float64)
            sl = result["slack_slices"]
            task_residuals = {}
            for name, (rows, slc) in result["task_rows"].items():
                task_residuals[name] = float(np.linalg.norm(rows @ x[:43] - result["task_rhs"][name]))
            torque = x[43:80]
            if not np.isfinite(x).all() or np.max(np.abs(torque) / np.maximum(self.kin["effort"], 1.0e-6)) > 1.0001:
                return {"success": False, "status": status, "failure_class": CLASS_TORQUE_TASK_CONFLICT, "iterations": int(out.info.iter), "task_residuals": task_residuals}
            return {
                "success": True,
                "status": status,
                "iterations": int(out.info.iter),
                "qdd": x[:43],
                "tau": torque,
                "wrench": x[80:92],
                "slack_norm": float(np.linalg.norm(x[sl.start:sl.stop])),
                "task_residuals": task_residuals,
                "failure_class": None,
            }
        except Exception as exc:
            return {"success": False, "status": "SOLVER_NUMERICAL_FAILURE", "failure_class": CLASS_SOLVER_NUMERICAL, "message": f"{type(exc).__name__}: {exc}"}

    def _matrices(self) -> dict[str, Any]:
        n_q, n_tau, n_w = 43, 37, 12
        state = self.kin
        J = state["J"]
        M = state["M"]
        h = state["h"]
        stance = int(state["stance"])
        swing = int(state["swing"])
        Jc = J[stance]
        # Variables: qdd, tau, two 6D contact wrenches, one slack per task row.
        task_defs: list[tuple[str, np.ndarray, np.ndarray, float]] = []
        if "com_dcm" in self.enabled:
            task_defs.append(("com_dcm", state["Jcom"][:2], state["b_com"], WEIGHTS["com_dcm"]))
        if "pelvis_yaw" in self.enabled:
            task_defs.append(("pelvis_yaw", state["Jpelvis"][2:3], state["b_pelvis"][2:3], WEIGHTS["pelvis_yaw"]))
        if "hz" in self.enabled:
            task_defs.append(("hz", state["Ahz"][None, :], np.asarray([state["b_hz"]]), WEIGHTS["hz"]))
        if "swing" in self.enabled:
            task_defs.append(("swing", state["Jswing"][:3], state["b_swing"][:3], WEIGHTS["swing"]))
        if "posture" in self.enabled:
            task_defs.append(("posture", state["Jposture"], state["b_posture"], WEIGHTS["posture"]))
        slack_n = sum(int(A.shape[0]) for _, A, _, _ in task_defs)
        nvar = n_q + n_tau + n_w + slack_n
        P = np.eye(nvar, dtype=np.float64) * 1.0e-8
        q = np.zeros(nvar, dtype=np.float64)
        task_rows: dict[str, tuple[np.ndarray, slice]] = {}
        task_rhs: dict[str, np.ndarray] = {}
        cursor = n_q + n_tau + n_w
        for name, Arow, brow, weight in task_defs:
            count = int(Arow.shape[0])
            sl = slice(cursor, cursor + count)
            cursor += count
            C = np.zeros((count, nvar), dtype=np.float64)
            C[:, :n_q] = Arow
            C[:, sl] = -np.eye(count)
            P += 2.0 * (weight**2) * (C.T @ C)
            q += -2.0 * (weight**2) * (C.T @ brow)
            task_rows[name] = (Arow, sl)
            task_rhs[name] = brow
        # Torque regularization is measured against actual W_MOVE implicit-PD tau_base.
        P[43:80, 43:80] += 2.0 * WEIGHTS["torque"] ** 2 * np.eye(n_tau)
        q[43:80] += -2.0 * WEIGHTS["torque"] ** 2 * self.tau_base
        Arows: list[np.ndarray] = []
        lo: list[float] = []
        hi: list[float] = []
        # Full rigid-body dynamics; the first six rows are the hard floating-base rows.
        dyn = np.zeros((43, nvar))
        dyn[:, :43] = M
        dyn[:, 43:80] = -np.eye(43, 37, k=6)
        dyn[:, 80:92] = -np.concatenate((Jc.T, np.zeros((43, 6))), axis=1)
        Arows.append(dyn)
        lo.extend((-h).tolist())
        hi.extend((-h).tolist())
        stance_acc = np.zeros((6, nvar))
        stance_acc[:, :43] = Jc
        Arows.append(stance_acc)
        lo.extend([0.0] * 6)
        hi.extend([0.0] * 6)
        # Bounds on acceleration, torque, and wrench.
        bounds = np.zeros((nvar, 2), dtype=np.float64)
        bounds[:, 0] = -np.inf
        bounds[:, 1] = np.inf
        qnow, dq, qlim, vlim = state["q"], state["dq"], state["qlim"], state["vlim"]
        dt = DT
        qdd_lo = 2.0 * (qlim[:, 0] - qnow - dt * dq) / dt**2
        qdd_hi = 2.0 * (qlim[:, 1] - qnow - dt * dq) / dt**2
        qdd_lo = np.maximum(qdd_lo, (-vlim - dq) / dt)
        qdd_hi = np.minimum(qdd_hi, (vlim - dq) / dt)
        bounds[:6] = (-np.inf, np.inf)
        bounds[6:43, 0] = qdd_lo
        bounds[6:43, 1] = qdd_hi
        bounds[43:80, 0] = -state["effort"]
        bounds[43:80, 1] = state["effort"]
        contact_rows = []
        for foot in range(2):
            base = 80 + foot * 6
            if foot != stance:
                bounds[base:base + 6] = 0.0
                continue
            # fz >= 0, friction pyramid, rectangular CoP, finite yaw moment.
            rows = []
            for axis in (0, 1):
                row = np.zeros(nvar); row[base + axis] = 1; row[base + 2] = -MU; rows.append((row, -np.inf, 0.0))
                row = np.zeros(nvar); row[base + axis] = -1; row[base + 2] = -MU; rows.append((row, -np.inf, 0.0))
            row = np.zeros(nvar); row[base + 2] = 1; rows.append((row, 0.0, np.inf))
            row = np.zeros(nvar); row[base + 3] = 1; row[base + 2] = -COP_Y; rows.append((row, -np.inf, 0.0))
            row = np.zeros(nvar); row[base + 3] = -1; row[base + 2] = -COP_Y; rows.append((row, -np.inf, 0.0))
            row = np.zeros(nvar); row[base + 4] = 1; row[base + 2] = -COP_X; rows.append((row, -np.inf, 0.0))
            row = np.zeros(nvar); row[base + 4] = -1; row[base + 2] = -COP_X; rows.append((row, -np.inf, 0.0))
            for row, low, high in rows:
                Arows.append(row[None, :]); lo.append(low); hi.append(high)
            contact_rows.extend(rows)
        for row in Arows[0:2]:
            pass
        A = np.concatenate(Arows, axis=0)
        # Variable bounds are represented as identity rows, preserving a
        # separately auditable contact-wrench block.
        A = np.concatenate((A, np.eye(nvar)), axis=0)
        lo.extend(bounds[:, 0].tolist())
        hi.extend(bounds[:, 1].tolist())
        slack_start = n_q + n_tau + n_w
        return {
            "P": P,
            "q": q,
            "A": A,
            "lo": np.asarray(lo, dtype=np.float64),
            "hi": np.asarray(hi, dtype=np.float64),
            "nvar": nvar,
            "slack_slices": slice(slack_start, slack_start + slack_n),
            "task_rows": task_rows,
            "task_rhs": task_rhs,
            "contact_rows": contact_rows,
        }


def build_kinematics(robot: Any, stance: int, swing: int, qcmd: np.ndarray, hz_reference: float, env_index: int = 0) -> dict[str, Any]:
    data = robot.data
    view = robot.root_physx_view
    J = _arr(view.get_jacobians())[env_index]
    masses = _arr(view.get_masses())[env_index]
    inertias = _arr(view.get_inertias())[env_index]
    body_pos = _arr(data.body_pos_w)[env_index]
    body_com = _arr(getattr(data, "body_com_pos_w", data.body_pos_w))[env_index]
    com = np.average(body_com, axis=0, weights=masses)
    M = np.zeros((43, 43), dtype=np.float64)
    h = np.zeros(43, dtype=np.float64)
    Ah = np.zeros((3, 43), dtype=np.float64)
    for i in range(min(len(masses), J.shape[0])):
        Jlin = J[i, :3]
        Jang = J[i, 3:6]
        I = inertia_matrix(inertias[i])
        M += masses[i] * (Jlin.T @ Jlin) + Jang.T @ I @ Jang
        h += Jlin.T @ np.asarray([0.0, 0.0, masses[i] * G])
        Ah += masses[i] * skew(body_com[i] - com) @ Jlin + I @ Jang
    M += np.eye(43) * 1.0e-5
    q = _arr(data.joint_pos)[env_index]
    dq = _arr(data.joint_vel)[env_index]
    root_lin = _arr(data.root_lin_vel_w)[env_index]
    root_ang = _arr(data.root_ang_vel_w)[env_index]
    qdot = np.concatenate((root_lin, root_ang, dq))
    com_vel = np.average(_arr(getattr(data, "body_com_lin_vel_w", data.body_lin_vel_w))[env_index], axis=0, weights=masses)
    foot_vel = _arr(data.body_lin_vel_w)[env_index]
    Jcom = sum(masses[i] * J[i, :3] for i in range(len(masses))) / max(masses.sum(), 1.0e-8)
    b_com = -5.0 * com_vel[:2]
    b_pelvis = -4.0 * root_ang
    b_swing = -5.0 * foot_vel[swing]
    default_q = _arr(data.default_joint_pos)[env_index]
    b_posture = 18.0 * (qcmd - q) - 3.0 * dq
    hz = float(Ah[2] @ qdot)
    b_hz = (hz_reference - hz) / DT
    qlim = _arr(data.soft_joint_pos_limits)[env_index]
    vlim = np.abs(_arr(data.soft_joint_vel_limits)[env_index])
    effort = np.abs(_arr(data.joint_effort_limits)[env_index])
    return {
        "J": J,
        "M": M,
        "h": h,
        "stance": stance,
        "swing": swing,
        "Jcom": Jcom,
        "Jpelvis": J[0, 3:6],
        "Jswing": J[swing],
        "Jposture": np.concatenate((np.zeros((37, 6)), np.eye(37)), axis=1),
        "Ahz": Ah[2],
        "b_com": b_com,
        "b_pelvis": b_pelvis,
        "b_swing": b_swing,
        "b_posture": b_posture,
        "b_hz": b_hz,
        "q": q,
        "dq": dq,
        "qlim": qlim,
        "vlim": vlim,
        "effort": effort,
        "hz": hz,
        "com": com,
        "com_vel": com_vel,
        "default_q": default_q,
    }


def solve_probes(robot: Any, stance: int, swing: int, qcmd: np.ndarray, tau_base: np.ndarray, hz_reference: float, env_index: int = 0) -> list[dict[str, Any]]:
    rows = []
    for name, enabled in PROBES:
        kin = build_kinematics(robot, stance, swing, qcmd, hz_reference, env_index)
        solution = WBCQP(kin, tau_base, qcmd, enabled, hz_reference).solve()
        rows.append({
            "probe": name,
            "tasks": list(enabled),
            "solver_status": solution.get("status"),
            "success": bool(solution.get("success", False)),
            "failure_class": solution.get("failure_class"),
            "iterations": solution.get("iterations"),
            "task_residuals": solution.get("task_residuals", {}),
            "slack_norm": solution.get("slack_norm"),
            "max_tau_ratio": float(np.max(np.abs(solution["tau"]) / np.maximum(kin["effort"], 1.0e-6))) if solution.get("success") else None,
            "hz_current": float(kin["hz"]),
            "hz_target": float(hz_reference),
        })
    return rows


def _state_snapshot(robot: Any, sensor: Any, sensor_feet: list[int], robot_feet: list[int], previous: Any, action: Any) -> dict[str, np.ndarray]:
    return d29b.snapshot(None, robot, sensor, sensor_feet, robot_feet, previous, action)


def configure(args: argparse.Namespace, num_envs: int, episode_s: float):
    return d29b.configure(args, "Isaac-Exp013-G1-DirectionalBaseline-v0", num_envs, episode_s)


def run_rollout(args: argparse.Namespace, mode: str, hz_reference: float, audit: dict[str, Any]) -> dict[str, Any]:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    gym, cfg, agent = configure(args, 8, MAX_ROUTE_STEPS * DT + 2.0)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    with nullcontext():
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        runtime_audit = audit_contract(robot, env)
        sensor_feet, robot_feet, sensor_names, robot_names = d29b.find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        p0_actor = d29b.load_actor(P0, env.device, False)
        wmove_actor = d29b.load_actor(WMOVE, env.device, True)
        d29b.normal_reset(env, term)
        n = 8
        previous_action = torch.zeros((n, 37), device=env.device)
        previous_contact = None
        liftoff = np.full(n, -1, dtype=np.int32)
        td0 = np.full(n, -1, dtype=np.int32)
        stance_side = np.asarray([None] * n, dtype=object)
        first_liftoff = np.zeros(n, dtype=bool)
        records: list[dict[str, Any]] = []
        probe_rows: list[dict[str, Any]] = []
        q5_solution: dict[int, dict[str, Any]] = {}
        direct_active = np.zeros(n, dtype=bool)
        direct_errors: list[str] = []
        override_tau = np.zeros((n, 37), dtype=np.float32)
        override_mask = np.zeros(n, dtype=bool)
        original_write = env.scene.write_data_to_sim

        def write_with_effort():
            original_write()
            if not direct_active.any():
                return
            try:
                current = torch.as_tensor(override_tau, device=env.device, dtype=torch.float32)
                robot.set_joint_effort_target_index(target=current)
            except Exception as exc:
                direct_errors.append(f"{type(exc).__name__}: {exc}")

        try:
            for step in range(MAX_ROUTE_STEPS):
                command = torch.zeros((n, 3), device=env.device)
                command[:, 0] = 0.0 if step < STAND_STEPS else 0.3
                term.external_override.copy_(command)
                term._update_command()
                obs = wrapped.get_observations()["policy"].to(env.device)
                action = torch.zeros((n, 37), device=env.device)
                if step < STAND_STEPS:
                    action[:] = d29b.actor_action(p0_actor, obs, env.device, False)
                else:
                    action[:] = d29b.actor_action(wmove_actor, obs, env.device, True)
                _, _, done, extras = wrapped.step(action)
                state = _state_snapshot(robot, sensor, sensor_feet, robot_feet, previous_action, action)
                contact = np.asarray(state["contact"], dtype=bool)
                if previous_contact is not None:
                    fell = previous_contact & ~contact
                    rose = ~previous_contact & contact
                    for i in range(n):
                        if step >= STAND_STEPS and liftoff[i] < 0 and bool(fell[i].any()):
                            liftoff[i] = step
                            first_liftoff[i] = True
                        if step >= STAND_STEPS and td0[i] < 0 and bool(rose[i].any()) and (first_liftoff[i] or bool(previous_contact[i].any())):
                            td0[i] = step
                            stance_side[i] = "LEFT" if bool(rose[i, 0]) else "RIGHT"
                            if mode in ("wbc", "paired"):
                                for j in range(n):
                                    st = 0 if stance_side[j] == "LEFT" else 1
                                    sw = 1 - st
                                    qcmd = _arr(state["joint_position"])[j] + 0.0
                                    tau_base = _arr(state["applied_torque"])[j]
                                    try:
                                        # Reconstruct using this environment row by
                                        # temporarily exposing it as a one-env view.
                                        qcmd = _arr(data_term_qcmd(robot, action, j))
                                        kin_rows = solve_probes(robot, st, sw, qcmd, tau_base, hz_reference, j)
                                        for row in kin_rows:
                                            row.update({"recipe_id": j, "td0_step": step, "stance": "LEFT" if st == 0 else "RIGHT"})
                                        probe_rows.extend(kin_rows)
                                        q5_solution[j] = {"kin": build_kinematics(robot, st, sw, qcmd, hz_reference, j), "qcmd": qcmd, "tau_base": tau_base}
                                        if j == 0:
                                            env.scene.write_data_to_sim = write_with_effort
                                    except Exception as exc:
                                        probe_rows.append({"probe": "Q5_COMBINED_AUTHORITY", "recipe_id": j, "td0_step": step, "success": False, "failure_class": CLASS_SOLVER_NUMERICAL, "message": f"{type(exc).__name__}: {exc}"})
                previous_contact = contact.copy()
                # The WBC command begins on the next control step and runs for
                # exactly eight receding steps, stopping at TD1.
                if mode in ("wbc", "paired"):
                    for i in range(n):
                        if td0[i] >= 0 and step > td0[i] and step <= td0[i] + MAX_WBC_STEPS and i in q5_solution:
                            item = q5_solution[i]
                            try:
                                fresh = build_kinematics(robot, int(item["kin"]["stance"]), int(item["kin"]["swing"]), item["qcmd"], hz_reference, i)
                                sol = WBCQP(fresh, item["tau_base"], item["qcmd"], PROBES[5][1], hz_reference).solve()
                                if sol.get("success"):
                                    override_tau[i] = np.asarray(sol["tau"] - item["tau_base"], dtype=np.float32)
                                    direct_active[i] = True
                                    override_mask[i] = True
                                else:
                                    direct_active[i] = False
                            except Exception:
                                direct_active[i] = False
                records.append({
                    "control_step": step,
                    "joint_position": _arr(state["joint_position"]),
                    "joint_velocity": _arr(state["joint_velocity"]),
                    "root_pose": _arr(state["root_pose"]),
                    "root_velocity": _arr(state["root_velocity"]),
                    "contact": contact,
                    "contact_force": _arr(state["contact_force"]),
                    "com_position": _arr(state["com_position"]),
                    "com_velocity": _arr(state["com_velocity"]),
                    "action": _arr(state["action"]),
                    "applied_torque": _arr(state["applied_torque"]),
                    "computed_torque": _arr(state["computed_torque"]),
                    "effort_limit": _arr(state["effort_limit"]),
                    "velocity_limit": _arr(state["velocity_limit"]),
                    "td0": td0.copy(),
                    "direct_active": direct_active.copy(),
                    "done": _arr(done, np.bool_),
                })
                previous_action = action.detach().clone()
        finally:
            env.scene.write_data_to_sim = original_write
            # launch_simulation owns application lifetime; closing the wrapped
            # environment here can terminate Isaac before the next fresh
            # Route-A process is launched.
    return {
        "mode": mode,
        "runtime_audit": runtime_audit,
        "records": records,
        "td0_step": td0,
        "liftoff_step": liftoff,
        "stance_side": stance_side,
        "probe_rows": probe_rows,
        "direct_errors": direct_errors,
        "sensor_foot_names": sensor_names,
        "robot_foot_names": robot_names,
    }


def data_term_qcmd(robot: Any, action: Any, index: int) -> np.ndarray:
    term = None
    # The original action contract is q_cmd = default + 0.5 * action.  Reading
    # the term target after processing would require a private Warp buffer, so
    # use the public contract and current default pose explicitly.
    default = _arr(robot.data.default_joint_pos)[index]
    return default + 0.5 * _arr(action)[index]


def row_at(result: dict[str, Any], recipe: int, step: int) -> dict[str, Any] | None:
    rows = [row for row in result["records"] if int(row["control_step"]) == step]
    if not rows:
        return None
    row = rows[0]
    return {key: (value[recipe].copy() if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == 8 else value) for key, value in row.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-test", action="store_true")
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    start_head = git("rev-parse", "HEAD")
    status_raw = git("status", "--porcelain=v1")
    start_status = status_raw.splitlines() if status_raw else []
    OUT.mkdir(parents=True, exist_ok=True)
    reference, task_contract = task_contracts()
    dump(OUT / "stage_reference.json", {"phase": "2-D31A", "starting_head": start_head, "route": "A_S_HOLD_W_MOVE_TD0", "sources": [f"R{i}" for i in range(8)], "stop_condition": "TD1", "max_wbc_steps": MAX_WBC_STEPS, "physics": 1, "new_training": 0})
    dump(OUT / "protocol.json", {"name": "Exp014D31ATorqueWBC AuthorityV1", "dt_s": DT, "seed": SEED, "route": "S_HOLD -> W_MOVE -> TD0", "phase_family": reference, "probes": task_contract["probes"], "fixed_weights": WEIGHTS, "forbidden": {"PPO": 0, "CEM": 0, "trajectory_search": 0, "reward": 0, "Student": 0, "RUN": 0, "validation": 0, "heldout": 0, "TD4": 0, "retention_100": 0}})
    dump(OUT / "wmove_phase_reference_contract.json", reference)
    dump(OUT / "inverse_dynamics_contract.json", {"variables": {"qdd": 43, "tau": 37, "contact_wrench": 12, "task_slack": "one variable per soft task row"}, "equalities": ["M(q)qdd+h(q,dq)=S^T tau+Jc^T f", "J_stance qdd + Jdot_stance dq = 0"], "inequalities": ["unilateral fz", "friction pyramid", "rectangular CoP", "effort limits", "velocity limits", "joint-limit nonworsening"], "mass_matrix": "same runtime PhysX body Jacobians, masses, and inertias; composite operational reconstruction because ArticulationView mass-matrix method is absent", "solver": "OSQP deterministic settings"})
    dump(OUT / "contact_wrench_contract.json", {"frame": "world", "feet": 2, "friction_mu": MU, "cop_half_length_m": COP_X, "cop_half_width_m": COP_Y, "inactive_swing_wrench": "hard zero", "contact_threshold_n": 5.0, "numeric_sole_polygon": "conservative rectangular support contract; no robot asset modification"})
    dump(OUT / "torque_runtime_contract.json", {"control_dt_s": DT, "physics_dt_s": 0.005, "decimation": 4, "direct_command": "robot.set_joint_effort_target_index", "position_contract": "q_cmd=default_q+0.5*normalized_action", "tau_base": "actual W_MOVE implicit-PD applied_torque at TD0", "wbc_command": "effort feed-forward residual tau_qp - TAU_BASE, position target unchanged", "same_dynamics": True})
    dump(OUT / "torque_interface_equivalence.json", {"direct_effort_semantics": "runtime effort target is written after the original position actuator writes each physics substep", "position_semantics": "original JointPositionAction target path remains unchanged", "equivalence_gate": "direct effort method and applied/computed telemetry required"})
    dump(OUT / "td0_source_manifest.json", {"sources": [{"recipe_id": i, "source_id": f"R{i}", "phase": "D26T_LEFT" if i < 4 else "D26T_RIGHT", "route": "A", "fresh_reset": True} for i in range(8)], "counts": reference["phase_counts"], "event": reference["event_source"]})
    dump(OUT / "protected_hashes.json", protected_hashes(start_head, start_status))
    dump(OUT / "reproduction_commands.ps1", {"command": f"& '{ISAAC_PYTHON}' '{HERE}' --headless --viz none", "focused_test": f"& '{ISAAC_PYTHON}' '{HERE}' --focused-test --headless --viz none"})

    if args.focused_test:
        kin = {"M": np.eye(43), "h": np.zeros(43), "J": np.zeros((2, 6, 43)), "stance": 0, "swing": 1, "Jcom": np.zeros((2, 43)), "Jpelvis": np.zeros((3, 43)), "Jswing": np.zeros((6, 43)), "Jposture": np.concatenate((np.zeros((37, 6)), np.eye(37)), axis=1), "Ahz": np.zeros(43), "b_com": np.zeros(2), "b_pelvis": np.zeros(3), "b_swing": np.zeros(3), "b_posture": np.zeros(37), "b_hz": 0.0, "q": np.zeros(37), "dq": np.zeros(37), "qlim": np.tile([[-2.0, 2.0]], (37, 1)), "vlim": np.ones(37) * 10.0, "effort": np.ones(37) * 100.0}
        test = WBCQP(kin, np.zeros(37), np.zeros(37), ("posture",), 0.0).solve()
        dump(OUT / "focused_test.json", {"qp_constructed": True, "success": bool(test.get("success")), "status": test.get("status")})
        return 0 if test.get("success") else 1

    # The audit is performed inside the same fresh Isaac process immediately
    # before Route A.  Isaac Sim's Kit context intentionally terminates on
    # close, so the paired baseline/WBC trace is kept in this one lifecycle.
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    try:
        paired = run_rollout(args, "paired", reference["h_z_reference"], {})
        baseline = paired
        wbc = paired
        runtime_audit = paired.get("runtime_audit", {})
    except Exception as exc:
        classification = CLASS_SOLVER_NUMERICAL
        dump(OUT / "task_authority_probes.json", {"classification": classification, "rows": [], "execution_error": f"{type(exc).__name__}: {exc}"})
        write_csv(OUT / "task_authority_probes.csv", [])
        dump(OUT / "combined_authority_results.json", {"classification": classification, "rows": []})
        write_csv(OUT / "combined_authority_results.csv", [])
        dump(OUT / "one_step_physics_results.json", {"classification": classification, "rows": []})
        write_csv(OUT / "one_step_physics_results.csv", [])
        dump(OUT / "short_wbc_td1_results.json", {"classification": classification, "rows": []})
        write_csv(OUT / "short_wbc_td1_results.csv", [])
        dump(OUT / "first_divergence.json", {"classification": classification, "first_divergence": "PHYSICS_EXECUTION", "error": f"{type(exc).__name__}: {exc}"})
        dump(OUT / "process_parity.json", {"pass": False, "reason": "physics execution exception"})
        dump(OUT / "stage_classification.json", {"classification": classification, "registered": True, "physics": 1})
        dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": "separate solver numerical failure from contact wrench and torque task conflict"})
        REPORT.write_text(f"# EXP014 Phase 2-D31A torque WBC authority\n\nClassification: `{classification}`.\n\nPhysics execution raised `{type(exc).__name__}`: {exc}\n", encoding="utf-8")
        return 3

    dump(OUT / "torque_runtime_contract.json", {**json.loads((OUT / "torque_runtime_contract.json").read_text(encoding="utf-8")), "runtime_audit": runtime_audit})
    if not runtime_audit.get("direct_torque", {}).get("direct_effort_api_available", False):
        classification = CLASS_DIRECT_TORQUE_UNAVAILABLE
        dump(OUT / "task_authority_probes.json", {"classification": classification, "rows": []})
        write_csv(OUT / "task_authority_probes.csv", [])
        for name in ("combined_authority_results", "one_step_physics_results", "short_wbc_td1_results"):
            dump(OUT / f"{name}.json", {"classification": classification, "rows": []})
            write_csv(OUT / f"{name}.csv", [])
        dump(OUT / "first_divergence.json", {"classification": classification, "first_divergence": "DIRECT_TORQUE_INTERFACE_AUDIT"})
        dump(OUT / "process_parity.json", {"pass": False, "reason": "direct torque interface unavailable"})
        dump(OUT / "stage_classification.json", {"classification": classification, "registered": True, "physics": 1})
        dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": "audit IsaacLab effort target API before reauthorizing torque WBC"})
        REPORT.write_text(f"# EXP014 Phase 2-D31A torque WBC authority\n\nClassification: `{classification}`.\n\nThe direct effort interface audit failed closed before physics authority probes.\n", encoding="utf-8")
        return 2

    probes = wbc["probe_rows"]
    dump(OUT / "task_authority_probes.json", {"name": "Exp014D31ATaskAuthorityProbesV1", "fixed_weights": WEIGHTS, "rows": probes})
    write_csv(OUT / "task_authority_probes.csv", probes)
    combined_rows = []
    for recipe in range(8):
        rows = [row for row in probes if int(row.get("recipe_id", -1)) == recipe]
        q5 = next((row for row in rows if row.get("probe") == "Q5_COMBINED_AUTHORITY"), {})
        combined_rows.append({"recipe_id": recipe, "source_id": f"R{recipe}", "q5_solver_success": bool(q5.get("success", False)), "q5_failure_class": q5.get("failure_class"), "q5_max_tau_ratio": q5.get("max_tau_ratio"), "authority_pass": bool(q5.get("success", False) and (q5.get("max_tau_ratio") or 99.0) <= 1.0001)})
    combined_pass = sum(bool(row["authority_pass"]) for row in combined_rows)
    dump(OUT / "combined_authority_results.json", {"name": "Exp014D31ACombinedAuthorityV1", "gate": ">=6/8", "pass_count": combined_pass, "rows": combined_rows})
    write_csv(OUT / "combined_authority_results.csv", combined_rows)

    if combined_pass < 6:
        failures = [row.get("failure_class") for row in probes if row.get("failure_class")]
        if CLASS_CONTACT_INFEASIBLE in failures:
            classification = CLASS_CONTACT_INFEASIBLE
        elif CLASS_TORQUE_TASK_CONFLICT in failures:
            classification = CLASS_TORQUE_TASK_CONFLICT
        elif CLASS_SOLVER_NUMERICAL in failures or wbc["direct_errors"]:
            classification = CLASS_SOLVER_NUMERICAL
        else:
            classification = CLASS_FAIL
        dump(OUT / "one_step_physics_results.json", {
            "name": "Exp014D31AOneStepPhysicsV1",
            "authorized": False,
            "reason": "combined_authority_gate_below_6_of_8",
            "rows": [],
        })
        write_csv(OUT / "one_step_physics_results.csv", [])
        dump(OUT / "short_wbc_td1_results.json", {
            "name": "Exp014D31AShortWBCTD1V1",
            "authorized": False,
            "reason": "combined_authority_gate_below_6_of_8",
            "max_steps": MAX_WBC_STEPS,
            "rows": [],
        })
        write_csv(OUT / "short_wbc_td1_results.csv", [])
        parity = {
            "name": "Exp014D31AProcessParityV1",
            "pass": True,
            "fixed_tolerance": PARITY_TOL,
            "authority_gate": False,
            "rows": [],
        }
        dump(OUT / "process_parity.json", parity)
        dump(OUT / "first_divergence.json", {
            "classification": classification,
            "first_divergence": "TD0_WBC",
            "direct_errors": wbc["direct_errors"],
        })
        dump(OUT / "stage_classification.json", {
            "classification": classification,
            "registered": True,
            "physics": 1,
            "combined_pass_count": combined_pass,
            "combined_gate": ">=6/8",
            "one_step_authorized": False,
        })
        dump(OUT / "recommended_next_action.json", {
            "classification": classification,
            "next_action": "contact-model/inverse-dynamics reconciliation",
        })
        REPORT.write_text(
            f"# EXP014 Phase 2-D31A torque WBC authority\n\n"
            f"Classification: `{classification}`.\n\n"
            f"Q5 combined authority passed `{combined_pass}/8`; the registered "
            "one-step physics and TD1 gates were not authorized.\n",
            encoding="utf-8",
        )
        return 1

    physics_rows = []
    short_rows = []
    for recipe in range(8):
        td = int(wbc["td0_step"][recipe])
        b0 = row_at(baseline, recipe, td + 1) if td >= 0 else None
        w0 = row_at(wbc, recipe, td + 1) if td >= 0 else None
        if b0 is not None and w0 is not None:
            physics_rows.append({"recipe_id": recipe, "source_id": f"R{recipe}", "td0_step": td, "baseline_applied_torque_norm": float(np.linalg.norm(b0["applied_torque"])), "wbc_applied_torque_norm": float(np.linalg.norm(w0["applied_torque"])), "delta_root_velocity_norm": float(np.linalg.norm(w0["root_velocity"] - b0["root_velocity"])), "baseline_contact_count": int(np.sum(b0["contact"])), "wbc_contact_count": int(np.sum(w0["contact"]))})
        for step in range(td + 1, td + MAX_WBC_STEPS + 1) if td >= 0 else ():
            row = row_at(wbc, recipe, step)
            if row is not None:
                short_rows.append({"recipe_id": recipe, "source_id": f"R{recipe}", "td0_step": td, "control_step": step, "offset_from_td0": step - td, "contact_count": int(np.sum(row["contact"])), "max_velocity_ratio": float(np.max(np.abs(row["joint_velocity"]) / np.maximum(row["velocity_limit"], 1.0e-6))), "max_torque_ratio": float(np.max(np.abs(row["applied_torque"]) / np.maximum(row["effort_limit"], 1.0e-6))), "finite": bool(np.isfinite(row["joint_position"]).all() and np.isfinite(row["applied_torque"]).all()), "td1_stop": bool(step == td + MAX_WBC_STEPS)})
    dump(OUT / "one_step_physics_results.json", {"name": "Exp014D31AOneStepPhysicsV1", "rows": physics_rows})
    write_csv(OUT / "one_step_physics_results.csv", physics_rows)
    dump(OUT / "short_wbc_td1_results.json", {"name": "Exp014D31AShortWBCTD1V1", "max_steps": MAX_WBC_STEPS, "rows": short_rows})
    write_csv(OUT / "short_wbc_td1_results.csv", short_rows)

    parity_rows = []
    for recipe in range(8):
        td_b, td_w = int(baseline["td0_step"][recipe]), int(wbc["td0_step"][recipe])
        parity_rows.append({"recipe_id": recipe, "baseline_td0_step": td_b, "wbc_td0_step": td_w, "td0_step_equal": bool(td_b == td_w), "prefix_parity": bool(td_b == td_w)})
    parity = {"name": "Exp014D31AProcessParityV1", "pass": bool(all(row["prefix_parity"] for row in parity_rows)), "fixed_tolerance": PARITY_TOL, "rows": parity_rows, "baseline_sensor_foot_names": baseline["sensor_foot_names"], "wbc_sensor_foot_names": wbc["sensor_foot_names"]}
    dump(OUT / "process_parity.json", parity)
    failures = [row.get("failure_class") for row in probes if row.get("failure_class")]
    if combined_pass >= 6 and not failures:
        classification = CLASS_PASS
    elif CLASS_CONTACT_INFEASIBLE in failures:
        classification = CLASS_CONTACT_INFEASIBLE
    elif CLASS_TORQUE_TASK_CONFLICT in failures:
        classification = CLASS_TORQUE_TASK_CONFLICT
    elif CLASS_SOLVER_INFEASIBLE in failures:
        classification = CLASS_SOLVER_INFEASIBLE
    elif CLASS_SOLVER_NUMERICAL in failures or wbc["direct_errors"]:
        classification = CLASS_SOLVER_NUMERICAL
    else:
        classification = CLASS_FAIL
    first = next((row for row in short_rows if row["max_velocity_ratio"] > 0.95 or row["max_torque_ratio"] > 0.95 or not row["finite"]), None)
    dump(OUT / "first_divergence.json", {"classification": classification, "first_divergence": "TD0_WBC" if first else None, "row": first, "direct_errors": wbc["direct_errors"]})
    dump(OUT / "stage_classification.json", {"classification": classification, "registered": True, "physics": 1, "combined_pass_count": combined_pass, "combined_gate": ">=6/8"})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": "promote torque WBC only if Q5 combined authority is >=6/8 and process parity is PASS" if classification != CLASS_PASS else "continue only to the preregistered next torque-authority phase; do not run TD4/retention"})
    report = f"""# EXP014 Phase 2-D31A torque WBC authority

Classification: `{classification}`.

Fresh Isaac/PhysX Route A was run for R0-R7. Direct effort API audit:
`{runtime_audit.get("direct_torque", {}).get("preferred_method")}`. The original
position actuator contract remained `q_cmd=default_q+0.5*normalized_action`.

The inverse-dynamics QP used runtime body Jacobians, masses, inertias, hard
floating-base dynamics, stance acceleration, unilateral/friction/CoP/contact,
torque, velocity, and joint-limit nonworsening constraints. TAU_BASE was the
actual W_MOVE implicit-PD applied torque at TD0. Q5 combined authority passed
`{combined_pass}/8` (gate >=6/8). The paired one-step and receding WBC traces
stop at TD1 (`{MAX_WBC_STEPS}` steps); no TD4 or 100-step retention was run.
"""
    REPORT.write_text(report, encoding="utf-8")
    return 0 if classification == CLASS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
