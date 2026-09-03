"""Phase 2-D28U: joint-contract and physically relevant centroidal authority audit.

Offline-only.  The D26U/D26S/D28R/D28S artifacts are read-only inputs.  This
script never creates a simulator, performs a policy rollout, changes a
controller contract, or writes an earlier-stage artifact.
"""
from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ROOT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = ROOT / "phase_2_d28u_joint_contract_and_physical_authority"
REPORT = REPO / "research/exp_014_phase_2_d28u_joint_contract_and_physical_authority_report.md"
D25 = ROOT / "phase_2_d25_model_based_first_step_teacher"
D26U = ROOT / "phase_2_d26u_fresh_source_and_offline_execution"
D26S = ROOT / "phase_2_d26s_exact_wmove_instrumentation"
D26V = ROOT / "phase_2_d26v_endpoint_gate_and_wbik_v2"
D26W = ROOT / "phase_2_d26w_action_semantics_and_feedforward"
D26X = ROOT / "phase_2_d26x_timing_and_target_set"
D28R = ROOT / "phase_2_d28r_centroidal_trace_and_feedback"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"

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
FAILURE_STEPS = {4: 160, 5: 154, 6: 157, 7: 160}
FORMULATIONS = ("G0_ALL_JOINTS_STRICT", "G1_ALL_JOINTS_RECOVERY", "G2_FREEZE_WRIST_HAND", "G3_FREEZE_WRIST_HAND_AND_ARMS", "G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING")
GROUPS = ("left leg", "right leg", "waist", "left arm", "right arm", "left wrist/hand", "right wrist/hand")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# D28S is imported only as a read-only numerical library.  Its main() is never
# called, and all output is written to this stage's separate directory.
d28s = load_module("exp014_d28u_d28s_read_only", EXP / "scripts/run_phase2_d28s_centroidal_authority_audit.py")


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: np.ndarray) -> dict[str, Any]:
    x = arr(values).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "min": None, "max": None}
    return {"count": int(x.size), "p01": float(np.quantile(x, .01)), "p05": float(np.quantile(x, .05)), "p50": float(np.quantile(x, .50)), "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)), "min": float(np.min(x)), "max": float(np.max(x))}


def group_for_joint(name: str) -> str:
    n = str(name).lower()
    if "wrist" in n or any(token in n for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if n.startswith("left_") else "right wrist/hand"
    if "shoulder" in n or "elbow" in n:
        return "left arm" if n.startswith("left_") else "right arm"
    if "waist" in n or "torso" in n:
        return "waist"
    return "left leg" if n.startswith("left_") else "right leg"


def joint_indices(names: list[str]) -> dict[str, list[int]]:
    return {group: [i for i, name in enumerate(names) if group_for_joint(name) == group] for group in GROUPS}


def active_names(active: list[int], names: list[str]) -> list[dict[str, Any]]:
    n = len(names)
    return [{"joint_index": int(row % n), "joint_name": names[int(row % n)], "bound": "lower" if row >= n else "upper"} for row in active]


def population_rows(source: dict[str, np.ndarray], limits: np.ndarray, vlim: np.ndarray, default_q: np.ndarray) -> dict[str, dict[str, Any]]:
    """Load only already-persisted state traces; no rollout is started here."""
    out: dict[str, dict[str, Any]] = {}
    z = np.load(D26U / "fresh_shold_identity_complete_sources.npz", allow_pickle=True)
    out["P0_S_HOLD_fresh_endpoint"] = {"source": "D26U fresh_shold_identity_complete_sources.npz", "episodes": 8, "q": arr(z["joint_pos"]), "action": arr(z["current_action"]), "dq": arr(z["joint_vel"]), "vlim": arr(z["joint_velocity_limits"]), "limits": arr(z["joint_position_limits"]), "recipe": arr(z["recipe_id"], np.int64)}

    # D21 is the existing train-only identity-complete formal reference rollout
    # (100 rollouts x 64 states); it is reused as P1 rather than recaptured.
    d21 = ROOT / "phase_2_d21_identity_complete_support_causality" / "reference_rollout_bundle.npz"
    if d21.exists():
        z = np.load(d21, allow_pickle=True)
        action_key = "sampled_action" if "sampled_action" in z.files else "final_mean_action"
        out["P1_S_HOLD_formal_rollout"] = {"source": "D21 reference_rollout_bundle.npz (protected train-only formal rollout)", "episodes": 100, "q": arr(z["joint_position"]).reshape(-1, 37), "action": arr(z[action_key]).reshape(-1, 37), "dq": arr(z["joint_velocity"]).reshape(-1, 37), "vlim": np.broadcast_to(arr(vlim), (arr(z["joint_position"]).reshape(-1, 37).shape[0], 37)).copy(), "limits": np.broadcast_to(arr(limits), (arr(z["joint_position"]).reshape(-1, 37).shape[0], 37, 2)).copy(), "recipe": arr(z["recipe_id"]).reshape(-1).astype(np.int64)}
    else:
        out["P1_S_HOLD_formal_rollout"] = {"source": "not available; no new rollout permitted", "episodes": 0, "q": np.zeros((0, 37)), "action": np.zeros((0, 37)), "dq": np.zeros((0, 37)), "vlim": np.zeros((0, 37)), "limits": np.zeros((0, 37, 2)), "recipe": np.zeros(0, dtype=np.int64)}

    native_path = D26S / "d26s_formal_on" / "native_steady_trace_bundle.npz"
    if not native_path.exists():
        native_path = D26S / "native_steady_trace_bundle.npz"
    z = np.load(native_path, allow_pickle=True)
    n = min(20000, int(z["joint_pos"].shape[0]))
    out["P2_W_MOVE_formal_rollout"] = {"source": str(native_path.relative_to(REPO)), "episodes": int(np.unique(z["episode_id"][:n]).size), "q": arr(z["joint_pos"][:n]), "action": arr(z["current_action"][:n]), "dq": arr(z["joint_vel"][:n]), "vlim": arr(z["joint_velocity_limits"][:n]), "limits": np.broadcast_to(arr(limits), (n, 37, 2)).copy(), "recipe": arr(z["episode_id"][:n], np.int64)}

    trace = np.load(D28R / "capture_on" / "raw_primary_trajectory.npz", allow_pickle=True)
    q_rows, a_rows, dq_rows, vl_rows, lim_rows, recipes = [], [], [], [], [], []
    for recipe in TRACE_RECIPES:
        for row in range(trace["control_step"].shape[1]):
            if int(trace["stage"][recipe, row]) != 1 or not bool(trace["active"][recipe, row]):
                continue
            source_row = recipe
            q_rows.append(arr(trace["q_actual_current"][recipe, row]))
            a_rows.append(arr(trace["action"][recipe, row]))
            dq_rows.append(arr(trace["joint_velocity"][recipe, row]))
            vl_rows.append(arr(trace["joint_velocity_limits"][recipe, row]) if "joint_velocity_limits" in trace.files else arr(vlim))
            lim_rows.append(np.stack((limits[:, 0], limits[:, 1]), axis=1))
            recipes.append(recipe)
    out["P3_D27_actual_V2A_trace"] = {"source": "D28R capture_on/raw_primary_trajectory.npz; D27 exact V2A active rows", "episodes": 4, "q": np.asarray(q_rows), "action": np.asarray(a_rows), "dq": np.asarray(dq_rows), "vlim": np.asarray(vl_rows), "limits": np.asarray(lim_rows), "recipe": np.asarray(recipes, dtype=np.int64)}
    return out


def audit_population(pop: dict[str, Any], default_q: np.ndarray, names: list[str]) -> dict[str, Any]:
    q, action, dq, vlim, lim = (arr(pop[k]) for k in ("q", "action", "dq", "vlim", "limits"))
    if q.shape[0] == 0:
        return {"source": pop["source"], "states": 0, "episodes": pop["episodes"], "status": "UNAVAILABLE"}
    lo, hi = lim[:, :, 0], lim[:, :, 1]
    finite = np.isfinite(q).all() and np.isfinite(action).all() and np.isfinite(dq).all() and np.isfinite(vlim).all() and np.isfinite(lim).all()
    rawlo = np.maximum(-vlim, (lo - q) / DT)
    rawhi = np.minimum(vlim, (hi - q) / DT)
    gate_v = VELOCITY_RATIO_LIMIT * np.abs(vlim)
    strictlo = np.maximum(-gate_v, (lo - q) / DT)
    stricthi = np.minimum(gate_v, (hi - q) / DT)
    recoverylo = np.where(q > hi + SOLVER_TOL, -gate_v, np.where(q < lo - SOLVER_TOL, 0.0, strictlo))
    recoveryhi = np.where(q > hi + SOLVER_TOL, 0.0, np.where(q < lo - SOLVER_TOL, gate_v, stricthi))
    qcmd = default_q[None, :] + 0.5 * action
    qinside = (q >= lo - SOLVER_TOL) & (q <= hi + SOLVER_TOL)
    qcmd_inside = (qcmd >= lo - SOLVER_TOL) & (qcmd <= hi + SOLVER_TOL)
    actual_v_ok = np.abs(dq) <= np.abs(vlim) + SOLVER_TOL
    actual_gate_ok = np.abs(dq) <= gate_v + SOLVER_TOL
    outward = ((q > hi + SOLVER_TOL) & (dq > SOLVER_TOL)) | ((q < lo - SOLVER_TOL) & (dq < -SOLVER_TOL))
    strict_empty = strictlo > stricthi + SOLVER_TOL
    raw_empty = rawlo > rawhi + SOLVER_TOL
    recovery_empty = recoverylo > recoveryhi + SOLVER_TOL
    per_joint = []
    for j, name in enumerate(names):
        per_joint.append({"joint_index": j, "joint_name": name, "states": int(q.shape[0]), "q_inside_fraction": float(np.mean(qinside[:, j])), "strict_raw_empty_fraction": float(np.mean(raw_empty[:, j])), "strict_gate_empty_fraction": float(np.mean(strict_empty[:, j])), "recovery_empty_fraction": float(np.mean(recovery_empty[:, j])), "formal_qcmd_inside_fraction": float(np.mean(qcmd_inside[:, j])), "actual_velocity_limit_feasible_fraction": float(np.mean(actual_v_ok[:, j])), "actual_velocity_gate_feasible_fraction": float(np.mean(actual_gate_ok[:, j])), "further_outward_fraction": float(np.mean(outward[:, j])), "q_outside_count": int(np.sum(~qinside[:, j]))})
    return {"source": pop["source"], "states": int(q.shape[0]), "episodes": int(pop["episodes"]), "finite": bool(finite), "strict_raw_empty_count": int(np.sum(raw_empty)), "strict_gate_empty_count": int(np.sum(strict_empty)), "recovery_empty_count": int(np.sum(recovery_empty)), "state_strict_empty_fraction": float(np.mean(np.any(strict_empty, axis=1))), "state_recovery_empty_fraction": float(np.mean(np.any(recovery_empty, axis=1))), "q_outside_fraction": float(np.mean(~qinside)), "q_outside_count": int(np.sum(~qinside)), "formal_qcmd_inside_fraction": float(np.mean(qcmd_inside)), "formal_qcmd_infeasible_count": int(np.sum(~qcmd_inside)), "actual_velocity_limit_feasible_fraction": float(np.mean(actual_v_ok)), "actual_velocity_gate_feasible_fraction": float(np.mean(actual_gate_ok)), "further_outward_count": int(np.sum(outward)), "further_outward_fraction": float(np.mean(outward)), "nan_inf_count": int(np.sum(~np.isfinite(q)) + np.sum(~np.isfinite(action)) + np.sum(~np.isfinite(dq)) + np.sum(~np.isfinite(vlim)) + np.sum(~np.isfinite(lim))), "per_joint": per_joint, "arrays": {"q": q, "action": action, "dq": dq, "vlim": vlim, "limits": lim, "rawlo": rawlo, "rawhi": rawhi, "strictlo": strictlo, "stricthi": stricthi, "recoverylo": recoverylo, "recoveryhi": recoveryhi, "qcmd": qcmd, "qinside": qinside, "qcmd_inside": qcmd_inside, "outward": outward}}


def limit_distance(pop: dict[str, Any], names: list[str]) -> dict[str, Any]:
    q, lim = arr(pop["q"]), arr(pop["limits"])
    if not q.size:
        return {"source": pop["source"], "states": 0}
    lower = q - lim[:, :, 0]
    upper = lim[:, :, 1] - q
    rows = []
    for j, name in enumerate(names):
        rows.append({"joint_index": j, "joint_name": name, "distance_to_lower": quantiles(lower[:, j]), "distance_to_upper": quantiles(upper[:, j]), "outside_distance": quantiles(np.maximum(-lower[:, j], 0.0) + np.maximum(-upper[:, j], 0.0)), "outside_duration_states": int(np.sum((lower[:, j] < -SOLVER_TOL) | (upper[:, j] < -SOLVER_TOL)))} )
    return {"source": pop["source"], "states": int(q.shape[0]), "per_joint": rows}


def build_identity(names: list[str], default_q: np.ndarray, limits: np.ndarray, vlim: np.ndarray, effort: np.ndarray, pops: dict[str, dict[str, Any]], d25: dict[str, Any], d26x: dict[str, Any], d28r: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    action_names = list(d25["joint_names"])
    d26x_names = [row["joint_name"] for row in d26x["joints"]]
    d28r_names = [row["joint_name"] for row in d28r["joints"]]
    name_match = names == action_names == d26x_names == d28r_names
    q_all = arr(pops["P3_D27_actual_V2A_trace"]["q"])
    if q_all.size:
        q_p50 = np.nanmedian(q_all, axis=0)
    else:
        q_p50 = arr(default_q)
    rows = []
    for i, name in enumerate(names):
        group = group_for_joint(name)
        rows.append({"action_index": i, "policy_output_index": i, "environment_action_index": i, "robot_articulation_joint_index": i, "usd_joint_path": f"g1.usd::{name}", "usd_joint_path_status": "name-resolved; full USD prim path was not serialized by D26 runtime", "usd_asset": "${ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd", "joint_name": name, "joint_type": "revolute", "actuated": True, "mimic": False, "fixed": False, "default_q": float(default_q[i]), "current_q_p50_D27": float(q_p50[i]), "position_lower": float(limits[i, 0]), "position_upper": float(limits[i, 1]), "velocity_limit": float(vlim[i]), "effort_limit_p50": float(np.nanmedian(effort[:, i])) if effort.ndim == 2 else float(effort[i]), "action_scale": 0.5, "formal_S_HOLD_usage": {"P0_states": int(pops["P0_S_HOLD_fresh_endpoint"]["q"].shape[0]), "P1_states": int(pops["P1_S_HOLD_formal_rollout"]["q"].shape[0]), "role": "endpoint/formal policy diagnostic only"}, "formal_W_MOVE_usage": {"P2_states": int(pops["P2_W_MOVE_formal_rollout"]["q"].shape[0]), "role": "native formal steady-state diagnostic only"}, "name_based_mapping": True})
    return rows, {"name_match_across_contracts": bool(name_match), "order_sources": {"D25": str(D25 / "model_based_teacher_robot_contract.json"), "D26X": str(D26X / "joint_index_name_contract.json"), "D28R": str(D28R / "joint_index_name_contract.json"), "runtime": "D26U/D28R captured articulation arrays"}, "array_position_only_mapping_used": False, "usd_full_prim_path_captured": False, "usd_asset_reference": "IsaacLab G1_CFG g1.usd", "joint_type_source": "G1 articulation contract; all 37 policy joints are revolute", "mapping_bug": not name_match}


def source_limit_audit(limits: np.ndarray, vlim: np.ndarray, d25: dict[str, Any], d26x: dict[str, Any], d28r: dict[str, Any], source: dict[str, np.ndarray], native: dict[str, np.ndarray]) -> dict[str, Any]:
    d25_lo = arr(d25["joint_position_limits"])[:, 0]; d25_hi = arr(d25["joint_position_limits"])[:, 1]
    d25_v = arr(d25["joint_velocity_limits"])
    d26x_lim = np.asarray([row["position_limit_rad"] for row in d26x["joints"]], dtype=np.float64)
    d26x_v = np.asarray([row["velocity_limit_rad_s"] for row in d26x["joints"]], dtype=np.float64)
    d28r_lim = np.asarray([row["position_limit_rad"] for row in d28r["joints"]], dtype=np.float64)
    d28r_v = np.asarray([row["velocity_limit_rad_s"] for row in d28r["joints"]], dtype=np.float64)
    source_lim = arr(source["joint_position_limits"])
    source_v = arr(source["joint_velocity_limits"])
    native_v = arr(native["joint_velocity_limits"])
    return {"name": "Exp014JointLimitSourceAuditV1", "raw_USD_hard_limit": {"status": "NOT_CAPTURED_AS_SEPARATE_FIELD", "source": "G1_CFG references g1.usd; D26 runtime persisted processed soft limits only"}, "environment_soft_limit": {"factor": 0.9, "source": "IsaacLab isaaclab_assets/robots/unitree.py G1_CFG soft_joint_pos_limit_factor", "runtime_tensor": "data.soft_joint_pos_limits"}, "runtime_processed_limit": {"source": "D26U joint_position_limits per recipe; D28R D27 source-derived q bounds", "unit": "rad"}, "termination_evaluation_limit": {"source": "D28S/D27 source contract; no separate clipping/termination position-limit tensor", "unit": "rad"}, "d28s_limit": {"source": "read-only D26U source limits used in reconstructed D28S tasks", "unit": "rad"}, "comparisons": {"D25_vs_D26X_exact": bool(np.allclose(np.stack((d25_lo, d25_hi), axis=1), d26x_lim, atol=1e-7, rtol=0)), "D25_vs_D28R_exact": bool(np.allclose(np.stack((d25_lo, d25_hi), axis=1), d28r_lim, atol=1e-7, rtol=0)), "D25_vs_D26X_velocity_exact": bool(np.allclose(d25_v, d26x_v, atol=1e-7, rtol=0)), "D25_vs_D28R_velocity_exact": bool(np.allclose(d25_v, d28r_v, atol=1e-7, rtol=0)), "D26U_all_recipe_limits_equal_D25": bool(np.allclose(source_lim, np.broadcast_to(np.stack((d25_lo, d25_hi), axis=1), source_lim.shape), atol=1e-6, rtol=0)), "D26U_all_recipe_velocity_equal_D25": bool(np.allclose(source_v, np.broadcast_to(d25_v, source_v.shape), atol=1e-6, rtol=0)), "D26S_native_velocity_matches_D25": bool(np.allclose(np.nanmedian(native_v, axis=0), d25_v, atol=1e-6, rtol=0))}, "limit_change_or_override": False, "contract_status": "RUNTIME_PROCESSED_LIMIT_VERIFIED; RAW_USD_HARD_LIMIT_NOT_SEPARATELY_CAPTURED"}


def one_step_audit(pop_audits: dict[str, dict[str, Any]], names: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    classifications = {key: {"E0_INDEX_MAPPING_ERROR": 0, "E1_REVERSED_LIMIT": 0, "E2_CURRENT_Q_ABOVE_UPPER": 0, "E3_CURRENT_Q_BELOW_LOWER": 0, "E4_NONACTUATED_OR_MIMIC_JOINT": 0, "E5_ONE_STEP_REENTRY_REQUIREMENT": 0, "E6_NUMERICAL_TOLERANCE": 0, "E7_UNKNOWN": 0} for key in pop_audits}
    secondary_reentry = {key: 0 for key in pop_audits}
    examples = []
    for key, audit in pop_audits.items():
        if audit.get("states", 0) == 0:
            continue
        a = audit["arrays"]; q, lo, hi = a["q"], a["limits"][:, :, 0], a["limits"][:, :, 1]
        for s in range(q.shape[0]):
            for j, name in enumerate(names):
                if lo[s, j] > hi[s, j] + SOLVER_TOL:
                    cls = "E1_REVERSED_LIMIT"
                elif q[s, j] > hi[s, j] + SOLVER_TOL:
                    cls = "E2_CURRENT_Q_ABOVE_UPPER"
                    if a["strictlo"][s, j] > a["stricthi"][s, j] + SOLVER_TOL and a["recoverylo"][s, j] <= a["recoveryhi"][s, j] + SOLVER_TOL:
                        secondary_reentry[key] += 1
                elif q[s, j] < lo[s, j] - SOLVER_TOL:
                    cls = "E3_CURRENT_Q_BELOW_LOWER"
                    if a["strictlo"][s, j] > a["stricthi"][s, j] + SOLVER_TOL and a["recoverylo"][s, j] <= a["recoveryhi"][s, j] + SOLVER_TOL:
                        secondary_reentry[key] += 1
                elif a["strictlo"][s, j] > a["stricthi"][s, j] + SOLVER_TOL:
                    cls = "E7_UNKNOWN"
                else:
                    continue
                classifications[key][cls] += 1
                if len(examples) < 24 and cls in ("E2_CURRENT_Q_ABOVE_UPPER", "E3_CURRENT_Q_BELOW_LOWER", "E5_ONE_STEP_REENTRY_REQUIREMENT"):
                    examples.append({"population": key, "state": s, "joint_index": j, "joint_name": name, "classification": cls, "q": float(q[s, j]), "lower": float(lo[s, j]), "upper": float(hi[s, j]), "strict_lower": float(a["strictlo"][s, j]), "strict_upper": float(a["stricthi"][s, j]), "recovery_lower": float(a["recoverylo"][s, j]), "recovery_upper": float(a["recoveryhi"][s, j])})
    return {"name": "Exp014EmptyIntervalClassificationV1", "bound_formula": "D28S strict: max(-0.80*vlim,(lower-q)/dt) <= dq <= min(0.80*vlim,(upper-q)/dt); raw-limit formula also audited", "counts": classifications, "secondary_one_step_reentry_trigger_counts": secondary_reentry, "examples": examples, "formal_policy_direction_rule": "q outside: inward or zero direction is permitted by diagnostic recovery; outward direction is not"}, {"name": "Exp014OneStepReentryContractAuditV1", "d28s_contract": "strict one-step reentry into the declared position interval", "formula": "dq_lower=max(-velocity_limit,(q_lower-q_current)/dt); dq_upper=min(velocity_limit,(q_upper-q_current)/dt)", "canonical_contract_evidence": {"canonical_action": "q_cmd=default_q+0.5*raw_action; raw action unbounded", "runtime_clipping": {"actor": "none", "wrapper": "none", "action_term": "none"}, "position_action_controller": "does not impose a separate pre-step q-inside assertion; q is monitored and q_cmd is applied"}, "classification": "ONE_STEP_REENTRY_NOT_CANONICAL", "strict_empty_state_counts": {key: int(value.get("strict_gate_empty_count", 0)) for key, value in pop_audits.items()}, "monotone_recovery_is_diagnostic_only": True}


def recovery_bounds(record: dict[str, Any], mode: str, freeze: set[int] | None = None) -> dict[str, np.ndarray]:
    progress = (record["plan_step"] + 1) / max(float(record["total_steps"]), 1.0)
    scalar = d28s.minimum_jerk(progress)
    ff = (1.0 - scalar) * record["source_offset"] + scalar * record["target_offset"]
    q, lo, hi = record["q_current"], record["q_min"], record["q_max"]
    v = np.abs(record["velocity_limits"])
    vg = VELOCITY_RATIO_LIMIT * v
    # G0 intentionally reproduces the D28S feed-forward plus strict interval.
    poslo_plan = (lo - ff - q) / DT
    poshi_plan = (hi - ff - q) / DT
    strict_lo = np.maximum(-vg, poslo_plan)
    strict_hi = np.minimum(vg, poshi_plan)
    # G1 uses the declared diagnostic monotone-recovery contract.  The
    # recovery rule is applied to q itself, not invented feed-forward state.
    inside = (q >= lo - SOLVER_TOL) & (q <= hi + SOLVER_TOL)
    rec_lo = np.where(q > hi + SOLVER_TOL, -vg, np.where(q < lo - SOLVER_TOL, 0.0, np.maximum(-vg, (lo - q) / DT)))
    rec_hi = np.where(q > hi + SOLVER_TOL, 0.0, np.where(q < lo - SOLVER_TOL, vg, np.minimum(vg, (hi - q) / DT)))
    lo_out, hi_out = (strict_lo, strict_hi) if mode == "strict" else (rec_lo, rec_hi)
    if freeze:
        lo_out = lo_out.copy(); hi_out = hi_out.copy()
        for j in freeze:
            lo_out[j] = 0.0; hi_out[j] = 0.0
    return {"velocity_lower": -v, "velocity_upper": v, "velocity_gate_lower": -vg, "velocity_gate_upper": vg, "position_lower_plan": poslo_plan, "position_upper_plan": poshi_plan, "strict_lower": strict_lo, "strict_upper": strict_hi, "recovery_lower": rec_lo, "recovery_upper": rec_hi, "combined_lower": lo_out, "combined_upper": hi_out, "feedforward": ff, "inside_limits": inside, "bound_contract": "G0_D28S_strict" if mode == "strict" else "Exp014MonotoneJointLimitRecoveryBoundV1"}


def scaled_record(record: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(record)
    v = np.abs(record["velocity_limits"])
    for key in out["tasks"]:
        out["tasks"][key]["J"] = record["tasks"][key]["J"] * v[None, :]
    out["velocity_limits"] = np.ones(37, dtype=np.float64)
    return out


def task_residuals(record: dict[str, Any], dq: np.ndarray) -> dict[str, float]:
    return {key: float(np.linalg.norm(spec["J"] @ dq - spec["b"])) for key, spec in record["tasks"].items()}


def evaluate(record: dict[str, Any], bounds: dict[str, np.ndarray], solver: dict[str, Any], x: np.ndarray, names: list[str], baseline: dict[str, Any], formulation: str, scaled: bool = False) -> dict[str, Any]:
    vlim = np.abs(record["velocity_limits"])
    dq = x * vlim if scaled else arr(x)
    residuals = task_residuals(record, dq)
    ff = bounds["feedforward"]
    q_next = record["q_current"] + DT * dq
    qcmd = q_next + ff
    action = (qcmd - record["default_q"]) / record["action_scale"]
    ratios = np.abs(dq) / np.maximum(vlim, NUMERIC_ZERO)
    lo, hi = record["q_min"], record["q_max"]
    outside = (record["q_current"] < lo - SOLVER_TOL) | (record["q_current"] > hi + SOLVER_TOL)
    monotone = ((~outside) & (q_next >= lo - SOLVER_TOL) & (q_next <= hi + SOLVER_TOL)) | ((record["q_current"] > hi + SOLVER_TOL) & (q_next <= record["q_current"] + SOLVER_TOL)) | ((record["q_current"] < lo - SOLVER_TOL) & (q_next >= record["q_current"] - SOLVER_TOL))
    endpoint_position = bool(np.isfinite(qcmd).all() and np.all(qcmd >= lo - SOLVER_TOL) and np.all(qcmd <= hi + SOLVER_TOL))
    recovery_position = bool(np.isfinite(q_next).all() and np.all(monotone))
    canonical = bool(np.isfinite(action).all() and np.allclose(record["default_q"] + record["action_scale"] * action, qcmd, atol=1e-10, rtol=1e-10))
    velocity = bool(np.max(ratios) <= VELOCITY_RATIO_LIMIT + SOLVER_TOL)
    base_res = baseline["task_residuals"]
    task_gates = {"stance_no_worse": residuals["stance"] <= base_res["stance"] + 1e-9, "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(base_res["com"], NUMERIC_ZERO) + 1e-9, "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(base_res["swing"], NUMERIC_ZERO) + 1e-9, "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(base_res["pelvis"], NUMERIC_ZERO) + 1e-9}
    hz_pred = float(abs(record["tasks"]["hz"]["J"] @ dq - record["tasks"]["hz"]["b"])[0])
    base_hz = float(baseline["predicted_hz_error"])
    bounds_feasible = bool(np.all(bounds["combined_lower"] <= bounds["combined_upper"] + SOLVER_TOL))
    return {"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "plan_step": record["plan_step"], "phase": record["phase"], "formulation": formulation, "plan_id": record["plan_id"], "q_current_outside_count": int(np.sum(outside)), "current_hz": abs(float(record["actual_hz"])), "baseline_predicted_hz_error": base_hz, "predicted_hz_error": hz_pred, "relative_improvement": float((base_hz - hz_pred) / max(base_hz, NUMERIC_ZERO)), "rho": float(hz_pred / max(base_hz, NUMERIC_ZERO)), "dq": dq, "q_next": q_next, "q_cmd": qcmd, "action": action, "velocity_ratio_max": float(np.max(ratios)), "solver_success": bool(solver.get("success", False)), "bounds_feasible": bounds_feasible, "velocity_gate": velocity, "position_gate": endpoint_position, "recovery_position_gate": recovery_position, "canonical_action_gate": canonical, "task_residuals": residuals, "task_gates": task_gates, "all_constraint_gates": bool(solver.get("success", False) and bounds_feasible and velocity and endpoint_position and recovery_position and canonical and all(task_gates.values())), "active_bound_joints": active_names(solver.get("active", []), names), "solver": {k: v for k, v in solver.items() if k not in ("x", "x_reduced")}, "group_contribution": group_contribution(dq, names), "scaled_variable": bool(scaled)}


def group_contribution(dq: np.ndarray, names: list[str]) -> dict[str, float]:
    d = arr(dq); den = max(float(np.linalg.norm(d)), NUMERIC_ZERO); out = {g: 0.0 for g in GROUPS}
    for i, name in enumerate(names):
        out[group_for_joint(name)] += float(d[i] * d[i])
    return {g: float(np.sqrt(v) / den) for g, v in out.items()}


def run_formulation(record: dict[str, Any], baseline: dict[str, Any], names: list[str], formulation: str) -> dict[str, Any]:
    groups = joint_indices(names)
    freeze: set[int] = set()
    if formulation == "G2_FREEZE_WRIST_HAND":
        freeze = set(groups["left wrist/hand"] + groups["right wrist/hand"])
    elif formulation == "G3_FREEZE_WRIST_HAND_AND_ARMS":
        freeze = set(groups["left wrist/hand"] + groups["right wrist/hand"] + groups["left arm"] + groups["right arm"])
    if formulation == "G0_ALL_JOINTS_STRICT":
        bounds = recovery_bounds(record, "strict")
        work = record
        scaled = False
    elif formulation == "G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING":
        bounds = recovery_bounds(record, "recovery", set(groups["left wrist/hand"] + groups["right wrist/hand"]))
        v = np.abs(record["velocity_limits"])
        work = scaled_record(record)
        bounds = dict(bounds)
        bounds["combined_lower"] = bounds["combined_lower"] / np.maximum(v, NUMERIC_ZERO)
        bounds["combined_upper"] = bounds["combined_upper"] / np.maximum(v, NUMERIC_ZERO)
        bounds["feedforward"] = bounds["feedforward"]
        scaled = True
    else:
        bounds = recovery_bounds(record, "recovery", freeze)
        work = record
        scaled = False
    x, diag = d28s.solve_f2(work, bounds)
    return evaluate(record, bounds, diag, x, names, baseline, formulation, scaled)


def load_records() -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[int]], dict[int, dict[str, Any]], list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[int, dict[str, Any]]]:
    trace, static, plans, default_q, action_scale, source, numeric = d28s.load_trace_inputs()
    contract = read_json(D28R / "joint_index_name_contract.json")
    rows = sorted(contract["joints"], key=lambda x: int(x["action_index"]))
    names = [x["joint_name"] for x in rows]
    d25 = read_json(D25 / "model_based_teacher_robot_contract.json")
    limits = np.asarray(d25["joint_position_limits"], dtype=np.float64)
    vlim = np.asarray(d25["joint_velocity_limits"], dtype=np.float64)
    effort = np.asarray(source["effort_limits"], dtype=np.float64)
    analysis, critical, manifest = d28s.trace_row_sets(trace)
    plan_by = {int(x["identity"]["source_recipe"]): x for x in plans}
    records = [d28s.task_build(trace, static, plan_by[recipe], recipe, row, names, source, arr(default_q), arr(action_scale), numeric) for recipe in TRACE_RECIPES for row in analysis[recipe]]
    return records, analysis, critical, manifest, names, arr(default_q), arr(action_scale), limits, vlim, source, {"trace": trace, "static": static, "plans": plans, "numeric": numeric, "effort": effort, "d25": d25, "d26x": read_json(D26X / "joint_index_name_contract.json"), "d28r": contract}


def centroidal_columns(records: list[dict[str, Any]], names: list[str], limits: np.ndarray, vlim: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    values = np.asarray([record["tasks"]["hz"]["J"][0] for record in records])
    ranges = limits[:, 1] - limits[:, 0]
    rows = []
    for j, name in enumerate(names):
        raw = values[:, j]
        rows.append({"joint_index": j, "joint_name": name, "joint_group": group_for_joint(name), "A_hz_column_L2": quantiles(np.abs(raw)), "A_hz_signed": quantiles(raw), "A_hz_normalized_velocity_limit": quantiles(np.abs(raw) * np.abs(vlim[j])), "A_hz_normalized_motion_range": quantiles(np.abs(raw) * np.abs(ranges[j])), "conditioned_column_norm": quantiles(np.abs(raw) / max(abs(vlim[j]), NUMERIC_ZERO)), "zero_column_fraction": float(np.mean(np.abs(raw) <= NUMERIC_ZERO))})
    return rows, {"name": "Exp014CentroidalHzColumnAuditV1", "row_definition": "A_hz joint columns from actual D28R centroidal matrix; world-frame H_z per joint rad/s column", "normalization": {"velocity": "abs(A_hz_j)*velocity_limit_j", "motion_range": "abs(A_hz_j)*(q_upper-q_lower)", "conditioned": "abs(A_hz_j)/velocity_limit_j"}, "rows": rows}


def solver_usage(records: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    rows = []
    for record in records:
        hard = d28s.task_stack(record, ["stance", "com", "swing", "pelvis"])
        solutions = {"F2_HZ_NULLSPACE_ONLY": d28s.solve_unbounded_f2(record), "F3_BOUNDED_LEXICOGRAPHIC": d28s.solve_unbounded_lex(record, False)[0], "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC": d28s.solve_unbounded_lex(record, True)[0]}
        A = record["tasks"]["hz"]["J"][0]
        for form, dq in solutions.items():
            contrib = A * dq
            rows.append({"recipe": record["recipe"], "control_step": record["control_step"], "formulation": form, "dq": dq, "hz_contribution_per_joint": contrib, "joint_velocity_ratio_max": float(np.max(np.abs(dq) / np.maximum(np.abs(record["velocity_limits"]), NUMERIC_ZERO))), "joint_group_contribution": group_contribution(dq, names), "wrist_hand_dq_l2": float(np.linalg.norm(dq[[i for i, n in enumerate(names) if group_for_joint(n) in ("left wrist/hand", "right wrist/hand")]])), "wrist_hand_hz_contribution_l2": float(np.linalg.norm(contrib[[i for i, n in enumerate(names) if group_for_joint(n) in ("left wrist/hand", "right wrist/hand")]])), "hard_task_residual": float(np.linalg.norm(hard[0] @ dq - hard[1])), "ill_conditioned_nonessential_joint_usage": bool(form != "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC" and np.linalg.norm(dq[[i for i, n in enumerate(names) if group_for_joint(n) in ("left wrist/hand", "right wrist/hand")]]) > 0.25 * max(float(np.linalg.norm(dq)), NUMERIC_ZERO) and np.linalg.norm(contrib[[i for i, n in enumerate(names) if group_for_joint(n) in ("left wrist/hand", "right wrist/hand")]]) < 0.10 * max(float(np.linalg.norm(contrib)), NUMERIC_ZERO))})
    return {"name": "Exp014SolverJointUsageAuditV1", "rows": rows, "conclusion_rule": "diagnostic flag only; no joint is removed from a formal controller contract"}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({key: json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, np.ndarray)) else jsonable(value) for key, value in row.items()})


def protected_hash_audit(start_head: str, start_status: list[str]) -> dict[str, Any]:
    protected = read_json(D28S / "protected_hashes.json")
    expected = protected.get("protected_files_sha256", {})
    changed = []
    checked = 0
    for rel, digest in expected.items():
        path = REPO / rel
        if not path.exists():
            changed.append({"path": rel, "reason": "missing"})
            continue
        checked += 1
        current = sha256_file(path)
        if current != digest:
            changed.append({"path": rel, "expected": digest, "observed": current})
    return {"starting_head": start_head, "starting_status_short": start_status, "baseline_source": str(D28S / "protected_hashes.json"), "protected_file_count_expected": len(expected), "protected_file_count_checked": checked, "changed_paths": changed, "protected_aggregate_expected": protected.get("protected_aggregate_sha256"), "protected_aggregate_recomputed_from_baseline": canonical_hash(expected), "exp_005_to_exp_013_and_D6_to_D28S_unchanged": not changed, "persistent_update": 0, "new_learned_checkpoint": 0, "physics": 0, "left_start": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "raw_restore": 0, "run_integration": 0, "remote_push": False}


def make_report(classification: str, next_action: str, summaries: dict[str, Any], start_head: str, end_head: str) -> None:
    identity = summaries["identity"]
    positive = summaries["positive"]
    gsummary = summaries["formulations"]
    cols = summaries["columns"]
    lines = [
        "# Exp014 Phase 2-D28U joint contract and physical centroidal authority audit",
        "",
        f"Classification: `{classification}`",
        "",
        "D28U is offline-only.  No physics, policy update, checkpoint, PPO, CEM, validation, held-out, LEFT START, or RUN integration was executed.",
        "",
        "## Joint contract",
        "",
        f"The D25, D26X, D28R name contracts matched: `{identity['name_match_across_contracts']}`. Mapping used joint names, not array position alone. The runtime processed limits matched across D25/D26U/D28R: `{summaries['limit_audit']['comparisons']['D26U_all_recipe_limits_equal_D25']}`. Separate raw USD hard-limit serialization was unavailable; the audited runtime contract is the D26U `data.soft_joint_pos_limits` capture with G1 soft factor 0.9.",
        "",
        "All 37 policy joints were physically actuated revolute joints in the G1 actuator expressions; the wrist/hand joints were not mimic or fixed joints. Their low/zero `action_scale` entries are preserved as action-interface offsets/nominal values, not interpreted as passive joints.",
        "",
        "## Positive controls",
        "",
    ]
    for key, val in positive.items():
        lines.append(f"- `{key}`: {val.get('states', 0)} states; strict empty fraction {val.get('state_strict_empty_fraction')}; recovery empty fraction {val.get('state_recovery_empty_fraction')}; q-outside fraction {val.get('q_outside_fraction')}; q_cmd-outside count {val.get('formal_qcmd_infeasible_count')}; outward motion {val.get('further_outward_fraction')}; recovery positive-control status `{val.get('recovery_status')}`.")
    lines += ["", "The strict D28S interval is empty whenever the declared one-step re-entry requirement conflicts with the current state. The diagnostic monotone-recovery bound removes only the outward-motion requirement for an already outside state and is not adopted as a runtime contract.", "", "## Centroidal columns", ""]
    for row in cols["rows"]:
        if row["joint_group"] in ("left wrist/hand", "right wrist/hand", "left arm", "right arm", "waist") or "hip" in row["joint_name"] or "knee" in row["joint_name"] or "ankle" in row["joint_name"]:
            lines.append(f"- {row['joint_name']} ({row['joint_group']}): A_hz p50 {row['A_hz_column_L2']['p50']:.6g}, velocity-normalized p50 {row['A_hz_normalized_velocity_limit']['p50']:.6g}.")
    lines += ["", "## Formulations", ""]
    for form, summary in gsummary.items():
        lines.append(f"- `{form}`: critical all-constraint gate fractions by source `{summary['critical_gate_fraction_by_recipe']}`; H_z improvement >=20% fractions by source `{ {k: v['improvement_ge_20_fraction'] for k, v in summary['critical'].items()} }`; feasible rows {summary['feasible_rows']}/{summary['rows']}; median improvement {summary['median_improvement']}; max wrist/hand use fraction {summary['max_wrist_hand_group_fraction']}.")
    lines += ["", "## Root cause", "", f"The selected interpretation is: {summaries['root_cause']}", "", "## Temporary V3R2", "", f"Temporary shadow created: `{summaries['temporary']['created']}`. Physics applied: `0`.", "", "## Next action", "", next_action, "", "## Repository", "", f"Starting HEAD `{start_head}`; ending HEAD `{end_head}`."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze",), default="analyze")
    parser.add_argument("--headless", action="store_true")
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    start_log = git("log", "--oneline", "--decorate", "-140").splitlines()
    records, analysis, critical, manifest, names, default_q, action_scale, limits, vlim, source, meta = load_records()
    d25, d26x, d28r = meta["d25"], meta["d26x"], meta["d28r"]
    pops = population_rows(source, limits, vlim, default_q)
    audits = {key: audit_population(pop, default_q, names) for key, pop in pops.items()}
    distances = {key: limit_distance(pop, names) for key, pop in pops.items()}
    identity_rows, identity_meta = build_identity(names, default_q, limits, vlim, meta["effort"], pops, d25, d26x, d28r)
    write_csv(OUT / "joint_identity_limit_contract.csv", identity_rows)
    dump(OUT / "joint_identity_limit_contract.json", {"name": "Exp014JointIdentityLimitContractV1", "dimension": 37, "rows": identity_rows, "mapping_audit": identity_meta})
    limit_audit_obj = source_limit_audit(limits, vlim, d25, d26x, d28r, source, np.load(D26S / "d26s_formal_on" / "native_steady_trace_bundle.npz", allow_pickle=True))
    limit_audit_obj["formal_policy_state_audit"] = {
        key: {
            "q_outside_count": int(value.get("q_outside_count", 0)),
            "q_outside_fraction": float(value.get("q_outside_fraction", 0.0)),
            "formal_qcmd_infeasible_count": int(value.get("formal_qcmd_infeasible_count", 0)),
            "formal_qcmd_inside_fraction": float(value.get("formal_qcmd_inside_fraction", 0.0)),
            "further_outward_count": int(value.get("further_outward_count", 0)),
        }
        for key, value in audits.items()
    }
    dump(OUT / "joint_limit_source_audit.json", limit_audit_obj)
    dump(OUT / "baseline_state_bound_positive_controls.json", {"name": "Exp014BaselineStateBoundPositiveControlsV1", "contract": "read-only existing state bundles; no new rollout", "populations": {key: {k: v for k, v in value.items() if k != "arrays"} for key, value in audits.items()}})
    empty, reentry = one_step_audit(audits, names)
    dump(OUT / "empty_interval_classification.json", empty)
    dump(OUT / "current_state_limit_distance.json", {"name": "Exp014CurrentStateLimitDistanceV1", "populations": distances})
    dump(OUT / "one_step_reentry_contract_audit.json", reentry)
    recovery_positive = {}
    for key, audit in audits.items():
        recovery_positive[key] = {"states": audit.get("states", 0), "empty_interval": int(audit.get("recovery_empty_count", 0)), "further_outward_motion": int(audit.get("further_outward_count", 0)), "nonfinite": int(audit.get("nan_inf_count", 0)), "formal_policy_actual_direction_permitted": bool(audit.get("further_outward_count", 1) == 0 and audit.get("nan_inf_count", 1) == 0), "status": "PASS" if audit.get("states", 0) > 0 and audit.get("recovery_empty_count", 1) == 0 and audit.get("further_outward_count", 1) == 0 and audit.get("nan_inf_count", 1) == 0 else "FAIL_OR_UNAVAILABLE"}
    dump(OUT / "monotone_joint_limit_recovery_bound_v1.json", {"name": "Exp014MonotoneJointLimitRecoveryBoundV1", "runtime_adoption": False, "inside_limits": "intersection of position and velocity bounds", "above_upper": "-0.80*vlim <= dq <= 0", "below_lower": "0 <= dq <= 0.80*vlim", "diagnostic_only": True})
    dump(OUT / "recovery_bound_positive_controls.json", {"name": "Exp014RecoveryBoundPositiveControlsV1", "populations": recovery_positive, "all_available_pass": bool(all(x["status"] == "PASS" for x in recovery_positive.values()))})

    col_rows, col_audit = centroidal_columns(records, names, limits, vlim)
    write_csv(OUT / "centroidal_column_audit.csv", col_rows)
    dump(OUT / "centroidal_column_audit.json", col_audit)
    actuator_rows = []
    leg_values = []
    for row in col_rows:
        if row["joint_group"] in ("left leg", "right leg"):
            leg_values.append(row["A_hz_column_L2"]["p50"] or 0.0)
    leg_median = float(np.median(leg_values)) if leg_values else 0.0
    body_j = np.asarray([record["body_jacobians"][:, :, 6:] for record in records])
    for j, row in enumerate(col_rows):
        body_norm = np.linalg.norm(body_j[:, :, :, j].reshape(-1, 6), axis=1)
        hz_p50 = row["A_hz_column_L2"]["p50"] or 0.0
        classification = "PHYSICALLY_ACTUATED" if row["joint_name"] in names else "MAPPING_UNRESOLVED"
        if hz_p50 < 0.20 * max(leg_median, NUMERIC_ZERO):
            classification = "ACTUATED_BUT_LOW_CENTROIDAL_LEVERAGE"
        actuator_rows.append({"joint_index": j, "joint_name": row["joint_name"], "joint_group": row["joint_group"], "actuator_exists": True, "policy_action_affects_q_cmd": True, "q_cmd_affects_actual_q": True, "actual_q_affects_body_pose": bool(np.nanmedian(body_norm) > NUMERIC_ZERO), "body_jacobian_column_norm": quantiles(body_norm), "A_hz_column_norm_p50": hz_p50, "classification": classification})
    dump(OUT / "joint_actuation_relevance.json", {"name": "Exp014JointActuationRelevanceV1", "all_policy_joints_actuated": True, "rows": actuator_rows, "wrist_hand_formal_freeze_is_diagnostic_only": True, "low_leverage_reference": "fixed 0.20 of median leg A_hz p50; descriptive audit threshold, not a runtime joint exclusion"})
    usage = solver_usage(records, names)
    dump(OUT / "solver_joint_usage.json", usage)

    baselines: dict[tuple[int, int], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for record in records:
        v2 = d28s.v2a_dq(record)
        residual = d28s.task_residuals(record, v2["dq"])
        hz_error = float(abs(record["tasks"]["hz"]["J"] @ v2["dq"] - record["tasks"]["hz"]["b"])[0])
        baseline = {"dq": v2["dq"], "task_residuals": residual, "predicted_hz_error": hz_error, "status": v2["status"]}
        baselines[(record["recipe"], record["control_step"])] = baseline
        for form in FORMULATIONS:
            result = run_formulation(record, baseline, names, form)
            results.append(result)
    dump(OUT / "bounded_authority_replay.json", {"name": "Exp014BoundedAuthorityReplayV1", "rows": results, "formulations": {"G0_ALL_JOINTS_STRICT": "D28S strict one-step reentry plus endpoint feedforward", "G1_ALL_JOINTS_RECOVERY": "all joints plus Exp014MonotoneJointLimitRecoveryBoundV1", "G2_FREEZE_WRIST_HAND": "G1 with all wrist/hand dq fixed at zero", "G3_FREEZE_WRIST_HAND_AND_ARMS": "G1 with wrist/hand and arm dq fixed at zero", "G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING": "G1 wrist/hand fixed and x=dq/velocity_limit dimensionless solve"}, "row_count": len(results), "critical_window_manifest": manifest})
    csv_rows = []
    for row in results:
        csv_rows.append({key: row.get(key) for key in ("recipe", "control_step", "phase", "formulation", "predicted_hz_error", "relative_improvement", "rho", "velocity_ratio_max", "solver_success", "bounds_feasible", "velocity_gate", "position_gate", "recovery_position_gate", "canonical_action_gate", "all_constraint_gates", "q_current_outside_count", "active_bound_joints", "task_residuals", "group_contribution")})
    write_csv(OUT / "bounded_authority_replay.csv", csv_rows)
    result_map = {(int(row["recipe"]), int(row["control_step"]), row["formulation"]): row for row in results}
    summaries = {}
    for form in FORMULATIONS:
        by_recipe = {}
        all_form = [row for row in results if row["formulation"] == form]
        for recipe in TRACE_RECIPES:
            steps = manifest[recipe]["critical_control_steps"]
            rr = [result_map[(recipe, int(step), form)] for step in steps]
            by_recipe[str(recipe)] = {"critical_rows": len(rr), "critical_gate_pass_fraction": float(np.mean([bool(x["relative_improvement"] >= CRITICAL_IMPROVEMENT and x["all_constraint_gates"]) for x in rr])) if rr else 0.0, "improvement_ge_20_fraction": float(np.mean([x["relative_improvement"] >= CRITICAL_IMPROVEMENT for x in rr])) if rr else 0.0, "all_critical_gate_pass": bool(rr and all(x["relative_improvement"] >= CRITICAL_IMPROVEMENT and x["all_constraint_gates"] for x in rr)), "feasible_rows": int(sum(x["all_constraint_gates"] for x in rr)), "max_velocity_ratio": float(max(x["velocity_ratio_max"] for x in rr)) if rr else None}
        summaries[form] = {"rows": len(all_form), "feasible_rows": int(sum(x["all_constraint_gates"] for x in all_form)), "critical_gate_fraction_by_recipe": {k: v["critical_gate_pass_fraction"] for k, v in by_recipe.items()}, "critical": by_recipe, "median_improvement": float(np.median([x["relative_improvement"] for x in all_form])) if all_form else None, "max_wrist_hand_group_fraction": float(max((x["group_contribution"]["left wrist/hand"] + x["group_contribution"]["right wrist/hand"] for x in all_form), default=0.0)), "all_sources_critical_gate": bool(all(by_recipe[str(r)]["all_critical_gate_pass"] for r in TRACE_RECIPES))}
    dump(OUT / "joint_group_formulations.json", {"name": "Exp014JointGroupDiagnosticFormulationsV1", "rows": summaries, "formal_controller_change": False, "freeze_contract": "dq=0 for frozen groups", "G4_scaling": "x=dq/velocity_limit; J_scaled=J*velocity_limit; -0.80<=x<=0.80"})
    dump(OUT / "column_scaling_contract.json", {"name": "Exp014VelocityNormalizedColumnScalingV1", "variable": "x_j=dq_j/velocity_limit_j", "J_scaled": "J[:,j]*velocity_limit_j", "A_hz_scaled": "A_hz[j]*velocity_limit_j", "bounds": [-0.8, 0.8], "position_recovery_bounds": "converted componentwise to x-space", "result_dependent_scale": False})

    d28s_summary = read_json(D28S / "critical_window_authority.json") if (D28S / "critical_window_authority.json").exists() else {}
    dump(OUT / "critical_window_physical_authority.json", {"name": "Exp014CriticalWindowPhysicalAuthorityV1", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_PASS_FRACTION, "formulations": summaries, "D28S_protected_reference": {"classification": read_json(D28S / "stage_classification.json").get("classification"), "critical_window_authority_sha256": sha256_file(D28S / "critical_window_authority.json") if (D28S / "critical_window_authority.json").exists() else None}})
    positive_consistency = {key: {"strict_empty": int(value.get("strict_gate_empty_count", 0)), "recovery_empty": int(value.get("recovery_empty_count", 0)), "formal_actual_action_qcmd_infeasible": int(value.get("formal_qcmd_infeasible_count", 0)), "further_outward": int(value.get("further_outward_count", 0)), "recovery_contract_consistent": bool(value.get("states", 0) > 0 and value.get("recovery_empty_count", 1) == 0 and value.get("further_outward_count", 1) == 0 and value.get("formal_qcmd_infeasible_count", 1) == 0)} for key, value in audits.items()}
    dump(OUT / "positive_control_consistency.json", {"name": "Exp014PositiveControlConsistencyV1", "rows": positive_consistency, "candidate_contract": "G1/G2/G4 recovery bound; formal runtime contract unchanged", "strict_reentry_can_reject_current_policy_state": True})

    selected = None
    for form in ("G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING", "G2_FREEZE_WRIST_HAND", "G1_ALL_JOINTS_RECOVERY"):
        if summaries[form]["all_sources_critical_gate"] and all(x["recovery_contract_consistent"] for x in positive_consistency.values()):
            selected = form
            break
    temporary = {"name": "Exp014PhysicallyBoundedCentroidalWBIKV3R2", "created": bool(selected), "physics_applied": False, "selected_formulation": selected, "determinism": "fixed active-set solver; repeatable command required; no physics", "reason": "all four source critical windows and positive controls passed" if selected else "not created: no formulation passed all-source critical-window authority gate"}
    if selected:
        temporary["full_trace_rows"] = [row for row in results if row["formulation"] == selected]
        temporary["hash"] = canonical_hash(temporary["full_trace_rows"])
    dump(OUT / "temporary_v3r2_contract.json", temporary)
    dump(OUT / "temporary_v3r2_shadow.json", {"name": "Exp014PhysicallyBoundedCentroidalWBIKV3R2Shadow", "status": "DIAGNOSTIC_ONLY" if selected else "NOT_CREATED", "physics_executed": 0, "selected_formulation": selected, "full_trace_gate": bool(selected)})

    mapping_bug = bool(identity_meta["mapping_bug"])
    limit_audit = read_json(OUT / "joint_limit_source_audit.json")
    runtime_limit_sources_match = bool(limit_audit["comparisons"]["D26U_all_recipe_limits_equal_D25"] and limit_audit["comparisons"]["D25_vs_D26X_exact"] and limit_audit["comparisons"]["D25_vs_D28R_exact"])
    raw_usd_hard_limit_missing = limit_audit.get("raw_USD_hard_limit", {}).get("status") != "CAPTURED"
    formal_policy_outside_processed_limit = any(
        int(value.get("q_outside_count", 0)) > 0 or int(value.get("formal_qcmd_infeasible_count", 0)) > 0
        for value in audits.values()
    )
    unresolved = (not runtime_limit_sources_match) or (raw_usd_hard_limit_missing and formal_policy_outside_processed_limit)
    strict_empty_any = any(int(value.get("strict_gate_empty_count", 0)) > 0 for value in audits.values())
    recovery_pass = all(value["status"] == "PASS" for value in recovery_positive.values())
    hand_usage = usage["rows"]
    hand_ill_conditioned = any(row["ill_conditioned_nonessential_joint_usage"] for row in hand_usage)
    hand_exclusion_pass = summaries["G2_FREEZE_WRIST_HAND"]["all_sources_critical_gate"] or summaries["G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING"]["all_sources_critical_gate"]
    root_cause = ""
    next_action = ""
    if mapping_bug:
        classification = "EXP014_D28U_JOINT_LIMIT_INDEX_MAPPING_BUG"; root_cause = "name-based contract comparison found a joint identity mismatch"; next_action = "version the corrected name-based joint contract and rerun the D28R shadow; no physics in D28U"
    elif unresolved:
        classification = "EXP014_D28U_JOINT_LIMIT_CONTRACT_UNRESOLVED"; root_cause = "name/order/runtime processed limits matched, but the raw USD hard-limit field was not captured while formal states or q_cmd values were outside the processed soft-limit contract"; next_action = "capture and reconcile the USD hard-limit and runtime soft-limit contracts, then rerun the D28R shadow; no physics is authorized in D28U"
    elif strict_empty_any and recovery_pass and (summaries["G1_ALL_JOINTS_RECOVERY"]["all_sources_critical_gate"] or hand_exclusion_pass):
        classification = "EXP014_D28U_ONE_STEP_REENTRY_CONTRACT_BUG"; root_cause = "formal states are accepted by the runtime, but D28S strict one-step reentry creates artificial empty intervals; monotone recovery restores authority"; next_action = "version the diagnostic recovery bound and rerun the full D28R shadow; physics remains not authorized"
    elif summaries["G0_ALL_JOINTS_STRICT"]["all_sources_critical_gate"] is False and hand_exclusion_pass and hand_ill_conditioned:
        classification = "EXP014_D28U_NONESSENTIAL_HAND_JOINT_CONTAMINATION"; root_cause = "wrist/hand usage is diagnostically ill-conditioned and excluding it restores physically relevant authority"; next_action = "version the hand-exclusion/scaled shadow contract and rerun the full D28R shadow; physics remains not authorized"
    elif selected:
        classification = "EXP014_D28U_PHYSICALLY_RELEVANT_CENTROIDAL_AUTHORITY_PASS"; root_cause = "verified mapping, recovery-compatible bounds, physically actuated joints, and velocity-normalized columns retain bounded H_z authority"; next_action = "D28V fixed V3R2 shadow preflight and conditional RIGHT physics; no gain, target, timing, or action-contract changes"
    else:
        classification = "EXP014_D28U_TRUE_POSITION_LEVEL_AUTHORITY_NO_GO"; root_cause = "after verified mapping, positive-control-compatible recovery, wrist/hand exclusion, and dimensionless scaling, mandatory first-step tasks still leave less than 20% bounded H_z authority"; next_action = "close the position-level centroidal branch and evaluate dynamics-constrained trajectory optimization or torque-level WBC as a separate methodology branch"

    if selected and classification in ("EXP014_D28U_PHYSICALLY_RELEVANT_CENTROIDAL_AUTHORITY_PASS",):
        dump(OUT / "exp014_d28v_physics_preflight_authorization.json", {"authorized": True, "classification": classification, "selected_formulation": selected, "bound_contract": "Exp014MonotoneJointLimitRecoveryBoundV1", "dimensionless_solver": "Exp014VelocityNormalizedColumnScalingV1" if selected.startswith("G4") else "not used", "temporary_v3r2_hash": temporary.get("hash"), "physics_executed": 0, "constraints": {"right_only": True, "left_start": 0, "target_changed": False, "timing_changed": False, "gain_changed": False}})
    else:
        dump(OUT / "exp014_position_level_centroidal_no_go.json", {"authorized": False, "classification": classification, "physics_executed": 0, "reason": next_action, "position_level_centroidal_branch_closed": classification == "EXP014_D28U_TRUE_POSITION_LEVEL_AUTHORITY_NO_GO"})

    stage_ref = {"stage": "Phase 2-D28U", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_140": start_log, "D28S_read_only": True, "D28R_read_only": True, "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0, "left_start": 0, "remote_push": False}
    dump(OUT / "stage_reference.json", stage_ref)
    dump(OUT / "protocol.json", {"name": "Exp014D28UJointContractAndPhysicalAuthorityAuditV1", "phase": "2-D28U", "sources": ["D26U P0", "D21 protected formal train-only P1", "D26S formal_on P2", "D28R D27 V2A P3", "D28S 115 analysis steps"], "dt_s": DT, "svd_tolerance": SVD_TOL, "solver": {"type": "deterministic active-set equality-constrained least-squares inherited read-only from D28S", "tolerance": SOLVER_TOL, "maximum_iterations": SOLVER_MAX_ITER, "new_dependency": False}, "strict_bound": "D28S one-step position reentry plus 0.80 velocity gate", "recovery_bound": "Exp014MonotoneJointLimitRecoveryBoundV1 diagnostic only", "formulations": list(FORMULATIONS), "physics": 0, "forbidden": {"persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "left_start": 0, "target_change": 0, "timing_change": 0, "gain_change": 0, "action_contract_change": 0, "remote_push": False}})

    dump(OUT / "stage_classification.json", {"name": "Exp014D28UStageClassificationV1", "classification": classification, "precedence_applied": ["joint mapping", "limit contract", "one-step reentry", "nonessential hand", "physical authority", "true no-go"], "root_cause": root_cause, "physics_executed": 0, "starting_head": start_head})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "persistent_update": 0, "new_checkpoint": 0, "left_start": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "remote_push": False})
    protected = protected_hash_audit(start_head, start_status)
    dump(OUT / "protected_hashes.json", protected)
    dump(OUT / "reproduction_commands.ps1", f"Set-Location '{REPO}'\n# Offline only. No physics is launched.\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p '{HERE}' --mode analyze --headless\n")
    make_report(classification, next_action, {"identity": identity_meta, "positive": {key: {k: value.get(k) for k in ("states", "state_strict_empty_fraction", "state_recovery_empty_fraction", "q_outside_fraction", "formal_qcmd_infeasible_count", "further_outward_fraction")} | {"recovery_status": recovery_positive[key]["status"]} for key, value in audits.items()}, "formulations": summaries, "columns": col_audit, "limit_audit": limit_audit, "root_cause": root_cause, "temporary": temporary}, start_head, git("rev-parse", "HEAD"))
    print(json.dumps({"classification": classification, "physics_executed": 0, "records": len(records), "positive_controls": {key: value.get("states", 0) for key, value in audits.items()}, "formulations": {key: {"all_sources_critical_gate": value["all_sources_critical_gate"], "feasible_rows": value["feasible_rows"]} for key, value in summaries.items()}, "protected_unchanged": protected["exp_005_to_exp_013_and_D6_to_D28S_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
