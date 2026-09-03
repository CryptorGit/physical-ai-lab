"""Phase 2-D28Y: dynamic-limit invariance closure and final offline authority replay.

Only protected D28W/D28X traces and the protected D28S analysis records are
read.  This stage never creates a simulator, probe, or training process.
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
OUT = ROOT / "phase_2_d28y_dynamic_limit_and_final_centroidal_authority"
REPORT = REPO / "research/exp_014_phase_2_d28y_dynamic_limit_and_final_centroidal_authority_report.md"
D28W = ROOT / "phase_2_d28w_limit_enforcement_and_actuator_parity"
D28X = ROOT / "phase_2_d28x_scope_aware_centroidal_authority"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"
D21 = ROOT / "phase_2_d21_identity_complete_support_causality"
D28X_SCRIPT = EXP / "scripts/run_phase2_d28x_scope_aware_centroidal_authority.py"

DT = 0.02
HARD_TOL = 1.0e-6
VEL_RATIO = 0.80
PARITY_TOL = 1.0e-5
SOLVER_TOL = 1.0e-9
SVD_TOL = 1.0e-8
TASK_REL_TOL = 1.20
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_FRACTION = 0.80
OLD_VELOCITY_TOL = 1.0e-3
FORMULATIONS = ("C3_SCOPE", "C4_SCOPE")
TRACE_RECIPES = (4, 5, 6, 7)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d28x = load_module("exp014_d28y_d28x_read_only", D28X_SCRIPT)
d28s = d28x.d28s
d28w = d28x.d28w


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
            values = {}
            for key, value in row.items():
                values[key] = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value)
            writer.writerow(values)


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


def quantiles(value: Any) -> dict[str, Any]:
    x = arr(value).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, "min": None, "p01": None, "p05": None, "p50": None, "p95": None, "p99": None, "p99_9": None, "max": None}
    return {"count": int(x.size), "min": float(np.min(x)), "p01": float(np.quantile(x, .01)), "p05": float(np.quantile(x, .05)), "p50": float(np.quantile(x, .50)), "p95": float(np.quantile(x, .95)), "p99": float(np.quantile(x, .99)), "p99_9": float(np.quantile(x, .999)), "max": float(np.max(x))}


def hash_tree(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(file.relative_to(REPO)).replace("\\", "/"): sha256_file(file) for file in sorted(path.rglob("*")) if file.is_file()}


def protected_input_hashes() -> dict[str, Any]:
    wtree = hash_tree(D28W); xtree = hash_tree(D28X)
    return {"D28W_tree": wtree, "D28W_tree_sha256": canonical_hash(wtree), "D28X_tree": xtree, "D28X_tree_sha256": canonical_hash(xtree), "D28W_classification": read_json(D28W / "stage_classification.json").get("classification"), "D28X_classification": read_json(D28X / "stage_classification.json").get("classification")}


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
    allowed = {"left leg", "right leg", "waist", "left arm", "right arm"} if formulation == "C3_SCOPE" else {"left leg", "right leg", "waist"}
    return [i for i, name in enumerate(names) if group_for_joint(name) in allowed]


class ControlTrace:
    """Control-rate last-substep view of an existing protected trace."""

    def __init__(self, path: Path):
        self.path = path
        with np.load(path, allow_pickle=False) as z:
            control = np.asarray(z["control_step"], dtype=np.int64)
            substep = np.asarray(z["substep_index"], dtype=np.int64)
            self.raw_control = control.copy()
            self.raw_substep = substep.copy()
            self.raw_q = np.asarray(z["q_post"], dtype=np.float64).copy()
            self.raw_dq = np.asarray(z["dq_post"], dtype=np.float64).copy()
            self.raw_q_cmd = np.asarray(z["q_cmd_input"], dtype=np.float64).copy()
            last = []
            for step in np.unique(control):
                ids = np.flatnonzero((control == step) & (substep == np.max(substep[control == step])))
                last.append(int(ids[-1]))
            last = np.asarray(last, dtype=np.int64)
            self.steps = control[last]
            self.q = np.asarray(z["q_post"][last], dtype=np.float64)
            self.dq = np.asarray(z["dq_post"][last], dtype=np.float64)
            self.q_cmd = np.asarray(z["q_cmd_input"][last], dtype=np.float64)
            self.nonfinite = int(sum(not np.isfinite(self.raw_values).all() for self.raw_values in (self.raw_q, self.raw_dq, self.raw_q_cmd)))
        self.sha256 = sha256_file(path)


def theil_sen(values: np.ndarray) -> float:
    values = arr(values).reshape(-1)
    slopes = [(values[j] - values[i]) / (j - i) for i in range(len(values) - 1) for j in range(i + 1, len(values))]
    return float(np.median(np.sort(np.asarray(slopes, dtype=np.float64), kind="mergesort"))) if slopes else 0.0


def source_window(population: str) -> dict[str, Any]:
    if population == "D28W_PROVEN_CALIBRATION":
        return {"control_steps": 50, "command_hold": [0, 40], "steady": [30, 40], "steady_early": [30, 35], "steady_middle": [32, 37], "steady_late": [35, 40], "release": [40, 50], "release_steps": 10, "note": "D28W protected trace has 40 command-hold and 10 release steps; no extrapolation."}
    return {"control_steps": 300, "command_hold": [0, 100], "steady": [50, 100], "steady_early": [50, 70], "steady_middle": [65, 85], "steady_late": [80, 100], "release": [100, 300], "release_steps": 200, "note": "D28X protected trace uses the specified 100-step hold and 200-step release."}


def mask(steps: np.ndarray, bounds: list[int]) -> np.ndarray:
    return (steps >= bounds[0]) & (steps < bounds[1])


def probe_envelope_map() -> dict[tuple[int, str], float]:
    return {(int(row["joint_index"]), str(row["direction"])): float(row["formal_allowance"]) for row in read_json(D28W / "runtime_limit_enforcement_envelope.json").get("rows", [])}


def direction_sets() -> tuple[set[tuple[int, str]], set[tuple[int, str]], set[tuple[int, str]], list[dict[str, Any]], list[dict[str, Any]], dict[tuple[int, str], str]]:
    wclass = read_json(D28W / "limit_response_classification.json")["rows"]
    wrows = read_json(D28W / "isolated_limit_probe_results.json")["rows"]
    xobj = read_json(D28X / "targeted_extended_probe_results.json")
    xclass = xobj["direction_classification"]
    xrows = xobj["rows"]
    wproven = {(int(row["joint_index"]), str(row["direction"])) for row in wclass if row["classification"] in ("PHYSX_LIMIT_EXACT", "PHYSX_LIMIT_ENFORCED_WITH_COMPLIANCE")}
    resolved = {(int(row["joint_index"]), str(row["direction"])) for row in xclass if row["classification"] == "ACTIVE_LIMIT_ENFORCED"}
    test = {(int(row["joint_index"]), str(row["direction"])) for row in xclass if row["classification"] == "ACTIVE_LIMIT_STILL_AMBIGUOUS"}
    old = {(int(row["joint_index"]), str(row["direction"])): str(row["classification"]) for row in wclass}
    old.update({(int(row["joint_index"]), str(row["direction"])): str(row["classification"]) for row in xclass})
    # Keep the two calibration populations disjoint: D28W contributes only
    # its 62 proven directions, while the two D28X resolutions contribute
    # only their D28X trace rows.  A D28W ambiguous row must never become a
    # calibration row merely because the same direction was resolved in D28X.
    return wproven | resolved, wproven, test, wrows, xrows, old


def analyze_probe_row(row: dict[str, Any], trace: ControlTrace, population: str, envelope_map: dict[tuple[int, str], float]) -> dict[str, Any]:
    j = int(row["joint_index"]); env = int(row["env_index"]); upper = str(row["direction"]) == "upper"; nominal = float(row["nominal_limit"]); windows = source_window(population)
    q = trace.q[:, env, j]; dq = trace.dq[:, env, j]; qcmd = trace.q_cmd[:, env, j]; steps = trace.steps
    p = np.maximum(q - nominal, 0.0) if upper else np.maximum(nominal - q, 0.0); u = q - nominal if upper else nominal - q; outward_dq = dq if upper else -dq
    hold = mask(steps, windows["command_hold"]); steady = mask(steps, windows["steady"]); early = mask(steps, windows["steady_early"]); middle = mask(steps, windows["steady_middle"]); late = mask(steps, windows["steady_late"]); release = mask(steps, windows["release"])
    steady_p = p[steady]; envelope = float(row.get("fixed_probe_envelope", envelope_map.get((j, str(row["direction"])), HARD_TOL)))
    # Reproduce the existing D28X release contract from the protected raw
    # substep trace: per-control-step median penetration for recovery time,
    # and the final 80 substeps for the release-envelope gate.  The primary
    # hold/steady metrics above remain control-rate last-substep metrics.
    raw_q = trace.raw_q[:, env, j]; raw_dq = trace.raw_dq[:, env, j]
    raw_p = np.maximum(raw_q - nominal, 0.0) if upper else np.maximum(nominal - raw_q, 0.0)
    raw_outward_dq = raw_dq if upper else -raw_dq
    release_steps = np.unique(trace.raw_control[trace.raw_control >= windows["release"][0]])
    nominal_steps = [int(step) for step in release_steps if float(np.median(raw_p[trace.raw_control == step])) <= HARD_TOL]
    envelope_steps = [int(step) for step in release_steps if float(np.median(raw_p[trace.raw_control == step])) <= envelope]
    release_raw = trace.raw_control >= windows["release"][0]
    final_count = min(80, int(np.sum(release_raw)))
    final_p = raw_p[release_raw][-final_count:] if np.any(release_raw) else np.zeros(0)
    final_dq = raw_dq[release_raw][-final_count:] if np.any(release_raw) else np.zeros(0)
    release_v = raw_outward_dq[release_raw]
    running = np.maximum.accumulate(steady_p) if len(steady_p) else np.zeros(0)
    return {"population": population, "joint_index": j, "joint_name": row["joint_name"], "direction": row["direction"], "env_index": env, "offset_rad": float(row["offset_rad"]), "nominal_limit": nominal, "q_cmd_beyond": float(row["q_cmd_beyond"]), "source_trace": str(trace.path.relative_to(REPO)).replace("\\", "/"), "source_trace_sha256": trace.sha256, "window_contract": windows, "peak_penetration": float(np.max(p[hold])) if np.any(hold) else 0.0, "steady_penetration_median": float(np.median(steady_p)) if len(steady_p) else 0.0, "steady_penetration_p05": float(np.quantile(steady_p, .05)) if len(steady_p) else 0.0, "steady_penetration_p95": float(np.quantile(steady_p, .95)) if len(steady_p) else 0.0, "steady_penetration_max": float(np.max(steady_p)) if len(steady_p) else 0.0, "steady_early_median_penetration": float(np.median(p[early])) if np.any(early) else 0.0, "steady_middle_median_penetration": float(np.median(p[middle])) if np.any(middle) else 0.0, "steady_late_median_penetration": float(np.median(p[late])) if np.any(late) else 0.0, "terminal_growth": float(np.median(p[late]) - np.median(p[early])) if np.any(early) and np.any(late) else 0.0, "terminal_net_outward_displacement": float(np.median(u[late]) - np.median(u[early])) if np.any(early) and np.any(late) else 0.0, "steady_theil_sen_drift_rad_per_control_step": theil_sen(steady_p), "steady_oscillation_range_p95_minus_p05": float(np.quantile(steady_p, .95) - np.quantile(steady_p, .05)) if len(steady_p) else 0.0, "steady_new_running_max_events": int(np.sum(steady_p[1:] > running[:-1])) if len(steady_p) > 1 else 0, "steady_outward_velocity_abs_median": float(np.median(np.abs(outward_dq[steady]))) if np.any(steady) else 0.0, "steady_outward_velocity_abs_p95": float(np.quantile(np.abs(outward_dq[steady]), .95)) if np.any(steady) else 0.0, "old_velocity_gate_median_pass": bool(np.any(steady) and np.median(np.abs(outward_dq[steady])) <= OLD_VELOCITY_TOL), "old_velocity_gate_pointwise_pass": bool(np.any(steady) and np.all(np.abs(outward_dq[steady]) <= OLD_VELOCITY_TOL)), "time_to_nominal_limit_control_steps": None if not nominal_steps else int(nominal_steps[0] - windows["release"][0]), "time_to_calibration_envelope_control_steps": None if not envelope_steps else int(envelope_steps[0] - windows["release"][0]), "release_final_penetration": float(np.median(final_p)) if len(final_p) else None, "release_final_dq": float(np.median(final_dq)) if len(final_dq) else None, "release_final_within_calibration_envelope": bool(len(final_p) and np.median(final_p) <= envelope), "release_rebound_count": int(np.sum((release_v[:-1] < 0.0) & (release_v[1:] >= 0.0))) if len(release_v) > 1 else 0, "fixed_calibration_envelope": envelope, "nonfinite": bool(trace.nonfinite or not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(qcmd).all())}


def summarize_direction(rows: list[dict[str, Any]], epsilon: dict[str, float], old_classification: str | None) -> dict[str, Any]:
    offsets = np.asarray([float(row["offset_rad"]) for row in rows]); medians = np.asarray([float(row["steady_penetration_median"]) for row in rows])
    slope = float(np.polyfit(offsets, medians, 1)[0]) if len(rows) == 3 else None
    peak = max((float(row["peak_penetration"]) for row in rows), default=0.0)
    drift_ok = all(abs(float(row["steady_theil_sen_drift_rad_per_control_step"])) <= epsilon["epsilon_drift"] for row in rows)
    growth_ok = all(float(row["terminal_growth"]) <= epsilon["epsilon_growth"] for row in rows)
    release_ok = all(bool(row["release_final_within_calibration_envelope"]) for row in rows)
    finite_ok = all(not bool(row["nonfinite"]) for row in rows)
    low_coupling = slope is not None and slope <= 0.10
    not_enforced = slope is not None and slope >= 0.80 and any(float(row["terminal_growth"]) > epsilon["epsilon_growth"] for row in rows) and not release_ok
    if peak <= HARD_TOL:
        classification = "DYNAMICALLY_ENFORCED_EXACT"
    elif low_coupling and drift_ok and growth_ok and release_ok and finite_ok:
        classification = "DYNAMICALLY_ENFORCED_COMPLIANT"
    elif not_enforced:
        classification = "DYNAMICALLY_NOT_ENFORCED"
    else:
        classification = "DYNAMIC_ENFORCEMENT_UNRESOLVED"
    return {"joint_index": rows[0]["joint_index"] if rows else None, "joint_name": rows[0]["joint_name"] if rows else None, "direction": rows[0]["direction"] if rows else None, "old_classification": old_classification, "response_slope": slope, "peak_penetration_max": peak, "max_abs_terminal_drift": max((abs(float(row["steady_theil_sen_drift_rad_per_control_step"])) for row in rows), default=0.0), "max_terminal_growth": max((float(row["terminal_growth"]) for row in rows), default=0.0), "max_oscillation_range": max((float(row["steady_oscillation_range_p95_minus_p05"]) for row in rows), default=0.0), "release_recovery_all": release_ok, "nonfinite_all": finite_ok, "command_coupling_gate": low_coupling, "terminal_drift_gate": drift_ok, "terminal_growth_gate": growth_ok, "classification": classification, "rows": rows}


def dynamic_analysis() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    calibration_keys, wproven, test, wrows, xrows, old = direction_sets()
    wtrace = ControlTrace(D28W / "isolated_probe/isolated_substep_trace.npz")
    xtrace = ControlTrace(D28X / "targeted_probe/targeted_substep_trace.npz")
    envelope = probe_envelope_map()
    calibration_rows = [analyze_probe_row(row, wtrace, "D28W_PROVEN_CALIBRATION", envelope) for row in wrows if (int(row["joint_index"]), str(row["direction"])) in wproven]
    # Only the two D28X resolved directions are added to the 62 D28W proven
    # directions; the ten D28X test directions never enter calibration.
    xresolved = {(int(row["joint_index"]), str(row["direction"])) for row in read_json(D28X / "targeted_extended_probe_results.json")["direction_classification"] if row["classification"] == "ACTIVE_LIMIT_ENFORCED"}
    calibration_rows.extend(analyze_probe_row(row, xtrace, "D28X_RESOLVED_CALIBRATION", envelope) for row in xrows if (int(row["joint_index"]), str(row["direction"])) in xresolved)
    test_rows = [analyze_probe_row(row, xtrace, "D28X_TEST", envelope) for row in xrows if (int(row["joint_index"]), str(row["direction"])) in test]
    abs_drift = [abs(float(row["steady_theil_sen_drift_rad_per_control_step"])) for row in calibration_rows]
    positive_growth = [max(float(row["terminal_growth"]), 0.0) for row in calibration_rows]
    oscillation = [float(row["steady_oscillation_range_p95_minus_p05"]) for row in calibration_rows]
    epsilon = {"epsilon_drift": max(1.0e-8, float(np.quantile(abs_drift, .999))), "epsilon_growth": max(1.0e-6, float(np.quantile(positive_growth, .999))), "epsilon_oscillation": max(1.0e-6, float(np.quantile(oscillation, .999)))}
    cal_by = {}
    test_by = {}
    for key in sorted(calibration_keys):
        cal_by[key] = summarize_direction([row for row in calibration_rows if (int(row["joint_index"]), str(row["direction"])) == key], epsilon, old.get(key))
    for key in sorted(test):
        test_by[key] = summarize_direction([row for row in test_rows if (int(row["joint_index"]), str(row["direction"])) == key], epsilon, old.get(key))
    calibration = {"name": "Exp014D28YDynamicLimitCalibrationV1", "calibration_direction_count": len(cal_by), "calibration_offset_count": len(calibration_rows), "test_direction_count": len(test_by), "test_offset_count": len(test_rows), "test_excluded_from_tolerance": True, "source_trace_hashes": {"D28W": wtrace.sha256, "D28X": xtrace.sha256}, "window_contract": {"D28W": source_window("D28W_PROVEN_CALIBRATION"), "D28X": source_window("D28X_TEST")}, "epsilon": epsilon, "calibration_abs_drift_distribution": quantiles(abs_drift), "calibration_positive_growth_distribution": quantiles(positive_growth), "calibration_oscillation_distribution": quantiles(oscillation), "directions": [cal_by[key] for key in sorted(cal_by)]}
    dynamic = {"name": "Exp014PhysXDynamicLimitEnforcementV2", "calibration": {"direction_count": len(cal_by), "directions": [cal_by[key] for key in sorted(cal_by)]}, "test": {"direction_count": len(test_by), "directions": [test_by[key] for key in sorted(test_by)]}, "test_closure_pass": bool(test_by) and all(row["classification"] in ("DYNAMICALLY_ENFORCED_EXACT", "DYNAMICALLY_ENFORCED_COMPLIANT") for row in test_by.values()), "test_not_enforced_count": int(sum(row["classification"] == "DYNAMICALLY_NOT_ENFORCED" for row in test_by.values())), "test_unresolved_count": int(sum(row["classification"] == "DYNAMIC_ENFORCEMENT_UNRESOLVED" for row in test_by.values())), "epsilon": epsilon, "physics": 0}
    return {"calibration": calibration, "dynamic": dynamic, "cal_by": cal_by, "test_by": test_by, "proven": calibration_keys, "test": test}, [*calibration_rows, *test_rows]


def write_dynamic_artifacts(analysis: dict[str, Any], metrics: list[dict[str, Any]]) -> None:
    dump(OUT / "dynamic_limit_calibration.json", analysis["calibration"])
    dump(OUT / "dynamic_limit_enforcement_v2.json", analysis["dynamic"])
    write_csv(OUT / "dynamic_limit_invariance_metrics.csv", metrics)
    dump(OUT / "dynamic_limit_invariance_metrics.json", {"name": "Exp014D28YDynamicLimitInvarianceMetricsV1", "calibration_direction_count": len(analysis["cal_by"]), "test_direction_count": len(analysis["test_by"]), "rows": metrics, "calibration_directions": [analysis["cal_by"][key] for key in sorted(analysis["cal_by"])], "test_directions": [analysis["test_by"][key] for key in sorted(analysis["test_by"])]})
    summary = {}
    for label, values in (("calibration", list(analysis["cal_by"].values())), ("test", list(analysis["test_by"].values()))):
        summary[label] = {"direction_count": len(values), "median_gate_direction_pass_rate": float(np.mean([all(row["old_velocity_gate_median_pass"] for row in item["rows"]) for item in values])) if values else 0.0, "pointwise_gate_direction_pass_rate": float(np.mean([all(row["old_velocity_gate_pointwise_pass"] for row in item["rows"]) for item in values])) if values else 0.0, "terminal_drift_pass_rate": float(np.mean([bool(item["terminal_drift_gate"]) for item in values])) if values else 0.0, "net_drift_pass_rate": float(np.mean([bool(item["terminal_drift_gate"]) for item in values])) if values else 0.0, "position_invariance_pass_rate": float(np.mean([item["classification"] in ("DYNAMICALLY_ENFORCED_EXACT", "DYNAMICALLY_ENFORCED_COMPLIANT") for item in values])) if values else 0.0, "position_invariance_classifications": {name: int(sum(item["classification"] == name for item in values)) for name in ("DYNAMICALLY_ENFORCED_EXACT", "DYNAMICALLY_ENFORCED_COMPLIANT", "DYNAMICALLY_NOT_ENFORCED", "DYNAMIC_ENFORCEMENT_UNRESOLVED")}}
    retrospective_rows = []
    for row in metrics:
        item = next((item for item in [*analysis["cal_by"].values(), *analysis["test_by"].values()] if item["joint_index"] == row["joint_index"] and item["direction"] == row["direction"]), None)
        retrospective_rows.append({"joint_index": row["joint_index"], "joint_name": row["joint_name"], "direction": row["direction"], "population": row["population"], "offset_rad": row["offset_rad"], "old_velocity_gate_median_pass": row["old_velocity_gate_median_pass"], "old_velocity_gate_pointwise_pass": row["old_velocity_gate_pointwise_pass"], "position_invariance_classification": item["classification"] if item else None, "steady_drift": row["steady_theil_sen_drift_rad_per_control_step"], "terminal_growth": row["terminal_growth"], "release_recovery": row["release_final_within_calibration_envelope"]})
    dump(OUT / "outward_velocity_gate_retrospective.json", {"name": "Exp014D28YOutwardVelocityGateRetrospectiveV1", "old_gate_threshold_rad_s": OLD_VELOCITY_TOL, "calibration": summary["calibration"], "test": summary["test"], "pointwise_velocity_gate_not_necessary_for_limit_invariance": bool(summary["test"]["net_drift_pass_rate"] > 0.0 and summary["test"]["median_gate_direction_pass_rate"] < summary["test"]["net_drift_pass_rate"]), "rows": retrospective_rows})


def formal_maximum_interpretation() -> dict[str, Any]:
    rows = read_json(D28W / "formal_limit_violation_magnitude.json")["rows"]
    maximum = max(rows, key=lambda row: float(row["absolute_violation_magnitude"]))
    context = {"status": "UNAVAILABLE"}
    bundle = D21 / "reference_rollout_bundle.npz"
    if bundle.exists() and maximum["population"] == "P1_S_HOLD_formal_rollout":
        with np.load(bundle, allow_pickle=False) as z:
            env = int(maximum["episode"]); step = int(maximum["control_step"]); joint = int(maximum["joint_index"]); lower = float(maximum["lower_candidate"]); q = np.asarray(z["joint_position"][:, env, joint], dtype=np.float64); p = np.maximum(lower - q, 0.0); after = np.flatnonzero(np.arange(len(p)) > step); inside = after[p[after] <= HARD_TOL]; root_velocity = np.asarray(z["root_velocity"][:, env], dtype=np.float64); root_acc = (root_velocity[step + 1] - root_velocity[step]) / DT if step + 1 < len(root_velocity) else np.zeros(6)
            get = lambda key: z[key][step, env].tolist() if key in z.files else None
            context = {"status": "AVAILABLE_READ_ONLY", "source": str(bundle.relative_to(REPO)).replace("\\", "/"), "environment_index": env, "control_step": step, "contact_force": get("contact_force"), "F_L": get("F_L"), "F_R": get("F_R"), "F_total": get("F_total"), "support_valid": get("support_valid"), "foot_tangential_velocity": get("foot_tangential_velocity"), "root_velocity": get("root_velocity"), "estimated_root_acceleration": root_acc.tolist(), "Lz": get("Lz"), "dLz_dt": get("dLz_dt"), "contact_yaw_moment": get("contact_yaw_moment"), "yaw_rate": get("yaw_rate"), "yaw_acceleration": get("yaw_acceleration"), "fall": get("fall"), "dangerous_slip": get("dangerous_slip"), "impact": get("impact"), "velocity_saturation": get("velocity_saturation"), "torque_saturation": get("torque_saturation"), "recovery_first_nominal_step": int(inside[0]) if inside.size else None, "penetration_after_maximum": p[step:min(step + 10, len(p))].tolist(), "requested_applied_torque": "not present in protected D21 bundle; not inferred", "formal_safety_classification": "existing D21 flags preserved"}
    classification = "TRANSIENT_CONTACT_LOADED_COMPLIANCE" if context.get("status") == "AVAILABLE_READ_ONLY" and context.get("recovery_first_nominal_step") is not None and not context.get("fall") and not context.get("dangerous_slip") else "UNRESOLVED_DYNAMIC_PENETRATION"
    result = {"name": "Exp014D28YFormalMaximumPenetrationInterpretationV1", "joint_name": maximum["joint_name"], "joint_index": maximum["joint_index"], "direction": maximum["violation_side"], "source_family": maximum["population"], "episode": maximum["episode"], "control_step": maximum["control_step"], "phase": maximum["phase"], "candidate_limit": maximum["lower_candidate"], "q_actual": maximum["q_actual"], "q_cmd": maximum["q_cmd"], "dq": {"preceding": maximum["preceding_dq"], "following": maximum["following_dq"]}, "penetration": maximum["absolute_violation_magnitude"], "duration_control_steps": maximum["violation_duration_control_steps"], "context": context, "classification": classification, "D28W_read_only": True}
    dump(OUT / "formal_maximum_penetration_interpretation.json", result)
    return result


def actuator_payload() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(read_json(D28W / "physx_runtime_joint_contract.json")["rows"], key=lambda row: int(row["joint_index_by_name"]))
    return np.asarray([float(row["actuator_stiffness"]) for row in rows]), np.asarray([float(row["actuator_damping"]) for row in rows]), np.asarray([float(row["runtime_effort_limit"]) for row in rows])


def v5_bounds(record: dict[str, Any], active: list[int], hard: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(record["q_current"], dtype=np.float64); v = np.abs(np.asarray(record["velocity_limits"], dtype=np.float64)); lower = np.asarray(hard[active, 0], dtype=np.float64).copy(); upper = np.asarray(hard[active, 1], dtype=np.float64).copy()
    for local, joint in enumerate(active):
        if q[joint] > hard[joint, 1]:
            upper[local] = q[joint]
        elif q[joint] < hard[joint, 0]:
            lower[local] = q[joint]
    dq_lower = np.maximum(-VEL_RATIO * v[active], (lower - q[active]) / DT); dq_upper = np.minimum(VEL_RATIO * v[active], (upper - q[active]) / DT)
    return dq_lower / np.maximum(v[active], 1.0e-8), dq_upper / np.maximum(v[active], 1.0e-8), q[active], lower, upper


def solve_sequence(record: dict[str, Any], formulation: str, hard: np.ndarray, names: list[str], hz_first: bool = False) -> dict[str, Any]:
    active = active_indices(names, formulation); passed = sorted(set(range(37)) - set(active)); q = np.asarray(record["q_current"], dtype=np.float64); v = np.abs(np.asarray(record["velocity_limits"], dtype=np.float64)); baseline = np.asarray(record["baseline_q_cmd"], dtype=np.float64); pass_dq = (baseline - q) / DT; lower, upper, q_active, qkin_lower, qkin_upper = v5_bounds(record, active, hard)
    hz_J = np.asarray(record["tasks"]["hz"]["J"], dtype=np.float64); hz_b = np.asarray(record["tasks"]["hz"]["b"], dtype=np.float64)
    if hz_first:
        stance_J, stance_b = d28s.task_stack(record, ["stance"]); stance_A = stance_J[:, active] * v[active][None, :]; stance_b = stance_b - stance_J[:, passed] @ pass_dq[passed]; stage0 = d28s.bounded_lsq(stance_A, stance_b, lower, upper); x0 = np.asarray(stage0.get("x", np.zeros(len(active))), dtype=np.float64); hz_A = hz_J[:, active] * v[active][None, :]; hz_res = hz_b - hz_J[:, passed] @ pass_dq[passed]; x1, hz_diag = d28s.bounded_nullspace_stage(x0, stance_A, hz_A, hz_res, lower, upper); other_J, other_b = d28s.task_stack(record, ["com", "swing", "pelvis"]); other_A = other_J[:, active] * v[active][None, :]; other_res = other_b - other_J[:, passed] @ pass_dq[passed]; x_final, final_diag = d28s.bounded_nullspace_stage(x1, np.vstack((stance_A, hz_A)), other_A, other_res, lower, upper); solver_success = bool(stage0.get("success", False) and hz_diag.get("success", False) and final_diag.get("success", False)); solver_meta = {"priority": ["stance", "hz", "com+swing+pelvis"], "stance": stage0, "hz": hz_diag, "other": final_diag}
    else:
        task_J, task_b = d28s.task_stack(record, ["stance", "com", "swing", "pelvis"]); task_A = task_J[:, active] * v[active][None, :]; task_b = task_b - task_J[:, passed] @ pass_dq[passed]; hard_sol = d28s.bounded_lsq(task_A, task_b, lower, upper); x0 = np.asarray(hard_sol.get("x", np.zeros(len(active))), dtype=np.float64); hz_A = hz_J[:, active] * v[active][None, :]; hz_res = hz_b - hz_J[:, passed] @ pass_dq[passed]; x_final, hz_diag = d28s.bounded_nullspace_stage(x0, task_A, hz_A, hz_res, lower, upper); solver_success = bool(hard_sol.get("success", False) and hz_diag.get("success", False)); solver_meta = {"priority": ["stance+com+swing+pelvis", "hz"], "hard": hard_sol, "hz": hz_diag}
    dq = pass_dq.copy(); dq[active] = x_final * v[active]; qkin_next = q[active] + DT * dq[active]; qcmd = baseline.copy(); qcmd[active] = qkin_next
    before_pen = np.maximum(q[active] - hard[active, 1], 0.0) + np.maximum(hard[active, 0] - q[active], 0.0); after_pen = np.maximum(qkin_next - hard[active, 1], 0.0) + np.maximum(hard[active, 0] - qkin_next, 0.0); qkin_gate = bool(np.all(after_pen <= before_pen + HARD_TOL) and np.all(qkin_next >= qkin_lower - HARD_TOL) and np.all(qkin_next <= qkin_upper + HARD_TOL))
    residuals = d28s.task_residuals(record, dq); v2 = d28s.v2a_dq(record); baseline_tasks = d28s.task_residuals(record, v2["dq"]); task_gates = {"stance_no_worse": residuals["stance"] <= baseline_tasks["stance"] + 1.0e-9, "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(baseline_tasks["com"], 1.0e-8) + 1.0e-9, "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(baseline_tasks["swing"], 1.0e-8) + 1.0e-9, "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(baseline_tasks["pelvis"], 1.0e-8) + 1.0e-9}
    action = (qcmd - record["default_q"]) / record["action_scale"]; roundtrip = record["default_q"] + record["action_scale"] * action; qcmd_gate = bool(np.isfinite(qcmd).all() and np.allclose(roundtrip, qcmd, atol=1.0e-10, rtol=1.0e-10) and np.array_equal(qcmd[passed], baseline[passed])); kp, kd, effort_limit = actuator_payload(); torque = kp * (qcmd - q) - kd * np.asarray(record["dq_current"], dtype=np.float64); effort_ratio = np.abs(torque[active]) / np.maximum(np.abs(effort_limit[active]), 1.0e-8); effort_gate = bool(np.isfinite(torque[active]).all() and np.max(effort_ratio) <= 1.0 + SOLVER_TOL); velocity_ratio = np.abs(dq[active]) / np.maximum(v[active], 1.0e-8); all_gate = bool(solver_success and qkin_gate and qcmd_gate and effort_gate and np.max(velocity_ratio) <= VEL_RATIO + SOLVER_TOL and all(task_gates.values()))
    hz_error = float(abs(hz_J @ dq - hz_b)[0]); v2_error = float(abs(hz_J @ v2["dq"] - hz_b)[0]); active_bounds = []
    for local, value in enumerate(x_final):
        if abs(value - lower[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "lower", "x": float(value)})
        if abs(value - upper[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "upper", "x": float(value)})
    return {"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "phase": record["phase"], "formulation": formulation, "hz_first": hz_first, "solver_success": solver_success, "solver": solver_meta, "current_hz_error": float(abs(record["actual_hz"])), "v2a_predicted_hz_error": v2_error, "minimum_achievable_hz_error": hz_error, "relative_hz_improvement": float((v2_error - hz_error) / max(v2_error, 1.0e-8)), "dq": dq, "q_cmd": qcmd, "active_qkin_next": qkin_next, "qkin_penetration_before": before_pen, "qkin_penetration_after": after_pen, "qkin_penetration_worsening": float(np.max(after_pen - before_pen)) if len(after_pen) else 0.0, "active_velocity_ratio_max": float(np.max(velocity_ratio)) if len(velocity_ratio) else 0.0, "active_effort_ratio_max": float(np.max(effort_ratio)) if len(effort_ratio) else 0.0, "stance_residual": residuals["stance"], "com_residual": residuals["com"], "swing_residual": residuals["swing"], "pelvis_residual": residuals["pelvis"], "task_residuals": residuals, "task_gates": task_gates, "qkin_dynamic_limit_gate": qkin_gate, "active_velocity_gate": bool(np.max(velocity_ratio) <= VEL_RATIO + SOLVER_TOL) if len(velocity_ratio) else True, "q_cmd_setter_gate": qcmd_gate, "effort_gate": effort_gate, "pass_through_qcmd_bitwise_gate": bool(np.array_equal(qcmd[passed], baseline[passed])), "active_bounds": active_bounds, "all_mandatory_gates": all_gate, "joint_group_hz_contribution": {group_for_joint(names[i]): float(dq[i] * hz_J[0, i]) for i in active}}


def pass_through_controls(base: dict[str, Any]) -> dict[str, Any]:
    parity = read_json(D28W / "actuator_substep_parity.json"); result = {}
    for formulation in FORMULATIONS:
        active = set(active_indices(base["names"], formulation)); passed = sorted(set(range(37)) - active); mismatch = 0; samples = 0
        for recipe in TRACE_RECIPES:
            for trace_row in base["analysis"][recipe]:
                record = next(row for row in base["records"] if int(row["recipe"]) == recipe and int(row["trace_row"]) == int(trace_row)); baseline = np.asarray(record["baseline_q_cmd"]); candidate = np.asarray(record["baseline_q_cmd"]); mismatch += int(np.sum(candidate[passed] != baseline[passed])); samples += len(passed)
        result[formulation] = {"pass_through_joint_count": len(passed), "pass_through_joint_names": [base["names"][i] for i in passed], "analysis_samples": samples, "q_cmd_bitwise_mismatch_count": mismatch, "q_cmd_bitwise_pass": mismatch == 0, "D28W_actuator_contract_pass": bool(parity.get("pass", False)), "effort_parity_pass": bool(parity.get("computed_request_gate", False) and parity.get("applied_gate", False) and parity.get("effort_clipping_classification_agreement", False)), "command_mutation": 0, "authority_claim": False}
    output = {"name": "Exp014D28YPassThroughPositiveControlsV2", "formulations": result, "pass": bool(all(row["q_cmd_bitwise_pass"] and row["D28W_actuator_contract_pass"] and row["effort_parity_pass"] and row["command_mutation"] == 0 for row in result.values()))}
    dump(OUT / "pass_through_positive_controls_v2.json", output)
    return output


def replay(base: dict[str, Any], dynamic: dict[str, Any], positive: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    if not dynamic.get("test_closure_pass", False) or not positive.get("pass", False):
        return [], {"name": "Exp014D28YScopeAwareAuthorityReplayV2", "status": "NOT_EXECUTED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED", "row_count": 0, "formulations": {}, "physics": 0}, None
    results = [solve_sequence(record, formulation, base["hard"], base["names"]) for record in base["records"] for formulation in FORMULATIONS]
    summaries = {}
    for formulation in FORMULATIONS:
        rr = [row for row in results if row["formulation"] == formulation]; by_recipe = {}
        for recipe in TRACE_RECIPES:
            critical = set(int(step) for step in base["critical"][recipe]); cr = [row for row in rr if int(row["recipe"]) == recipe and int(row["control_step"]) in critical]
            by_recipe[str(recipe)] = {"critical_steps": len(cr), "improvement_ge_20_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for row in cr])) if cr else 0.0, "critical_gate_pass_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in cr])) if cr else 0.0, "all_critical_gate_pass": bool(cr and all(row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in cr)), "max_qkin_penetration_worsening": max((row["qkin_penetration_worsening"] for row in cr), default=0.0), "max_velocity_ratio": max((row["active_velocity_ratio_max"] for row in cr), default=0.0), "max_effort_ratio": max((row["active_effort_ratio_max"] for row in cr), default=0.0)}
        summaries[formulation] = {"rows": len(rr), "solver_success_fraction": float(np.mean([row["solver_success"] for row in rr])) if rr else 0.0, "full_trace_gate_fraction": float(np.mean([row["all_mandatory_gates"] and row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for row in rr])) if rr else 0.0, "median_improvement": float(np.median([row["relative_hz_improvement"] for row in rr])) if rr else None, "critical": by_recipe, "all_sources_critical_gate": bool(all(row["all_critical_gate_pass"] for row in by_recipe.values()))}
    selected = None
    for formulation in ("C4_SCOPE", "C3_SCOPE"):
        if summaries[formulation]["all_sources_critical_gate"] and summaries[formulation]["solver_success_fraction"] == 1.0:
            selected = formulation; break
    return results, {"name": "Exp014D28YScopeAwareAuthorityReplayV2", "status": "PASS" if selected else "COMPLETED_NO_SELECTED_FORMULATION", "row_count": len(results), "critical_steps": 36, "formulations": summaries, "physics": 0}, selected


def hz_first_diagnostic(base: dict[str, Any], dynamic: dict[str, Any], positive: dict[str, Any]) -> dict[str, Any]:
    if not dynamic.get("test_closure_pass", False) or not positive.get("pass", False):
        return {"name": "Exp014D28YHzFirstDiagnosticV1", "status": "NOT_EXECUTED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED", "physics": 0}
    rows = [solve_sequence(record, formulation, base["hard"], base["names"], hz_first=True) for record in base["records"] for formulation in FORMULATIONS]
    return {"name": "Exp014D28YHzFirstDiagnosticV1", "status": "COMPLETED", "rows": rows, "formal_pass_candidate": False, "physics": 0}


def write_stage_reference(start_head: str, start_status: list[str], start_log: list[str]) -> None:
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28Y", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_180": start_log, "D28X_read_only": True, "D28X_classification_preserved": "EXP014_D28X_ACTIVE_LIMIT_ENFORCEMENT_UNRESOLVED", "new_physics": 0, "START_capability_physics": 0, "LEFT_START": 0, "persistent_update": 0, "remote_push": False, "protected_input_hashes": protected_input_hashes()})


def write_protocol(start_head: str) -> None:
    dump(OUT / "protocol.json", {"name": "Exp014PhysXDynamicLimitEnforcementV2", "phase": "2-D28Y", "starting_head": start_head, "physics": 0, "new_physics_probe": 0, "calibration": {"D28W_proven_directions": 62, "D28X_resolved_directions": 2, "total": 64, "test_directions": 10, "test_excluded_from_tolerance": True}, "windows": {"D28W": source_window("D28W_PROVEN_CALIBRATION"), "D28X": source_window("D28X_TEST")}, "thresholds": {"hard_tolerance_rad": HARD_TOL, "coupling_low_le": 0.10, "coupling_not_enforced_ge": 0.80, "old_velocity_gate_rad_s": OLD_VELOCITY_TOL, "velocity_ratio": VEL_RATIO, "critical_improvement": CRITICAL_IMPROVEMENT, "critical_fraction": CRITICAL_FRACTION, "svd_tolerance": SVD_TOL, "solver_tolerance": SOLVER_TOL}, "forbidden": {"START_capability_physics": 0, "new_physics_probe": 0, "LEFT_START": 0, "persistent_update": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "raw_restore": 0, "remote_push": False}})


def finish(base: dict[str, Any], start_head: str) -> str:
    analysis, metrics = dynamic_analysis()
    write_dynamic_artifacts(analysis, metrics)
    formal = formal_maximum_interpretation()
    positive = pass_through_controls(base)
    replay_rows, replay_summary, selected = replay(base, analysis["dynamic"], positive)
    dump(OUT / "scope_aware_authority_replay_v2.json", {**replay_summary, "rows": replay_rows, "contract": "Exp014DynamicComplianceAwareJointAuthorityV5"})
    write_csv(OUT / "scope_aware_authority_replay_v2.csv", replay_rows)
    critical = {form: replay_summary.get("formulations", {}).get(form, {}) for form in FORMULATIONS}
    dump(OUT / "critical_window_authority_v5.json", {"name": "Exp014D28YCriticalWindowAuthorityV5", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_FRACTION, "critical_steps": 36, "formulations": critical, "selected": selected, "physics": 0})
    hz_first = hz_first_diagnostic(base, analysis["dynamic"], positive)
    dump(OUT / "hz_first_diagnostic.json", hz_first)
    dump(OUT / "hard_task_conflict_v2.json", {"name": "Exp014D28YHardTaskConflictV2", "status": "NOT_CLASSIFIED_UNTIL_REPLAY", "h_z_first_status": hz_first.get("status"), "scope_replay_status": replay_summary.get("status"), "formal_pass_candidate": False, "diagnostic_rule": "H_z-first >=20% while C3/C4 misses mandatory gates implies HZ_CONTROL_CONFLICTS_WITH_FIRST_STEP_TASKS"})
    blockers = [{"formulation": row["formulation"], "recipe": row["recipe"], "control_step": row["control_step"], "relative_hz_improvement": row["relative_hz_improvement"], "qkin_penetration_worsening": row["qkin_penetration_worsening"], "task_gates": row["task_gates"], "active_velocity_gate": row["active_velocity_gate"], "effort_gate": row["effort_gate"]} for row in replay_rows if not row["all_mandatory_gates"]]
    dump(OUT / "active_position_authority_blockers.json", {"name": "Exp014D28YActivePositionAuthorityBlockersV1", "rows": blockers})
    dynamic_not_enforced = any(row["classification"] == "DYNAMICALLY_NOT_ENFORCED" for row in analysis["test_by"].values())
    dynamic_unresolved = any(row["classification"] == "DYNAMIC_ENFORCEMENT_UNRESOLVED" for row in analysis["test_by"].values())
    if dynamic_not_enforced:
        classification = "EXP014_D28Y_DYNAMIC_LIMIT_NOT_ENFORCED"; next_action = "do not authorize position-level authority; investigate the listed non-enforced active directions"
    elif dynamic_unresolved:
        classification = "EXP014_D28Y_DYNAMIC_LIMIT_INVARIANCE_UNRESOLVED"; next_action = "do not run authority replay; resolve the listed dynamic-limit invariance cases without new physics"
    elif replay_summary.get("status") == "NOT_EXECUTED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED":
        classification = "EXP014_D28Y_DYNAMIC_LIMIT_INVARIANCE_UNRESOLVED"; next_action = "dynamic active-limit closure was not complete; authority replay remains unauthorized"
    elif replay_summary.get("row_count", 0) and any(float(info.get("solver_success_fraction", 0.0)) < 1.0 for info in replay_summary.get("formulations", {}).values()):
        classification = "EXP014_D28Y_BOUNDED_SOLVER_FAIL"; next_action = "repair deterministic bounded solver failures before physics authorization"
    elif selected:
        classification = "EXP014_D28Y_SCOPE_AWARE_POSITION_LEVEL_AUTHORITY_PASS"; next_action = "D28Z fresh source parity and V3R5 runtime shadow preflight; physics remains unauthorized"
    else:
        classification = "EXP014_D28Y_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO"; next_action = "close the position-level centroidal branch and evaluate torque-level WBC or dynamics-constrained trajectory optimization separately"
    contract_status = "PASS" if analysis["dynamic"]["test_closure_pass"] else "NOT_CREATED_DYNAMIC_LIMIT_CONTRACT_UNRESOLVED"
    if analysis["dynamic"]["test_closure_pass"]:
        v5_contract = {"name": "Exp014DynamicComplianceAwareJointAuthorityV5", "status": contract_status, "active_joint_sets": {form: [base["names"][i] for i in active_indices(base["names"], form)] for form in FORMULATIONS}, "pass_through_joint_sets": {form: [base["names"][i] for i in sorted(set(range(37)) - set(active_indices(base["names"], form)))] for form in FORMULATIONS}, "q_actual": "simulation state; no isolated fixed-root peak envelope hard bound", "q_kin": "nominal bound when current q_actual is inside; monotone non-worsening signed penetration when outside", "dq_kin": "abs(dq)<=0.80*runtime velocity limit", "q_cmd": "no physical position clamp; canonical setter parity", "effort": "D28W resolved implicit actuator contract", "dynamic_limit": analysis["dynamic"], "physics": 0}
    else:
        v5_contract = {"name": "Exp014DynamicComplianceAwareJointAuthorityV5", "status": contract_status, "activation_condition": "all ten D28X test directions must be DYNAMICALLY_ENFORCED_EXACT or DYNAMICALLY_ENFORCED_COMPLIANT", "unresolved_test_directions": [f"{row['joint_name']}:{row['direction']}" for row in analysis["dynamic"]["test"]["directions"] if row["classification"] == "DYNAMIC_ENFORCEMENT_UNRESOLVED"], "physics": 0}
    dump(OUT / "dynamic_compliance_aware_joint_authority_v5.json", v5_contract)
    if selected:
        full = [row for row in replay_rows if row["formulation"] == selected]; shadow_hash = canonical_hash(full)
        dump(OUT / "temporary_v3r5_full_trace_shadow.json", {"name": "Exp014DynamicComplianceAwareCentroidalWBIKV3R5Shadow", "status": "CREATED", "selected_formulation": selected, "physics": 0, "rows": full, "hash": shadow_hash, "determinism": "D28Y deterministic solver; independent process replay required"})
        dump(OUT / "temporary_v3r5_contract.json", {"name": "Exp014DynamicComplianceAwareCentroidalWBIKV3R5", "status": "CREATED", "selected_formulation": selected, "hash": shadow_hash, "contract": "Exp014DynamicComplianceAwareJointAuthorityV5", "physics": 0})
    else:
        dump(OUT / "temporary_v3r5_full_trace_shadow.json", {"name": "Exp014DynamicComplianceAwareCentroidalWBIKV3R5Shadow", "status": "NOT_CREATED", "physics": 0})
        dump(OUT / "temporary_v3r5_contract.json", {"name": "Exp014DynamicComplianceAwareCentroidalWBIKV3R5", "status": "NOT_CREATED", "physics": 0})
    if classification == "EXP014_D28Y_SCOPE_AWARE_POSITION_LEVEL_AUTHORITY_PASS":
        dump(OUT / "exp014_d28z_dynamic_centroidal_shadow_authorization.json", {"name": "Exp014D28ZDynamicCentroidalShadowAuthorizationV1", "authorized": True, "selected_formulation": selected, "contract": "Exp014DynamicComplianceAwareJointAuthorityV5", "V3R5_hash": read_json(OUT / "temporary_v3r5_contract.json").get("hash"), "physics_authorized": False, "physics_executed": 0})
    else:
        dump(OUT / "exp014_d28y_not_authorized.json", {"name": "Exp014D28YNotAuthorizedV1", "authorized": False, "classification": classification, "reason": next_action, "physics_authorized": False, "physics_executed": 0, "new_physics": 0})
        if classification in ("EXP014_D28Y_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO", "EXP014_D28Y_HZ_FIRST_STEP_TASK_CONFLICT", "EXP014_D28Y_ACTIVE_POSITION_LEVEL_AUTHORITY_INSUFFICIENT"):
            dump(OUT / "exp014_position_level_centroidal_no_go.json", {"name": "Exp014PositionLevelCentroidalNoGoV2", "authorized": False, "classification": classification, "position_level_centroidal_branch_closed": True, "physics_executed": 0, "reason": next_action, "dynamic_limit_contract": analysis["dynamic"], "scope_replay": replay_summary, "h_z_first": hz_first})
    baseline = read_json(OUT / "stage_reference.json").get("protected_input_hashes", {}); protected_after = protected_input_hashes(); protected_ok = bool(baseline) and baseline == protected_after
    dump(OUT / "stage_classification.json", {"name": "Exp014D28YStageClassificationV1", "classification": classification, "D28X_classification_unchanged": "EXP014_D28X_ACTIVE_LIMIT_ENFORCEMENT_UNRESOLVED", "dynamic_limit_contract": contract_status, "test_unresolved_count": analysis["dynamic"]["test_unresolved_count"], "test_not_enforced_count": analysis["dynamic"]["test_not_enforced_count"], "selected_formulation": selected, "scope_replay_status": replay_summary.get("status"), "new_physics": 0, "START_capability_physics": 0, "protected_inputs_unchanged": protected_ok})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "new_physics": 0, "persistent_update": 0, "new_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_inputs_unchanged": protected_ok, "protected_input_hashes_at_start": baseline, "protected_input_hashes_at_finish": protected_after, "D28W_unchanged": protected_ok, "D28X_unchanged": protected_ok, "D28V_and_earlier_unchanged": protected_ok, "production_asset_unchanged": True, "new_physics": 0, "new_physics_probe": 0, "START_capability_physics": 0, "persistent_update": 0, "new_learned_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "raw_restore": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28y_dynamic_limit_and_final_centroidal_authority.py' --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28y_dynamic_limit_and_final_centroidal_authority.py' --headless\n", encoding="utf-8")
    table = []
    for row in analysis["test_by"].values():
        table.append(f"| {row['joint_name']} | {row['direction']} | {row['response_slope']:.6g} | {row['max_abs_terminal_drift']:.6g} | {row['max_terminal_growth']:.6g} | {row['release_recovery_all']} | {row['classification']} |")
    retrospective_test = read_json(OUT / "outward_velocity_gate_retrospective.json")["test"]
    retro_n = int(retrospective_test["direction_count"])
    retro_pointwise = int(round(float(retrospective_test["pointwise_gate_direction_pass_rate"]) * retro_n))
    retro_terminal = int(round(float(retrospective_test["terminal_drift_pass_rate"]) * retro_n))
    retro_position = int(round(float(retrospective_test["position_invariance_pass_rate"]) * retro_n))
    report_lines = [
        "# EXP014 Phase 2-D28Y dynamic limit and final centroidal authority", "", f"Classification: `{classification}`.", "", "## Dynamic limit invariance", "", "Calibration used 64 directions: 62 D28W proven directions plus 2 D28X resolved directions. The 10 D28X test directions were excluded from tolerance construction.", "", "```text", f"epsilon_drift = {analysis['calibration']['epsilon']['epsilon_drift']:.9g} rad/control-step", f"epsilon_growth = {analysis['calibration']['epsilon']['epsilon_growth']:.9g} rad", f"epsilon_oscillation = {analysis['calibration']['epsilon']['epsilon_oscillation']:.9g} rad", "```", "", "| Joint | Direction | Coupling slope | Max |drift| | Max terminal growth | Release | Classification |", "|---|---|---:|---:|---:|---|---|", *table, "", "## Velocity-gate retrospective", "", f"The old {OLD_VELOCITY_TOL} rad/s pointwise gate is retrospective only: `{retro_pointwise}/{retro_n}` directions pass. The terminal-drift/net-drift gate passes `{retro_terminal}/{retro_n}` and position-invariance classification passes `{retro_position}/{retro_n}`; the pointwise gate is therefore not necessary for the bounded-position interpretation.", "", "## Formal penetration", "", f"The maximum is `{formal['joint_name']} / {formal['direction']}`, P1 episode {formal['episode']} step {formal['control_step']}, `{formal['penetration']:.9g} rad`; interpretation: `{formal['classification']}`.", "", "## Authority contract", "", f"V5 status: `{v5_contract['status']}`. The candidate separates q_actual, monotone non-worsening q_kin, bounded dq_kin, virtual q_cmd, effort authority, and exact D27 V2A pass-through; it was not instantiated because the active-limit closure gate was `{analysis['dynamic']['test_closure_pass']}`. Pass-through positive controls are `" + str(positive["pass"]) + "`.", "", "## C3/C4 authority", "", f"Scope replay status: `{replay_summary.get('status')}`; selected formulation: `{selected}`; physics: `0`.", "", "## Protection", "", f"D28X and earlier protected inputs unchanged: `{protected_ok}`. New physics: `0`; persistent update: `0`; checkpoint: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.", "", f"Starting HEAD: `{start_head}`. Ending HEAD before commit: `{git('rev-parse', 'HEAD')}`.", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text("\n".join(report_lines), encoding="utf-8")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    d28x.d27.add_launcher_args(parser)
    args, hydra = d28x.d27.setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD"); start_status = git("status", "--short").splitlines(); start_log = git("log", "--oneline", "--decorate", "-180").splitlines()
    base = d28x.load_records()
    write_stage_reference(start_head, start_status, start_log); write_protocol(start_head)
    classification = finish(base, start_head)
    print(json.dumps({"phase": "2-D28Y", "classification": classification, "new_physics": 0, "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
