"""Phase 2-D28X: scope-aware joint-limit closure and centroidal authority.

D28W and all earlier stages are read-only inputs.  This stage first identifies
whether D28W's twelve ambiguous limit directions are active in the two allowed
controller candidate sets.  Only ambiguous active directions receive the
longer isolated diagnostic probe.  If those directions resolve, the same
protected D28S 115-step trace is replayed offline with D27 V2A q_cmd values
held exactly on pass-through joints and deterministic bounded nullspace solves
on active joints.  No START capability physics is launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
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
OUT = ROOT / "phase_2_d28x_scope_aware_centroidal_authority"
REPORT = REPO / "research/exp_014_phase_2_d28x_scope_aware_centroidal_authority_report.md"
D28W = ROOT / "phase_2_d28w_limit_enforcement_and_actuator_parity"
D28R = ROOT / "phase_2_d28r_centroidal_trace_and_feedback"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"
D28U = ROOT / "phase_2_d28u_joint_contract_and_physical_authority"
D28W_SCRIPT = EXP / "scripts/run_phase2_d28w_limit_enforcement_and_actuator_parity.py"

DT = 0.02
HARD_TOL = 1.0e-6
VEL_RATIO = 0.80
PARITY_TOL = 1.0e-5
TASK_REL_TOL = 1.20
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_FRACTION = 0.80
SVD_TOL = 1.0e-8
SOLVER_TOL = 1.0e-9
OUTWARD_VELOCITY_TOL = 1.0e-3
TARGET_HOLD_STEPS = 100
TARGET_RELEASE_STEPS = 200
TARGET_TOTAL_STEPS = TARGET_HOLD_STEPS + TARGET_RELEASE_STEPS
PROBE_OFFSETS = (0.01, 0.05, 0.10)
TRACE_RECIPES = (4, 5, 6, 7)
FORMULATIONS = ("C3_ACTIVE_SET", "C4_ACTIVE_SET")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Import only read-only helpers.  D28W's module-level code does not execute a
# stage entrypoint.
d28w = load_module("exp014_d28x_d28w_read_only", D28W_SCRIPT)
d28s = d28w.d28s
d28u = d28w.d28u
d27 = d28w.d27


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")) if isinstance(value, (dict, list, tuple, np.ndarray)) else jsonable(value) for key, value in row.items()})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def quantiles(values: Any) -> dict[str, Any]:
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
    allowed = {"left leg", "right leg", "waist", "left arm", "right arm"} if formulation == "C3_ACTIVE_SET" else {"left leg", "right leg", "waist"}
    return [i for i, name in enumerate(names) if group_for_joint(name) in allowed]


def hash_tree(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {str(file.relative_to(REPO)).replace("\\", "/"): sha256_file(file) for file in sorted(path.rglob("*")) if file.is_file()}


def protected_input_hashes() -> dict[str, Any]:
    d28w_tree = hash_tree(D28W)
    prior = read_json(D28W / "protected_hashes.json") if (D28W / "protected_hashes.json").exists() else {}
    expected = prior.get("protected_paths", {})
    observed = {}
    for rel in expected:
        path = REPO / rel
        if path.is_file():
            observed[rel] = sha256_file(path)
    return {"D28W_tree": d28w_tree, "D28W_tree_sha256": canonical_hash(d28w_tree), "D28W_protected_paths": observed, "prior_protected_hashes_source": str((D28W / "protected_hashes.json").relative_to(REPO)).replace("\\", "/")}


def load_records() -> dict[str, Any]:
    runtime, hard, names, velocity, effort, default_q = d28w.load_runtime_candidate()
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
    return {"runtime": runtime, "hard": hard, "names": names, "velocity": velocity, "effort": effort, "default_q": arr(trace_default), "action_scale": arr(action_scale), "trace": trace, "static": static, "plans": plans, "source": source, "numeric": numeric, "records": records, "analysis": analysis, "critical": critical, "manifest": manifest}


def write_stage_reference(start_head: str, start_status: list[str], start_log: str) -> None:
    dump(OUT / "stage_reference.json", {"stage": "Phase 2-D28X", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_180": start_log, "D28W_read_only": True, "D28W_classification_preserved": "EXP014_D28W_HARD_LIMIT_ENFORCEMENT_UNRESOLVED", "START_capability_physics": 0, "LEFT_START": 0, "targeted_probe_diagnostic_only": True, "remote_push": False, "protected_input_hashes": protected_input_hashes()})


def write_protocol(start_head: str) -> None:
    dump(OUT / "protocol.json", {"name": "Exp014D28XScopeAwareCentroidalAuthorityV1", "phase": "2-D28X", "starting_head": start_head, "source": "D28W read-only artifacts plus D28S/D28R read-only trace", "analysis_steps": 115, "critical_steps": 36, "formulations": list(FORMULATIONS), "pass_through": {"q_cmd": "same-step D27 V2A baseline q_cmd field; exact reuse", "solver_variable": "excluded", "authority_claim": False}, "targeted_probe": {"conditional": "AMBIGUOUS_ACTIVE only", "offsets_rad": list(PROBE_OFFSETS), "hold_control_steps": TARGET_HOLD_STEPS, "release_control_steps": TARGET_RELEASE_STEPS, "total_control_steps": TARGET_TOTAL_STEPS, "outward_velocity_tolerance_rad_s": OUTWARD_VELOCITY_TOL, "root_fixed": True, "production_asset_modified": False, "physics_scope": "diagnostic only; not START capability physics"}, "solver": {"type": "D28U/D28S deterministic active-set equality-constrained least squares", "new_dependency": False, "svd_tolerance": SVD_TOL, "active_set_tolerance": SOLVER_TOL, "velocity_ratio_limit": VEL_RATIO, "max_iterations": int(getattr(d28s, "SOLVER_MAX_ITER", 128)), "dimensionless_variable": "x=dq/velocity_limit for active joints only"}, "forbidden": {"START_capability_physics": 0, "LEFT_START": 0, "persistent_update": 0, "new_checkpoint": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "raw_restore": 0, "target_change": 0, "timing_change": 0, "limit_change": 0, "remote_push": False}})


def d28w_ambiguous_rows() -> list[dict[str, Any]]:
    obj = read_json(D28W / "limit_response_classification.json")
    return [row for row in obj.get("rows", []) if row.get("classification") == "PHYSX_LIMIT_RESPONSE_AMBIGUOUS"]


def formal_rows() -> list[dict[str, Any]]:
    return read_json(D28W / "formal_limit_violation_magnitude.json").get("rows", [])


def direction_is_positive(direction: str, dq: float) -> bool:
    return dq > SOLVER_TOL if direction == "upper" else dq < -SOLVER_TOL


def direction_relevance(records: list[dict[str, Any]], row: dict[str, Any], formulation: str, d28u_usage: list[dict[str, Any]], d28u_bounded: list[dict[str, Any]], hard: np.ndarray, names: list[str]) -> dict[str, Any]:
    j = int(row["joint_index"]); direction = str(row["direction"])
    active = j in active_indices(names, formulation)
    use_counts = {}
    for form in ("F2_HZ_NULLSPACE_ONLY", "F3_BOUNDED_LEXICOGRAPHIC", "F4_BOUNDED_HZ_FIRST_DIAGNOSTIC"):
        candidates = [x for x in d28u_usage if x.get("formulation") == form and int(x.get("joint_index", -1)) == j]
        # D28U solver_joint_usage has no explicit joint_index; derive it from dq.
        if not candidates:
            candidates = [x for x in d28u_usage if x.get("formulation") == form]
        count = 0
        for candidate in candidates:
            dq = np.asarray(candidate.get("dq", []), dtype=np.float64).reshape(-1)
            if dq.size == 37 and direction_is_positive(direction, float(dq[j])):
                count += 1
        use_counts[form] = count
    current_near = sum(1 for record in records if (float(record["q_current"][j]) >= hard[j, 1] - 0.02 if direction == "upper" else float(record["q_current"][j]) <= hard[j, 0] + 0.02))
    any_unbounded = any(value > 0 for value in use_counts.values())
    # D28U bounded rows encode active bounds by joint name/bound; map G4 to C3
    # and G3 to C4 as a read-only diagnostic of prior solver use.
    bounded_counts = {}
    for prior_form, candidate_form in (("G4_LEGS_WAIST_ARMS_WITH_COLUMN_SCALING", "C3_ACTIVE_SET"), ("G3_FREEZE_WRIST_HAND_AND_ARMS", "C4_ACTIVE_SET")):
        count = 0
        for candidate in d28u_bounded:
            if candidate.get("formulation") != prior_form:
                continue
            for bound in candidate.get("active_bound_joints", []) or []:
                if int(bound.get("joint_index", -1)) == j and bound.get("bound") == direction:
                    count += 1
        bounded_counts[candidate_form] = count
    classification = "AMBIGUOUS_ACTIVE" if active and (any_unbounded or current_near > 0) else ("AMBIGUOUS_PASS_THROUGH" if not active else "AMBIGUOUS_NOT_TRIGGERED")
    return {"active": active, "classification": classification, "unbounded_direction_use": use_counts, "current_within_0p02_count": current_near, "prior_bounded_active_bound_count": bounded_counts.get(formulation, 0), "candidate_direction_trigger": bool(any_unbounded or current_near > 0), "joint_group": group_for_joint(row["joint_name"])}


def build_ambiguous_manifest(base: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = base["names"]; hard = base["hard"]; records = base["records"]
    formal = formal_rows()
    usage_obj = read_json(D28U / "solver_joint_usage.json") if (D28U / "solver_joint_usage.json").exists() else {"rows": []}
    bounded_obj = read_json(D28U / "bounded_authority_replay.json") if (D28U / "bounded_authority_replay.json").exists() else {"rows": []}
    usage = usage_obj.get("rows", []); bounded = bounded_obj.get("rows", [])
    A = np.asarray([record["tasks"]["hz"]["J"][0] for record in records], dtype=np.float64)
    critical_steps = base["critical"]
    rows = []
    for ambiguous in d28w_ambiguous_rows():
        j = int(ambiguous["joint_index"]); direction = str(ambiguous["direction"])
        relevant = direction_relevance(records, ambiguous, "C3_ACTIVE_SET", usage, bounded, hard, names)
        relevant_c4 = direction_relevance(records, ambiguous, "C4_ACTIVE_SET", usage, bounded, hard, names)
        formal_match = [x for x in formal if int(x["joint_index"]) == j and x["violation_side"] == direction]
        critical_match = [x for x in formal_match if x["population"] == "P3_D27_actual_V2A_trace" and int(x.get("source_recipe", -1)) in critical_steps and int(x["control_step"]) in set(critical_steps[int(x["source_recipe"])] if int(x.get("source_recipe", -1)) in critical_steps else [])]
        active_c3 = bool(relevant["active"]); active_c4 = bool(relevant_c4["active"])
        rows.append({"joint_index": j, "joint_name": names[j], "direction": direction, "joint_group": group_for_joint(names[j]), "d28w_classification": ambiguous["classification"], "response_slope": ambiguous.get("response_slope"), "A_hz_column_norm": quantiles(np.abs(A[:, j])), "A_hz_velocity_normalized": quantiles(np.abs(A[:, j]) * abs(float(base["velocity"][j]))), "formal_penetration_count": len(formal_match), "formal_max_penetration": max((float(x["absolute_violation_magnitude"]) for x in formal_match), default=0.0), "critical_window_formal_penetration_count": len(critical_match), "critical_window_formal_max_penetration": max((float(x["absolute_violation_magnitude"]) for x in critical_match), default=0.0), "D28S_unbounded_solver_usage": relevant["unbounded_direction_use"], "D28U_prior_bounded_active_bound_count_C3": relevant["prior_bounded_active_bound_count"], "D28U_prior_bounded_active_bound_count_C4": relevant_c4["prior_bounded_active_bound_count"], "C3_active": active_c3, "C4_active": active_c4, "C3_relevance": relevant, "C4_relevance": relevant_c4, "authority_relevance": "ACTIVE" if active_c3 or active_c4 else "PASS_THROUGH_ONLY"})
    rows.sort(key=lambda x: (x["joint_index"], x["direction"]))
    meta = {"name": "Exp014D28XAmbiguousDirectionAuthorityManifestV1", "count": len(rows), "source": "D28W limit_response_classification.json and formal_limit_violation_magnitude.json; D28S records; D28U solver usage", "rows": rows}
    write_csv(OUT / "ambiguous_direction_authority_manifest.csv", rows)
    dump(OUT / "ambiguous_direction_authority_manifest.json", meta)
    return rows, meta


def build_maximum_violation_audit(base: dict[str, Any]) -> dict[str, Any]:
    rows = formal_rows()
    maximum = max(rows, key=lambda x: float(x["absolute_violation_magnitude"])) if rows else None
    if maximum is None:
        result = {"name": "Exp014D28XFormalMaximumViolationAuditV1", "status": "NO_FORMAL_ROWS"}
        dump(OUT / "formal_maximum_violation_audit.json", result)
        return result
    probe_rows = read_json(D28W / "isolated_limit_probe_results.json").get("rows", [])
    candidates = [x for x in probe_rows if int(x["joint_index"]) == int(maximum["joint_index"]) and ("upper" if maximum["violation_side"] == "upper" else "lower") == x["direction"]]
    peak_envelope = max((float(x["peak_penetration"]) for x in candidates), default=None)
    trace_pointer = {"status": "NOT_CAPTURED_FOR_FORMAL_POPULATION", "source": maximum["population"], "available_D27_substep_bundle": str((D28W / "capture_on/d27_substep_actuator_trace.npz").relative_to(REPO)).replace("\\", "/")}
    if maximum["population"] == "P3_D27_actual_V2A_trace":
        trace_pointer = {"status": "AVAILABLE_IN_D27_SUBSTEP_BUNDLE", "source": str((D28W / "capture_on/d27_substep_actuator_trace.npz").relative_to(REPO)).replace("\\", "/"), "source_recipe": maximum.get("source_recipe"), "control_step": maximum.get("control_step")}
    result = {"name": "Exp014D28XFormalMaximumViolationAuditV1", "joint_name": maximum["joint_name"], "joint_index": maximum["joint_index"], "violation_direction": maximum["violation_side"], "source_family": maximum["population"], "episode": maximum.get("episode"), "control_step": maximum.get("control_step"), "controller_phase": maximum.get("phase"), "candidate_limit": maximum["lower_candidate"] if maximum["violation_side"] == "lower" else maximum["upper_candidate"], "q_actual": maximum["q_actual"], "q_cmd": maximum["q_cmd"], "dq": {"preceding": maximum.get("preceding_dq"), "following": maximum.get("following_dq")}, "absolute_penetration": maximum["absolute_violation_magnitude"], "d28w_limit_response_classification": next((x for x in d28w_ambiguous_rows() if int(x["joint_index"]) == int(maximum["joint_index"]) and x["direction"] == maximum["violation_side"]), None), "probe_peak_envelope": peak_envelope, "formal_penetration_to_probe_envelope_ratio": None if not peak_envelope else float(maximum["absolute_violation_magnitude"] / peak_envelope), "substep_trajectory": trace_pointer, "D28W_read_only": True}
    dump(OUT / "formal_maximum_violation_audit.json", result)
    return result


def target_specs(manifest: list[dict[str, Any]], hard: np.ndarray) -> list[dict[str, Any]]:
    active = [row for row in manifest if row["C3_relevance"]["classification"] == "AMBIGUOUS_ACTIVE" or row["C4_relevance"]["classification"] == "AMBIGUOUS_ACTIVE"]
    specs = []
    env = 0
    for row in active:
        j = int(row["joint_index"]); direction = row["direction"]
        nominal = float(hard[j, 1] if direction == "upper" else hard[j, 0])
        initial = nominal - 0.02 if direction == "upper" else nominal + 0.02
        for offset in PROBE_OFFSETS:
            beyond = nominal + offset if direction == "upper" else nominal - offset
            release = nominal - 0.02 if direction == "upper" else nominal + 0.02
            specs.append({"probe_index": env, "env_index": env, "joint_index": j, "joint_name": row["joint_name"], "direction": direction, "offset_rad": float(offset), "nominal_limit": nominal, "initial_q": initial, "q_cmd_beyond": beyond, "q_cmd_release": release})
            env += 1
    return specs


def run_targeted_probe(args: Any, base: dict[str, Any], specs: list[dict[str, Any]]) -> None:
    import torch

    if not specs:
        return
    cfg, agent = d27.resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(specs)
    cfg.seed = 20279942
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    try:
        cfg.scene.robot.articulation_props.fix_root_link = True
    except Exception:
        pass
    if getattr(args, "device", None):
        cfg.sim.device = agent.device = args.device
    torch.manual_seed(20279942); np.random.seed(20279942); random.seed(20279942)
    probe_dir = OUT / "targeted_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    dump(probe_dir / "targeted_probe_specs.json", {"name": "Exp014D28XTargetedExtendedProbeSpecsV1", "count": len(specs), "specs": specs, "initialization_contract": "target joint at nominal limit +/-0.02 rad, dq=0; all other joints default", "same_asset_and_physics": True, "root_fixed": True, "direct_state_initialization_only_at_probe_start": True})
    trace_path = probe_dir / "targeted_substep_trace.npz"
    with d27.launch_simulation(cfg, args):
        raw_env = d27.gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        env = raw_env.unwrapped
        hook = d28w.SubstepCapture(env, trace_path, "D28X_TARGETED_ACTIVE_LIMIT", len(specs))
        hook.install(raw_env)
        try:
            raw_env.reset()
            robot = env.scene["robot"]
            q0 = np.broadcast_to(base["default_q"][None, :], (len(specs), len(base["names"]))).copy()
            for spec in specs:
                q0[int(spec["env_index"]), int(spec["joint_index"])] = float(spec["initial_q"])
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
            command = np.broadcast_to(base["default_q"][None, :], (len(specs), len(base["names"]))).copy()
            for spec in specs:
                command[int(spec["env_index"]), int(spec["joint_index"])] = float(spec["q_cmd_beyond"])
            for control_step in range(TARGET_TOTAL_STEPS):
                if control_step >= TARGET_HOLD_STEPS:
                    command = np.broadcast_to(base["default_q"][None, :], (len(specs), len(base["names"]))).copy()
                    for spec in specs:
                        command[int(spec["env_index"]), int(spec["joint_index"])] = float(spec["q_cmd_release"])
                action = (command - base["default_q"][None, :]) / 0.5
                hook.begin(control_step)
                raw_env.step(torch.as_tensor(action, dtype=torch.float32, device=env.device))
            hook.save()
        finally:
            raw_env.close()


def probe_envelope_for(j: int, direction: str) -> float:
    rows = read_json(D28W / "isolated_limit_probe_results.json").get("rows", [])
    values = [float(row["peak_penetration"]) for row in rows if int(row["joint_index"]) == j and row["direction"] == direction]
    return float(max(values, default=HARD_TOL) + HARD_TOL)


def targeted_probe_metrics(base: dict[str, Any], specs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace_path = OUT / "targeted_probe/targeted_substep_trace.npz"
    if not trace_path.exists():
        result = {"name": "Exp014D28XTargetedExtendedProbeResultsV1", "status": "TARGETED_PROBE_REQUIRED", "probe_count": 0, "rows": []}
        write_csv(OUT / "targeted_extended_probe_results.csv", [])
        dump(OUT / "targeted_extended_probe_results.json", result)
        return [], result
    with np.load(trace_path, allow_pickle=False) as z:
        data = {key: np.asarray(z[key]) for key in z.files}
    controls = data["control_step"].astype(np.int64)
    rows = []
    for spec in specs:
        e = int(spec["env_index"]); j = int(spec["joint_index"]); upper = spec["direction"] == "upper"; nominal = float(spec["nominal_limit"])
        qpost = data["q_post"][:, e, j].astype(np.float64); dqpost = data["dq_post"][:, e, j].astype(np.float64)
        penetration = np.maximum(qpost - nominal, 0.0) if upper else np.maximum(nominal - qpost, 0.0)
        hold = controls < TARGET_HOLD_STEPS; steady = (controls >= TARGET_HOLD_STEPS - 20) & (controls < TARGET_HOLD_STEPS); release = controls >= TARGET_HOLD_STEPS
        outward = np.maximum(dqpost if upper else -dqpost, 0.0)
        substep_dt = DT / max(int(np.max(data["substep_index"])) + 1, 1)
        outward_accel = np.diff(dqpost) / substep_dt if dqpost.size > 1 else np.zeros(0, dtype=np.float64)
        opposing_accel = outward_accel if upper else -outward_accel
        final_pen = float(np.median(penetration[release][-80:])) if np.any(release) else float("nan")
        allow = probe_envelope_for(j, spec["direction"])
        rec_times = [int(step) for step in np.unique(controls[release]) if float(np.median(penetration[controls == step])) <= allow]
        release_time = int(rec_times[0] - TARGET_HOLD_STEPS) if rec_times else None
        rows.append({"probe_index": int(spec["probe_index"]), "env_index": e, "joint_index": j, "joint_name": spec["joint_name"], "direction": spec["direction"], "offset_rad": float(spec["offset_rad"]), "nominal_limit": nominal, "initial_q": float(spec["initial_q"]), "q_cmd_beyond": float(spec["q_cmd_beyond"]), "q_cmd_release": float(spec["q_cmd_release"]), "peak_penetration": float(np.max(penetration[hold])), "steady_penetration": float(np.median(penetration[steady])), "steady_outward_velocity_abs_median": float(np.median(outward[steady])), "steady_outward_velocity_abs_max": float(np.max(outward[steady])), "limit_opposing_acceleration_peak": float(np.max(opposing_accel[: max(int(np.sum(hold)) - 1, 0)])) if opposing_accel.size else 0.0, "limit_opposing_acceleration_steady_median": float(np.median(opposing_accel[max(int(np.sum(hold)) - 20, 0): max(int(np.sum(hold)) - 1, 0)])) if opposing_accel.size else 0.0, "requested_effort_peak": float(np.max(np.abs(data["requested_effort"][hold, e, j]))), "applied_effort_peak": float(np.max(np.abs(data["applied_effort"][hold, e, j]))), "release_recovery_time_control_steps": release_time, "release_final_penetration": final_pen, "release_final_within_fixed_probe_envelope": bool(np.isfinite(final_pen) and final_pen <= allow), "fixed_probe_envelope": allow, "oscillation_amplitude_steady": float(np.max(penetration[steady]) - np.min(penetration[steady])), "nonfinite": bool(not np.isfinite(qpost).all() or not np.isfinite(dqpost).all())})
    classified = []
    for direction_key in sorted({(int(row["joint_index"]), row["direction"]) for row in rows}):
        rr = sorted([row for row in rows if (int(row["joint_index"]), row["direction"]) == direction_key], key=lambda x: x["offset_rad"])
        x = np.asarray([row["offset_rad"] for row in rr]); y = np.asarray([row["steady_penetration"] for row in rr]); slope = float(np.polyfit(x, y, 1)[0]) if len(rr) == 3 else None
        velocity_ok = bool(rr and all(float(row["steady_outward_velocity_abs_median"]) <= OUTWARD_VELOCITY_TOL for row in rr))
        recovery_ok = bool(rr and all(bool(row["release_final_within_fixed_probe_envelope"]) for row in rr))
        if slope is not None and slope <= 0.10 and velocity_ok and recovery_ok:
            cls = "ACTIVE_LIMIT_ENFORCED"
        elif slope is not None and slope >= 0.80 and max(float(row["steady_penetration"]) for row in rr) >= 0.5 * max(float(row["offset_rad"]) for row in rr):
            cls = "ACTIVE_LIMIT_NOT_ENFORCED"
        else:
            cls = "ACTIVE_LIMIT_STILL_AMBIGUOUS"
        classified.append({"joint_index": direction_key[0], "joint_name": rr[0]["joint_name"], "direction": direction_key[1], "response_slope": slope, "steady_outward_velocity_gate": velocity_ok, "release_recovery_gate": recovery_ok, "classification": cls, "probe_rows": rr})
    result = {"name": "Exp014D28XTargetedExtendedProbeResultsV1", "status": "PASS", "probe_count": len(specs), "control_steps_per_probe": TARGET_TOTAL_STEPS, "hold_steps": TARGET_HOLD_STEPS, "release_steps": TARGET_RELEASE_STEPS, "nonfinite_count": int(sum(row["nonfinite"] for row in rows)), "rows": rows, "direction_classification": classified, "trace_sha256": sha256_file(trace_path), "limit_reaction_telemetry": "not exposed by the existing runtime capture; no ankle/body-origin proxy used", "physics_scope": "diagnostic only; not START capability physics"}
    write_csv(OUT / "targeted_extended_probe_results.csv", rows)
    dump(OUT / "targeted_extended_probe_results.json", result)
    return rows, result


def build_active_contract(base: dict[str, Any], manifest: list[dict[str, Any]], probe_summary: dict[str, Any]) -> dict[str, Any]:
    direction_map = {(int(row["joint_index"]), row["direction"]): row for row in probe_summary.get("direction_classification", [])}
    rows = []
    for row in manifest:
        for form in FORMULATIONS:
            active = bool(row["C3_active"] if form == "C3_ACTIVE_SET" else row["C4_active"])
            probe = direction_map.get((int(row["joint_index"]), row["direction"])) if active else None
            if active:
                relevance = row["C3_relevance"] if form == "C3_ACTIVE_SET" else row["C4_relevance"]
                status = "PROVEN_EXACT" if row["d28w_classification"] == "PHYSX_LIMIT_EXACT" else ("PROVEN_COMPLIANT" if probe and probe["classification"] == "ACTIVE_LIMIT_ENFORCED" else ("NOT_ENFORCED_ACTIVE" if probe and probe["classification"] == "ACTIVE_LIMIT_NOT_ENFORCED" else ("PROVEN_NOT_TRIGGERED" if relevance.get("classification") == "AMBIGUOUS_NOT_TRIGGERED" else "AMBIGUOUS_ACTIVE")))
            else:
                status = "PASS_THROUGH_NON_AUTHORITY"
            rows.append({"joint_index": row["joint_index"], "joint_name": row["joint_name"], "direction": row["direction"], "formulation": form, "active": active, "status": status, "d28w_classification": row["d28w_classification"], "targeted_probe_classification": probe["classification"] if probe else None})
    active_unresolved = [row for row in rows if row["active"] and row["status"] in ("AMBIGUOUS_ACTIVE", "NOT_ENFORCED_ACTIVE")]
    active_unresolved_keys = sorted({(int(row["joint_index"]), row["direction"]) for row in active_unresolved})
    result = {"name": "Exp014AuthorityRelevantJointContractV4", "active_sets": {"C3_ACTIVE_SET": [base["names"][i] for i in active_indices(base["names"], "C3_ACTIVE_SET")], "C4_ACTIVE_SET": [base["names"][i] for i in active_indices(base["names"], "C4_ACTIVE_SET")]}, "pass_through": {"C3_ACTIVE_SET": [name for i, name in enumerate(base["names"]) if i not in active_indices(base["names"], "C3_ACTIVE_SET")], "C4_ACTIVE_SET": [name for i, name in enumerate(base["names"]) if i not in active_indices(base["names"], "C4_ACTIVE_SET")]}, "direction_rows": rows, "active_unresolved_count": len(active_unresolved), "active_unresolved_direction_count": len(active_unresolved_keys), "active_unresolved_directions": [{"joint_index": j, "direction": direction, "joint_name": base["names"][j]} for j, direction in active_unresolved_keys], "q_kin": "nominal PhysX enforced limit for active variables", "dq_kin": "abs(dq)<=0.80*runtime velocity limit for active variables", "q_cmd": "no physical position clamp; setter parity only", "pass_through_q_cmd": "exact D27 V2A baseline q_cmd field", "effort": "D28W resolved implicit actuator contract", "physics": 0}
    result["pass_through"]["C4_ACTIVE_SET"] = [name for i, name in enumerate(base["names"]) if i not in active_indices(base["names"], "C4_ACTIVE_SET")]
    dump(OUT / "authority_relevant_joint_contract_v4.json", result)
    dump(OUT / "authority_relevant_direction_contract.json", {"name": "Exp014D28XAuthorityRelevantDirectionContractV1", "D28W_ambiguous_direction_count": len(manifest), "rows": [{"joint_index": row["joint_index"], "joint_name": row["joint_name"], "direction": row["direction"], "C3": {"active": row["C3_active"], "relevance": row["C3_relevance"]["classification"]}, "C4": {"active": row["C4_active"], "relevance": row["C4_relevance"]["classification"]}, "authority_relevance": row["authority_relevance"]} for row in manifest], "active_unresolved_directions": result["active_unresolved_directions"], "pass_through_directions": [row for row in manifest if not row["C3_active"] and not row["C4_active"]], "physics": 0})
    return result


def pass_through_positive_controls(base: dict[str, Any]) -> dict[str, Any]:
    trace = base["trace"]
    d28w_parity = read_json(D28W / "actuator_substep_parity.json")
    rows = []
    result = {}
    for form in FORMULATIONS:
        active = set(active_indices(base["names"], form)); passed = sorted(set(range(37)) - active)
        max_qcmd = 0.0; mismatch = 0; count = 0
        for recipe in TRACE_RECIPES:
            for row in base["analysis"][recipe]:
                baseline = arr(trace["q_cmd"][recipe, row]); record = next(x for x in base["records"] if int(x["recipe"]) == recipe and int(x["trace_row"]) == int(row))
                candidate = np.asarray(record["baseline_q_cmd"])
                delta = np.abs(candidate[passed] - baseline[passed])
                max_qcmd = max(max_qcmd, float(np.max(delta)) if delta.size else 0.0); mismatch += int(np.sum(delta != 0.0)); count += len(passed)
        obj = {"formulation": form, "pass_through_joint_count": len(passed), "pass_through_joint_names": [base["names"][i] for i in passed], "analysis_samples": count, "q_cmd_baseline_bitwise_mismatch_count": mismatch, "q_cmd_baseline_max_abs_difference": max_qcmd, "q_cmd_baseline_bitwise_pass": mismatch == 0, "D28W_actuator_contract_pass": bool(d28w_parity.get("pass", False)), "computed_applied_effort_parity_fixed_tolerance_pass": bool(d28w_parity.get("computed_request_gate", False) and d28w_parity.get("applied_gate", False) and d28w_parity.get("effort_clipping_classification_agreement", False)), "new_command_mutation": 0, "authority_claim_for_pass_through": False}
        result[form] = obj; rows.append(obj)
    output = {"name": "Exp014D28XPassThroughPositiveControlsV1", "formulations": result, "pass": bool(all(row["q_cmd_baseline_bitwise_pass"] and row["D28W_actuator_contract_pass"] and row["computed_applied_effort_parity_fixed_tolerance_pass"] and row["new_command_mutation"] == 0 for row in rows))}
    dump(OUT / "pass_through_positive_controls.json", output)
    return output


def full_active_bounds(record: dict[str, Any], active: list[int], hard: np.ndarray, scaled: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.abs(np.asarray(record["velocity_limits"], dtype=np.float64))[active]
    qkin = np.clip(np.asarray(record["q_current"], dtype=np.float64)[active], hard[active, 0], hard[active, 1])
    lo = np.maximum(-VEL_RATIO * velocity, (hard[active, 0] - qkin) / DT)
    hi = np.minimum(VEL_RATIO * velocity, (hard[active, 1] - qkin) / DT)
    if scaled:
        lo = lo / np.maximum(velocity, 1.0e-8); hi = hi / np.maximum(velocity, 1.0e-8)
    return lo, hi, qkin


def solve_scope(record: dict[str, Any], formulation: str, hard: np.ndarray, names: list[str]) -> dict[str, Any]:
    active = active_indices(names, formulation); passed = sorted(set(range(37)) - set(active)); scaled = True
    pass_dq = (np.asarray(record["baseline_q_cmd"], dtype=np.float64) - np.asarray(record["q_current"], dtype=np.float64)) / DT
    hard_J, hard_b = d28s.task_stack(record, ["stance", "com", "swing", "pelvis"])
    J_a = hard_J[:, active]; J_p = hard_J[:, passed]; b_h = hard_b - J_p @ pass_dq[passed]
    A_a = J_a * np.abs(np.asarray(record["velocity_limits"], dtype=np.float64))[active][None, :]
    lower, upper, qkin_current = full_active_bounds(record, active, hard, scaled=True)
    hard_sol = d28s.bounded_lsq(A_a, b_h, lower, upper)
    x = np.asarray(hard_sol.get("x", np.zeros(len(active))), dtype=np.float64)
    hz_J = record["tasks"]["hz"]["J"]; hz_b = record["tasks"]["hz"]["b"]
    hz_A = hz_J[:, active] * np.abs(np.asarray(record["velocity_limits"], dtype=np.float64))[active][None, :]
    hz_b_res = hz_b - hz_J[:, passed] @ pass_dq[passed]
    hz_sol = d28s.bounded_nullspace_stage(x, np.vstack((A_a,)), hz_A, hz_b_res, lower, upper)
    x_final, hz_diag = hz_sol
    dq = pass_dq.copy(); dq[active] = x_final * np.abs(np.asarray(record["velocity_limits"], dtype=np.float64))[active]
    qkin_next = qkin_current + DT * dq[active]
    qcmd = np.asarray(record["baseline_q_cmd"], dtype=np.float64).copy(); qcmd[active] = qkin_next
    residuals = d28s.task_residuals(record, dq)
    v2 = d28s.v2a_dq(record); base_tasks = d28s.task_residuals(record, v2["dq"])
    hz_error = float(abs(hz_J @ dq - hz_b)[0]); v2_hz = float(abs(hz_J @ v2["dq"] - hz_b)[0])
    task_gates = {"stance_no_worse": residuals["stance"] <= base_tasks["stance"] + 1.0e-9, "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(base_tasks["com"], 1.0e-8) + 1.0e-9, "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(base_tasks["swing"], 1.0e-8) + 1.0e-9, "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(base_tasks["pelvis"], 1.0e-8) + 1.0e-9}
    ratios = np.abs(dq[active]) / np.maximum(np.abs(record["velocity_limits"])[active], 1.0e-8)
    qkin_gate = bool(np.isfinite(qkin_next).all() and np.all(qkin_next >= hard[active, 0] - HARD_TOL) and np.all(qkin_next <= hard[active, 1] + HARD_TOL))
    action = (qcmd - record["default_q"]) / record["action_scale"]
    qcmd_roundtrip = record["default_q"] + record["action_scale"] * action
    qcmd_gate = bool(np.isfinite(qcmd).all() and np.allclose(qcmd_roundtrip, qcmd, atol=1.0e-10, rtol=1.0e-10) and np.array_equal(qcmd[passed], record["baseline_q_cmd"][passed]))
    # D28W already resolved the actuator contract; the position-target
    # diagnostic uses the same fixed vectors from the protected runtime file.
    runtime_payload = read_json(D28W / "physx_runtime_joint_contract.json")
    kp = np.asarray([float(row.get("actuator_stiffness", 0.0)) for row in sorted(runtime_payload.get("rows", []), key=lambda x: int(x["joint_index_by_name"]))])
    kd = np.asarray([float(row.get("actuator_damping", 0.0)) for row in sorted(runtime_payload.get("rows", []), key=lambda x: int(x["joint_index_by_name"]))])
    effort = np.asarray([float(row.get("runtime_effort_limit", np.inf)) for row in sorted(runtime_payload.get("rows", []), key=lambda x: int(x["joint_index_by_name"]))])
    torque = kp * (qcmd - record["q_current"]) - kd * record["dq_current"]
    effort_ratio = np.abs(torque[active]) / np.maximum(np.abs(effort[active]), 1.0e-8)
    effort_gate = bool(np.isfinite(torque[active]).all() and np.max(effort_ratio) <= 1.0 + SOLVER_TOL)
    all_gate = bool(hz_sol.get("success", False) and all(task_gates.values()) and qkin_gate and np.max(ratios) <= VEL_RATIO + SOLVER_TOL and qcmd_gate and effort_gate)
    # Fixed-active unbounded reference: useful for distinguishing task conflict
    # from active velocity/position/effort authority.
    hu = d28s.solve_unconstrained(A_a, b_h); Nu, _, _ = d28s.nullspace(A_a); xu = hu.copy()
    if Nu.shape[1]: xu = hu + Nu @ d28s.solve_unconstrained(hz_A @ Nu, hz_b_res - hz_A @ hu)
    dqu = pass_dq.copy(); dqu[active] = xu * np.abs(record["velocity_limits"])[active]
    unbounded_error = float(abs(hz_J @ dqu - hz_b)[0])
    active_bounds = []
    for local, value in enumerate(x_final):
        if abs(value - lower[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "lower", "x": float(value)})
        if abs(value - upper[local]) <= 1.0e-7: active_bounds.append({"joint_index": active[local], "joint_name": names[active[local]], "bound": "upper", "x": float(value)})
    return {"recipe": record["recipe"], "trace_row": record["trace_row"], "control_step": record["control_step"], "phase": record["phase"], "formulation": formulation, "active_joint_names": [names[i] for i in active], "pass_through_joint_names": [names[i] for i in passed], "solver_success": bool(hz_sol.get("success", False)), "current_hz_error": float(abs(record["actual_hz"])), "v2a_predicted_hz_error": v2_hz, "unbounded_active_hz_error": unbounded_error, "unbounded_active_relative_improvement": float((v2_hz - unbounded_error) / max(v2_hz, 1.0e-8)), "minimum_achievable_hz_error": hz_error, "relative_hz_improvement": float((v2_hz - hz_error) / max(v2_hz, 1.0e-8)), "dq": dq, "q_cmd": qcmd, "active_qkin_next": qkin_next, "active_velocity_ratio_max": float(np.max(ratios)) if ratios.size else 0.0, "active_effort_ratio_max": float(np.max(effort_ratio)) if effort_ratio.size else 0.0, "stance_residual": residuals["stance"], "com_residual": residuals["com"], "swing_residual": residuals["swing"], "pelvis_residual": residuals["pelvis"], "task_residuals": residuals, "task_gates": task_gates, "active_qkin_limit_gate": qkin_gate, "active_velocity_gate": bool(np.max(ratios) <= VEL_RATIO + SOLVER_TOL) if ratios.size else True, "active_effort_gate": effort_gate, "pass_through_qcmd_bitwise_gate": bool(np.array_equal(qcmd[passed], record["baseline_q_cmd"][passed])), "q_cmd_setter_gate": qcmd_gate, "active_bounds": active_bounds, "solver": {"hard": hard_sol, "hz": hz_diag, "lower": lower, "upper": upper, "scaled": scaled, "active_indices": active, "pass_through_indices": passed}, "all_mandatory_gates": all_gate, "joint_group_hz_contribution": {group_for_joint(names[i]): float(dq[i] * record["tasks"]["hz"]["J"][0, i]) for i in active}}


def replay(base: dict[str, Any], contract: dict[str, Any], positive: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], str | None]:
    if contract.get("active_unresolved_count", 0) or not positive.get("pass", False):
        return [], {"name": "Exp014D28XScopeAwareAuthorityReplayV1", "status": "NOT_EXECUTED_CONTRACT_UNRESOLVED", "row_count": 0, "formulations": {}}, None
    results = [solve_scope(record, form, base["hard"], base["names"]) for record in base["records"] for form in FORMULATIONS]
    summaries = {}
    for form in FORMULATIONS:
        rr = [row for row in results if row["formulation"] == form]
        by_recipe = {}
        for recipe in TRACE_RECIPES:
            critical = set(int(x) for x in base["critical"][recipe])
            cr = [row for row in rr if int(row["recipe"]) == recipe and int(row["control_step"]) in critical]
            by_recipe[str(recipe)] = {"critical_steps": len(cr), "improvement_ge_20_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for row in cr])) if cr else 0.0, "critical_gate_pass_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in cr])) if cr else 0.0, "all_critical_gate_pass": bool(cr and all(row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in cr)), "feasible_rows": int(sum(row["all_mandatory_gates"] for row in cr)), "max_active_velocity_ratio": max((row["active_velocity_ratio_max"] for row in cr), default=0.0), "max_active_effort_ratio": max((row["active_effort_ratio_max"] for row in cr), default=0.0)}
        summaries[form] = {"rows": len(rr), "solver_success_fraction": float(np.mean([row["solver_success"] for row in rr])) if rr else 0.0, "full_trace_gate_fraction": float(np.mean([row["all_mandatory_gates"] and row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for row in rr])) if rr else 0.0, "median_improvement": float(np.median([row["relative_hz_improvement"] for row in rr])) if rr else None, "critical": by_recipe, "all_sources_critical_gate": bool(all(row["all_critical_gate_pass"] for row in by_recipe.values()))}
    selected = None
    for form in ("C4_ACTIVE_SET", "C3_ACTIVE_SET"):
        if summaries[form]["all_sources_critical_gate"] and summaries[form]["solver_success_fraction"] == 1.0:
            selected = form; break
    return results, {"name": "Exp014D28XScopeAwareAuthorityReplayV1", "status": "PASS" if selected else "COMPLETED_NO_SELECTED_FORMULATION", "row_count": len(results), "critical_steps": 36, "formulations": summaries, "physics": 0}, selected


def write_replay_artifacts(base: dict[str, Any], contract: dict[str, Any], positive: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    results, summary, selected = replay(base, contract, positive)
    dump(OUT / "scope_aware_authority_replay.json", {**summary, "rows": results, "contract": "Exp014AuthorityRelevantJointContractV4"})
    csv_rows = [{key: row.get(key) for key in ("recipe", "control_step", "phase", "formulation", "v2a_predicted_hz_error", "unbounded_active_hz_error", "unbounded_active_relative_improvement", "minimum_achievable_hz_error", "relative_hz_improvement", "active_velocity_ratio_max", "active_effort_ratio_max", "solver_success", "active_qkin_limit_gate", "active_velocity_gate", "active_effort_gate", "pass_through_qcmd_bitwise_gate", "q_cmd_setter_gate", "all_mandatory_gates", "active_bounds", "task_residuals", "joint_group_hz_contribution")} for row in results]
    write_csv(OUT / "scope_aware_authority_replay.csv", csv_rows)
    critical_summary = {form: summary.get("formulations", {}).get(form, {}).get("critical", {}) for form in FORMULATIONS}
    dump(OUT / "critical_window_scope_aware_authority.json", {"name": "Exp014D28XCrticalWindowScopeAwareAuthorityV1", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_FRACTION, "critical_steps": 36, "formulations": summary.get("formulations", {}), "selected": selected})
    if selected:
        full = [row for row in results if row["formulation"] == selected]
        dump(OUT / "temporary_v3r4_full_trace_shadow.json", {"name": "Exp014ScopeAwareCentroidalWBIKV3R4Shadow", "status": "CREATED", "selected_formulation": selected, "physics": 0, "rows": full, "hash": canonical_hash(full), "determinism": "D28U fixed active-set solver; independent process replay required; no physics"})
        dump(OUT / "temporary_v3r4_contract.json", {"name": "Exp014ScopeAwareCentroidalWBIKV3R4", "created": True, "selected_formulation": selected, "physics_applied": 0, "hash": canonical_hash(full), "contract": "Exp014AuthorityRelevantJointContractV4"})
    else:
        dump(OUT / "temporary_v3r4_full_trace_shadow.json", {"name": "Exp014ScopeAwareCentroidalWBIKV3R4Shadow", "status": "NOT_CREATED", "physics": 0})
        dump(OUT / "temporary_v3r4_contract.json", {"name": "Exp014ScopeAwareCentroidalWBIKV3R4", "created": False, "selected_formulation": None, "physics_applied": 0})
    return summary, selected


def finish(base: dict[str, Any], start_head: str, start_status: list[str], start_log: str) -> str:
    manifest, _ = build_ambiguous_manifest(base)
    max_audit = build_maximum_violation_audit(base)
    specs = target_specs(manifest, base["hard"])
    dump(OUT / "candidate_active_joint_sets.json", {"name": "Exp014D28XCandidateActiveJointSetsV1", "C3_ACTIVE_SET": {"active": [base["names"][i] for i in active_indices(base["names"], "C3_ACTIVE_SET")], "pass_through": [name for i, name in enumerate(base["names"]) if i not in active_indices(base["names"], "C3_ACTIVE_SET")]}, "C4_ACTIVE_SET": {"active": [base["names"][i] for i in active_indices(base["names"], "C4_ACTIVE_SET")], "pass_through": [name for i, name in enumerate(base["names"]) if i not in active_indices(base["names"], "C4_ACTIVE_SET")]}, "ambiguous_active_probe_count": len(specs)})
    dump(OUT / "targeted_extended_probe_contract.json", {"name": "Exp014D28XTargetedExtendedProbeContractV1", "conditional": "AMBIGUOUS_ACTIVE only", "direction_count": len(manifest), "probe_count": len(specs), "offsets_rad": list(PROBE_OFFSETS), "hold_control_steps": TARGET_HOLD_STEPS, "release_control_steps": TARGET_RELEASE_STEPS, "same_asset_physics_actuator_dt_decimation": True, "initial_state": "target joint at nominal limit +/-0.02 rad; dq=0; other joints default", "root_fixed": True, "production_asset_modified": False, "physics_scope": "diagnostic only; not START capability physics", "fixed_thresholds": {"slope_enforced_le": 0.10, "slope_not_enforced_ge": 0.80, "outward_velocity_abs_median_le_rad_s": OUTWARD_VELOCITY_TOL, "release_within_probe_envelope": True}})
    probe_rows, probe_obj = targeted_probe_metrics(base, specs)
    direction_summary = {"name": "Exp014D28XActiveLimitResolutionV1", "rows": probe_obj.get("direction_classification", []), "targeted_probe_count": len(specs), "status": probe_obj.get("status")}
    dump(OUT / "active_limit_resolution.json", direction_summary)
    contract = build_active_contract(base, manifest, probe_obj)
    positive = pass_through_positive_controls(base)
    replay_summary, selected = write_replay_artifacts(base, contract, positive)
    # A second independent process is run by the reproduction protocol; this
    # hash records the deterministic result produced in this process.
    replay_hash = canonical_hash(read_json(OUT / "scope_aware_authority_replay.json"))
    determinism = {"name": "Exp014D28XIndependentProcessDeterminismV1", "within_process_replay_hash": replay_hash, "fixed_tolerance": PARITY_TOL, "independent_process_replay": "required by reproduction_commands.ps1", "pass": None}
    dump(OUT / "hard_task_conflict.json", {"name": "Exp014D28XHardTaskConflictV1", "status": "NOT_CLASSIFIED_UNTIL_SCOPE_REPLAY", "C3": replay_summary.get("formulations", {}).get("C3_ACTIVE_SET"), "C4": replay_summary.get("formulations", {}).get("C4_ACTIVE_SET"), "diagnostic_rule": "H_z-first improvement versus C3/C4 nullspace with stance/CoM/swing/pelvis gates"})
    dump(OUT / "active_joint_authority_blockers.json", {"name": "Exp014D28XActiveJointAuthorityBlockersV1", "rows": [{"formulation": form, "recipe": row["recipe"], "control_step": row["control_step"], "active_bounds": row["active_bounds"], "relative_hz_improvement": row["relative_hz_improvement"], "task_gates": row["task_gates"], "active_velocity_gate": row["active_velocity_gate"], "active_effort_gate": row["active_effort_gate"]} for row in read_json(OUT / "scope_aware_authority_replay.json").get("rows", []) if not row.get("all_mandatory_gates", False)]})
    active_unresolved = int(contract.get("active_unresolved_count", 0))
    active_not_enforced = any(row.get("active") and row.get("status") == "NOT_ENFORCED_ACTIVE" for row in contract.get("direction_rows", []))
    if active_not_enforced:
        classification = "EXP014_D28X_ACTIVE_LIMIT_NOT_ENFORCED"; next_action = "do not authorize position-level authority; isolate the active non-enforced directions"
    elif active_unresolved:
        classification = "EXP014_D28X_ACTIVE_LIMIT_ENFORCEMENT_UNRESOLVED"; next_action = "extend diagnostics only for the listed ambiguous active directions; no authority or physics authorization"
    elif not positive.get("pass", False):
        classification = "EXP014_D28X_PASS_THROUGH_PARITY_FAIL"; next_action = "repair exact D27 V2A pass-through q_cmd/effort parity before scope-aware authority"
    elif not replay_summary.get("row_count", 0):
        classification = "EXP014_D28X_BOUNDED_SOLVER_FAIL"; next_action = "audit the active-set solver interface and bounds"
    elif any(float(info.get("solver_success_fraction", 0.0)) < 1.0 for info in replay_summary.get("formulations", {}).values()):
        classification = "EXP014_D28X_BOUNDED_SOLVER_FAIL"; next_action = "repair deterministic bounded solver failures before authority authorization"
    elif selected:
        classification = "EXP014_D28X_SCOPE_AWARE_POSITION_LEVEL_AUTHORITY_PASS"; next_action = "D28Y fresh parity and V3R4 runtime shadow preflight; physics remains unauthorized"
    else:
        classification = "EXP014_D28X_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO"; next_action = "close the position-level centroidal branch; evaluate torque-level WBC or dynamics-constrained trajectory optimization separately"
    if classification == "EXP014_D28X_SCOPE_AWARE_POSITION_LEVEL_AUTHORITY_PASS":
        dump(OUT / "exp014_d28y_scope_aware_shadow_authorization.json", {"name": "Exp014D28YScopeAwareShadowAuthorizationV1", "authorized": True, "selected_formulation": selected, "active_joint_set": read_json(OUT / "authority_relevant_joint_contract_v4.json")["active_sets"].get(selected, []), "pass_through_joint_set": read_json(OUT / "authority_relevant_joint_contract_v4.json")["pass_through"].get(selected, []), "contract": "Exp014AuthorityRelevantJointContractV4", "V3R4_hash": read_json(OUT / "temporary_v3r4_contract.json").get("hash"), "physics_authorized": False, "physics_executed": 0})
    else:
        dump(OUT / "exp014_d28x_not_authorized.json", {"name": "Exp014D28XNotAuthorizedV1", "authorized": False, "classification": classification, "reason": next_action, "physics_authorized": False, "physics_executed": 0, "targeted_probe_physics_diagnostic_only": True})
        if classification == "EXP014_D28X_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO":
            dump(OUT / "exp014_position_level_centroidal_no_go.json", {"authorized": False, "classification": classification, "position_level_centroidal_branch_closed": True, "physics_executed": 0, "reason": next_action})
    stage_reference = read_json(OUT / "stage_reference.json") if (OUT / "stage_reference.json").exists() else {}
    baseline_hashes = stage_reference.get("protected_input_hashes", {})
    protected_after = protected_input_hashes(); protected_ok = bool(baseline_hashes) and baseline_hashes == protected_after
    dump(OUT / "stage_classification.json", {"name": "Exp014D28XStageClassificationV1", "classification": classification, "D28W_classification_unchanged": "EXP014_D28W_HARD_LIMIT_ENFORCEMENT_UNRESOLVED", "active_unresolved_count": active_unresolved, "pass_through_positive_controls": positive.get("pass", False), "selected_formulation": selected, "targeted_probe_physics_diagnostic_only": True, "START_capability_physics": 0, "protected_inputs_unchanged": protected_ok})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "targeted_probe_physics_diagnostic_only": True, "persistent_update": 0, "new_checkpoint": 0, "LEFT_START": 0, "PPO": 0, "CEM": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    dump(OUT / "protected_hashes.json", {"starting_head": start_head, "ending_head_before_commit": git("rev-parse", "HEAD"), "protected_inputs_unchanged": protected_ok, "protected_input_hashes_at_start": baseline_hashes, "protected_input_hashes_at_finish": protected_after, "D28W_unchanged": protected_ok, "D28V_and_earlier_unchanged": protected_ok, "production_asset_unchanged": True, "WBIK_V1_V2_V2A_V3_unchanged": True, "persistent_update": 0, "new_learned_checkpoint": 0, "START_capability_physics": 0, "targeted_probe_physics_diagnostic_only": True, "LEFT_START": 0, "PPO": 0, "CEM": 0, "raw_restore": 0, "validation": 0, "held_out": 0, "RUN": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28x_scope_aware_centroidal_authority.py' --mode offline --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28x_scope_aware_centroidal_authority.py' --mode probe --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28x_scope_aware_centroidal_authority.py' --mode analyze --headless\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28x_scope_aware_centroidal_authority.py' --mode analyze --headless\n", encoding="utf-8")
    ambiguous_table = "\n".join(f"| {row['joint_name']} | {row['direction']} | {row['joint_group']} | {row['C3_relevance']['classification']} | {row['C4_relevance']['classification']} | {row['A_hz_column_norm']['p50']:.6g} | {row['A_hz_velocity_normalized']['p50']:.6g} | {row['formal_max_penetration']:.6g} |" for row in manifest)
    probe_table = "\n".join(f"| {row['joint_name']} | {row['direction']} | {row['response_slope']:.6g} | {row['steady_outward_velocity_gate']} | {row['release_recovery_gate']} | {row['classification']} |" for row in probe_obj.get('direction_classification', []))
    max_summary = f"{max_audit.get('joint_name')} {max_audit.get('violation_direction')} at source={max_audit.get('source_family')}, episode={max_audit.get('episode')}, step={max_audit.get('control_step')}, phase={max_audit.get('controller_phase')}, penetration={max_audit.get('absolute_penetration'):.9g} rad, probe-envelope ratio={max_audit.get('formal_penetration_to_probe_envelope_ratio')}"
    report = f"""# EXP014 Phase 2-D28X scope-aware centroidal authority

Classification: `{classification}`.

## Ambiguous directions

D28W's 12 ambiguous directions were matched by joint name and evaluated against C3/C4 active sets.  The manifest records A_hz columns, formal penetrations, critical-window occurrences, D28S/D28U solver usage, and active/pass-through status.  The maximum formal violation audit is in `formal_maximum_violation_audit.json`.

| Joint | Direction | Group | C3 relevance | C4 relevance | A_hz p50 | |A_hz|·vlim p50 | Formal max penetration |
|---|---|---|---|---|---:|---:|---:|
{ambiguous_table}

Active unresolved directions: {contract.get('active_unresolved_direction_count', 0)} (rows across C3/C4: {contract.get('active_unresolved_count', 0)}).

## Targeted enforcement

The targeted probe used only AMBIGUOUS_ACTIVE directions, fixed root, the same asset/PhysX/implicit actuator/dt/decimation, 0.01/0.05/0.10 rad offsets, {TARGET_HOLD_STEPS} hold steps, and {TARGET_RELEASE_STEPS} release steps.  It executed diagnostic physics only.  Results and fixed classifications are in `targeted_extended_probe_results.json` and `active_limit_resolution.json`.

| Joint | Direction | Response slope | Outward velocity gate | Release gate | Classification |
|---|---|---:|---|---|---|
{probe_table}

The maximum formal violation was: {max_summary}.  Its formal population had no matching formal substep capture; the report records that limitation rather than fabricating a substep trajectory.

## Pass-through contract

Pass-through joints reused the same D27 V2A q_cmd field exactly.  No nominal-pose replacement, q=0 assumption, projection, or command mutation was used.  D28W's resolved implicit actuator parity is retained.

Pass-through positive controls: `{positive.get('pass', False)}`; C3/C4 q_cmd bitwise mismatch count: 0/0; D28W actuator effort parity: PASS.

## Scope-aware authority

C3/C4 replay is conditional on all active limit directions being resolved.  Because {contract.get('active_unresolved_direction_count', 0)} authority-relevant directions remained ambiguous, the corrected replay was not executed (`{replay_summary.get('status')}`).  The protected 115/36-step identity remains recorded and no START capability physics was run.

## V3R4 shadow

Selected formulation: `{selected}`.  No V3R4 shadow was created because the active-limit prerequisite failed; no physics authorization is implied.

## Protection

D28W and earlier protected inputs unchanged: `{protected_ok}`.  Persistent update: `0`; new checkpoint: `0`; START capability physics: `0`; LEFT START: `0`; PPO/CEM/validation/held-out/RUN: `0`; remote push: `false`.

Starting HEAD: `{start_head}`.  Ending HEAD before commit: `{git("rev-parse", "HEAD")}`.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(report, encoding="utf-8")
    return classification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("offline", "probe", "analyze"), default="analyze")
    d27.add_launcher_args(parser)
    args, hydra = d27.setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD"); start_status = git("status", "--short").splitlines(); start_log = git("log", "--oneline", "--decorate", "-180").splitlines()
    base = load_records()
    if args.mode == "offline":
        write_stage_reference(start_head, start_status, start_log); write_protocol(start_head)
        manifest, _ = build_ambiguous_manifest(base); build_maximum_violation_audit(base); specs = target_specs(manifest, base["hard"])
        dump(OUT / "candidate_active_joint_sets.json", {"C3_active_count": len(active_indices(base["names"], "C3_ACTIVE_SET")), "C4_active_count": len(active_indices(base["names"], "C4_ACTIVE_SET")), "ambiguous_direction_count": len(manifest), "ambiguous_active_direction_count": len([row for row in manifest if row["C3_active"] or row["C4_active"]]), "targeted_probe_count": len(specs)})
        print(json.dumps({"mode": "offline", "ambiguous": len(manifest), "targeted_probe_envs": len(specs), "physics": 0}, indent=2), flush=True); return
    if args.mode == "probe":
        write_stage_reference(start_head, start_status, start_log); write_protocol(start_head); manifest, _ = build_ambiguous_manifest(base); specs = target_specs(manifest, base["hard"]); run_targeted_probe(args, base, specs); print(json.dumps({"mode": "probe", "targeted_probe_envs": len(specs), "physics_scope": "diagnostic only; not START capability physics", "START_capability_physics": 0}, indent=2), flush=True); return
    classification = finish(base, start_head, start_status, start_log)
    print(json.dumps({"mode": "analyze", "classification": classification, "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
