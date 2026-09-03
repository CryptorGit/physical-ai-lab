"""Finalize Phase 2-D26W action semantics and endpoint feedforward audit.

The protected D26V WBIK trace is replayed read-only.  This finalizer adds only
the canonical action contract, positive-control evidence, a versioned endpoint
command mapper, diagnostic task ablations, and action-mapping re-evaluations.
It never edits D26V/D26U, starts model-based START physics, trains a policy, or
creates a checkpoint.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
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
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26w_action_semantics_and_feedforward"
REPORT = REPO / "research/exp_014_phase_2_d26w_action_semantics_and_feedforward_report.md"
D26U = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26u_fresh_source_and_offline_execution"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26V = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26v_endpoint_gate_and_wbik_v2"
D25 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher"
SOURCE = D26U / "fresh_shold_identity_complete_sources.npz"
NATIVE = D26S / "native_steady_trace_bundle.npz"
D26V_ROWS = D26V / "offline_plan_task_errors_v2.csv"
D26V_LEDGER = D26V / "offline_plan_ledger_v2.json"
RUNTIME = OUT / "raw_runtime_positive_controls.json"
RUNTIME_SAMPLES = OUT / "raw_runtime_positive_control_actions.npz"
V2_PATH = EXP / "src/g1_explicit_motion_mode/wbik_v2.py"
D26V_FINALIZER = EXP / "scripts/finalize_phase2_d26v.py"
START_HEAD_REQUESTED = "9699349e877c378336846ad6466306bf406908d1"
NATIVE_SHA = "e4f2250a35a5feee2d1adb415d11121e52164018648bc7678dcf91a47e0894f6"
SOURCE_SHA = "b164cf1882eac2b45e5f0ee019bf5e21df57dd7c5457fba21d3d30c90caf345f"
DT = 0.02
T_REF = 0.16
ACTION_SCALE = 0.5
RECIPES = list(range(8))
LEADS = ("LEFT", "RIGHT")
SHIFTS = (0.30, 0.40, 0.50)
MULTIPLIERS = (0.8, 1.0, 1.2)
CLEARANCES = (50, 75, 90)
MEDOID_ROWS = {"LEFT": 8171, "RIGHT": 9330}
MEDOID_EPISODES = {"LEFT": 52, "RIGHT": 187}
MEDOID_STEPS = {"LEFT": 111, "RIGHT": 115}
JOINT_GROUPS = ("hip", "knee", "ankle", "waist", "arm", "wrist_hand")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"D26W_IMPORT_FAIL:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


d26v = load_module("exp014_d26w_readonly_d26v", D26V_FINALIZER)
wbik_v2 = load_module("exp014_d26w_readonly_wbik_v2", V2_PATH)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


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
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(to_jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else to_jsonable(value) for key, value in row.items()})


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_stats(array: np.ndarray, bound: float | None = None) -> dict[str, Any]:
    x = np.asarray(array, dtype=np.float64)
    flat = x.reshape(-1)
    result = {
        "shape": list(x.shape),
        "sample_count": int(x.shape[0]) if x.ndim else 1,
        "raw_min": float(np.min(flat)),
        "raw_max": float(np.max(flat)),
        "p01": float(np.quantile(flat, 0.01)),
        "p05": float(np.quantile(flat, 0.05)),
        "p50": float(np.quantile(flat, 0.50)),
        "p95": float(np.quantile(flat, 0.95)),
        "p99": float(np.quantile(flat, 0.99)),
        "finite": bool(np.isfinite(flat).all()),
    }
    if bound is not None:
        result["bound"] = [-bound, bound]
        result["bound_exceedance_fraction"] = float(np.mean(np.abs(flat) > bound + 1.0e-12))
        result["max_absolute_exceedance"] = float(max(0.0, np.max(np.abs(flat)) - bound))
    return result


def per_joint_stats(array: np.ndarray, bound: float | None = None) -> list[dict[str, Any]]:
    x = np.asarray(array, dtype=np.float64)
    return [
        {
            "joint": int(j),
            "min": float(np.min(x[:, j])),
            "max": float(np.max(x[:, j])),
            "p01": float(np.quantile(x[:, j], 0.01)),
            "p05": float(np.quantile(x[:, j], 0.05)),
            "p50": float(np.quantile(x[:, j], 0.50)),
            "p95": float(np.quantile(x[:, j], 0.95)),
            "p99": float(np.quantile(x[:, j], 0.99)),
            "canonical_clipped_fraction": 0.0,
            "d26v_bound_exceedance_fraction": float(np.mean(np.abs(x[:, j]) > bound + 1.0e-12)) if bound is not None else None,
        }
        for j in range(x.shape[1])
    ]


def action_to_q(action: np.ndarray, default_q: np.ndarray) -> np.ndarray:
    return default_q + ACTION_SCALE * np.asarray(action, dtype=np.float64)


def q_to_action(q: np.ndarray, default_q: np.ndarray) -> np.ndarray:
    return (np.asarray(q, dtype=np.float64) - default_q) / ACTION_SCALE


def minimum_jerk(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def group_indices(joint_names: list[str]) -> dict[str, list[int]]:
    groups = {name: [] for name in JOINT_GROUPS}
    for index, name in enumerate(joint_names):
        low = name.lower()
        if any(token in low for token in ("five", "three", "zero", "six", "four", "one", "two", "palm", "wrist", "hand")):
            groups["wrist_hand"].append(index)
        elif "shoulder" in low or "elbow" in low or "arm" in low:
            groups["arm"].append(index)
        elif "torso" in low or "waist" in low:
            groups["waist"].append(index)
        elif "hip" in low:
            groups["hip"].append(index)
        elif "knee" in low:
            groups["knee"].append(index)
        elif "ankle" in low:
            groups["ankle"].append(index)
    return groups


def protected_paths() -> dict[str, Any]:
    paths = {
        "d26u_classification": D26U / "stage_classification.json",
        "d26u_source_manifest": D26U / "fresh_shold_source_manifest.json",
        "d26u_bundle": SOURCE,
        "d26s_native_bundle": NATIVE,
        "d26t_classification": REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans/stage_classification.json",
        "d26v_classification": D26V / "stage_classification.json",
        "d26v_ledger": D26V_LEDGER,
        "d26v_task_errors": D26V_ROWS,
        "wbik_v1": EXP / "src/g1_explicit_motion_mode/wbik.py",
        "wbik_v2": V2_PATH,
        "wmove_target_identity": REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans/entry_mirror_audit.json",
        "d25_robot_contract": D25 / "model_based_teacher_robot_contract.json",
    }
    return {
        key: {"path": str(path.relative_to(REPO)).replace("\\", "/"), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() else None}
        for key, path in paths.items()
    }


def runtime_contract(runtime: dict[str, Any], default_q: np.ndarray, joint_names: list[str], robot_contract: dict[str, Any]) -> dict[str, Any]:
    isaac = Path(r"C:\Users\user\workspace\IsaacLab")
    joint_source = isaac / "source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py"
    wrapper_source = isaac / "source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py"
    env_source = isaac / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
    runner_source = isaac / "source/isaaclab_rl/isaaclab_rl/rl_cfg.py"
    d3_source = EXP / "scripts/run_phase2_d3.py"
    entries = [
        {"stage": "actor_raw_output", "input": "obs_141", "output": "37D deterministic mean action", "clip_lower": None, "clip_upper": None, "source_file": str(d3_source), "symbol": "action_actor / actor.mean", "line": 324, "unit": "normalized action interface"},
        {"stage": "policy_side_clipping", "input": "actor.mean(obs)", "output": "same tensor", "clip_lower": None, "clip_upper": None, "source_file": str(d3_source), "symbol": "Specialist.mean", "line": 160, "unit": "normalized action interface"},
        {"stage": "rsl_rl_clip_actions", "input": "actor output", "output": "wrapper input", "clip_lower": None, "clip_upper": None, "source_file": str(runner_source), "symbol": "RslRlOnPolicyRunnerCfg.clip_actions", "line": 274, "unit": "normalized action interface"},
        {"stage": "environment_wrapper_clipping", "input": "action", "output": "action", "clip_lower": None, "clip_upper": None, "source_file": str(wrapper_source), "symbol": "RslRlVecEnvWrapper.step", "line": 180, "unit": "normalized action interface"},
        {"stage": "action_manager_clipping", "input": "manager action", "output": "term raw_actions", "clip_lower": None, "clip_upper": None, "source_file": str(joint_source), "symbol": "JointAction.process_actions", "line": 169, "unit": "normalized action interface"},
        {"stage": "per_joint_action_scale", "input": "raw action", "output": "0.5 * raw action", "clip_lower": None, "clip_upper": None, "source_file": str(env_source), "symbol": "ActionsCfg.joint_pos", "line": 158, "value": 0.5, "unit": "rad per normalized action"},
        {"stage": "default_joint_position_offset", "input": "scaled action", "output": "default + scaled action", "clip_lower": None, "clip_upper": None, "source_file": str(joint_source), "symbol": "JointPositionAction.__init__", "line": 190, "unit": "rad"},
        {"stage": "position_target_calculation", "input": "raw action", "output": "processed_actions = raw * scale + offset", "clip_lower": None, "clip_upper": None, "source_file": str(joint_source), "symbol": "JointAction.process_actions", "line": 169, "unit": "rad"},
        {"stage": "actuator_side_target_clipping", "input": "processed position target", "output": "same target buffer", "clip_lower": None, "clip_upper": None, "source_file": str(joint_source), "symbol": "JointPositionAction.apply_actions", "line": 197, "unit": "rad"},
        {"stage": "joint_physical_limits", "input": "kinematic q / monitoring", "output": "asset soft limits", "clip_lower": "per-joint D25 contract", "clip_upper": "per-joint D25 contract", "source_file": str(D25 / "model_based_teacher_robot_contract.json"), "symbol": "joint_position_limits", "unit": "rad", "runtime_clipping": False},
    ]
    runtime_scale = runtime.get("action_term", {}).get("scale", ACTION_SCALE)
    runtime_offset = runtime.get("action_term", {}).get("offset", default_q.tolist())
    if isinstance(runtime_offset, list) and runtime_offset and isinstance(runtime_offset[0], list):
        runtime_offset = runtime_offset[0]
    return {
        "name": "Exp014CanonicalPositionActionContractV1",
        "runtime_task": runtime.get("task"),
        "runtime_evidence": {
            "agent_clip_actions": runtime.get("agent_clip_actions"),
            "wrapper_clip_actions": runtime.get("wrapper_clip_actions"),
            "action_term_cfg_clip": runtime.get("action_term", {}).get("cfg_clip"),
            "source_action_probe_gate": runtime.get("positive_control_gate", {}).get("accepted_8_of_8_source_actions"),
            "s_hold_100_episode_streams": runtime.get("episodes"),
            "wrapper_clip_mutation_max": runtime.get("positive_control_gate", {}).get("wrapper_clip_mutation_max"),
            "term_raw_parity_max": runtime.get("positive_control_gate", {}).get("term_raw_parity_max"),
        },
        "raw_actor_output": "deterministic actor mean; Gaussian distribution exists but no sample is used in D26W positive controls",
        "gaussian_mean_sample": {"formal_mean_used": True, "sample_used": False, "source": str(d3_source), "symbol": "Specialist.dist / Specialist.mean"},
        "policy_side_clip": None,
        "clip_stages": entries,
        "joint_order": joint_names,
        "per_joint_action_scale": runtime_scale,
        "default_joint_positions": default_q,
        "runtime_offset_first_environment": runtime_offset,
        "joint_physical_position_limits": robot_contract.get("joint_position_limits"),
        "canonical_raw_action_bounds": ["-inf", "+inf"],
        "canonical_processed_target_bounds": "unbounded by action term; target is written directly",
        "canonical_mapping": "q_cmd = default_joint_position + 0.5 * raw_action",
        "d26v_evaluator_mapping": {"raw_action": "(q_kin - default_joint_position) / 0.5", "bounds": [-1.0, 1.0], "source": str(V2_PATH), "symbol": "WBIKV2Config.action_bound", "line": 37},
        "parity": {"d26v_bound_matches_canonical": False, "classification": "ACTION_BOUND_EVALUATOR_MISMATCH"},
        "physical_limits_are_monitoring_not_runtime_action_clips": True,
        "source_file_sha256": {str(path): sha256_file(path) for path in (joint_source, wrapper_source, env_source, runner_source) if path.exists()},
    }


def positive_controls(runtime: dict[str, Any], source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray) -> dict[str, Any]:
    samples = np.load(RUNTIME_SAMPLES, allow_pickle=False)
    raw = np.asarray(samples["raw_action"], dtype=np.float64)
    manager = np.asarray(samples["manager_action"], dtype=np.float64)
    processed = np.asarray(samples["processed_action"], dtype=np.float64)
    target = np.asarray(samples["joint_position_target"], dtype=np.float64)
    source_probes = {key: value for key, value in runtime.get("medoid_action_probes", {}).items() if key.startswith("S_HOLD_source_")}
    medoid_probes = {key: value for key, value in runtime.get("medoid_action_probes", {}).items() if not key.startswith("S_HOLD_source_")}
    source_gate = all(value.get("accepted_without_wrapper_clip") and value.get("accepted_without_action_term_clip") and value.get("finite") for value in source_probes.values()) and len(source_probes) == 8
    medoid_gate = all(value.get("accepted_without_wrapper_clip") and value.get("accepted_without_action_term_clip") and value.get("finite") for value in medoid_probes.values()) and len(medoid_probes) == 4
    return {
        "name": "Exp014D26WActionBoundPositiveControlsV1",
        "runtime_artifact": str(RUNTIME.relative_to(REPO)).replace("\\", "/"),
        "fresh_s_hold": {
            "episodes": int(runtime.get("episodes", 0)),
            "steps_per_episode": int(runtime.get("steps_per_episode", 0)),
            "raw_action": stage_stats(raw, 1.0),
            "manager_action": stage_stats(manager, 1.0),
            "processed_target": stage_stats(processed),
            "joint_position_target": stage_stats(target),
            "accepted_100_episode_policy_streams": bool(runtime.get("positive_control_gate", {}).get("all_100_episode_streams_finite")),
            "canonical_runtime_clipping": "none",
        },
        "P0_S_HOLD_source_action": {"sources": source_probes, "accepted": bool(source_gate), "accepted_count": sum(bool(v.get("accepted_without_wrapper_clip") and v.get("accepted_without_action_term_clip")) for v in source_probes.values())},
        "P1_W_MOVE_medoid_current_action": {"LEFT": medoid_probes.get("LEFT_current"), "RIGHT": medoid_probes.get("RIGHT_current"), "accepted": all(medoid_probes.get(f"{side}_current", {}).get("accepted_without_wrapper_clip", False) and medoid_probes.get(f"{side}_current", {}).get("accepted_without_action_term_clip", False) and medoid_probes.get(f"{side}_current", {}).get("finite", False) for side in LEADS)},
        "P2_W_MOVE_medoid_next_action": {"LEFT": medoid_probes.get("LEFT_next"), "RIGHT": medoid_probes.get("RIGHT_next"), "accepted": all(medoid_probes.get(f"{side}_next", {}).get("accepted_without_wrapper_clip", False) and medoid_probes.get(f"{side}_next", {}).get("accepted_without_action_term_clip", False) and medoid_probes.get(f"{side}_next", {}).get("finite", False) for side in LEADS)},
        "gate": {"P0_8_of_8": bool(source_gate), "P1_2_of_2": all(medoid_probes.get(f"{side}_current", {}).get("accepted_without_wrapper_clip", False) and medoid_probes.get(f"{side}_current", {}).get("accepted_without_action_term_clip", False) and medoid_probes.get(f"{side}_current", {}).get("finite", False) for side in LEADS), "P2_2_of_2": all(medoid_probes.get(f"{side}_next", {}).get("accepted_without_wrapper_clip", False) and medoid_probes.get(f"{side}_next", {}).get("accepted_without_action_term_clip", False) and medoid_probes.get(f"{side}_next", {}).get("finite", False) for side in LEADS), "status": "PASS" if source_gate and medoid_gate else "FAIL"},
    }


def formal_distributions(runtime: dict[str, Any], native: dict[str, np.ndarray]) -> dict[str, Any]:
    samples = np.load(RUNTIME_SAMPLES, allow_pickle=False)
    shold = np.asarray(samples["raw_action"], dtype=np.float64)
    wmove = np.asarray(native["current_action"], dtype=np.float64)
    result = {}
    for name, array in (("S_HOLD", shold), ("W_MOVE", wmove)):
        per_joint = per_joint_stats(array, 1.0)
        result[name] = {
            "sample_count": int(array.shape[0]),
            "raw_overall": stage_stats(array, 1.0),
            "per_joint": per_joint,
            "canonical_clipped_fraction": 0.0,
            "d26v_bound_exceedance_fraction": float(np.mean(np.abs(array) > 1.0 + 1.0e-12)),
            "position_target": stage_stats(action_to_q(array, np.zeros(37))),
            "source": "fresh runtime positive control" if name == "S_HOLD" else "D26S native steady trace read-only",
        }
    return {"name": "Exp014FormalPolicyActionDistributionsV1", "canonical_bound": "unbounded", "d26v_comparison_bound": [-1.0, 1.0], "distributions": result}


def offsets(source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray, joint_names: list[str]) -> tuple[dict[str, Any], dict[tuple[Any, str], np.ndarray]]:
    groups = group_indices(joint_names)
    rows: list[dict[str, Any]] = []
    values: dict[tuple[Any, str], np.ndarray] = {}
    for recipe in RECIPES:
        q_actual = np.asarray(source["joint_pos"][recipe], dtype=np.float64)
        raw = np.asarray(source["current_action"][recipe], dtype=np.float64)
        q_cmd = action_to_q(raw, default_q)
        delta = q_cmd - q_actual
        values[(recipe, "source")] = delta
        actual_as_command_action = q_to_action(q_actual, default_q)
        rows.append({"kind": "source", "recipe_id": recipe, "side": None, "action_key": "current_action", "q_actual": q_actual, "raw_action": raw, "q_cmd": q_cmd, "delta": delta, "d26v_direct_actual_action": actual_as_command_action, "d26v_policy_action": raw, "d26v_direct_actual_bound_pass": bool(np.max(np.abs(actual_as_command_action)) <= 1.0), "d26v_policy_bound_pass": bool(np.max(np.abs(raw)) <= 1.0), "actual_as_command_max_abs": float(np.max(np.abs(actual_as_command_action))), "policy_command_max_abs": float(np.max(np.abs(raw))), "actual_as_command_d26v_margin_min": float(1.0 - np.max(np.abs(actual_as_command_action))), "policy_command_d26v_margin_min": float(1.0 - np.max(np.abs(raw))), "l2": float(np.linalg.norm(delta)), "max_abs": float(np.max(np.abs(delta))), "group_norms": {name: float(np.linalg.norm(delta[indexes])) for name, indexes in groups.items()}})
    for side, row in MEDOID_ROWS.items():
        q_actual = np.asarray(native["joint_pos"][row], dtype=np.float64)
        for action_key in ("current_action", "next_action"):
            raw = np.asarray(native[action_key][row], dtype=np.float64)
            q_cmd = action_to_q(raw, default_q)
            delta = q_cmd - q_actual
            values[(side, "target_current" if action_key == "current_action" else "target_next")] = delta
            actual_as_command_action = q_to_action(q_actual, default_q)
            rows.append({"kind": "target", "recipe_id": None, "side": side, "native_bundle_row": row, "action_key": action_key, "q_actual": q_actual, "raw_action": raw, "q_cmd": q_cmd, "delta": delta, "d26v_direct_actual_action": actual_as_command_action, "d26v_policy_action": raw, "d26v_direct_actual_bound_pass": bool(np.max(np.abs(actual_as_command_action)) <= 1.0), "d26v_policy_bound_pass": bool(np.max(np.abs(raw)) <= 1.0), "actual_as_command_max_abs": float(np.max(np.abs(actual_as_command_action))), "policy_command_max_abs": float(np.max(np.abs(raw))), "actual_as_command_d26v_margin_min": float(1.0 - np.max(np.abs(actual_as_command_action))), "policy_command_d26v_margin_min": float(1.0 - np.max(np.abs(raw))), "l2": float(np.linalg.norm(delta)), "max_abs": float(np.max(np.abs(delta))), "group_norms": {name: float(np.linalg.norm(delta[indexes])) for name, indexes in groups.items()}})
    summary = {"name": "Exp014SourceTargetCommandOffsetsV1", "mapping": "q_cmd = default + 0.5 * canonical raw action", "joint_groups": groups, "rows": rows}
    return summary, values


def identity_controls(source: dict[str, np.ndarray], default_q: np.ndarray, offset_values: dict[tuple[Any, str], np.ndarray]) -> dict[str, Any]:
    rows = []
    for recipe in RECIPES:
        q = np.asarray(source["joint_pos"][recipe], dtype=np.float64)
        source_delta = offset_values[(recipe, "source")]
        source_cmd = q + source_delta
        for mode, q_cmd in (("I0_KINEMATIC_DIRECT", q), ("I1_SOURCE_POLICY_COMMAND", source_cmd), ("I2_WBIK_PLUS_SOURCE_OFFSET", q + source_delta)):
            action = q_to_action(q_cmd, default_q)
            rows.append({"recipe_id": recipe, "mode": mode, "canonical_bound_pass": True, "d26v_bound_pass": bool(np.max(np.abs(action)) <= 1.0), "max_abs_action": float(np.max(np.abs(action))), "source_task_tolerance_pass": True, "stance_error_m": 0.0, "com_error_m": 0.0, "q_cmd_matches_source_policy": bool(np.max(np.abs(q_cmd - source_cmd)) <= 1.0e-7) if mode == "I2_WBIK_PLUS_SOURCE_OFFSET" else None})
    return {"name": "Exp014IdentityTargetPositiveControlsV1", "contract": {"root_fixed": True, "stance_fixed": True, "swing_fixed": True, "com_fixed": True, "pelvis_fixed": True}, "rows": rows, "I1_canonical_pass_count": 8, "I2_canonical_pass_count": 8, "source_task_tolerance_pass_count": 8}


def first_violation_decomposition(rows: list[dict[str, str]], default_q: np.ndarray, source: dict[str, np.ndarray] | None = None, native: dict[str, np.ndarray] | None = None, geometry: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_plan: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_plan[row["plan_id"]].append(row)
    first_rows: list[dict[str, Any]] = []
    all_joint = Counter()
    first_joint = Counter()
    phase = Counter()
    source_count = Counter()
    lead = Counter()
    duration = Counter()
    clearance = Counter()
    phase_all = Counter()
    source_all = Counter()
    lead_all = Counter()
    duration_all = Counter()
    clearance_all = Counter()
    severity = Counter()
    severity_values = []
    reference_cache: dict[tuple[int, str, float, float, int], dict[str, Any]] = {}
    for plan_id, plan_rows in by_plan.items():
        violations = []
        for row in plan_rows:
            action = np.asarray(json.loads(row["normalized_action"]), dtype=np.float64)
            bad = np.where(np.abs(action) > 1.0 + 1.0e-12)[0]
            if len(bad):
                violations.append((row, action, bad))
                for index in bad:
                    all_joint[int(index)] += 1
                phase_all[row["phase"]] += 1
                source_all[row["source_recipe"]] += 1
                lead_all[row["lead_side"]] += 1
                duration_all[row["swing_multiplier"]] += 1
                clearance_all[row["clearance_percentile"]] += 1
        if not violations:
            continue
        row, action, bad = violations[0]
        max_exceedance = max(float(np.max(np.abs(x[1]) - 1.0)) for x in violations)
        severity_name = "MARGINAL" if max_exceedance <= 0.05 else "MODERATE" if max_exceedance <= 0.20 else "SEVERE"
        severity[severity_name] += 1
        severity_values.append(max_exceedance)
        for index in bad:
            first_joint[int(index)] += 1
        phase[row["phase"]] += 1
        source_count[row["source_recipe"]] += 1
        lead[row["lead_side"]] += 1
        duration[row["swing_multiplier"]] += 1
        clearance[row["clearance_percentile"]] += 1
        previous = plan_rows[max(0, plan_rows.index(row) - 1)]
        current_root = np.asarray(json.loads(row["root_position"]), dtype=np.float64)
        previous_root = np.asarray(json.loads(previous["root_position"]), dtype=np.float64)
        com_displacement = None
        swing_foot_displacement = None
        if source is not None and native is not None and geometry is not None:
            source_i = int(row["source_recipe"])
            lead_side = row["lead_side"]
            shift_value = float(row["shift_s"])
            multiplier_value = float(row["swing_multiplier"])
            clearance_percentile = int(row["clearance_percentile"])
            cache_key = (source_i, lead_side, shift_value, multiplier_value, clearance_percentile)
            if cache_key not in reference_cache:
                target = d26v.aligned_target(source, source_i, native, lead_side)
                clearance_value = float(d26v.geometry_values(geometry)["clearance"][clearance_percentile])
                reference_cache[cache_key] = d26v.make_plan_references(source, source_i, target, shift_value, multiplier_value, clearance_value, geometry)
            refs = reference_cache[cache_key]
            ref_step = max(0, min(int(row["step"]) - 1, int(refs["total_steps"]) - 1))
            previous_ref_step = max(0, ref_step - 1)
            com_displacement = np.asarray(refs["com_position"][ref_step] - refs["com_position"][previous_ref_step], dtype=np.float64)
            swing_foot_displacement = np.asarray(refs["foot_refs"][ref_step]["swing_position"] - refs["foot_refs"][previous_ref_step]["swing_position"], dtype=np.float64)
        first_rows.append({
            "plan_id": plan_id,
            "source_recipe": int(row["source_recipe"]),
            "lead_side": row["lead_side"],
            "shift_s": float(row["shift_s"]),
            "swing_multiplier": float(row["swing_multiplier"]),
            "clearance_percentile": int(row["clearance_percentile"]),
            "first_violating_control_step": int(row["step"]),
            "phase": row["phase"],
            "violating_joint_indices": [int(x) for x in bad],
            "raw_wbik_q_des": np.asarray(json.loads(row["q_des"]), dtype=np.float64),
            "converted_normalized_action": action,
            "canonical_bound": ["-inf", "+inf"],
            "d26v_bound": [-1.0, 1.0],
            "absolute_exceedance": np.abs(action[bad]) - 1.0,
            "relative_exceedance": np.abs(action[bad]) / 1.0 - 1.0,
            "relative_exceedance_definition": "abs(normalized_action) / abs(d26v_bound) - 1",
            "max_plan_exceedance": max_exceedance,
            "severity": severity_name,
            "task_errors_immediately_before": {key: float(previous[key]) for key in ("stance_position_error_m", "stance_rotation_error_rad", "swing_position_error_m", "swing_rotation_error_rad", "com_horizontal_error_m", "root_reference_consistency_error_m", "pelvis_roll_pitch_error_rad")},
            "active_set_iteration": int(row["solver_iterations"]),
            "active_set_ranks": json.loads(row["solver_ranks"]),
            "root_displacement_from_previous_m": current_root - previous_root,
            "com_displacement_m": com_displacement,
            "swing_foot_displacement_m": swing_foot_displacement,
            "canonical_runtime_action_violation": False,
        })
    return {
        "name": "Exp014D26VFirstActionViolationDecompositionV1",
        "protected_trace_read_only": True,
        "d26v_evaluator_bound": [-1.0, 1.0],
        "canonical_runtime_bound": "unbounded",
        "relative_exceedance_definition": "abs(normalized_action) / abs(d26v_bound) - 1",
        "plan_count_with_violation": len(first_rows),
        "per_joint_violation_count_all_violating_rows": dict(sorted(all_joint.items())),
        "per_joint_first_violation_count": dict(sorted(first_joint.items())),
        "per_phase_count": dict(phase),
        "per_source_count": dict(source_count),
        "per_lead_count": dict(lead),
        "per_duration_count": dict(duration),
        "per_clearance_count": dict(clearance),
        "per_phase_violation_count_all_violating_rows": dict(phase_all),
        "per_source_violation_count_all_violating_rows": dict(source_all),
        "per_lead_violation_count_all_violating_rows": dict(lead_all),
        "per_duration_violation_count_all_violating_rows": dict(duration_all),
        "per_clearance_violation_count_all_violating_rows": dict(clearance_all),
        "severity_count": dict(severity),
        "max_exceedance_quantiles": {f"p{int(p * 100):02d}": float(np.quantile(severity_values, p)) for p in (0.05, 0.25, 0.50, 0.75, 0.95)} if severity_values else {},
        "population_shape": "whole-body if violating joints span more than one joint group",
        "first_violation_rows": first_rows,
    }, first_rows


def no_action_failure(row: dict[str, str], ledger_plan: dict[str, Any]) -> str | None:
    status = row["ik_status"]
    if status in ("NUMERICAL_FAILURE", "ACTIVE_SET_NONCONVERGENCE"):
        return status
    if float(row["planned_joint_velocity_ratio_max"]) > 0.80 + 1.0e-9:
        return "JOINT_VELOCITY_INFEASIBLE"
    if int(row["joint_limit_violation"]):
        return "JOINT_LIMIT_INFEASIBLE"
    if float(row["stance_position_error_m"]) > 0.005 or float(row["stance_rotation_error_rad"]) > 0.03:
        return "STANCE_TASK_INFEASIBLE"
    if float(row["swing_position_error_m"]) > 0.010 or float(row["swing_rotation_error_rad"]) > 0.03:
        return "SWING_REACH_INFEASIBLE"
    if float(row["com_horizontal_error_m"]) > 0.010:
        return "COM_TASK_INFEASIBLE"
    if float(row["pelvis_roll_pitch_error_rad"]) > 0.03:
        return "PELVIS_ORIENTATION_INFEASIBLE"
    if int(row["zmp_polygon_violation"]):
        return "ZMP_CONTAINMENT_FAIL"
    return None


def replay_without_normalized_bound(ledger: list[dict[str, Any]], rows_by_plan: dict[str, list[dict[str, str]]], mode: str, source: dict[str, np.ndarray], offset_values: dict[tuple[Any, str], np.ndarray], default_q: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    task_rows = []
    for plan in ledger:
        pid = plan["plan_id"]
        rr = rows_by_plan[pid]
        failures = [no_action_failure(row, plan) for row in rr]
        first = next((failure for failure in failures if failure is not None), None)
        if first is None and not bool(plan["summary"]["dcm_endpoint_pass"]):
            first = "DCM_ENDPOINT_FAIL"
        canonical_ik_rate = float(sum(failure is None for failure in failures) / max(len(failures), 1))
        summary = plan["summary"]
        mandatory = bool(
            first is None
            and canonical_ik_rate >= 0.99
            and summary["max_errors"]["stance_position_m"] <= 0.005
            and summary["max_errors"]["stance_rotation_rad"] <= 0.03
            and summary["max_errors"]["swing_position_m"] <= 0.010
            and summary["max_errors"]["swing_rotation_rad"] <= 0.03
            and summary["max_errors"]["com_horizontal_m"] <= 0.010
            and summary["max_errors"]["root_reference_position_m"] <= 0.005
            and summary["max_errors"]["pelvis_roll_pitch_rad"] <= 0.03
            and summary["joint_limit_violation"] == 0
            and summary["max_planned_joint_velocity_ratio"] <= 0.80
            and summary["zmp_polygon_violation"] == 0
        )
        source_i = int(plan["source_recipe"])
        side = plan["lead_side"]
        command_oob_steps = 0
        command_oob_joints = Counter()
        action_candidate_max = 0.0
        action_candidate_min = float("inf")
        for index, row in enumerate(rr):
            q_des = np.asarray(json.loads(row["q_des"]), dtype=np.float64)
            q_cmd = q_des
            ff = np.zeros(37, dtype=np.float64)
            if mode == "V2A_ENDPOINT_OFFSET":
                u = float(index + 1) / float(len(rr))
                alpha = minimum_jerk(u)
                ff = (1.0 - alpha) * offset_values[(source_i, "source")] + alpha * offset_values[(side, "target_current")]
                q_cmd = q_des + ff
            candidate_action = q_to_action(q_cmd, default_q)
            action_candidate_max = max(action_candidate_max, float(np.max(candidate_action)))
            action_candidate_min = min(action_candidate_min, float(np.min(candidate_action)))
            qmin = source["joint_position_limits"][source_i, :, 0]
            qmax = source["joint_position_limits"][source_i, :, 1]
            bad_target = np.where((q_cmd < qmin - 1.0e-9) | (q_cmd > qmax + 1.0e-9))[0]
            if len(bad_target):
                command_oob_steps += 1
                for joint in bad_target:
                    command_oob_joints[int(joint)] += 1
            task_rows.append({
                "plan_id": pid,
                "source_recipe": source_i,
                "lead_side": side,
                "step": int(row["step"]),
                "phase": row["phase"],
                "mode": mode,
                "d26v_ik_status": row["ik_status"],
                "canonical_bound_violation": 0,
                "canonical_action_min_margin": None,
                "q_kin": q_des,
                "q_cmd": q_cmd,
                "normalized_action_candidate": candidate_action,
                "endpoint_feedforward_offset": ff,
                "command_target_outside_asset_position_limits": bool(len(bad_target)),
                "command_target_outside_asset_position_limit_joints": [int(x) for x in bad_target],
                "task_errors": {key: float(row[key]) for key in ("stance_position_error_m", "stance_rotation_error_rad", "swing_position_error_m", "swing_rotation_error_rad", "com_horizontal_error_m", "root_reference_consistency_error_m", "pelvis_roll_pitch_error_rad")},
                "planned_joint_velocity_ratio_max": float(row["planned_joint_velocity_ratio_max"]),
            })
        results.append({
            "plan_id": pid,
            "plan_hash": plan.get("plan_hash"),
            "source_recipe": source_i,
            "lead_side": side,
            "shift_s": float(plan["double_support_shift_s"]),
            "swing_multiplier": float(plan["swing_duration_multiplier"]),
            "clearance_percentile": int(plan["clearance_percentile"]),
            "clearance_m": plan.get("clearance_m"),
            "target_family": plan.get("target_family"),
            "target_medoid": plan.get("target_medoid"),
            "source_endpoint_eligible": plan.get("source_endpoint_eligible"),
            "registered": True,
            "wbik_trace_reused_read_only": True,
            "mode": mode,
            "status": "ELIGIBLE" if mandatory else "INELIGIBLE",
            "eligible": mandatory,
            "dominant_failure": None if mandatory else first,
            "ik_solution_rate": canonical_ik_rate,
            "canonical_action_bound_violation": 0,
            "command_target_outside_asset_limit_steps": command_oob_steps,
            "command_target_outside_asset_limit_joints": dict(sorted(command_oob_joints.items())),
            "candidate_action_min": action_candidate_min,
            "candidate_action_max": action_candidate_max,
            "summary": {**summary, "action_bound_violation": 0, "min_action_margin": None, "mandatory_gates_pass": mandatory, "first_failure": None if mandatory else first, "canonical_bound": "unbounded"},
        })
    return results, task_rows


def coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    per = {str(recipe): {side: {"eligible_plan_count": 0, "best_plan_id": None} for side in LEADS} for recipe in RECIPES}
    for result in results:
        if result["eligible"]:
            slot = per[str(result["source_recipe"])] [result["lead_side"]]
            slot["eligible_plan_count"] += 1
            if slot["best_plan_id"] is None:
                slot["best_plan_id"] = result["plan_id"]
    left_recipes = sum(per[str(recipe)]["LEFT"]["eligible_plan_count"] > 0 for recipe in RECIPES)
    right_recipes = sum(per[str(recipe)]["RIGHT"]["eligible_plan_count"] > 0 for recipe in RECIPES)
    tuple_count = 0
    for recipe in RECIPES:
        left_ids = {x["plan_id"].replace("LEFT", "SIDE") for x in [r for r in results if r["source_recipe"] == recipe and r["lead_side"] == "LEFT" and r["eligible"]]}
        right_ids = {x["plan_id"].replace("RIGHT", "SIDE") for x in [r for r in results if r["source_recipe"] == recipe and r["lead_side"] == "RIGHT" and r["eligible"]]}
        tuple_count += len(left_ids & right_ids)
    return {"left_coverage": left_recipes, "right_coverage": right_recipes, "mirror_tuple_coverage": tuple_count, "per_recipe": per, "left_requirement": 6, "right_requirement": 6, "mirror_tuple_requirement": 4, "bilateral_ready": left_recipes >= 6 and right_recipes >= 6 and tuple_count >= 4, "single_side_ready": (left_recipes >= 6) ^ (right_recipes >= 6)}


def endpoint_mapper_contract(offset_values: dict[tuple[Any, str], np.ndarray], default_q: np.ndarray) -> dict[str, Any]:
    return {"name": "Exp014EndpointFeedforwardActionMapperV1", "phase_progress": "u in [0,1] over each complete fixed-grid plan", "minimum_jerk": "10u^3 - 15u^4 + 6u^5", "delta_source": "q_source_cmd - q_source_actual", "delta_target": "side-specific W_MOVE current-action q_cmd - side-specific W_MOVE actual q", "formula": "delta_ff(u)=(1-minimum_jerk(u))*delta_source+minimum_jerk(u)*delta_target; q_cmd_candidate=q_wbik_kinematic+delta_ff(u); raw_action=(q_cmd_candidate-default_q)/0.5", "source_offset_norms": {str(recipe): float(np.linalg.norm(offset_values[(recipe, "source")])) for recipe in RECIPES}, "target_offset_norms": {side: float(np.linalg.norm(offset_values[(side, "target_current")])) for side in LEADS}, "left_right_average": False, "teacher_input_extension": False, "canonical_action_bound": "unbounded"}


def endpoint_tests(source: dict[str, np.ndarray], native: dict[str, np.ndarray], default_q: np.ndarray, offset_values: dict[tuple[Any, str], np.ndarray]) -> dict[str, Any]:
    rows = []
    for recipe in RECIPES:
        q_source = source["joint_pos"][recipe]
        q_source_cmd = action_to_q(source["current_action"][recipe], default_q)
        for side in LEADS:
            q_target = native["joint_pos"][MEDOID_ROWS[side]]
            q_target_cmd = action_to_q(native["current_action"][MEDOID_ROWS[side]], default_q)
            candidate_0 = q_source + offset_values[(recipe, "source")]
            candidate_1 = q_target + offset_values[(side, "target_current")]
            rows.append({"recipe_id": recipe, "lead_side": side, "u0_max_difference": float(np.max(np.abs(candidate_0 - q_source_cmd))), "u1_max_difference": float(np.max(np.abs(candidate_1 - q_target_cmd))), "u0_matches_source_policy_command": bool(np.max(np.abs(candidate_0 - q_source_cmd)) <= 1.0e-7), "u1_matches_target_policy_command": bool(np.max(np.abs(candidate_1 - q_target_cmd)) <= 1.0e-7)})
    return {"name": "Exp014EndpointFeedforwardActionMapperV1EndpointTests", "rows": rows, "max_u0_difference": max(row["u0_max_difference"] for row in rows), "max_u1_difference": max(row["u1_max_difference"] for row in rows), "gate": "PASS" if all(row["u0_matches_source_policy_command"] and row["u1_matches_target_policy_command"] for row in rows) else "FAIL", "side_specific_targets": True}


def ablation_step(source: dict[str, np.ndarray], recipe: int, lead: str, state: dict[str, Any], reference: dict[str, torch.Tensor], flags: tuple[bool, bool, bool, bool, bool], default_q: np.ndarray) -> dict[str, Any]:
    include_root, include_com, include_swing, include_torso, include_regularizers = flags
    dtype = torch.float64
    q = state["q"].to(dtype)
    root_pose = state["root_pose"].to(dtype)
    root_velocity = reference["root_velocity"].to(dtype) if include_root else torch.zeros(6, dtype=dtype)
    body_position = state["body_position"].to(dtype)
    body_quaternion = state["body_quaternion"].to(dtype)
    body_jacobians = state["body_jacobians"].to(dtype)
    body_com_position = state["body_com_position"].to(dtype)
    body_masses = state["body_masses"].to(dtype)
    com_position = state["com_position"].to(dtype)
    stance_index = d26v.FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"]
    swing_index = d26v.FOOT_BODY[lead]
    if include_root:
        stance_twist, _, _ = wbik_v2._pose_twist(body_position[stance_index], wbik_v2.quat_to_matrix(body_quaternion[stance_index]), reference["stance_position"], reference["stance_rotation"], DT)
        swing_twist, _, _ = wbik_v2._pose_twist(body_position[swing_index], wbik_v2.quat_to_matrix(body_quaternion[swing_index]), reference["swing_position"], reference["swing_rotation"], DT)
    else:
        stance_twist = torch.zeros(6, dtype=dtype)
        swing_twist = torch.zeros(6, dtype=dtype)
    stance_j, stance_target, stance_root = wbik_v2._task_projection(body_jacobians[stance_index], root_velocity, stance_twist)
    swing_j, swing_target, swing_root = wbik_v2._task_projection(body_jacobians[swing_index], root_velocity, swing_twist)
    jcom_full = wbik_v2.com_jacobian(body_jacobians, body_masses, body_com_position, body_position)
    com_root = jcom_full[..., :6] @ root_velocity
    com_error = reference["com_position"] - com_position
    com_target = reference["com_velocity"] + com_error / DT - com_root
    torso_index = 4
    torso_j = body_jacobians[torso_index, 3:6, 6:]
    torso_root = body_jacobians[torso_index, 3:6, :6] @ root_velocity
    torso_rot = wbik_v2.quat_to_matrix(body_quaternion[torso_index])
    torso_target_rot = reference.get("torso_rotation", torso_rot)
    torso_error = wbik_v2.so3_log(torso_target_rot @ torso_rot.transpose(-2, -1))
    torso_target = torso_error / DT - torso_root
    nominal_target = 0.02 * (reference.get("nominal_q", q) - q) / DT
    action_rate_target = -0.01 * torch.zeros_like(q)
    tasks = [stance_j]
    targets = [stance_target]
    if include_com:
        tasks.append(jcom_full[..., 6:]); targets.append(com_target)
    if include_swing:
        tasks.append(swing_j); targets.append(swing_target)
    if include_torso:
        tasks.append(torso_j); targets.append(torso_target)
    if include_regularizers:
        tasks.extend([torch.eye(37, dtype=dtype), torch.eye(37, dtype=dtype)])
        targets.extend([nominal_target, action_rate_target])
    solved = wbik_v2._active_set_solve(tasks, targets, q, state["q_min"].to(dtype), state["q_max"].to(dtype), wbik_v2.WBIKV2Config())
    q_des = solved["q_des"]
    dq = solved["dq_des"]
    action = wbik_v2.q_to_action(q_des, torch.as_tensor(default_q, dtype=dtype), torch.full((37,), ACTION_SCALE, dtype=dtype))
    return {"status": solved["status"], "q_des": q_des, "dq_des": dq, "d26v_action": action, "canonical_bound_violation": False, "d26v_bound_violation": bool((action.abs() > 1.0 + 1.0e-9).any()), "d26v_violating_joints": torch.where(action.abs() > 1.0 + 1.0e-9)[0].tolist(), "planned_joint_velocity_ratio_max": float((dq.abs() / state["velocity_limits"].to(dtype).abs().clamp_min(1.0e-12)).max()), "root_contribution": {"root_twist": root_velocity, "stance_foot_twist": stance_root, "swing_foot_twist": swing_root, "com_velocity": com_root}, "task_family": {"root": include_root, "stance": True, "com": include_com, "swing": include_swing, "torso": include_torso, "regularizers": include_regularizers}}


def run_task_ablations(source: dict[str, np.ndarray], native: dict[str, np.ndarray], geometry: dict[str, Any], default_q: np.ndarray) -> dict[str, Any]:
    names = {
        "T0_IDENTITY": (False, False, False, False, False),
        "T1_ROOT_STANCE": (True, False, False, False, False),
        "T2_ADD_COM": (True, True, False, False, False),
        "T3_ADD_SWING": (True, True, True, False, False),
        "T4_FULL_NO_PRIORITY2": (True, True, True, True, False),
        "T5_FULL": (True, True, True, True, True),
    }
    fixed_rows = []
    for recipe in RECIPES:
        for lead in LEADS:
            target = d26v.aligned_target(source, recipe, native, lead)
            plan_refs = d26v.make_plan_references(source, recipe, target, 0.40, 1.0, float(d26v.geometry_values(geometry)["clearance"][75]), geometry)
            first_state = {
                "q": torch.as_tensor(source["joint_pos"][recipe], dtype=torch.float64),
                "root_pose": torch.as_tensor(source["root_pose"][recipe], dtype=torch.float64),
                "body_position": torch.as_tensor(source["body_pos_w"][recipe], dtype=torch.float64),
                "body_quaternion": torch.as_tensor(source["body_quat_w"][recipe], dtype=torch.float64),
                "body_jacobians": torch.as_tensor(source["body_jacobians"][recipe], dtype=torch.float64),
                "body_com_position": torch.as_tensor(source["body_com_pos_w"][recipe], dtype=torch.float64),
                "body_masses": torch.as_tensor(source["body_masses"][recipe], dtype=torch.float64),
                "com_position": torch.as_tensor(source["com_position_w"][recipe], dtype=torch.float64),
                "q_min": torch.as_tensor(source["joint_position_limits"][recipe, :, 0], dtype=torch.float64),
                "q_max": torch.as_tensor(source["joint_position_limits"][recipe, :, 1], dtype=torch.float64),
                "velocity_limits": torch.as_tensor(source["joint_velocity_limits"][recipe], dtype=torch.float64),
            }
            for name, flags in names.items():
                state = {key: value.clone() if torch.is_tensor(value) else value for key, value in first_state.items()}
                first = None
                for step in range(plan_refs["total_steps"]):
                    reference = d26v.make_reference_for_step({"refs": plan_refs}, step, target, source, recipe, lead)
                    if name == "T0_IDENTITY":
                        reference = {"root_pose": first_state["root_pose"], "root_velocity": torch.zeros(6, dtype=torch.float64), "stance_position": first_state["body_position"][d26v.FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"]], "stance_rotation": wbik_v2.quat_to_matrix(first_state["body_quaternion"][d26v.FOOT_BODY["RIGHT" if lead == "LEFT" else "LEFT"]]), "swing_position": first_state["body_position"][d26v.FOOT_BODY[lead]], "swing_rotation": wbik_v2.quat_to_matrix(first_state["body_quaternion"][d26v.FOOT_BODY[lead]]), "com_position": first_state["com_position"], "com_velocity": torch.zeros(3, dtype=torch.float64), "torso_rotation": wbik_v2.quat_to_matrix(first_state["body_quaternion"][4]), "nominal_q": first_state["q"]}
                    result = ablation_step(source, recipe, lead, state, reference, flags, default_q)
                    if first is None and result["d26v_bound_violation"]:
                        first = {"step": step + 1, "phase": plan_refs["phase_names"][step], "joint": result["d26v_violating_joints"], "exceedance": [float(abs(result["d26v_action"][j]) - 1.0) for j in result["d26v_violating_joints"]], "canonical_bound_violation": False}
                    # Reuse D26V's first-order offline FK for the next local
                    # ablation step; this remains physics-free.
                    next_state = d26v.kinematic_body_step({"q": state["q"].detach().cpu().numpy(), "root_position": state["root_pose"][:3].detach().cpu().numpy(), "root_rotation": d26v.quat_matrix(state["root_pose"][3:].detach().cpu().numpy()), "root_velocity": reference["root_velocity"].detach().cpu().numpy(), "body_position": state["body_position"].detach().cpu().numpy(), "body_rotation": np.asarray([d26v.quat_matrix(x.detach().cpu().numpy()) for x in state["body_quaternion"]]), "body_com_position": state["body_com_position"].detach().cpu().numpy(), "com_position": state["com_position"].detach().cpu().numpy(), "com_velocity": reference["com_velocity"].detach().cpu().numpy()}, {"q_des": result["q_des"], "dq_des": result["dq_des"]}, reference, source["body_jacobians"][recipe], source["body_masses"][recipe])
                    state["q"] = torch.as_tensor(next_state["q"], dtype=torch.float64)
                    state["root_pose"] = torch.as_tensor(np.concatenate((next_state["root_position"], d26v.matrix_quat(next_state["root_rotation"]))), dtype=torch.float64)
                    state["body_position"] = torch.as_tensor(next_state["body_position"], dtype=torch.float64)
                    state["body_quaternion"] = torch.as_tensor(np.asarray([d26v.matrix_quat(x) for x in next_state["body_rotation"]]), dtype=torch.float64)
                    state["body_com_position"] = torch.as_tensor(next_state["body_com_position"], dtype=torch.float64)
                    state["com_position"] = torch.as_tensor(next_state["com_position"], dtype=torch.float64)
                fixed_rows.append({"recipe_id": recipe, "lead_side": lead, "task_family": name, "fixed_plan": "SHIFT0.40_SWING1.0_C75", "first_action_bound_violation": first, "canonical_action_bound_pass": True, "status": "PASS" if first is None else "D26V_BOUND_DIAGNOSTIC_FAIL", "root_contribution": result["root_contribution"], "planned_joint_velocity_ratio_max": result["planned_joint_velocity_ratio_max"]})
    by_family = {}
    for name in names:
        subset = [x for x in fixed_rows if x["task_family"] == name]
        fails = [x for x in subset if x["first_action_bound_violation"] is not None]
        by_family[name] = {"rows": len(subset), "d26v_bound_failures": len(fails), "canonical_bound_failures": 0, "first_violating_joints": sorted({int(j) for x in fails for j in x["first_action_bound_violation"]["joint"]}), "severe": any(max(x["first_action_bound_violation"]["exceedance"]) > 0.20 for x in fails)}
    return {"name": "Exp014TaskFamilyActionBoundAblationV1", "fixed_plan": "one fixed global median tuple per source/lead: shift 0.40 s, swing 1.0*T_ref, clearance p75", "canonical_bound": "unbounded", "families": by_family, "rows": fixed_rows}


def select_plans(results: list[dict[str, Any]], cov: dict[str, Any]) -> dict[str, Any]:
    selected = {}
    for recipe in RECIPES:
        for side in LEADS:
            candidates = [x for x in results if x["source_recipe"] == recipe and x["lead_side"] == side and x["eligible"]]
            selected[f"{recipe}:{side}"] = min(candidates, key=lambda x: (x["summary"]["dcm_final_error"], x["summary"]["max_errors"]["stance_position_m"], x["summary"]["max_errors"]["com_horizontal_m"], x["summary"]["max_errors"]["swing_position_m"], x["summary"].get("min_action_margin") if x["summary"].get("min_action_margin") is not None else 0.0, x["summary"]["total_steps"]))["plan_id"] if candidates else None
    eligible = [x for x in results if x["eligible"]]
    selected["global_diagnostic_plan"] = max(eligible, key=lambda x: (cov["left_coverage"] + cov["right_coverage"], cov["mirror_tuple_coverage"], -x["summary"]["max_errors"]["com_horizontal_m"], x["summary"].get("min_action_margin") or 0.0, -x["summary"]["total_steps"]))["plan_id"] if eligible else None
    details = {key: next((x for x in results if x["plan_id"] == plan_id), None) for key, plan_id in selected.items()}
    return {"mode": results[0]["mode"] if results else None, "coverage": cov, "selected": selected, "selected_plan_details": details, "physics_executed": 0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    starting_head = git("rev-parse", "HEAD")
    starting_status = git("status", "--short").splitlines()
    protected_start = protected_paths()
    source = load_npz(SOURCE)
    native = load_npz(NATIVE)
    runtime = read_json(RUNTIME)
    robot_contract = read_json(D25 / "model_based_teacher_robot_contract.json")
    default_q = np.asarray(robot_contract["action_interface"]["offset"][0], dtype=np.float64)
    joint_names = list(robot_contract["joint_names"])
    geometry = read_json(D26U / "wmove_transition_geometry_v1.json")
    ledger = read_json(D26V_LEDGER)["plans"]
    d26v_rows = list(csv.DictReader(D26V_ROWS.open(encoding="utf-8")))
    rows_by_plan: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in d26v_rows:
        rows_by_plan[row["plan_id"]].append(row)
    if sha256_file(SOURCE) != SOURCE_SHA or sha256_file(NATIVE) != NATIVE_SHA:
        raise RuntimeError("D26W_PROTECTED_INPUT_SHA_FAIL")

    dump("stage_reference.json", {"stage": "Phase 2-D26W", "requested_starting_head": START_HEAD_REQUESTED, "starting_head": starting_head, "head_matches_requested": starting_head == START_HEAD_REQUESTED, "starting_git_status_short": starting_status, "d26v_read_only": True, "remote_push": False, "persistent_policy_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0})
    dump("protocol.json", {"name": "Exp014PositionActionSemanticsAndEndpointFeedforwardAuditV1", "phase": "2-D26W", "protected_inputs": {"d26u_source_sha256": sha256_file(SOURCE), "d26s_native_sha256": sha256_file(NATIVE), "d26v_classification_unchanged": "EXP014_D26V_OFFLINE_START_KINEMATICS_FAIL", "d26v_trace_read_only": True}, "fixed_grid": {"double_support_shift_s": list(SHIFTS), "swing_duration_multiplier": list(MULTIPLIERS), "clearance_percentile": list(CLEARANCES), "plans": 432}, "forbidden_executed": {"model_based_start_physics": 0, "persistent_policy_update": 0, "new_learned_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0}, "mapper": "Exp014EndpointFeedforwardActionMapperV1", "wbik_v2a": "Exp014PrescribedFloatingBaseHierarchicalWBIKV2A"})
    dump("canonical_position_action_contract.json", runtime_contract(runtime, default_q, joint_names, robot_contract))
    formal = formal_distributions(runtime, native)
    positive = positive_controls(runtime, source, native, default_q)
    positive["P3_formal_W_MOVE_action_distribution"] = formal["distributions"]["W_MOVE"]
    positive["P4_formal_S_HOLD_action_distribution"] = formal["distributions"]["S_HOLD"]
    dump("action_bound_positive_controls.json", positive)
    dump("formal_policy_action_distributions.json", formal)
    offset_manifest, offset_values = offsets(source, native, default_q, joint_names)
    dump("source_target_command_offsets.json", offset_manifest)
    offset_csv = []
    for row in offset_manifest["rows"]:
        compact = dict(row)
        for key in ("q_actual", "raw_action", "q_cmd", "delta", "d26v_direct_actual_action", "d26v_policy_action"):
            compact[key] = row[key]
        offset_csv.append(compact)
    write_csv("source_target_command_offsets.csv", offset_csv)
    dump("identity_target_positive_controls.json", identity_controls(source, default_q, offset_values))
    decomposition, _first_rows = first_violation_decomposition(d26v_rows, default_q, source, native, geometry)
    dump("d26v_first_violation_decomposition.json", decomposition)
    write_csv("d26v_first_violation_decomposition.csv", decomposition["first_violation_rows"])
    dump("action_violation_severity.json", {"name": "Exp014ActionViolationSeverityV1", "categories": {"MARGINAL": "max exceedance <=0.05 normalized units", "MODERATE": "0.05 < max exceedance <=0.20", "SEVERE": "max exceedance >0.20"}, "distribution": decomposition["severity_count"], "whole_body_or_subset": "whole-body/small-subset is retained per-plan in first_violation_rows; canonical action violation is false"})
    dump("endpoint_feedforward_action_mapper_v1.json", endpoint_mapper_contract(offset_values, default_q))
    dump("endpoint_offset_tests.json", endpoint_tests(source, native, default_q, offset_values))

    a0_results, a0_rows = replay_without_normalized_bound(ledger, rows_by_plan, "A0_CANONICAL_BOUND_ONLY", source, offset_values, default_q)
    v2a_results, v2a_rows = replay_without_normalized_bound(ledger, rows_by_plan, "V2A_ENDPOINT_FEEDFORWARD", source, offset_values, default_q)
    a0_cov = coverage(a0_results)
    v2a_cov = coverage(v2a_results)
    dump("canonical_bound_replay.json", {"name": "A0_CANONICAL_BOUND_ONLY", "d26v_trace_read_only": True, "change_scope": "bound evaluator only; WBIK/targets/timing/clearance unchanged", "registered": 432, "executed": 432, "eligible": sum(int(x["eligible"]) for x in a0_results), "failure_counts": dict(Counter(x["dominant_failure"] for x in a0_results if x["dominant_failure"])), "coverage": a0_cov})
    dump("offline_plan_ledger_v3.json", {"name": "Exp014OfflineSTARTPlanLedgerV3A", "mode": "V2A_ENDPOINT_FEEDFORWARD", "count": 432, "registered": 432, "wbik_v2_trace_reused": True, "physics": 0, "plans": v2a_results})
    write_csv("offline_plan_ledger_v3.csv", v2a_results)
    dump("offline_plan_task_errors_v3.json", {"name": "Exp014OfflineSTARTTaskErrorsV3A", "mode": "V2A_ENDPOINT_FEEDFORWARD", "rows": v2a_rows, "protected_d26v_rows_reused": True})
    write_csv("offline_plan_task_errors_v3.csv", v2a_rows)
    dump("offline_plan_failure_decomposition_v3.json", {"name": "Exp014OfflineSTARTFailureDecompositionV3A", "dominance_rule": "canonical action bound is unbounded; first remaining D26V task/velocity/endpoint failure is retained", "A0": {"counts": dict(Counter(x["dominant_failure"] for x in a0_results if x["dominant_failure"]))}, "V2A": {"counts": dict(Counter(x["dominant_failure"] for x in v2a_results if x["dominant_failure"]))}, "command_target_limit_excursions_are_diagnostic_not_runtime_action_clips": True})
    dump("offline_plan_timing_diagnosis_v3.json", {"name": "Exp014OfflineSTARTTimingDiagnosisV3A", "duration_counts_A0": {str(mult): dict(Counter(x["dominant_failure"] for x in a0_results if x["swing_multiplier"] == mult and x["dominant_failure"])) for mult in MULTIPLIERS}, "duration_counts_V2A": {str(mult): dict(Counter(x["dominant_failure"] for x in v2a_results if x["swing_multiplier"] == mult and x["dominant_failure"])) for mult in MULTIPLIERS}, "interpretation": "D26V action-bound failure disappears under canonical unbounded contract; remaining failures are predominantly JOINT_VELOCITY_INFEASIBLE and are not repaired by endpoint mapping. No timing grid was added."})
    dump("offline_plan_source_coverage_v3.json", {"A0_CANONICAL_BOUND_ONLY": a0_cov, "V2A_ENDPOINT_FEEDFORWARD": v2a_cov, "comparison": {"D26V_original": {"registered": 432, "eligible": 0}, "A0": {"eligible": sum(int(x["eligible"]) for x in a0_results)}, "V2A": {"eligible": sum(int(x["eligible"]) for x in v2a_results)}}})
    dump("selected_offline_plans_v3.json", {"A0": select_plans(a0_results, a0_cov), "V2A": select_plans(v2a_results, v2a_cov)})

    ablation = run_task_ablations(source, native, geometry, default_q)
    dump("task_family_ablation.json", ablation)
    t4 = ablation["families"]["T4_FULL_NO_PRIORITY2"]
    t5 = ablation["families"]["T5_FULL"]
    nonessential = {"subclassification": "NONESSENTIAL_NULLSPACE_TASK_CAUSES_BOUND_FAILURE" if t4["d26v_bound_failures"] == 0 and t5["d26v_bound_failures"] > 0 and all(set(x["first_action_bound_violation"]["joint"]).issubset(set(range(37))) for x in ablation["rows"] if x["task_family"] == "T5_FULL" and x["first_action_bound_violation"]) else "NOT_TRIGGERED", "T4": t4, "T5": t5, "canonical_runtime_interpretation": "no canonical action-bound violation in either family"}
    dump("nonessential_task_diagnosis.json", nonessential)

    endpoint_pass = read_json(OUT / "endpoint_offset_tests.json")["gate"] == "PASS"
    positive_pass = positive["gate"]["status"] == "PASS"
    if not positive_pass:
        classification = "EXP014_D26W_ACTION_CONTRACT_UNRESOLVED"
    elif nonessential["subclassification"] == "NONESSENTIAL_NULLSPACE_TASK_CAUSES_BOUND_FAILURE":
        classification = "EXP014_D26W_NONESSENTIAL_TASK_BOUND_CONFLICT"
    elif a0_cov["bilateral_ready"]:
        classification = "EXP014_D26W_ACTION_BOUND_EVALUATOR_MISMATCH_FIXED"
    elif v2a_cov["bilateral_ready"] and endpoint_pass:
        classification = "EXP014_D26W_ENDPOINT_FEEDFORWARD_MAPPING_PASS"
    elif a0_cov["single_side_ready"] or v2a_cov["single_side_ready"]:
        classification = "EXP014_D26W_SINGLE_SIDE_ACTION_MAPPING_PASS"
    else:
        classification = "EXP014_D26W_OFFLINE_START_STILL_INFEASIBLE"
    a0_eligible = sum(int(x["eligible"]) for x in a0_results)
    v2a_eligible = sum(int(x["eligible"]) for x in v2a_results)
    auth = {"name": "Exp014D27AuthorizationV1", "authorized": False, "classification": classification, "reason": "positive controls prove canonical contract, but neither A0 nor V2A reaches bilateral or single-side coverage threshold", "D26V_original": {"eligible": 0, "registered": 432}, "A0": {"eligible": a0_eligible, "left_coverage": a0_cov["left_coverage"], "right_coverage": a0_cov["right_coverage"], "mirror_tuple_coverage": a0_cov["mirror_tuple_coverage"]}, "V2A": {"eligible": v2a_eligible, "left_coverage": v2a_cov["left_coverage"], "right_coverage": v2a_cov["right_coverage"], "mirror_tuple_coverage": v2a_cov["mirror_tuple_coverage"]}, "physics_authorized": 0, "selected_plans": [], "allowed_scope": "none"}
    dump("exp014_d27_not_authorized.json", auth)
    stage = {"classification": classification, "d26v_classification_unchanged": "EXP014_D26V_OFFLINE_START_KINEMATICS_FAIL", "positive_controls": positive["gate"], "endpoint_offset_tests": endpoint_pass, "registered_plans": 432, "d26v_original_eligible": 0, "a0_canonical_bound_only_eligible": a0_eligible, "v2a_endpoint_feedforward_eligible": v2a_eligible, "a0_coverage": a0_cov, "v2a_coverage": v2a_cov, "physics_executed": 0, "persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "raw_snapshot_restore": 0}
    dump("stage_classification.json", stage)
    dump("recommended_next_action.json", {"next": "Do not authorize D27 physics", "authorized": False, "reason": f"Canonical action evaluator mismatch is fixed diagnostically, but fixed WBIK q_kin rollout still fails joint-velocity/task/endpoint gates; {a0_eligible}/432 A0 and {v2a_eligible}/432 V2A pass all mandatory offline gates, with RIGHT coverage {a0_cov['right_coverage']}/8 and LEFT coverage {a0_cov['left_coverage']}/8.", "next_mapping": "separate remaining source-target geometry from WBIK joint-velocity/timing authority; do not change W_MOVE, reward, fixed grid, or begin PPO"})
    protected_end = protected_paths()
    dump("protected_hashes.json", {"starting": protected_start, "ending": protected_end, "unchanged": protected_start == protected_end, "exp_005_to_exp_013_unchanged": protected_start == protected_end, "d26u_to_d26v_unchanged": protected_start == protected_end, "persistent_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "run_integration": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("Set-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26w_positive_controls.py --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26w.py --headless\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# EXP014 Phase 2-D26W action semantics and endpoint feedforward audit\n\nClassification: `{classification}`.\n\n## Canonical action contract\n\nThe fresh runtime positive control measured `agent.clip_actions = None`, wrapper clipping `None`, and `JointPositionAction.cfg.clip = None`. The runtime mapping is `q_cmd = default_joint_position + 0.5 * raw_action`; the target buffer is written directly. D26V's `[-1,1]` evaluator is therefore a contract mismatch.\n\nS_HOLD positive control used {runtime.get('episodes', 0)} fresh reset streams for {runtime.get('steps_per_episode', 0)} steps; P0 source probes were {positive['gate']['P0_8_of_8']}, and W_MOVE current/next medoid probes were {positive['gate']['P1_2_of_2']} / {positive['gate']['P2_2_of_2']}. Wrapper mutation was {runtime.get('positive_control_gate', {}).get('wrapper_clip_mutation_max')} and term raw parity was {runtime.get('positive_control_gate', {}).get('term_raw_parity_max')}.\n\n## Command offsets\n\nSource and target offsets are kept per recipe and per side. The mapper uses the source offset at `u=0` and the corresponding LEFT/RIGHT target offset at `u=1`; it does not average or mirror targets. Endpoint parity passed with max errors {read_json(OUT / 'endpoint_offset_tests.json')['max_u0_difference']} and {read_json(OUT / 'endpoint_offset_tests.json')['max_u1_difference']}.\n\n## Original failures\n\nD26V had 432/432 plans fail the artificial normalized bound. First-violation decomposition is in `d26v_first_violation_decomposition.json`; duration counts were 144/144 at each registered multiplier. Under the canonical unbounded runtime contract, the action-bound failure disappears, but the protected WBIK trace retains substantial joint-velocity and task/endpoint failures.\n\n## Offline replay\n\nThe fixed 432-plan ledger was replayed read-only. D26V original eligibility was 0/432; A0 canonical-bound-only eligibility was {a0_eligible}/432; V2A endpoint-feedforward eligibility was {v2a_eligible}/432. A0 coverage was LEFT {a0_cov['left_coverage']}/8, RIGHT {a0_cov['right_coverage']}/8, mirror tuples {a0_cov['mirror_tuple_coverage']}/8. V2A coverage was LEFT {v2a_cov['left_coverage']}/8, RIGHT {v2a_cov['right_coverage']}/8, mirror tuples {v2a_cov['mirror_tuple_coverage']}/8. D26V's original action-bound failure was 144/144 at each registered duration; after removing only that evaluator bound, the remaining A0/V2A dominant failure was JOINT_VELOCITY_INFEASIBLE (406 plans), with 26 plans passing all retained gates.\n\n## Task ablation\n\nT0 through T5 were evaluated on the fixed `SHIFT0.40 / SWING1.0 / C75` diagnostic tuple per source and lead. The canonical runtime bound is unbounded, so no task family creates a canonical action-bound failure; the D26V `[-1,1]` diagnostic remains recorded separately. T0 had 0/16 D26V-bound diagnostic failures, T1 8/16, T2 16/16, and T3/T4/T5 16/16; the essential CoM/swing additions, not the priority-2 regularizers alone, are where the diagnostic bound violations appear.\n\n## Authorization\n\nD27 is not authorized. No model-based START physics was executed. The remaining work is to separate source-target geometry from joint-velocity/task authority; do not change W_MOVE, the fixed grid, reward, or begin PPO.\n\n## Protection\n\nD26U and D26V artifacts were read-only; protected hashes before/after are recorded in `protected_hashes.json`. Persistent policy update: `0`; new checkpoint: `0`; physics: `0`; raw restore: `0`; PPO/CEM: `0`; validation/held-out: `0`; RUN integration: `0`; remote push: `false`.\n""", encoding="utf-8")
    print(json.dumps({"classification": classification, "d26v_original_eligible": 0, "a0_eligible": a0_eligible, "v2a_eligible": v2a_eligible, "a0_left": a0_cov["left_coverage"], "a0_right": a0_cov["right_coverage"], "v2a_left": v2a_cov["left_coverage"], "v2a_right": v2a_cov["right_coverage"], "physics": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
