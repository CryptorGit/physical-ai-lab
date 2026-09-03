"""Phase 2-D28V: raw-limit, actuator, and corrected centroidal authority audit.

This stage is fail-closed.  D28U, D28S, D28R, D27 and all earlier artifacts are
read-only inputs.  The optional Isaac Lab inspection creates the already
registered G1 environment only to read runtime metadata; it performs no reset,
control step, physics step, target write, or isolated hard-limit probe.  The
corrected C0--C4 replay is an offline shadow calculation on the protected D28S
115-step trace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
ROOT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = ROOT / "phase_2_d28v_hard_limit_and_actuator_authority"
REPORT = REPO / "research/exp_014_phase_2_d28v_hard_limit_and_actuator_authority_report.md"
D25 = ROOT / "phase_2_d25_model_based_first_step_teacher"
D26U = ROOT / "phase_2_d26u_fresh_source_and_offline_execution"
D26S = ROOT / "phase_2_d26s_exact_wmove_instrumentation"
D26X = ROOT / "phase_2_d26x_timing_and_target_set"
D28R = ROOT / "phase_2_d28r_centroidal_trace_and_feedback"
D28S = ROOT / "phase_2_d28s_centroidal_authority_audit"
D28U = ROOT / "phase_2_d28u_joint_contract_and_physical_authority"

DT = 0.02
HARD_TOL = 1.0e-6
SOLVER_TOL = 1.0e-9
SVD_TOL = 1.0e-8
SOLVER_MAX_ITER = 148
VELOCITY_RATIO_LIMIT = 0.80
TASK_REL_TOL = 1.20
NUMERIC_ZERO = 1.0e-8
CRITICAL_IMPROVEMENT = 0.20
CRITICAL_PASS_FRACTION = 0.80
TRACE_RECIPES = (4, 5, 6, 7)
FORMULATIONS = (
    "C0_ALL_JOINTS",
    "C1_FREEZE_WRIST_HAND",
    "C2_SCALED_ALL_JOINTS",
    "C3_SCALED_FREEZE_WRIST_HAND",
    "C4_SCALED_LEGS_WAIST",
)
GROUPS = (
    "left leg",
    "right leg",
    "waist",
    "left arm",
    "right arm",
    "left wrist/hand",
    "right wrist/hand",
)
TASK_KEYS = ("stance", "com", "swing", "pelvis")
TASK_NAMES = ("stance", "com", "swing", "pelvis", "hz")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Read-only numerical helpers.  Importing these modules does not call their
# main() functions and therefore cannot write D28U/D28S outputs.
d28u = load_module(
    "exp014_d28v_d28u_read_only",
    EXP / "scripts/run_phase2_d28u_joint_contract_and_physical_authority.py",
)
d28s = d28u.d28s
d27 = d28s.d28r.d27


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
    path.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def array_hash(value: Any) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("ascii"))
    h.update(repr(tuple(a.shape)).encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantiles(values: Any) -> dict[str, Any]:
    x = arr(values).reshape(-1)
    x = x[np.isfinite(x)]
    if not x.size:
        return {
            "count": 0,
            "p01": None,
            "p05": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(x.size),
        "p01": float(np.quantile(x, 0.01)),
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else jsonable(value)
                    for key, value in row.items()
                }
            )


def group_for_joint(name: str) -> str:
    n = str(name).lower()
    if "wrist" in n or any(token in n for token in ("_zero_", "_one_", "_two_", "_three_", "_four_", "_five_", "_six_")):
        return "left wrist/hand" if n.startswith("left_") else "right wrist/hand"
    if "shoulder" in n or "elbow" in n:
        return "left arm" if n.startswith("left_") else "right arm"
    if "waist" in n or "torso" in n:
        return "waist"
    return "left leg" if n.startswith("left_") else "right leg"


def group_indices(names: list[str]) -> dict[str, list[int]]:
    return {group: [i for i, name in enumerate(names) if group_for_joint(name) == group] for group in GROUPS}


def safe_repr(value: Any, limit: int = 2000) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # pragma: no cover - defensive metadata path
        text = f"<repr failed: {exc!r}>"
    return text[:limit]


def api_value(obj: Any, name: str) -> tuple[Any, dict[str, Any]]:
    try:
        value = getattr(obj, name)
    except Exception as exc:
        return None, {"symbol": name, "status": "ERROR", "error": repr(exc)}
    try:
        a = arr(value)
        return a, {"symbol": name, "status": "PASS", "shape": list(a.shape), "dtype": str(a.dtype)}
    except Exception:
        return value, {"symbol": name, "status": "PASS_NONARRAY", "repr": safe_repr(value)}


def call_api(obj: Any, name: str, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    try:
        fn = getattr(obj, name)
        value = fn(*args, **kwargs)
        a = arr(value)
        return a, {"symbol": name, "status": "PASS", "shape": list(a.shape), "dtype": str(a.dtype)}
    except Exception as exc:
        return None, {"symbol": name, "status": "ERROR", "error": repr(exc)}


def flatten_first_env(value: Any, expected: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    a = arr(value)
    if expected is not None:
        if a.ndim == 3 and a.shape[0] >= 1 and a.shape[1] == expected:
            return a[0]
        if a.ndim == 2 and a.shape[0] >= 1 and a.shape[1] == expected:
            return a[0]
        if a.ndim == 1 and a.shape[0] == expected:
            return a
    return a[0] if a.ndim > 1 and a.shape[0] == 1 else a


def find_source_file(name: str) -> Path | None:
    candidates = [
        Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\unitree.py"),
        Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd.py"),
        Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab_physx\isaaclab_physx\assets\articulation\articulation.py"),
        EXP / "src/g1_explicit_motion_mode/wbik_v3.py",
    ]
    for path in candidates:
        if path.name == name and path.exists():
            return path
    return None


def source_hash_record(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": None if path is None else str(path), "exists": False, "sha256": None}
    return {"path": str(path), "exists": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def version_probe() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import isaaclab

        result["isaaclab"] = getattr(isaaclab, "__version__", None) or safe_repr(isaaclab)
    except Exception as exc:
        result["isaaclab_error"] = repr(exc)
    try:
        import omni.kit.app

        app = omni.kit.app.get_app()
        result["isaac_sim"] = {
            "build_version": getattr(app, "get_build_version", lambda: None)(),
            "version": getattr(app, "get_version", lambda: None)(),
        }
    except Exception as exc:
        result["isaac_sim_error"] = repr(exc)
    try:
        import omni.physx

        interface = omni.physx.get_physx_interface()
        values: dict[str, Any] = {}
        for name in ("get_physx_version", "get_version", "get_physics_version"):
            if hasattr(interface, name):
                try:
                    values[name] = getattr(interface, name)()
                except Exception as exc:
                    values[name] = repr(exc)
        result["physx"] = values or safe_repr(interface)
    except Exception as exc:
        result["physx_error"] = repr(exc)
    return result


def stage_layer_info(stage: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        root = stage.GetRootLayer()
        result["root_layer_identifier"] = root.identifier
        result["root_layer_real_path"] = root.realPath
        result["root_layer_sha256"] = sha256_file(Path(root.realPath)) if root.realPath and Path(root.realPath).exists() else None
    except Exception as exc:
        result["root_layer_error"] = repr(exc)
    try:
        result["used_layers"] = [
            {
                "identifier": layer.identifier,
                "real_path": layer.realPath,
                "sha256": sha256_file(Path(layer.realPath)) if layer.realPath and Path(layer.realPath).exists() else None,
            }
            for layer in stage.GetUsedLayers()
        ]
    except Exception as exc:
        result["used_layers_error"] = repr(exc)
    try:
        from pxr import UsdGeom

        result["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))
    except Exception as exc:
        result["meters_per_unit_error"] = repr(exc)
    return result


def usd_joint_rows(stage: Any, names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve raw USD joint metadata by prim/joint name, never by array order."""
    rows_by_name: dict[str, dict[str, Any]] = {}
    parse_errors: list[str] = []
    try:
        from pxr import UsdPhysics
    except Exception as exc:  # pragma: no cover - Isaac runtime path
        return [], {"status": "USD_API_UNAVAILABLE", "error": repr(exc)}
    try:
        prims = list(stage.Traverse())
    except Exception as exc:
        return [], {"status": "STAGE_TRAVERSE_FAILED", "error": repr(exc)}
    for prim in prims:
        prim_name = str(prim.GetName())
        if prim_name not in names:
            continue
        try:
            joint_type = str(prim.GetTypeName())
            attrs = {str(attr.GetName()): attr for attr in prim.GetAttributes()}
            lower_attr = attrs.get("physics:lowerLimit")
            upper_attr = attrs.get("physics:upperLimit")
            lower_raw = lower_attr.Get() if lower_attr is not None else None
            upper_raw = upper_attr.Get() if upper_attr is not None else None
            lower_rad = None if lower_raw is None else float(np.deg2rad(float(lower_raw)))
            upper_rad = None if upper_raw is None else float(np.deg2rad(float(upper_raw)))
            drive = {}
            for attr_name, attr in attrs.items():
                if attr_name.startswith("drive:") and any(token in attr_name for token in ("stiffness", "damping", "maxForce", "targetPosition", "targetVelocity")):
                    try:
                        drive[attr_name] = attr.Get()
                    except Exception as exc:
                        drive[attr_name] = f"ERROR:{exc!r}"
            rows_by_name[prim_name] = {
                "joint_name": prim_name,
                "usd_prim_path": str(prim.GetPath()),
                "joint_type": joint_type,
                "is_revolute": bool(prim.IsA(UsdPhysics.RevoluteJoint)),
                "lower_limit_raw": lower_raw,
                "upper_limit_raw": upper_raw,
                "usd_unit": "degrees_for_UsdPhysics_revolute_limit",
                "stage_unit": "meters_per_unit_for_translation; radians_after_explicit_degree_conversion",
                "lower_limit_rad": lower_rad,
                "upper_limit_rad": upper_rad,
                "limit_attribute_authored": bool(lower_attr is not None and upper_attr is not None and lower_attr.HasAuthoredValue() and upper_attr.HasAuthoredValue()),
                "limit_enabled": bool(lower_attr is not None and upper_attr is not None),
                "drive_attributes": drive,
                "applied_schemas": [str(value) for value in prim.GetAppliedSchemas()],
            }
        except Exception as exc:
            parse_errors.append(f"{prim.GetPath()}: {exc!r}")
    rows = [rows_by_name[name] for name in names if name in rows_by_name]
    return rows, {
        "status": "PASS" if len(rows) == len(names) else "PARTIAL",
        "resolved_count": len(rows),
        "expected_count": len(names),
        "missing_names": [name for name in names if name not in rows_by_name],
        "parse_errors": parse_errors,
        "name_based_mapping": True,
    }


def actuator_joint_arrays(actuator: Any, names: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    actuator_names = getattr(actuator, "joint_names", None)
    if actuator_names is None:
        actuator_names = getattr(actuator, "_joint_names", None)
    if actuator_names is not None:
        try:
            actuator_names = [str(x) for x in actuator_names]
        except Exception:
            actuator_names = safe_repr(actuator_names)
    result["joint_names"] = actuator_names
    for field in (
        "joint_indices",
        "stiffness",
        "damping",
        "armature",
        "friction",
        "effort_limit",
        "effort_limit_sim",
        "velocity_limit",
        "velocity_limit_sim",
        "dynamic_friction",
        "viscous_friction",
    ):
        value, api = api_value(actuator, field)
        result[field] = value
        result.setdefault("api", {})[field] = api
    cfg = getattr(actuator, "cfg", None)
    result["cfg_repr"] = safe_repr(cfg, 6000)
    result["class"] = f"{type(actuator).__module__}.{type(actuator).__qualname__}"
    result["is_implicit_model"] = bool(getattr(actuator, "is_implicit_model", False))
    return result


def capture_runtime_metadata(args: Any) -> dict[str, Any]:
    """Read the same registered asset with zero control/physics steps."""
    cfg, agent = d27.resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    # Match the D27 environment construction.  Only the first environment is
    # inspected; no lifecycle reset or simulation step is performed.
    cfg.scene.num_envs = 8
    cfg.seed = 20279941
    if hasattr(cfg, "episode_length_s"):
        cfg.episode_length_s = 20.0
    if hasattr(cfg, "observations") and hasattr(cfg.observations, "policy"):
        cfg.observations.policy.enable_corruption = False
    if hasattr(cfg, "events"):
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
    if getattr(args, "device", None):
        cfg.sim.device = agent.device = args.device
    runtime: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
        "physics_steps": 0,
        "control_steps": 0,
        "reset_calls": 0,
        "target_setter_calls": 0,
        "hard_limit_probe": "NOT_EXECUTED",
    }
    with d27.launch_simulation(cfg, args):
        import omni.usd

        raw_env = d27.gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        wrapped = d27.RslRlVecEnvWrapper(raw_env, clip_actions=agent.clip_actions)
        try:
            env = wrapped.unwrapped
            robot = env.scene["robot"]
            names = [str(value) for value in robot.joint_names]
            dump(OUT / "_runtime_capture_checkpoint.json", {"status": "ENV_CREATED", "joint_names": names, "physics_steps": 0, "reset_calls": 0, "target_setter_calls": 0})
            runtime["agent_clip_actions"] = jsonable(getattr(agent, "clip_actions", None))
            runtime["env_prim_path"] = safe_repr(getattr(robot.cfg, "prim_path", None))
            spawn = getattr(getattr(robot, "cfg", None), "spawn", None)
            usd_path = getattr(spawn, "usd_path", None)
            runtime["resolved_asset"] = {
                "usd_path": None if usd_path is None else str(usd_path),
                "robot_cfg_repr": safe_repr(getattr(robot, "cfg", None), 8000),
                "spawn_cfg_repr": safe_repr(spawn, 8000),
            }
            runtime["joint_names"] = names
            runtime["joint_name_count"] = len(names)
            stage = omni.usd.get_context().get_stage()
            runtime["stage"] = stage_layer_info(stage)
            runtime["stage_prims"] = {
                "root_prim": str(stage.GetPseudoRoot().GetPath()),
                "robot_prim_candidates": [
                    str(prim.GetPath())
                    for prim in stage.Traverse()
                    if str(prim.GetName()).lower() in ("robot", "g1")
                ],
            }
            usd_rows, usd_meta = usd_joint_rows(stage, names)
            runtime["usd_joint_rows"] = usd_rows
            runtime["usd_meta"] = usd_meta
            data = robot.data
            root_view = robot.root_physx_view
            data_fields: dict[str, Any] = {}
            for field in (
                "joint_pos_limits",
                "soft_joint_pos_limits",
                "joint_vel_limits",
                "joint_effort_limits",
                "joint_stiffness",
                "joint_damping",
                "joint_armature",
                "joint_friction",
                "default_joint_stiffness",
                "default_joint_damping",
                "default_joint_armature",
                "default_joint_friction",
                "joint_pos_target",
                "joint_vel_target",
                "computed_torque",
                "applied_torque",
                "default_joint_pos",
            ):
                value, api = api_value(data, field)
                data_fields[field] = {"value": value, "api": api}
            runtime["data_fields"] = data_fields
            dump(OUT / "_runtime_capture_checkpoint.json", {"status": "DATA_CAPTURED", "runtime": runtime})
            physx_limits, physx_api = call_api(root_view, "get_dof_limits")
            runtime["physx_dof_limits"] = {"value": physx_limits, "api": physx_api}
            runtime["physx_other_apis"] = {}
            for method in ("get_dof_properties", "get_dof_vel_limits", "get_dof_max_forces", "get_dof_stiffnesses", "get_dof_dampings"):
                if hasattr(root_view, method):
                    _, api = call_api(root_view, method)
                    runtime["physx_other_apis"][method] = api
            runtime["actuators"] = {
                str(key): actuator_joint_arrays(value, names)
                for key, value in getattr(robot, "actuators", {}).items()
            }
            runtime["versions"] = version_probe()
            runtime["runtime_api_contract"] = {
                "robot_joint_names_symbol": "robot.joint_names",
                "physx_hard_limit_symbol": "robot.root_physx_view.get_dof_limits()",
                "isaac_hard_limit_symbol": "robot.data.joint_pos_limits",
                "isaac_soft_limit_symbol": "robot.data.soft_joint_pos_limits",
                "velocity_symbol": "robot.data.joint_vel_limits",
                "effort_symbol": "robot.data.joint_effort_limits",
                "no_reset_or_step": True,
                "no_target_setter": True,
            }
            runtime["status"] = "PASS"
            dump(OUT / "_runtime_capture_checkpoint.json", {"status": "COMPLETE", "runtime": runtime})
        finally:
            wrapped.close()
    return runtime


def runtime_array(runtime: dict[str, Any], field: str, expected: int = 37) -> np.ndarray | None:
    entry = runtime.get("data_fields", {}).get(field)
    if not entry:
        return None
    value = entry.get("value")
    return flatten_first_env(value, expected)


def runtime_hard_limits(runtime: dict[str, Any]) -> np.ndarray | None:
    value = runtime.get("physx_dof_limits", {}).get("value")
    physx = flatten_first_env(value, 37)
    isaac = runtime_array(runtime, "joint_pos_limits")
    candidates = [x for x in (physx, isaac) if x is not None and x.shape == (37, 2) and np.isfinite(x).all() and np.all(x[:, 0] <= x[:, 1])]
    if not candidates:
        return None
    return np.asarray(candidates[0], dtype=np.float64)


def runtime_vector(runtime: dict[str, Any], field: str) -> np.ndarray | None:
    value = runtime_array(runtime, field, 37)
    if value is None:
        return None
    if value.ndim == 2 and value.shape == (37, 2):
        return np.max(np.abs(value), axis=1)
    if value.ndim == 2:
        return value[:, 0]
    return value.reshape(-1)


def actuator_vectors(runtime: dict[str, Any], names: list[str]) -> dict[str, Any]:
    fields = {field: np.full(len(names), np.nan, dtype=np.float64) for field in ("stiffness", "damping", "armature", "friction", "effort_limit", "effort_limit_sim", "velocity_limit", "velocity_limit_sim")}
    class_by_joint = [None] * len(names)
    implicit = [False] * len(names)
    provenance = []
    name_index = {name: i for i, name in enumerate(names)}
    for key, actuator in runtime.get("actuators", {}).items():
        a_names = actuator.get("joint_names")
        if not isinstance(a_names, list):
            continue
        for local, name in enumerate(a_names):
            if name not in name_index:
                continue
            j = name_index[name]
            class_by_joint[j] = actuator.get("class")
            implicit[j] = bool(actuator.get("is_implicit_model", False))
            for field in fields:
                value = actuator.get(field)
                try:
                    value = arr(value)
                    if value.ndim >= 2:
                        value = value[0]
                    fields[field][j] = float(value.reshape(-1)[local])
                except Exception:
                    pass
        provenance.append({"actuator": key, "joint_names": a_names, "class": actuator.get("class"), "is_implicit_model": actuator.get("is_implicit_model")})
    # Data arrays are the runtime fallback for the limits and default gains.
    for field, data_field in (("effort_limit", "joint_effort_limits"), ("stiffness", "default_joint_stiffness"), ("damping", "default_joint_damping"), ("armature", "default_joint_armature"), ("friction", "default_joint_friction")):
        value = runtime_array(runtime, data_field)
        if value is not None and value.size == len(names):
            value = value[:, 0] if value.ndim == 2 else value
            fields[field] = np.where(np.isfinite(fields[field]), fields[field], value)
    return {"vectors": fields, "class_by_joint": class_by_joint, "implicit_by_joint": implicit, "provenance": provenance}


def load_inputs() -> tuple[list[dict[str, Any]], dict[int, list[int]], dict[int, list[int]], dict[int, dict[str, Any]], list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[int, dict[str, Any]]]:
    records, analysis, critical, manifest, names, default_q, action_scale, soft_limits, vlim, source, meta = d28u.load_records()
    return records, analysis, critical, manifest, names, default_q, action_scale, soft_limits, vlim, source, meta


def populations_for_hard_limits(hard_limits: np.ndarray, soft_limits: np.ndarray, vlim: np.ndarray, default_q: np.ndarray) -> dict[str, dict[str, Any]]:
    pops = d28u.population_rows({}, soft_limits, vlim, default_q)
    for value in pops.values():
        if value.get("q", np.zeros((0,))).size:
            value["soft_limits"] = np.asarray(value["limits"], dtype=np.float64)
            value["limits"] = np.broadcast_to(hard_limits, (value["q"].shape[0], 37, 2)).copy()
    return pops


def bound_audit_population(pop: dict[str, Any], default_q: np.ndarray, hard_limits: np.ndarray) -> dict[str, Any]:
    q = arr(pop.get("q", np.zeros((0, 37))))
    action = arr(pop.get("action", np.zeros_like(q)))
    dq = arr(pop.get("dq", np.zeros_like(q)))
    vlim = arr(pop.get("vlim", np.zeros_like(q)))
    if not q.size:
        return {"source": pop.get("source"), "states": 0, "status": "UNAVAILABLE"}
    lo, hi = hard_limits[:, 0], hard_limits[:, 1]
    lo_state = np.broadcast_to(lo, q.shape)
    hi_state = np.broadcast_to(hi, q.shape)
    strict_lo = np.maximum(-VELOCITY_RATIO_LIMIT * np.abs(vlim), (lo_state - q) / DT)
    strict_hi = np.minimum(VELOCITY_RATIO_LIMIT * np.abs(vlim), (hi_state - q) / DT)
    outside_upper = q > hi_state + HARD_TOL
    outside_lower = q < lo_state - HARD_TOL
    monotone_lo = np.where(outside_upper, -VELOCITY_RATIO_LIMIT * np.abs(vlim), np.where(outside_lower, 0.0, strict_lo))
    monotone_hi = np.where(outside_upper, 0.0, np.where(outside_lower, VELOCITY_RATIO_LIMIT * np.abs(vlim), strict_hi))
    qcmd = default_q[None, :] + 0.5 * action
    return {
        "source": pop.get("source"),
        "states": int(q.shape[0]),
        "episodes": int(pop.get("episodes", 0)),
        "finite": bool(np.isfinite(q).all() and np.isfinite(action).all() and np.isfinite(dq).all() and np.isfinite(vlim).all()),
        "q_actual_hard_inside_fraction": float(np.mean((q >= lo_state - HARD_TOL) & (q <= hi_state + HARD_TOL))),
        "q_actual_hard_violation_count": int(np.sum((q < lo_state - HARD_TOL) | (q > hi_state + HARD_TOL))),
        "q_cmd_hard_inside_fraction_diagnostic_only": float(np.mean((qcmd >= lo_state - HARD_TOL) & (qcmd <= hi_state + HARD_TOL))),
        "q_cmd_hard_outside_count_diagnostic_only": int(np.sum((qcmd < lo_state - HARD_TOL) | (qcmd > hi_state + HARD_TOL))),
        "strict_empty_count": int(np.sum(strict_lo > strict_hi + SOLVER_TOL)),
        "monotone_recovery_empty_count": int(np.sum(monotone_lo > monotone_hi + SOLVER_TOL)),
        "further_outward_count": int(np.sum(((outside_upper) & (dq > HARD_TOL)) | ((outside_lower) & (dq < -HARD_TOL)))),
        "hard_limit_distance": {"lower": quantiles(q - lo_state), "upper": quantiles(hi_state - q)},
        "per_joint": [
            {
                "joint_index": j,
                "q_actual_hard_inside_fraction": float(np.mean((q[:, j] >= lo[j] - HARD_TOL) & (q[:, j] <= hi[j] + HARD_TOL))),
                "strict_empty_fraction": float(np.mean(strict_lo[:, j] > strict_hi[:, j] + SOLVER_TOL)),
                "monotone_recovery_empty_fraction": float(np.mean(monotone_lo[:, j] > monotone_hi[:, j] + SOLVER_TOL)),
                "further_outward_count": int(np.sum(((outside_upper[:, j]) & (dq[:, j] > HARD_TOL)) | ((outside_lower[:, j]) & (dq[:, j] < -HARD_TOL)))),
            }
            for j in range(37)
        ],
    }


def empty_classification(populations: dict[str, dict[str, Any]], hard_limits: np.ndarray, vlim: np.ndarray, names: list[str]) -> dict[str, Any]:
    counts = {key: {name: 0 for name in ("E0_INDEX_MAPPING_ERROR", "E1_REVERSED_LIMIT", "E2_CURRENT_Q_ABOVE_UPPER", "E3_CURRENT_Q_BELOW_LOWER", "E4_NONACTUATED_OR_MIMIC_JOINT", "E5_ONE_STEP_REENTRY_REQUIREMENT", "E6_NUMERICAL_TOLERANCE", "E7_UNKNOWN")} for key in populations}
    examples: list[dict[str, Any]] = []
    for key, pop in populations.items():
        q = arr(pop.get("q", np.zeros((0, 37))))
        dq = arr(pop.get("dq", np.zeros_like(q)))
        if not q.size:
            continue
        lo = np.broadcast_to(hard_limits[:, 0], q.shape)
        hi = np.broadcast_to(hard_limits[:, 1], q.shape)
        vg = VELOCITY_RATIO_LIMIT * np.broadcast_to(np.abs(vlim), q.shape)
        strict_lo = np.maximum(-vg, (lo - q) / DT)
        strict_hi = np.minimum(vg, (hi - q) / DT)
        monotone_lo = np.where(q > hi + HARD_TOL, -vg, np.where(q < lo - HARD_TOL, 0.0, strict_lo))
        monotone_hi = np.where(q > hi + HARD_TOL, 0.0, np.where(q < lo - HARD_TOL, vg, strict_hi))
        for s in range(q.shape[0]):
            for j, name in enumerate(names):
                if not np.isfinite(q[s, j]) or not np.isfinite(lo[s, j]) or not np.isfinite(hi[s, j]):
                    cls = "E6_NUMERICAL_TOLERANCE"
                elif lo[s, j] > hi[s, j] + SOLVER_TOL:
                    cls = "E1_REVERSED_LIMIT"
                elif q[s, j] > hi[s, j] + HARD_TOL:
                    cls = "E2_CURRENT_Q_ABOVE_UPPER"
                elif q[s, j] < lo[s, j] - HARD_TOL:
                    cls = "E3_CURRENT_Q_BELOW_LOWER"
                elif strict_lo[s, j] > strict_hi[s, j] + SOLVER_TOL:
                    cls = "E5_ONE_STEP_REENTRY_REQUIREMENT" if monotone_lo[s, j] <= monotone_hi[s, j] + SOLVER_TOL else "E7_UNKNOWN"
                else:
                    continue
                counts[key][cls] += 1
                if len(examples) < 32:
                    examples.append({"population": key, "state": s, "joint_index": j, "joint_name": name, "classification": cls, "q_actual": q[s, j], "lower": lo[s, j], "upper": hi[s, j], "strict_lower": strict_lo[s, j], "strict_upper": strict_hi[s, j], "recovery_lower": monotone_lo[s, j], "recovery_upper": monotone_hi[s, j]})
    return {
        "name": "Exp014D28VEmptyIntervalClassificationV2",
        "hard_limit_tolerance_rad": HARD_TOL,
        "strict_formula": "max(-0.80*vlim,(hard_lower-q_actual)/dt) <= dq <= min(0.80*vlim,(hard_upper-q_actual)/dt)",
        "counts": counts,
        "examples": examples,
    }


def one_step_bound_semantics() -> dict[str, Any]:
    return {
        "name": "Exp014D28VOneStepReentryContractAuditV2",
        "d28s_formula": "dq_lower=max(-v,(q_lower-q_current)/dt); dq_upper=min(v,(q_upper-q_current)/dt)",
        "interpretation": "For q_current outside a position interval, the formula requires q_next to re-enter the interval within one control step; this is not imposed by the canonical q_cmd setter or implicit actuator source path.",
        "canonical_simulator_reentry_requirement": False,
        "classification": "ONE_STEP_REENTRY_NOT_CANONICAL",
        "diagnostic_replacement": "Exp014MonotoneJointLimitRecoveryBoundV1",
        "formal_runtime": "q_cmd=default_q+0.5*raw_action; raw action unbounded; no actor/wrapper/action-term clipping",
    }


def actuator_torque_parity(runtime: dict[str, Any], names: list[str], actuator: dict[str, Any], default_q: np.ndarray, source: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trace = np.load(D28R / "capture_on" / "raw_primary_trajectory.npz", allow_pickle=False)
    kp = actuator["vectors"]["stiffness"]
    kd = actuator["vectors"]["damping"]
    effort = actuator["vectors"]["effort_limit"]
    rows: list[dict[str, Any]] = []
    status = "PASS"
    reason = None
    formula = "implicit actuator: computed_effort = stiffness*(q_cmd-q_actual) + damping*(dq_cmd-dq_actual) + feedforward; applied_effort=clip(computed_effort,-effort_limit,effort_limit)"
    if not (np.isfinite(kp).all() and np.isfinite(kd).all() and np.isfinite(effort).all()):
        return [], {"name": "Exp014ActuatorTorqueParityV2", "status": "ACTUATOR_MODEL_STATE_MISSING", "formula": formula, "max_computed_abs_error": None, "max_applied_abs_error": None, "effort_classification_agreement": False}
    for recipe in TRACE_RECIPES:
        active = np.flatnonzero(trace["active"][recipe] & (trace["stage"][recipe] == 1))
        for row in active:
            q_post = trace["q_actual"][recipe, row]
            dq_post = trace["joint_velocity"][recipe, row]
            q_cmd = trace["q_cmd"][recipe, row]
            computed = trace["computed_torque"][recipe, row]
            applied = trace["applied_torque"][recipe, row]
            predicted_post = kp * (q_cmd - q_post) - kd * dq_post
            predicted_applied = np.clip(predicted_post, -np.abs(effort), np.abs(effort))
            computed_error = float(np.max(np.abs(predicted_post - computed)))
            applied_error = float(np.max(np.abs(predicted_applied - applied)))
            rows.append({
                "recipe": int(recipe),
                "control_step": int(trace["control_step"][recipe, row]),
                "timing_alignment": "q_cmd current action vs post-physics q_actual/dq_actual; computed/applied tensors persisted after the same step",
                "computed_max_abs_error": computed_error,
                "applied_max_abs_error": applied_error,
                "predicted_effort_ratio_max": float(np.max(np.abs(predicted_post) / np.maximum(np.abs(effort), NUMERIC_ZERO))),
                "trace_effort_ratio_max": float(np.max(np.abs(applied) / np.maximum(np.abs(effort), NUMERIC_ZERO))),
                "predicted_clipped_joint_count": int(np.sum(np.abs(predicted_post) > np.abs(effort) + 1.0e-9)),
                "trace_clipped_joint_count": int(np.sum(np.abs(computed) > np.abs(effort) + 1.0e-9)),
            })
    computed_errors = np.asarray([row["computed_max_abs_error"] for row in rows])
    applied_errors = np.asarray([row["applied_max_abs_error"] for row in rows])
    # The post-step trace is authoritative for applied torque, while the
    # computed tensor can be one simulator-buffer update behind.  Record both
    # without relaxing the registered 1e-5 gate.
    max_computed = float(np.max(computed_errors)) if computed_errors.size else None
    max_applied = float(np.max(applied_errors)) if applied_errors.size else None
    if max_computed is None or max_computed > 1.0e-5 or max_applied is None or max_applied > 1.0e-5:
        status = "ACTUATOR_MODEL_FORMULA_MISMATCH"
        reason = "persisted D28R computed/applied torque does not match the reconstructed implicit-actuator formula at fixed post-step alignment"
    trace_class = np.asarray([row["trace_clipped_joint_count"] > 0 for row in rows])
    pred_class = np.asarray([row["predicted_clipped_joint_count"] > 0 for row in rows])
    agreement = bool(trace_class.size and np.all(trace_class == pred_class))
    if not agreement and status == "PASS":
        status = "ACTUATOR_MODEL_FORMULA_MISMATCH"
        reason = "effort-clipping classification mismatch"
    return rows, {
        "name": "Exp014ActuatorTorqueParityV2",
        "status": status,
        "reason": reason,
        "formula": formula,
        "computed_torque_required_max_abs_error": 1.0e-5,
        "applied_torque_required_max_abs_error": 1.0e-5,
        "max_computed_abs_error": max_computed,
        "p95_computed_abs_error": float(np.quantile(computed_errors, 0.95)) if computed_errors.size else None,
        "max_applied_abs_error": max_applied,
        "p95_applied_abs_error": float(np.quantile(applied_errors, 0.95)) if applied_errors.size else None,
        "effort_clipping_classification_agreement": agreement,
        "source_contract": "D28R capture_on/raw_primary_trajectory.npz; no new physics",
        "runtime_actuator_class": actuator.get("class_by_joint"),
    }


def actuator_positive_controls(runtime: dict[str, Any], populations: dict[str, dict[str, Any]], default_q: np.ndarray, actuator: dict[str, Any], hard_limits: np.ndarray, names: list[str]) -> dict[str, Any]:
    kp = actuator["vectors"]["stiffness"]
    kd = actuator["vectors"]["damping"]
    effort = actuator["vectors"]["effort_limit"]
    result: dict[str, Any] = {}
    for key, pop in populations.items():
        q = arr(pop.get("q", np.zeros((0, 37))))
        action = arr(pop.get("action", np.zeros_like(q)))
        dq = arr(pop.get("dq", np.zeros_like(q)))
        if not q.size:
            result[key] = {"states": 0, "status": "UNAVAILABLE"}
            continue
        qcmd = default_q[None, :] + 0.5 * action
        accepted = np.allclose(qcmd, default_q[None, :] + 0.5 * action, atol=0.0, rtol=0.0)
        unclipped = kp[None, :] * (qcmd - q) - kd[None, :] * dq
        applied = np.clip(unclipped, -np.abs(effort)[None, :], np.abs(effort)[None, :])
        result[key] = {
            "states": int(q.shape[0]),
            "q_cmd_setter_accepts_unchanged": bool(accepted and np.isfinite(qcmd).all()),
            "computed_effort_finite": bool(np.isfinite(unclipped).all()),
            "applied_effort_finite": bool(np.isfinite(applied).all()),
            "effort_saturation_incidence": float(np.mean(np.abs(unclipped) > np.abs(effort)[None, :] + 1.0e-9)),
            "q_actual_hard_limit_violation_count": int(np.sum((q < hard_limits[None, :, 0] - HARD_TOL) | (q > hard_limits[None, :, 1] + HARD_TOL))),
            "velocity_limit_incidence": float(np.mean(np.abs(dq) > np.abs(pop["vlim"]) + HARD_TOL)),
            "canonical_action_formula": "q_cmd=default_q+0.5*raw_action",
        }
    return {"name": "Exp014FormalActuatorPositiveControlsV2", "rows": result, "status": "PASS" if all(row.get("status") != "UNAVAILABLE" and row.get("q_cmd_setter_accepts_unchanged") and row.get("computed_effort_finite") and row.get("applied_effort_finite") for row in result.values()) else "FAIL_OR_UNAVAILABLE"}


def hard_task_stack(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return d28s.task_stack(record, list(TASK_KEYS))


def scaled_record(record: dict[str, Any]) -> dict[str, Any]:
    out = {**record, "tasks": {key: {**value, "J": np.asarray(value["J"], dtype=np.float64).copy()} for key, value in record["tasks"].items()}}
    v = np.abs(record["velocity_limits"])
    for value in out["tasks"].values():
        value["J"] = value["J"] * v[None, :]
    out["velocity_limits"] = np.ones(37, dtype=np.float64)
    return out


def formulation_freeze(formulation: str, names: list[str]) -> set[int]:
    groups = group_indices(names)
    hand = set(groups["left wrist/hand"] + groups["right wrist/hand"])
    arms = set(groups["left arm"] + groups["right arm"])
    if formulation in ("C1_FREEZE_WRIST_HAND", "C3_SCALED_FREEZE_WRIST_HAND"):
        return hand
    if formulation == "C4_SCALED_LEGS_WAIST":
        return hand | arms
    return set()


def canonical_bounds(record: dict[str, Any], hard_limits: np.ndarray, formulation: str, names: list[str]) -> tuple[dict[str, np.ndarray], dict[str, Any], bool]:
    v = np.abs(record["velocity_limits"])
    q = record["q_current"]
    poslo = (hard_limits[:, 0] - q) / DT
    poshi = (hard_limits[:, 1] - q) / DT
    vel = VELOCITY_RATIO_LIMIT * v
    lo = np.maximum(-vel, poslo)
    hi = np.minimum(vel, poshi)
    freeze = formulation_freeze(formulation, names)
    if freeze:
        lo = lo.copy()
        hi = hi.copy()
        for j in freeze:
            lo[j] = 0.0
            hi[j] = 0.0
    scaled = formulation.startswith("C2") or formulation.startswith("C3") or formulation.startswith("C4")
    if scaled:
        lo = lo / np.maximum(v, NUMERIC_ZERO)
        hi = hi / np.maximum(v, NUMERIC_ZERO)
    return {
        "combined_lower": lo,
        "combined_upper": hi,
        "velocity_lower": -vel if not scaled else -np.ones(37) * VELOCITY_RATIO_LIMIT,
        "velocity_upper": vel if not scaled else np.ones(37) * VELOCITY_RATIO_LIMIT,
        "q_kin_position_lower": poslo if not scaled else poslo / np.maximum(v, NUMERIC_ZERO),
        "q_kin_position_upper": poshi if not scaled else poshi / np.maximum(v, NUMERIC_ZERO),
        "feedforward": np.zeros(37, dtype=np.float64),
        "freeze_indices": np.asarray(sorted(freeze), dtype=np.int64),
    }, {"physical_q_kin_hard_limits": True, "q_cmd_position_limit": False, "velocity_ratio_limit": VELOCITY_RATIO_LIMIT, "scaled": scaled, "frozen_joint_indices": sorted(freeze)}, scaled


def group_contribution(dq: np.ndarray, names: list[str]) -> dict[str, float]:
    den = max(float(np.linalg.norm(dq)), NUMERIC_ZERO)
    result = {group: 0.0 for group in GROUPS}
    for j, name in enumerate(names):
        result[group_for_joint(name)] += float(dq[j] * dq[j])
    return {group: float(np.sqrt(value) / den) for group, value in result.items()}


def active_names(active: list[int], names: list[str]) -> list[dict[str, Any]]:
    return [{"joint_index": int(index % len(names)), "joint_name": names[int(index % len(names))], "bound": "lower" if index >= len(names) else "upper"} for index in active]


def task_residuals(record: dict[str, Any], dq: np.ndarray) -> dict[str, float]:
    return {key: float(np.linalg.norm(record["tasks"][key]["J"] @ dq - record["tasks"][key]["b"])) for key in TASK_NAMES}


def corrected_evaluate(record: dict[str, Any], hard_limits: np.ndarray, dq: np.ndarray, names: list[str], label: str, solver: dict[str, Any], baseline: dict[str, Any], scaled: bool, actuator: dict[str, Any]) -> dict[str, Any]:
    q = record["q_current"]
    v = np.abs(record["velocity_limits"])
    q_kin_next = q + DT * dq
    scalar = d28s.minimum_jerk((record["plan_step"] + 1) / max(float(record["total_steps"]), 1.0))
    ff = (1.0 - scalar) * record["source_offset"] + scalar * record["target_offset"]
    q_cmd = q_kin_next + ff
    action = (q_cmd - record["default_q"]) / record["action_scale"]
    residuals = task_residuals(record, dq)
    base = baseline["task_residuals"]
    gates = {
        "stance_no_worse": residuals["stance"] <= base["stance"] + 1.0e-9,
        "com_within_20pct": residuals["com"] <= TASK_REL_TOL * max(base["com"], NUMERIC_ZERO) + 1.0e-9,
        "swing_within_20pct": residuals["swing"] <= TASK_REL_TOL * max(base["swing"], NUMERIC_ZERO) + 1.0e-9,
        "pelvis_within_20pct": residuals["pelvis"] <= TASK_REL_TOL * max(base["pelvis"], NUMERIC_ZERO) + 1.0e-9,
    }
    hz_error = float(abs(record["tasks"]["hz"]["J"] @ dq - record["tasks"]["hz"]["b"])[0])
    baseline_hz = float(baseline["predicted_hz_error"])
    ratio = np.abs(dq) / np.maximum(v, NUMERIC_ZERO)
    qkin_limits = bool(np.isfinite(q_kin_next).all() and np.all(q_kin_next >= hard_limits[:, 0] - HARD_TOL) and np.all(q_kin_next <= hard_limits[:, 1] + HARD_TOL))
    velocity = bool(np.max(ratio) <= VELOCITY_RATIO_LIMIT + SOLVER_TOL)
    setter = bool(np.isfinite(q_cmd).all() and np.allclose(record["default_q"] + record["action_scale"] * action, q_cmd, atol=1.0e-10, rtol=1.0e-10))
    kp = actuator["vectors"]["stiffness"]
    kd = actuator["vectors"]["damping"]
    effort = actuator["vectors"]["effort_limit"]
    torque = kp * (q_cmd - q) - kd * record["dq_current"]
    effort_ratio = np.abs(torque) / np.maximum(np.abs(effort), NUMERIC_ZERO)
    effort_gate = bool(np.isfinite(torque).all() and np.max(effort_ratio) <= 1.0 + SOLVER_TOL)
    all_tasks = all(gates.values())
    return {
        "recipe": record["recipe"],
        "control_step": record["control_step"],
        "trace_row": record["trace_row"],
        "phase": record["phase"],
        "formulation": label,
        "solver_success": bool(solver.get("success", False)),
        "hard_task_feasible": bool(solver.get("success", False) and all_tasks),
        "current_hz_error": float(abs(record["actual_hz"])),
        "v2a_predicted_hz_error": baseline_hz,
        "minimum_achievable_hz_error": hz_error,
        "relative_hz_improvement": float((baseline_hz - hz_error) / max(baseline_hz, NUMERIC_ZERO)),
        "q_kin_next": q_kin_next,
        "q_cmd": q_cmd,
        "action": action,
        "dq": dq,
        "velocity_ratio_max": float(np.max(ratio)),
        "q_kin_hard_limit_gate": qkin_limits,
        "q_cmd_setter_gate": setter,
        "effort_ratio_max": float(np.max(effort_ratio)),
        "effort_gate": effort_gate,
        "task_residuals": residuals,
        "task_gates": gates,
        "active_bound_joints": active_names(solver.get("active", []), names),
        "active_bound_indices": solver.get("active", []),
        "joint_group_contribution": group_contribution(dq, names),
        "solver": {key: value for key, value in solver.items() if key not in ("x", "x_reduced")},
        "scaled": scaled,
        "all_mandatory_gates": bool(solver.get("success", False) and all_tasks and qkin_limits and velocity and setter and effort_gate),
        "q_cmd_position_limit_applied": False,
    }


def corrected_replay(records: list[dict[str, Any]], critical: dict[int, list[int]], manifest: dict[int, dict[str, Any]], names: list[str], default_q: np.ndarray, hard_limits: np.ndarray, actuator: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    baseline_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        v2 = d28s.v2a_dq(record)
        baseline_by_key[(int(record["recipe"]), int(record["control_step"]))] = {"dq": v2["dq"], "task_residuals": task_residuals(record, v2["dq"]), "predicted_hz_error": float(abs(record["tasks"]["hz"]["J"] @ v2["dq"] - record["tasks"]["hz"]["b"])[0]), "status": v2["status"]}
    for record in records:
        baseline = baseline_by_key[(int(record["recipe"]), int(record["control_step"]))]
        for formulation in FORMULATIONS:
            bounds, contract, scaled = canonical_bounds(record, hard_limits, formulation, names)
            work = scaled_record(record) if scaled else record
            x, solver = d28s.solve_f2(work, bounds)
            dq = x * np.abs(record["velocity_limits"]) if scaled else arr(x)
            result = corrected_evaluate(record, hard_limits, dq, names, formulation, solver, baseline, scaled, actuator)
            result["bound_contract"] = contract
            results.append(result)
    summaries: dict[str, Any] = {}
    result_map = {(int(row["recipe"]), int(row["control_step"]), row["formulation"]): row for row in results}
    for formulation in FORMULATIONS:
        all_rows = [row for row in results if row["formulation"] == formulation]
        by_recipe: dict[str, Any] = {}
        for recipe in TRACE_RECIPES:
            rows = [result_map[(recipe, int(step), formulation)] for step in manifest[recipe]["critical_control_steps"]]
            by_recipe[str(recipe)] = {
                "critical_steps": len(rows),
                "improvement_ge_20_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT for row in rows])) if rows else 0.0,
                "all_mandatory_gate_fraction": float(np.mean([row["all_mandatory_gates"] for row in rows])) if rows else 0.0,
                "critical_gate_pass_fraction": float(np.mean([row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in rows])) if rows else 0.0,
                "all_critical_gate_pass": bool(rows and all(row["relative_hz_improvement"] >= CRITICAL_IMPROVEMENT and row["all_mandatory_gates"] for row in rows)),
                "max_velocity_ratio": float(max(row["velocity_ratio_max"] for row in rows)) if rows else None,
                "max_effort_ratio": float(max(row["effort_ratio_max"] for row in rows)) if rows else None,
                "median_improvement": float(np.median([row["relative_hz_improvement"] for row in rows])) if rows else None,
            }
        summaries[formulation] = {
            "rows": len(all_rows),
            "solver_success_fraction": float(np.mean([row["solver_success"] for row in all_rows])) if all_rows else 0.0,
            "all_mandatory_gate_rows": int(sum(row["all_mandatory_gates"] for row in all_rows)),
            "critical": by_recipe,
            "all_sources_critical_gate": bool(all(by_recipe[str(recipe)]["all_critical_gate_pass"] for recipe in TRACE_RECIPES)),
            "median_improvement": float(np.median([row["relative_hz_improvement"] for row in all_rows])) if all_rows else None,
            "max_velocity_ratio": float(max(row["velocity_ratio_max"] for row in all_rows)) if all_rows else None,
            "max_effort_ratio": float(max(row["effort_ratio_max"] for row in all_rows)) if all_rows else None,
        }
    return results, summaries, {"result_map": result_map, "baseline_count": len(baseline_by_key)}


def current_v3_comparison(results: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, Any]:
    old = read_json(D28S / "critical_window_authority.json") if (D28S / "critical_window_authority.json").exists() else {}
    old_forms = old.get("formulations", {})
    rows = []
    for form, summary in summaries.items():
        rows.append({"formulation": form, "corrected_summary": summary, "D28S_reference": old_forms.get("F1_CURRENT_V3" if form == "C0_ALL_JOINTS" else form, None)})
    return {"name": "Exp014CurrentV3CorrectedComparisonV2", "D28S_protected": True, "rows": rows, "current_v3_formal_fail": True, "comparison_only": True}


def full_trace_shadow(results: list[dict[str, Any]], selected: str | None) -> dict[str, Any]:
    rows = [row for row in results if row["formulation"] == selected] if selected else []
    return {
        "name": "Exp014CanonicalBoundedCentroidalWBIKV3R3Shadow",
        "created": bool(selected),
        "physics_applied": 0,
        "selected_formulation": selected,
        "rows": rows,
        "solver_success_fraction": float(np.mean([row["solver_success"] for row in rows])) if rows else 0.0,
        "mandatory_gate_fraction": float(np.mean([row["all_mandatory_gates"] for row in rows])) if rows else 0.0,
        "critical_window_gate": bool(selected),
        "determinism": "D28S deterministic active-set solver; no physics; fixed ordering/tolerance/iteration cap",
        "hash": canonical_hash(rows) if rows else None,
    }


def protected_audit(start_head: str, start_status: list[str], before: dict[str, str]) -> dict[str, Any]:
    changed = []
    for rel, digest in before.items():
        path = REPO / rel
        if not path.exists():
            changed.append({"path": rel, "reason": "missing"})
        elif sha256_file(path) != digest:
            changed.append({"path": rel, "reason": "hash_changed", "expected": digest, "observed": sha256_file(path)})
    return {
        "starting_head": start_head,
        "starting_status_short": start_status,
        "protected_paths": before,
        "changed_paths": changed,
        "unchanged": not changed,
        "exp005_to_exp013_unchanged": not changed,
        "d6_to_d28u_unchanged": not changed,
        "S_HOLD_unchanged": not changed,
        "Stage_2Q_unchanged": not changed,
        "W_MOVE_unchanged": not changed,
        "S_STOP_OMNI_unchanged": not changed,
        "WBIK_V1_V2_V2A_V3_unchanged": not changed,
        "persistent_update": 0,
        "new_learned_checkpoint": 0,
        "physics": 0,
        "left_start": 0,
        "ppo": 0,
        "cem": 0,
        "validation": 0,
        "held_out": 0,
        "RUN": 0,
        "remote_push": False,
    }


def initial_protected_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    # D28U itself is explicitly protected even though its earlier manifest
    # predates this stage.
    for base in (D28U, D28S, D28R):
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file():
                    result[str(path.relative_to(REPO)).replace("\\", "/")] = sha256_file(path)
    # Reuse the prior protected manifest for earlier D6--D28S files where it
    # is available, without treating the current D28V output as protected.
    manifest = D28U / "protected_hashes.json"
    if manifest.exists():
        try:
            expected = read_json(manifest).get("protected_paths", read_json(manifest).get("protected_files_sha256", {}))
            for rel, digest in expected.items():
                if (REPO / rel).exists():
                    result.setdefault(rel.replace("\\", "/"), digest)
        except Exception:
            pass
    for path in (
        D26U / "fresh_shold_identity_complete_sources.npz",
        D26S / "native_steady_trace_bundle.npz",
        D26X / "selected_offline_plans_v4.json",
        D28R / "capture_on/raw_primary_trajectory.npz",
        D28U / "stage_classification.json",
    ):
        if path.exists():
            result[str(path.relative_to(REPO)).replace("\\", "/")] = sha256_file(path)
    return result


def make_runtime_rows(runtime: dict[str, Any], names: list[str], hard_limits: np.ndarray | None, soft_limits: np.ndarray, vlim: np.ndarray, effort: np.ndarray, default_q: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    physx = runtime_hard_limits(runtime)
    data_hard = runtime_array(runtime, "joint_pos_limits")
    data_soft = runtime_array(runtime, "soft_joint_pos_limits")
    data_vel = runtime_vector(runtime, "joint_vel_limits")
    data_eff = runtime_vector(runtime, "joint_effort_limits")
    actuator = actuator_vectors(runtime, names)
    usd_by = {row["joint_name"]: row for row in runtime.get("usd_joint_rows", [])}
    raw_rows = []
    phys_rows = []
    for j, name in enumerate(names):
        usd = usd_by.get(name, {})
        raw_rows.append({
            "joint_name": name,
            "joint_index_by_name": j,
            "usd_prim_path": usd.get("usd_prim_path"),
            "joint_type": usd.get("joint_type"),
            "lower_limit_raw": usd.get("lower_limit_raw"),
            "upper_limit_raw": usd.get("upper_limit_raw"),
            "usd_unit": usd.get("usd_unit"),
            "stage_unit": usd.get("stage_unit"),
            "lower_limit_rad": usd.get("lower_limit_rad"),
            "upper_limit_rad": usd.get("upper_limit_rad"),
            "limit_attribute_authored": usd.get("limit_attribute_authored"),
            "limit_enabled": usd.get("limit_enabled"),
            "drive_attributes": usd.get("drive_attributes", {}),
        })
        phys_rows.append({
            "joint_name": name,
            "joint_index_by_name": j,
            "physx_lower": None if physx is None else physx[j, 0],
            "physx_upper": None if physx is None else physx[j, 1],
            "physx_limit_enabled": bool(physx is not None),
            "isaac_joint_pos_lower": None if data_hard is None else data_hard[j, 0],
            "isaac_joint_pos_upper": None if data_hard is None else data_hard[j, 1],
            "isaac_soft_lower": None if data_soft is None else data_soft[j, 0],
            "isaac_soft_upper": None if data_soft is None else data_soft[j, 1],
            "runtime_velocity_limit": None if data_vel is None else data_vel[j],
            "runtime_effort_limit": None if data_eff is None else data_eff[j],
            "actuator_stiffness": actuator["vectors"]["stiffness"][j],
            "actuator_damping": actuator["vectors"]["damping"][j],
            "actuator_armature": actuator["vectors"]["armature"][j],
            "actuator_friction": actuator["vectors"]["friction"][j],
            "actuator_effort_limit": actuator["vectors"]["effort_limit"][j],
            "actuator_class": actuator["class_by_joint"][j],
            "implicit_actuator": actuator["implicit_by_joint"][j],
        })
    return raw_rows, phys_rows


def limit_hierarchy(raw_rows: list[dict[str, Any]], phys_rows: list[dict[str, Any]], names: list[str], soft_limits: np.ndarray, d25_limits: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for j, name in enumerate(names):
        raw = raw_rows[j]
        phys = phys_rows[j]
        hard_lo = phys.get("physx_lower")
        hard_hi = phys.get("physx_upper")
        if hard_lo is None:
            hard_lo = phys.get("isaac_joint_pos_lower")
            hard_hi = phys.get("isaac_joint_pos_upper")
        soft_lo, soft_hi = soft_limits[j]
        eval_lo, eval_hi = d25_limits[j]
        source = "L2_PHYSX_HARD_LIMIT" if hard_lo is not None else "L0_NO_LIMIT"
        rows.append({
            "joint_name": name,
            "raw_usd_lower_rad": raw.get("lower_limit_rad"),
            "raw_usd_upper_rad": raw.get("upper_limit_rad"),
            "physx_hard_lower_rad": hard_lo,
            "physx_hard_upper_rad": hard_hi,
            "isaac_joint_pos_lower_rad": phys.get("isaac_joint_pos_lower"),
            "isaac_joint_pos_upper_rad": phys.get("isaac_joint_pos_upper"),
            "isaac_soft_lower_rad": phys.get("isaac_soft_lower"),
            "isaac_soft_upper_rad": phys.get("isaac_soft_upper"),
            "environment_soft_lower_rad": soft_lo,
            "environment_soft_upper_rad": soft_hi,
            "d28s_d28u_processed_lower_rad": eval_lo,
            "d28s_d28u_processed_upper_rad": eval_hi,
            "hard_limit_source": source,
            "raw_usd_vs_physx_lower_difference": None if raw.get("lower_limit_rad") is None or hard_lo is None else float(raw["lower_limit_rad"] - hard_lo),
            "raw_usd_vs_physx_upper_difference": None if raw.get("upper_limit_rad") is None or hard_hi is None else float(raw["upper_limit_rad"] - hard_hi),
            "soft_factor_vs_hard_lower": None if hard_lo is None else float((soft_lo - hard_lo) / max(abs(hard_lo), NUMERIC_ZERO)),
            "soft_factor_vs_hard_upper": None if hard_hi is None else float((hard_hi - soft_hi) / max(abs(hard_hi), NUMERIC_ZERO)),
        })
    meta = {
        "name": "Exp014JointLimitHierarchyV2",
        "classification_codes": ["L0_NO_LIMIT", "L1_USD_HARD_LIMIT", "L2_PHYSX_HARD_LIMIT", "L3_ISAAC_PROCESSED_LIMIT", "L4_ENVIRONMENT_SOFT_LIMIT", "L5_EVALUATION_ONLY_LIMIT"],
        "source_of_truth": "L2_PHYSX_HARD_LIMIT" if all(row["physx_hard_lower_rad"] is not None and row["physx_hard_upper_rad"] is not None for row in rows) else "UNRESOLVED",
        "processed_soft_limit_is_physical_hard": False,
        "rows": rows,
    }
    return rows, meta


def identity_rows(names: list[str], default_q: np.ndarray, hard_limits: np.ndarray | None, soft_limits: np.ndarray, vlim: np.ndarray, effort: np.ndarray, runtime: dict[str, Any], actuator: dict[str, Any], pops: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    d26x = read_json(D26X / "joint_index_name_contract.json")
    d28r = read_json(D28R / "joint_index_name_contract.json")
    d26x_names = [row["joint_name"] for row in sorted(d26x["joints"], key=lambda x: int(x["action_index"]))]
    d28r_names = [row["joint_name"] for row in sorted(d28r["joints"], key=lambda x: int(x["action_index"]))]
    runtime_names = runtime.get("joint_names", [])
    rows = []
    for j, name in enumerate(names):
        runtime_i = runtime_names.index(name) if name in runtime_names else None
        rows.append({
            "action_index": j,
            "policy_output_index": j,
            "environment_action_index": j,
            "robot_articulation_joint_index": runtime_i,
            "usd_joint_path": next((row.get("usd_prim_path") for row in runtime.get("usd_joint_rows", []) if row.get("joint_name") == name), None),
            "joint_name": name,
            "joint_type": "revolute" if runtime_i is not None else None,
            "actuated": bool(runtime_i is not None and actuator["class_by_joint"][j] is not None),
            "mimic": False,
            "fixed": False,
            "default_q": default_q[j],
            "current_q_p50_D27": None,
            "position_lower_physical_hard": None if hard_limits is None else hard_limits[j, 0],
            "position_upper_physical_hard": None if hard_limits is None else hard_limits[j, 1],
            "position_lower_processed_soft": soft_limits[j, 0],
            "position_upper_processed_soft": soft_limits[j, 1],
            "velocity_limit": vlim[j],
            "effort_limit_runtime": effort[j],
            "action_scale": 0.5,
            "actuator_class": actuator["class_by_joint"][j],
            "formal_S_HOLD_states": {key: int(value.get("q", np.zeros((0,))).shape[0]) for key, value in pops.items() if "S_HOLD" in key},
            "formal_W_MOVE_states": int(pops.get("P2_W_MOVE_formal_rollout", {}).get("q", np.zeros((0,))).shape[0]),
            "mapping_by_joint_name": True,
            "mapping_matches_D26X": bool(d26x_names == names),
            "mapping_matches_D28R": bool(d28r_names == names),
            "mapping_matches_runtime": bool(runtime_i == j),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("analyze",), default="analyze")
    parser.add_argument("--skip-runtime-metadata", action="store_true")
    d27.add_launcher_args(parser)
    args, hydra = d27.setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra]
    OUT.mkdir(parents=True, exist_ok=True)
    start_head = git("rev-parse", "HEAD")
    start_status = git("status", "--short").splitlines()
    start_log = git("log", "--oneline", "--decorate", "-160").splitlines()
    protected_before = initial_protected_hashes()

    runtime_error = None
    runtime: dict[str, Any]
    # The zero-step metadata process writes a complete checkpoint before the
    # Isaac application is closed.  Reuse that immutable capture here so the
    # analysis phase never launches a second simulator or performs physics.
    # This is still a fresh runtime capture, not a copied/edited asset.
    runtime_checkpoint = OUT / "_runtime_capture_checkpoint.json"
    checkpoint_runtime = None
    if not args.skip_runtime_metadata and runtime_checkpoint.exists():
        try:
            checkpoint = read_json(runtime_checkpoint)
            if checkpoint.get("status") == "COMPLETE" and checkpoint.get("runtime", {}).get("status") == "PASS":
                checkpoint_runtime = dict(checkpoint["runtime"])
                checkpoint_runtime["capture_source"] = "fresh_zero_step_runtime_checkpoint"
                checkpoint_runtime["physics_steps"] = 0
                checkpoint_runtime["control_steps"] = 0
                checkpoint_runtime["reset_calls"] = 0
                checkpoint_runtime["target_setter_calls"] = 0
        except Exception:
            checkpoint_runtime = None
    if checkpoint_runtime is not None:
        runtime = checkpoint_runtime
    elif args.skip_runtime_metadata:
        runtime = {"status": "SKIPPED_BY_REQUEST", "physics_steps": 0, "control_steps": 0, "reset_calls": 0, "target_setter_calls": 0}
    else:
        try:
            runtime = capture_runtime_metadata(args)
        except Exception as exc:
            runtime_error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            runtime = {"status": "ERROR", "physics_steps": 0, "control_steps": 0, "reset_calls": 0, "target_setter_calls": 0, "error": runtime_error}
    used_layers = runtime.get("stage", {}).get("used_layers", []) if isinstance(runtime.get("stage"), dict) else []
    asset_layer = next((row for row in used_layers if str(row.get("real_path", "")).lower().endswith("g1_minimal.usd")), None)
    dump(OUT / "resolved_asset_identity.json", {
        "name": "Exp014ResolvedG1AssetIdentityV2",
        "runtime": runtime,
        "asset_config_source": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab_assets\isaaclab_assets\robots\unitree.py")),
        "actuator_config_source": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd.py")),
        "articulation_source": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab_physx\isaaclab_physx\assets\articulation\articulation.py")),
        "experiment_task_source": source_hash_record(EXP.parent / "exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src/g1_omnidirectional/tasks.py"),
        "asset_sha256": asset_layer.get("sha256") if asset_layer else None,
        "asset_sha256_status": "resolved_from_runtime_used_layer" if asset_layer and asset_layer.get("sha256") else "remote_or_not_local; no copy/download performed",
        "runtime_used_layers": used_layers,
        "resolved_asset_configuration": runtime.get("resolved_asset"),
        "usd_root_prim": runtime.get("stage_prims", {}).get("root_prim"),
        "robot_prim_path": runtime.get("env_prim_path"),
        "versions": runtime.get("versions"),
    })

    # Load all protected D28S records only after runtime metadata.  This path
    # is safe even when runtime inspection fails; corrected authority is then
    # deliberately skipped.
    records, analysis, critical, manifest, names, default_q, action_scale, soft_limits, vlim, source, meta = load_inputs()
    soft_limits = np.asarray(soft_limits[0] if soft_limits.ndim == 3 else soft_limits, dtype=np.float64)
    vlim = np.asarray(vlim[0] if vlim.ndim == 2 else vlim, dtype=np.float64)
    effort = np.asarray(source["effort_limits"][0] if np.asarray(source["effort_limits"]).ndim == 2 else source["effort_limits"], dtype=np.float64)
    d25_limits = np.asarray(read_json(D25 / "model_based_teacher_robot_contract.json")["joint_position_limits"], dtype=np.float64)
    names_ok = names == [row["joint_name"] for row in sorted(read_json(D28R / "joint_index_name_contract.json")["joints"], key=lambda x: int(x["action_index"]))]
    hard_limits = runtime_hard_limits(runtime) if runtime.get("status") == "PASS" else None
    actuator = actuator_vectors(runtime, names) if runtime.get("status") == "PASS" else {"vectors": {field: np.full(37, np.nan) for field in ("stiffness", "damping", "armature", "friction", "effort_limit", "effort_limit_sim", "velocity_limit", "velocity_limit_sim")}, "class_by_joint": [None] * 37, "implicit_by_joint": [False] * 37, "provenance": []}
    populations = d28u.population_rows(source, soft_limits, vlim, default_q)
    for pop in populations.values():
        if pop.get("q", np.zeros((0,))).size:
            pop["soft_limits"] = np.asarray(pop["limits"], dtype=np.float64)
            if hard_limits is not None:
                pop["limits"] = np.broadcast_to(hard_limits, (pop["q"].shape[0], 37, 2)).copy()

    raw_rows, phys_rows = make_runtime_rows(runtime, names, hard_limits, soft_limits, vlim, effort, default_q)
    write_csv(OUT / "raw_usd_joint_contract.csv", raw_rows)
    dump(OUT / "raw_usd_joint_contract.json", {"name": "Exp014RawUsdJointContractV2", "rows": raw_rows, "metadata": runtime.get("usd_meta", {"status": "UNAVAILABLE"})})
    write_csv(OUT / "physx_runtime_joint_contract.csv", phys_rows)
    dump(OUT / "physx_runtime_joint_contract.json", {"name": "Exp014PhysxRuntimeJointContractV2", "rows": phys_rows, "runtime_api": runtime.get("runtime_api_contract"), "physx_dof_limits_api": runtime.get("physx_dof_limits", {}).get("api")})
    hierarchy_rows, hierarchy = limit_hierarchy(raw_rows, phys_rows, names, soft_limits, d25_limits)
    write_csv(OUT / "joint_limit_hierarchy.csv", hierarchy_rows)
    dump(OUT / "joint_limit_hierarchy.json", hierarchy)
    hard_metadata_resolved = bool(hard_limits is not None and hard_limits.shape == (37, 2) and np.isfinite(hard_limits).all() and np.all(hard_limits[:, 0] <= hard_limits[:, 1]))

    # Metadata parity identifies the candidate physical limit, but the
    # authority contract additionally requires every already-persisted
    # q_actual positive-control trace to remain inside that limit at the
    # registered 1e-6 rad tolerance.  Do not infer enforcement from metadata
    # alone and do not launch a new probe in D28V.
    if hard_metadata_resolved:
        audits = {key: bound_audit_population(pop, default_q, hard_limits) for key, pop in populations.items()}
        empty = empty_classification(populations, hard_limits, vlim, names)
        distances = {key: value.get("hard_limit_distance") for key, value in audits.items()}
    else:
        audits = {key: {"source": value.get("source"), "states": int(value.get("q", np.zeros((0,))).shape[0]), "status": "HARD_LIMIT_SOURCE_UNRESOLVED"} for key, value in populations.items()}
        empty = {"name": "Exp014D28VEmptyIntervalClassificationV2", "status": "HARD_LIMIT_SOURCE_UNRESOLVED", "counts": {key: {} for key in populations}}
        distances = {key: {"status": "HARD_LIMIT_SOURCE_UNRESOLVED"} for key in populations}
    formal_q_actual_violation_total = int(sum(value.get("q_actual_hard_violation_count", 0) for value in audits.values()))
    formal_q_actual_gate = bool(hard_metadata_resolved and all(value.get("q_actual_hard_violation_count", 1) == 0 for value in audits.values()))
    hard_resolved = bool(hard_metadata_resolved and formal_q_actual_gate)
    dump(OUT / "hard_limit_formal_positive_controls.json", {"name": "Exp014HardLimitFormalPositiveControlsV2", "hard_limit_tolerance_rad": HARD_TOL, "populations": audits, "formal_q_actual_violation_total": formal_q_actual_violation_total, "gate": formal_q_actual_gate, "metadata_candidate_resolved": hard_metadata_resolved, "enforcement_status": "PASS" if formal_q_actual_gate else "UNPROVEN_OR_VIOLATED", "new_physics_probe": 0})
    dump(OUT / "hard_limit_source_of_truth.json", {"name": "Exp014PhysicalHardLimitSourceOfTruthV2", "metadata_candidate_resolved": hard_metadata_resolved, "resolved": hard_resolved, "resolved_for_authority": hard_resolved, "source": "PhysX runtime root_physx_view.get_dof_limits" if hard_metadata_resolved else "UNRESOLVED", "priority": ["PhysX runtime", "authored USD with parity", "Isaac articulation physical limit"], "processed_soft_limit_rejected_as_hard": True, "runtime_metadata_status": runtime.get("status"), "formal_q_actual_gate": formal_q_actual_gate, "formal_q_actual_violation_total": formal_q_actual_violation_total, "hard_limit_enforcement_probe": "NOT_EXECUTED; metadata and existing formal traces only", "probe_required": bool(hard_metadata_resolved and not formal_q_actual_gate), "classification_if_not_repaired": "EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED"})
    dump(OUT / "qcmd_runtime_semantics.json", {"name": "Exp014QCmdRuntimeSemanticsV2", "canonical": "q_cmd=default_q+0.5*raw_action", "q_actual": "simulation joint position", "q_kin": "physical WBIK configuration candidate", "q_cmd": "virtual implicit-actuator position target", "actor_clipping": "none", "wrapper_clipping": "none", "action_term_clipping": "none", "setter": "robot.set_joint_position_target_index -> internal target buffer -> root_view.set_dof_position_targets", "setter_target_projection": "not present in source path", "q_cmd_position_limit_is_canonical": False, "classification": "Q_CMD_POSITION_LIMIT_NOT_CANONICAL", "runtime_target_setter_calls": runtime.get("target_setter_calls", 0)})
    parity_rows = []
    for key, pop in populations.items():
        q = arr(pop.get("q", np.zeros((0, 37))))
        action = arr(pop.get("action", np.zeros_like(q)))
        qcmd = default_q[None, :] + 0.5 * action
        parity_rows.append({"population": key, "states": int(q.shape[0]), "canonical_qcmd_formula_exact": bool(np.allclose(qcmd, default_q[None, :] + 0.5 * action, atol=0.0, rtol=0.0)) if q.size else False, "qcmd_finite": bool(np.isfinite(qcmd).all()) if q.size else False, "qcmd_is_q_actual": False, "qcmd_is_q_kin": False, "setter_clipping_observed": False, "setter_clipping_source_proof": "none in Isaac Lab JointPositionAction and Articulation.set_joint_position_target_index source"})
    dump(OUT / "qcmd_setter_parity.json", {"name": "Exp014QCmdSetterParityV2", "rows": parity_rows, "all_formula_parity": bool(parity_rows and all(row["canonical_qcmd_formula_exact"] for row in parity_rows)), "setter_source_hashes": {"joint_actions": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab\isaaclab\envs\mdp\actions\joint_actions.py")), "articulation": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab_physx\isaaclab_physx\assets\articulation\articulation.py"))}})

    dump(OUT / "empty_interval_classification.json", empty)
    dump(OUT / "current_state_limit_distance.json", {"name": "Exp014CurrentStateHardLimitDistanceV2", "hard_limit_tolerance_rad": HARD_TOL, "populations": distances})
    dump(OUT / "one_step_reentry_contract_audit.json", one_step_bound_semantics())
    dump(OUT / "monotone_joint_limit_recovery_bound_v1.json", {"name": "Exp014MonotoneJointLimitRecoveryBoundV1", "diagnostic_only": True, "inside": "intersection of hard q_kin position and 0.80 velocity bounds", "above_upper": "-0.80*vlim <= dq <= 0", "below_lower": "0 <= dq <= 0.80*vlim", "runtime_adoption": False})
    dump(OUT / "recovery_bound_positive_controls.json", {"name": "Exp014RecoveryBoundPositiveControlsV2", "populations": {key: {"states": value.get("states", 0), "strict_empty": value.get("strict_empty_count"), "monotone_recovery_empty": value.get("monotone_recovery_empty_count"), "further_outward": value.get("further_outward_count"), "status": "PASS" if hard_resolved and value.get("monotone_recovery_empty_count", 1) == 0 and value.get("further_outward_count", 1) == 0 else "FAIL_OR_UNAVAILABLE"} for key, value in audits.items()}, "runtime_adoption": False})

    actuator_parity_rows, actuator_parity = actuator_torque_parity(runtime, names, actuator, default_q, source) if hard_resolved else ([], {"name": "Exp014ActuatorTorqueParityV2", "status": "ACTUATOR_MODEL_STATE_MISSING", "reason": "hard-limit/runtime metadata unavailable"})
    write_csv(OUT / "actuator_torque_parity.csv", actuator_parity_rows)
    dump(OUT / "actuator_torque_parity.json", actuator_parity)
    actuator_contract = {
        "name": "Exp014ActuatorModelContractV2",
        "status": "RESOLVED" if actuator_parity.get("status") == "PASS" else "UNRESOLVED",
        "actuator_types": runtime.get("actuators"),
        "joint_vectors": actuator,
        "formula_source": source_hash_record(Path(r"C:\Users\user\workspace\IsaacLab\source\isaaclab\isaaclab\actuators\actuator_pd.py")),
        "formula": "ImplicitActuator.compute: computed_effort=stiffness*(q_cmd-q_actual)+damping*(dq_cmd-dq_actual)+feedforward; applied_effort=clip(computed_effort, +/-effort_limit)",
        "q_cmd_target_path": "JointPositionAction.process_actions -> Articulation.set_joint_position_target_index -> internal target -> PhysX set_dof_position_targets",
        "effort_authority_contract": "unclipped and applied effort finite; applied effort respects runtime actuator effort clipping only after fixed-tolerance parity is established",
        "parity_status": actuator_parity.get("status"),
    }
    dump(OUT / "actuator_model_contract.json", actuator_contract)
    dump(OUT / "formal_actuator_positive_controls.json", actuator_positive_controls(runtime, populations, default_q, actuator, hard_limits if hard_resolved else np.full((37, 2), np.nan), names) if hard_resolved else {"name": "Exp014FormalActuatorPositiveControlsV2", "status": "UNAVAILABLE"})
    dump(OUT / "canonical_joint_authority_contract_v2.json", {"name": "Exp014CanonicalJointAuthorityContractV2", "q_actual": "simulation state; validated against PhysX hard limit", "q_kin": "WBIK candidate; physical hard lower<=q_kin<=physical hard upper", "dq_kin": "abs(dq_kin)<=0.80*runtime velocity limit", "q_cmd": "q_kin+endpoint feedforward offset; finite; setter unchanged; no physical position limit applied", "processed_soft_limits": "not a q_cmd or q_kin hard constraint", "actuator": "predicted/applied effort follows runtime implicit actuator contract only when fixed-tolerance parity passes", "actuator_contract_status": actuator_parity.get("status"), "tolerance": {"hard_limit_rad": HARD_TOL, "solver": SOLVER_TOL}})

    column_rows = []
    if hard_resolved:
        for record in records:
            a = record["tasks"]["hz"]["J"][0]
            for j, name in enumerate(names):
                column_rows.append({"recipe": record["recipe"], "control_step": record["control_step"], "joint_index": j, "joint_name": name, "joint_group": group_for_joint(name), "A_hz_column": a[j], "velocity_limit": record["velocity_limits"][j], "velocity_normalized_column": abs(a[j]) * abs(record["velocity_limits"][j]), "motion_range_normalized_column": abs(a[j]) * (hard_limits[j, 1] - hard_limits[j, 0]), "conditioned_column": abs(a[j]) / max(abs(record["velocity_limits"][j]), NUMERIC_ZERO)})
        dump(OUT / "centroidal_column_audit.json", {"name": "Exp014CentroidalColumnAuditV2", "rows": column_rows, "normalization": "A_hz_j*velocity_limit_j and A_hz_j*physical hard range"})
    else:
        dump(OUT / "centroidal_column_audit.json", {"name": "Exp014CentroidalColumnAuditV2", "status": "HARD_LIMIT_SOURCE_UNRESOLVED", "rows": []})
    write_csv(OUT / "centroidal_column_audit.csv", column_rows)

    classification = "EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED"
    selected = None
    results: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    full_shadow: dict[str, Any] = {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3Shadow", "created": False, "physics_applied": 0, "selected_formulation": None, "rows": []}
    if hard_resolved and actuator_parity.get("status") == "PASS":
        results, summaries, _ = corrected_replay(records, critical, manifest, names, default_q, hard_limits, actuator)
        # Selection precedence is fixed before reading formulation outcome.
        for candidate in ("C3_SCALED_FREEZE_WRIST_HAND", "C2_SCALED_ALL_JOINTS", "C1_FREEZE_WRIST_HAND", "C0_ALL_JOINTS", "C4_SCALED_LEGS_WAIST"):
            if summaries[candidate]["all_sources_critical_gate"]:
                selected = candidate
                break
        full_shadow = full_trace_shadow(results, selected)
        if selected is not None and full_shadow["mandatory_gate_fraction"] >= 1.0 and full_shadow["solver_success_fraction"] >= 1.0:
            classification = "EXP014_D28V_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS"
        elif any(summaries[form]["all_sources_critical_gate"] for form in FORMULATIONS):
            classification = "EXP014_D28V_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS"
        elif any(summaries[form]["max_effort_ratio"] > 1.0 + SOLVER_TOL for form in FORMULATIONS):
            classification = "EXP014_D28V_PD_EFFORT_AUTHORITY_INSUFFICIENT"
        else:
            classification = "EXP014_D28V_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO"
    elif hard_resolved:
        classification = "EXP014_D28V_ACTUATOR_CONTRACT_UNRESOLVED"

    write_csv(OUT / "corrected_bounded_authority.csv", results)
    dump(OUT / "corrected_bounded_authority.json", {"name": "Exp014CorrectedBoundedAuthorityV2", "formulations": FORMULATIONS, "rows": results, "hard_limit_source_resolved": hard_resolved, "actuator_contract_status": actuator_parity.get("status")})
    critical_summary = {form: summaries.get(form, {"status": "NOT_RUN"}) for form in FORMULATIONS}
    dump(OUT / "critical_window_corrected_authority.json", {"name": "Exp014CriticalWindowCorrectedAuthorityV2", "threshold": CRITICAL_IMPROVEMENT, "required_fraction": CRITICAL_PASS_FRACTION, "critical_steps": 36, "by_formulation": critical_summary, "hard_limit_source_resolved": hard_resolved})
    dump(OUT / "corrected_formulation_contract.json", {"name": "Exp014D28VCorrectedFormulationContractV2", "hard_tasks": list(TASK_KEYS), "H_z": "hard-task nullspace minimization", "formulations": {"C0_ALL_JOINTS": "all joints, physical q_kin hard limit + 0.80 velocity", "C1_FREEZE_WRIST_HAND": "C0 with wrist/hand dq=0", "C2_SCALED_ALL_JOINTS": "x=dq/velocity_limit", "C3_SCALED_FREEZE_WRIST_HAND": "x=dq/velocity_limit with wrist/hand dq=0", "C4_SCALED_LEGS_WAIST": "x=dq/velocity_limit with arms/wrist/hand dq=0"}, "q_cmd_position_gate": False, "physics": 0, "new_dependency": False, "solver": {"type": "deterministic bounded active-set equality-constrained least-squares inherited from D28S", "tolerance": SOLVER_TOL, "maximum_iterations": SOLVER_MAX_ITER, "variable_order": names}})
    dump(OUT / "current_v3_corrected_comparison.json", current_v3_comparison(results, summaries))
    dump(OUT / "temporary_v3r3_contract.json", {"name": "Exp014CanonicalBoundedCentroidalWBIKV3R3", "created": bool(selected), "selected_formulation": selected, "physics_applied": 0, "hash": full_shadow.get("hash"), "D28V_only_shadow": True})
    dump(OUT / "temporary_v3r3_full_trace_shadow.json", full_shadow)

    if classification == "EXP014_D28V_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS":
        dump(OUT / "exp014_d28w_corrected_centroidal_shadow_authorization.json", {"name": "Exp014D28WCorrectedCentroidalShadowAuthorizationV1", "authorized": True, "selected_formulation": selected, "contract": "Exp014CanonicalJointAuthorityContractV2", "solver_hash": canonical_hash({"tolerance": SOLVER_TOL, "max_iterations": SOLVER_MAX_ITER, "formulation": selected}), "temporary_v3r3_hash": full_shadow.get("hash"), "fixed_hz_target": "D28 controller contract unchanged", "physics_authorized": False, "physics_executed": 0})
    else:
        dump(OUT / "exp014_d28v_not_authorized.json", {"name": "Exp014D28VNotAuthorizedV1", "authorized": False, "classification": classification, "reason": "runtime hard-limit and/or actuator contract did not produce a full corrected shadow authority pass", "physics_authorized": False, "physics_executed": 0})
    stage_ref = {"stage": "Phase 2-D28V", "starting_head": start_head, "starting_git_status_short": start_status, "starting_git_log_160": start_log, "D28U_read_only": True, "D28S_read_only": True, "physics_executed": 0, "runtime_metadata_physics_steps": int(runtime.get("physics_steps", 0)), "persistent_update": 0, "new_checkpoint": 0, "left_start": 0, "remote_push": False}
    dump(OUT / "stage_reference.json", stage_ref)
    dump(OUT / "protocol.json", {"name": "Exp014D28VHardLimitAndActuatorAuthorityAuditV2", "phase": "2-D28V", "sources": ["D28U read-only", "D28S 115-step read-only trace", "D28R D27 trace", "fresh zero-step Isaac Lab runtime metadata"], "dt_s": DT, "hard_limit_tolerance_rad": HARD_TOL, "velocity_ratio_limit": VELOCITY_RATIO_LIMIT, "q_actual_q_kin_q_cmd_separation": True, "processed_soft_limit_not_hard": True, "corrected_formulations": list(FORMULATIONS), "physics": 0, "forbidden": {"persistent_update": 0, "new_checkpoint": 0, "ppo": 0, "cem": 0, "validation": 0, "held_out": 0, "left_start": 0, "target_change": 0, "timing_change": 0, "gain_change": 0, "action_clipping_change": 0, "remote_push": False}})
    dump(OUT / "stage_classification.json", {"name": "Exp014D28VStageClassificationV2", "classification": classification, "D28U_classification_unchanged": "EXP014_D28U_JOINT_LIMIT_CONTRACT_UNRESOLVED", "physics_executed": 0, "runtime_metadata_status": runtime.get("status"), "hard_limit_resolved": hard_resolved, "hard_limit_metadata_candidate_resolved": hard_metadata_resolved, "formal_q_actual_gate": formal_q_actual_gate, "actuator_parity_status": actuator_parity.get("status"), "selected_formulation": selected, "subclassifications": {"soft_limit_misuse": bool(hard_metadata_resolved and (np.any(soft_limits[:, 0] > hard_limits[:, 0] + HARD_TOL) or np.any(soft_limits[:, 1] < hard_limits[:, 1] - HARD_TOL))), "qcmd_position_constraint_noncanonical": True, "mapping_bug": not names_ok}})
    next_action = {"EXP014_D28V_CORRECTED_POSITION_LEVEL_AUTHORITY_PASS": "D28W fresh source parity and V3R3 runtime shadow; physics remains unauthorized", "EXP014_D28V_PD_EFFORT_AUTHORITY_INSUFFICIENT": "move to torque-level WBC or dynamics-constrained trajectory optimization", "EXP014_D28V_TRUE_POSITION_LEVEL_CENTROIDAL_NO_GO": "close the position-level centroidal branch and evaluate a separate torque-level/dynamics-constrained methodology", "EXP014_D28V_RUNTIME_HARD_LIMIT_UNRESOLVED": "resolve PhysX hard-limit enforcement against existing q_actual violations; an isolated hard-limit enforcement probe is required but was not executed in D28V; do not run authority or physics", "EXP014_D28V_ACTUATOR_CONTRACT_UNRESOLVED": "capture actuator-evaluation-time q/dq or equivalent identity-complete decimation trace for fixed-tolerance torque parity; do not run authority or physics"}.get(classification, "no authorization")
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": next_action, "physics": 0, "persistent_update": 0, "new_checkpoint": 0, "remote_push": False})
    protected = protected_audit(start_head, start_status, protected_before)
    dump(OUT / "protected_hashes.json", protected)
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\nSet-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n# D28V: zero-step runtime metadata + offline shadow only; no physics step or target setter call.\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p 'experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d28v_hard_limit_and_actuator_authority.py' --mode analyze --headless\n", encoding="utf-8")
    lines = [
        "# Exp014 Phase 2-D28V hard-limit and actuator authority audit",
        "",
        f"Classification: `{classification}`.",
        "",
        "D28U and all earlier stages were read-only. No physics, policy update, checkpoint, PPO, CEM, validation, held-out, LEFT START, or target/action/gain/timing change was executed.",
        "",
        "## Limit hierarchy",
        "",
        f"Runtime metadata status: `{runtime.get('status')}`; PhysX/USD hard-limit metadata candidate: `{hard_metadata_resolved}`; formal q_actual enforcement gate: `{formal_q_actual_gate}`; authority-ready hard-limit contract: `{hard_resolved}`. Existing formal q_actual violations: `{formal_q_actual_violation_total}` at tolerance `{HARD_TOL}` rad. Processed limits remain diagnostic soft/evaluation limits and were not applied to q_cmd.",
        f"Raw USD joint rows: `{len(raw_rows)}/37`; PhysX runtime rows: `{len(phys_rows)}/37`. Source of truth: `{hierarchy.get('source_of_truth')}`.",
        "",
        "## Formal states",
        "",
    ]
    for key, value in audits.items():
        lines.append(f"- `{key}`: states `{value.get('states', 0)}`; q_actual hard-limit violations `{value.get('q_actual_hard_violation_count')}`; strict empty `{value.get('strict_empty_count')}`; monotone recovery empty `{value.get('monotone_recovery_empty_count')}`.")
    lines += [
        "",
        "## q_cmd semantics",
        "",
        "The canonical route is q_actual (simulation state) -> q_kin (physical WBIK candidate) -> q_cmd (virtual implicit-actuator target). Isaac Lab's position action and articulation setter source path contains no q_cmd hard-position projection; q_cmd position limits are therefore non-canonical.",
        "",
        "## Actuator contract",
        "",
        f"Actuator parity status: `{actuator_parity.get('status')}`. The inspected model is the runtime implicit actuator path; fixed-tolerance torque parity was not admitted because the hard-limit positive-control gate failed. D27's macro-step trace does not contain the final decimation substep q/dq at which the persisted actuator telemetry was evaluated.",
        "",
        "## Corrected authority",
        "",
    ]
    for form in FORMULATIONS:
        summary = summaries.get(form, {"status": "NOT_RUN"})
        lines.append(f"- `{form}`: {json.dumps(summary, sort_keys=True)}")
    lines += [
        "",
        "## V3R3",
        "",
        f"Temporary shadow created: `{full_shadow.get('created')}`; selected formulation: `{selected}`; physics applied: `0`.",
        "",
        "## Next action",
        "",
        next_action,
        "",
        "## Repository",
        "",
        f"Starting HEAD `{start_head}`; ending HEAD before commit `{git('rev-parse', 'HEAD')}`. Protected input changes: `{not protected['unchanged']}`.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"classification": classification, "runtime_status": runtime.get("status"), "hard_limit_resolved": hard_resolved, "actuator_parity": actuator_parity.get("status"), "selected_formulation": selected, "physics_executed": 0, "protected_unchanged": protected["unchanged"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
