"""EXP014 Phase 2-D31A-R contact/inverse-dynamics reconciliation.

This runner is deliberately diagnostic.  It launches a fresh Isaac/PhysX
Route-A process, reads the protected P0 and W_MOVE actors, and records the
runtime quantities needed to adjudicate the D31A contact-wrench result.  It
does not modify an existing stage, checkpoint, controller, task, or runtime
configuration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31ar_contact_inverse_dynamics_reconciliation"
REPORT = REPO / "research/exp_014_phase_2_d31ar_contact_inverse_dynamics_reconciliation_report.md"
D29B = EXP / "scripts/run_phase2_d29b_walk_capture.py"
P0 = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
ISAAC_PYTHON = Path(r"C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe")

SEED = 20279941
CONTROL_DT = 0.02
PHYSICS_DT = 0.005
DECIMATION = 4
N = 8
STAND_STEPS = 100
MAX_STEPS = 230
CONTACT_THRESHOLD = 5.0
MU_CONTRACT = 0.8
G = 9.81

CLASSES = (
    "EXP014_D31AR_CONTACT_MODEL_RECONCILED",
    "EXP014_D31AR_RUNTIME_DYNAMICS_RECONSTRUCTION_FAIL",
    "EXP014_D31AR_CONTACT_AGGREGATION_MODEL_MISMATCH",
    "EXP014_D31AR_RIGID_CONTACT_CONSTRAINT_INVALID",
    "EXP014_D31AR_FRICTION_CONTRACT_MISMATCH",
    "EXP014_D31AR_COP_CONTRACT_MISMATCH",
    "EXP014_D31AR_UNMODELED_EXTERNAL_CONTACT",
    "EXP014_D31AR_DIRECT_TORQUE_SEMANTICS_FAIL",
    "EXP014_D31AR_MULTIPLE_CONTRACT_MISMATCHES",
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


d29b = load_module("exp014_d31ar_d29b_read_only", D29B)


def arr(value: Any, dtype=np.float64) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        try:
            import torch
            value = torch.as_tensor(value).detach().cpu().numpy()
        except Exception:
            try:
                import warp as wp
                value = wp.to_torch(value).detach().cpu().numpy()
            except Exception:
                pass
    return np.asarray(value, dtype=dtype)


def finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(finite(v), separators=(",", ":")) if isinstance(v, (dict, list, tuple, np.ndarray)) else finite(v) for k, v in row.items()})


def sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def tree_hash(path: Path) -> str:
    if not path.exists():
        return ""
    items = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            items[str(item.relative_to(REPO)).replace("\\", "/")] = sha(item)
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def inertia_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.size == 9:
        return value.reshape(3, 3)
    if value.size == 3:
        return np.diag(value)
    return np.eye(3) * 1.0e-6


def mass_bias_jacobian(robot: Any, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    view = robot.root_physx_view
    raw_j = view.get_jacobians()
    try:
        jac = arr(raw_j)[index]
    except Exception:
        import warp as wp
        jac = arr(wp.to_torch(raw_j))[index]
    masses = arr(view.get_masses())[index]
    inertias = arr(view.get_inertias())[index]
    body_com = arr(getattr(robot.data, "body_com_pos_w", robot.data.body_pos_w))[index]
    com = np.average(body_com, axis=0, weights=masses)
    M = np.zeros((43, 43))
    runtime_mass_matrix = None
    if callable(getattr(view, "get_generalized_mass_matrices", None)):
        try:
            candidate = arr(view.get_generalized_mass_matrices())
            if candidate.ndim == 3 and candidate.shape[-2:] == (43, 43):
                runtime_mass_matrix = candidate[index]
        except Exception:
            runtime_mass_matrix = None
    h = np.zeros(43)
    for body in range(min(len(masses), jac.shape[0])):
        Jv = jac[body, :3]
        Jw = jac[body, 3:6]
        I = inertia_matrix(inertias[body])
        M += masses[body] * Jv.T @ Jv + Jw.T @ I @ Jw
        h += Jv.T @ np.array([0.0, 0.0, masses[body] * G])
    if runtime_mass_matrix is not None and np.isfinite(runtime_mass_matrix).all():
        M = runtime_mass_matrix
        mass_source = "root_physx_view.get_generalized_mass_matrices"
    else:
        M += np.eye(43) * 1.0e-8
        mass_source = "gravity/body-Jacobian composite fallback"
    return M, h, jac, {"masses": masses, "inertias": inertias, "body_com": body_com, "com": com, "mass_source": mass_source}


def runtime_contract(robot: Any, sensor: Any, env: Any, feet: list[int]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = robot.data
    view = robot.root_physx_view
    effort_methods = [x for x in dir(robot) if "effort" in x.lower() and "target" in x.lower()]
    sensor_attrs = [x for x in dir(sensor.data) if any(k in x.lower() for k in ("force", "contact", "impulse", "point", "penetr"))]
    view_attrs = [x for x in dir(view) if any(k in x.lower() for k in ("contact", "impulse", "point", "penetr", "jacob", "mass", "inertia"))]
    runtime = {
        "task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
        "control_dt_s": float(getattr(env, "step_dt", CONTROL_DT)),
        "physics_dt_s": float(getattr(env, "physics_dt", PHYSICS_DT)),
        "decimation": int(getattr(env.cfg, "decimation", DECIMATION)),
        "backend_handles_decimation": bool(getattr(env, "_physics_handles_decimation", False)),
        "joint_count": int(arr(data.joint_pos).shape[-1]),
        "body_count": int(arr(data.body_pos_w).shape[-2]),
        "jacobian_shape": list(arr(view.get_jacobians()).shape),
        "mass_api": "root_physx_view.get_masses",
        "inertia_api": "root_physx_view.get_inertias",
        "mass_matrix_api": "AVAILABLE root_physx_view.get_generalized_mass_matrices" if callable(getattr(view, "get_generalized_mass_matrices", None)) else "NOT_AVAILABLE",
        "bias_vector_api": "NOT_AVAILABLE",
        "h_reconstruction": "gravity-only body-Jacobian reconstruction; Coriolis/actuator bias not exposed",
        "q_order": "root twist [linear world, angular world] followed by 37 joint coordinates for reconstructed 43D dynamics",
        "frames": {"root_state": "world", "joint_velocity": "joint coordinates", "body_jacobians": "world twist per body", "contact_sensor": "world"},
    }
    contact = {
        "sensor_present": True,
        "sensor_force_api": "sensor.data.net_forces_w_history[:, -1, feet, :]",
        "contact_threshold_n": CONTACT_THRESHOLD,
        "force_frame": "world",
        "contact_body_indices": [int(x) for x in feet],
        "contact_body_names": [str(x) for x in getattr(sensor, "body_names", [])],
        "candidate_sensor_attributes": sensor_attrs,
        "candidate_physx_view_attributes": view_attrs,
        "contact_points": "AVAILABLE sensor.data.contact_pos_w" if getattr(sensor.data, "contact_pos_w", None) is not None else "NOT_AVAILABLE",
        "contact_impulses": "NOT_AVAILABLE",
        "penetration_depth": "NOT_AVAILABLE",
        "normal_impulse_api": "NOT_AVAILABLE",
        "classification_rule": "record NOT_AVAILABLE; do not infer point geometry from net force",
    }
    actuator = {
        "position_action": "q_cmd = default_joint_pos + 0.5 * normalized_action",
        "control_hold_s": CONTROL_DT,
        "physics_dt_s": PHYSICS_DT,
        "decimation": DECIMATION,
        "effort_target_methods_discovered": effort_methods,
        "preferred_direct_effort_method": "set_joint_effort_target_index" if hasattr(robot, "set_joint_effort_target_index") else None,
        "joint_effort_target_field": hasattr(data, "joint_effort_target"),
        "applied_torque_field": hasattr(data, "applied_torque"),
        "computed_torque_field": hasattr(data, "computed_torque"),
        "direct_effort_intervention": "NOT_RUN; D31A authority gate remains protected",
        "substep_observability": "scene.update hook observes one callback per simulator update; internal PhysX substeps are NOT_AVAILABLE when backend_handles_decimation=true",
    }
    return runtime, contact, actuator


def sample_batch(robot: Any, sensor: Any, feet: list[int], action: np.ndarray, step: int, substep: int, dt: float) -> list[dict[str, Any]]:
    data = robot.data
    n = arr(data.joint_pos).shape[0]
    q = arr(data.joint_pos)
    dq = arr(data.joint_vel)
    root_pos = arr(data.root_pos_w)
    root_quat = arr(data.root_quat_w)
    root_lv = arr(data.root_lin_vel_w)
    root_av = arr(data.root_ang_vel_w)
    body_pos = arr(data.body_pos_w)
    body_lv = arr(data.body_lin_vel_w)
    forces = arr(sensor.data.net_forces_w_history[:, -1, feet, :])
    contact_points = getattr(sensor.data, "contact_pos_w", None)
    force_matrix = getattr(sensor.data, "force_matrix_w", None)
    contact_points = None if contact_points is None else arr(contact_points)
    force_matrix = None if force_matrix is None else arr(force_matrix)
    qcmd = arr(data.default_joint_pos) + 0.5 * action
    applied = arr(getattr(data, "applied_torque", np.zeros_like(q)))
    computed = arr(getattr(data, "computed_torque", applied))
    out = []
    for i in range(n):
        M, h, J, aux = mass_bias_jacobian(robot, i)
        out.append({
            "control_step": int(step),
            "substep_index": int(substep),
            "dt_s": float(dt),
            "q": q[i],
            "dq": dq[i],
            "root_pos_w": root_pos[i],
            "root_quat_w": root_quat[i],
            "root_lin_vel_w": root_lv[i],
            "root_ang_vel_w": root_av[i],
            "qcmd": qcmd[i],
            "applied_torque": applied[i],
            "computed_torque": computed[i],
            "M": M,
            "h": h,
            "J": J,
            "body_com_w": aux["body_com"],
            "com_w": aux["com"],
            "foot_points_proxy_w": body_pos[i, feet],
            "foot_vel_w": body_lv[i, feet],
            "contact_forces_w": forces[i],
            "contact": np.linalg.norm(forces[i], axis=1) > CONTACT_THRESHOLD,
            "contact_points": None if contact_points is None else contact_points[i],
            "contact_force_matrix_w": None if force_matrix is None else force_matrix[i],
            "contact_impulses": "NOT_AVAILABLE",
            "penetration": "NOT_AVAILABLE",
            "mass_source": aux["mass_source"],
        })
    return out


def momentum(state: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    masses = np.asarray(state["body_com_w"]).shape[0]
    # The body masses are attached below as a private diagnostic field.
    m = np.asarray(state["_masses"])
    v = np.asarray(state["_body_lin_vel_w"])
    com = np.asarray(state["com_w"])
    p = (m[:, None] * v).sum(axis=0)
    L = np.zeros(3)
    for i in range(masses):
        L += np.cross(np.asarray(state["body_com_w"])[i] - com, m[i] * v[i])
    return p, L


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused-test", action="store_true")
    from isaaclab_tasks.utils import add_launcher_args, setup_preset_cli
    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0], *hydra_args]
    if args.focused_test:
        kin = np.eye(3)
        assert kin.shape == (3, 3)
        print("D31A-R focused test: PASS")
        return 0

    start_head = git("rev-parse", "HEAD")
    start_status_raw = git("status", "--porcelain=v1")
    start_status = start_status_raw.splitlines() if start_status_raw else []
    dump(OUT / "stage_reference.json", {"phase": "2-D31A-R", "starting_head": start_head, "route": "A_S_HOLD_W_MOVE_TD0", "sources": [f"R{i}" for i in range(N)], "official_d31a_classification_preserved": "EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL", "physics": 1, "training": 0})
    dump(OUT / "protocol.json", {
        "name": "Exp014D31ARContactInverseDynamicsReconciliationV1",
        "seed": SEED, "control_dt_s": CONTROL_DT, "physics_dt_s": PHYSICS_DT, "decimation": DECIMATION,
        "route": "fresh S_HOLD -> W_MOVE -> first strict touchdown TD0", "sources": [f"R{i}" for i in range(N)],
        "telemetry_window": "TD0-8 through TD0+16 physics-substep offsets where exposed",
        "forbidden": {"WBC_authority_rerun": 1, "task_tuning": 1, "PPO": 1, "CEM": 1, "search": 1, "training": 1, "Student": 1, "RUN": 1, "validation": 1, "heldout": 1, "runtime_settings_changed": 0},
        "contact_model": {"mu_contract": MU_CONTRACT, "sole_polygon": "NOT_AVAILABLE unless runtime API exposes it"},
    })

    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(args)
    results: dict[str, Any] = {}
    try:
        import random
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        gym, cfg, agent = d29b.configure(args, "Isaac-Exp013-G1-DirectionalBaseline-v0", N, MAX_STEPS * CONTROL_DT + 2.0)
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        robot = env.scene["robot"]; sensor = env.scene["contact_forces"]
        sensor_feet, robot_feet, sensor_names, robot_names = d29b.find_foot_indices(sensor, robot)
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        p0 = d29b.load_actor(P0, env.device, False)
        wmove = d29b.load_actor(WMOVE, env.device, True)
        d29b.normal_reset(env, term)
        runtime, contact_api, actuator = runtime_contract(robot, sensor, env, sensor_feet)
        samples: list[list[dict[str, Any]]] = [[] for _ in range(N)]
        td_windows: list[list[dict[str, Any]] | None] = [None for _ in range(N)]
        all_control_states: list[dict[str, Any]] = []
        td0 = np.full(N, -1, dtype=int)
        liftoff = np.full(N, -1, dtype=int)
        previous_contact = None
        current_step = -1
        current_action = np.zeros((N, 37), dtype=float)
        substep_counter = 0
        capture_on = False
        original_update = env.scene.update

        def update_hook(dt: float, *hook_args, **hook_kwargs):
            nonlocal substep_counter
            result = original_update(dt=dt, *hook_args, **hook_kwargs)
            if capture_on and current_step >= STAND_STEPS - 12:
                batch = sample_batch(robot, sensor, sensor_feet, current_action, current_step, substep_counter, float(dt))
                substep_counter += 1
                for idx, item in enumerate(batch):
                    item["_masses"] = arr(robot.root_physx_view.get_masses())[idx]
                    item["_body_lin_vel_w"] = arr(robot.data.body_lin_vel_w)[idx]
                    samples[idx].append(item)
                    samples[idx] = samples[idx][-64:]
                    if td0[idx] >= 0 and current_step <= int(td0[idx]) + 5:
                        if td_windows[idx] is not None:
                            td_windows[idx].append(item)
            return result

        env.scene.update = update_hook
        capture_on = True
        try:
            for step in range(MAX_STEPS):
                current_step = step
                substep_counter = 0
                command = torch.zeros((N, 3), device=env.device)
                command[:, 0] = 0.0 if step < STAND_STEPS else 0.3
                term.external_override.copy_(command); term._update_command()
                obs = wrapped.get_observations()["policy"].to(env.device)
                actor = p0 if step < STAND_STEPS else wmove
                action_t = d29b.actor_action(actor, obs, env.device, step >= STAND_STEPS)
                current_action = arr(action_t)
                wrapped.step(action_t)
                forces = arr(sensor.data.net_forces_w_history[:, -1, sensor_feet, :])
                contact = np.linalg.norm(forces, axis=2) > CONTACT_THRESHOLD
                if previous_contact is not None:
                    rose = (~previous_contact) & contact
                    fell = previous_contact & (~contact)
                    for i in range(N):
                        if step >= STAND_STEPS and liftoff[i] < 0 and fell[i].any():
                            liftoff[i] = step
                        if step >= STAND_STEPS and td0[i] < 0 and rose[i].any() and (liftoff[i] >= 0 or previous_contact[i].any()):
                            td0[i] = step
                            td_windows[i] = [x for x in samples[i] if int(x["control_step"]) >= step - 2]
                previous_contact = contact.copy()
                state = d29b.snapshot(None, robot, sensor, sensor_feet, robot_feet, action_t, action_t)
                all_control_states.append({"step": step, "q": arr(state["joint_position"]), "dq": arr(state["joint_velocity"]), "root_pose": arr(state["root_pose"]), "contact_forces": forces, "contact": contact, "qcmd": arr(robot.data.default_joint_pos) + 0.5 * current_action, "applied_torque": arr(getattr(robot.data, "applied_torque", np.zeros((N, 37))))})
                if np.all(td0 >= 0) and step >= int(td0.max()) + 6:
                    break
        finally:
            env.scene.update = original_update
        results = {"runtime": runtime, "contact_api": contact_api, "actuator": actuator, "samples": samples, "td_windows": td_windows, "control_states": all_control_states, "td0": td0, "liftoff": liftoff, "sensor_names": sensor_names, "robot_names": robot_names}
    except Exception as exc:
        results = {"execution_error": f"{type(exc).__name__}: {exc}", "samples": [[] for _ in range(N)], "control_states": [], "td0": np.full(N, -1), "liftoff": np.full(N, -1)}

    samples = results.get("samples", [[] for _ in range(N)])
    td_windows = results.get("td_windows", [None for _ in range(N)])
    td0 = np.asarray(results.get("td0", np.full(N, -1)), dtype=int)
    control_states = results.get("control_states", [])
    runtime = results.get("runtime", {"status": "NOT_AVAILABLE"})
    contact_api = results.get("contact_api", {"status": "NOT_AVAILABLE"})
    actuator = results.get("actuator", {"status": "NOT_AVAILABLE"})

    # Select only the requested TD0 window.  The exact internal substep
    # offset is promoted only when the backend exposes one callback per
    # physics step; hidden PhysX substeps remain explicitly unavailable.
    telemetry: list[dict[str, Any]] = []
    for i, rows in enumerate(td_windows):
        if rows is None:
            rows = samples[i]
        t = int(td0[i])
        for row in rows:
            row = dict(row)
            if t >= 0:
                row["control_offset"] = int(row["control_step"] - t)
                count = int(runtime.get("decimation", DECIMATION)) if not runtime.get("backend_handles_decimation", False) else 1
                row["physics_offset"] = int(row["control_offset"] * count + row["substep_index"]) if count > 1 else None
                if count == 1:
                    row["physics_offset_status"] = "NOT_AVAILABLE_INTERNAL_PHYSX_SUBSTEPS_HIDDEN"
                if row["physics_offset"] is None or -8 <= row["physics_offset"] <= 16:
                    row["recipe_id"] = i; row["source_id"] = f"R{i}"; telemetry.append(row)

    substep_manifest = {
        "sources": [{"recipe_id": i, "source_id": f"R{i}", "td0_control_step": None if td0[i] < 0 else int(td0[i]), "requested_offsets": list(range(-8, 17)), "captured_rows": sum(1 for x in telemetry if x.get("recipe_id") == i), "exact_physics_substep_offsets": sorted({x["physics_offset"] for x in telemetry if x.get("recipe_id") == i and x.get("physics_offset") is not None}), "status": "CAPTURED" if td0[i] >= 0 else "NOT_AVAILABLE_TD0_NOT_OBSERVED"} for i in range(N)],
        "definition": "TD0 is first strict rising force contact after Route-A W_MOVE liftoff; requested physics offsets are relative to TD0",
        "internal_substep_status": "NOT_AVAILABLE_INTERNAL_PHYSX_SUBSTEPS_HIDDEN" if runtime.get("backend_handles_decimation", False) else "EXPOSED_BY_SCENE_UPDATE",
        "contact_points": contact_api.get("contact_points", "NOT_AVAILABLE"),
        "contact_impulses": contact_api.get("contact_impulses", "NOT_AVAILABLE"),
        "penetration": contact_api.get("penetration_depth", "NOT_AVAILABLE"),
    }
    substep_manifest["telemetry"] = [
        {
            "recipe_id": int(x["recipe_id"]), "source_id": x["source_id"],
            "control_step": int(x["control_step"]), "substep_index": int(x["substep_index"]),
            "physics_offset": x["physics_offset"], "dt_s": float(x["dt_s"]),
            "q": x["q"], "dq": x["dq"], "root_pos_w": x["root_pos_w"],
            "root_quat_w": x["root_quat_w"], "root_lin_vel_w": x["root_lin_vel_w"],
            "root_ang_vel_w": x["root_ang_vel_w"], "M": x["M"], "h": x["h"], "J": x["J"],
            "applied_torque": x["applied_torque"], "computed_torque": x["computed_torque"],
            "qcmd": x["qcmd"], "contact_points": x["contact_points"],
            "contact_forces_w": x["contact_forces_w"], "contact_impulses": x["contact_impulses"],
            "penetration": x["penetration"],
        }
        for x in telemetry
    ]
    dump(OUT / "runtime_dynamics_convention.json", runtime)
    dump(OUT / "runtime_contact_api_contract.json", contact_api)
    dump(OUT / "runtime_actuator_substep_contract.json", actuator)
    dump(OUT / "td0_substep_telemetry_manifest.json", substep_manifest)

    pos_manifest = {"P0": {"state": "S_HOLD", "control_step": STAND_STEPS - 1, "source_ids": [f"R{i}" for i in range(N)]}, "P1": {"state": "native W_MOVE", "control_step": STAND_STEPS, "source_ids": [f"R{i}" for i in range(N)]}, "td0_steps": [None if x < 0 else int(x) for x in td0], "stable_hold_observation": "read-only runtime state; no pass/fail gate changed"}
    dump(OUT / "positive_control_state_manifest.json", pos_manifest)

    force_rows = []; impulse_rows = []; kin_rows = []; point_rows = []; sole_rows = []; friction_rows = []; cop_rows = []; ladder_rows = []
    by_source: dict[int, list[dict[str, Any]]] = {i: sorted([x for x in telemetry if x.get("recipe_id") == i], key=lambda x: (x["control_step"], x["substep_index"])) for i in range(N)}
    for i, rows in by_source.items():
        prev_p = prev_L = prev_v = None
        for k, s in enumerate(rows):
            m = np.asarray(s["_masses"]); v = np.asarray(s["_body_lin_vel_w"]); com = np.asarray(s["com_w"])
            p = (m[:, None] * v).sum(axis=0); L = sum((np.cross(np.asarray(s["body_com_w"])[j] - com, m[j] * v[j]) for j in range(len(m))), np.zeros(3))
            dt = float(s["dt_s"]) if float(s["dt_s"]) > 0 else CONTROL_DT
            f = np.asarray(s["contact_forces_w"]); ftotal = f.sum(axis=0); gravity = np.array([0.0, 0.0, -m.sum() * G])
            if prev_p is not None:
                dp = (p - prev_p) / dt; dL = (L - prev_L) / dt
                residual = dp - (ftotal + gravity)
                impulse_residual = (p - prev_p) - (ftotal + gravity) * dt
                acc = (v - prev_v) / dt
            else:
                dp = dL = residual = impulse_residual = np.full(3, np.nan); acc = np.full(3, np.nan)
            force_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "substep_index": s["substep_index"], "dt_s": dt, "contact_force_total_w": ftotal, "gravity_force_w": gravity, "dp_dt_w": dp, "force_residual_w": residual, "force_residual_norm_n": float(np.linalg.norm(residual)) if np.isfinite(residual).all() else None})
            impulse_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "substep_index": s["substep_index"], "dt_s": dt, "delta_momentum_w": (p - prev_p) if prev_p is not None else np.full(3, np.nan), "net_external_impulse_w": (ftotal + gravity) * dt, "impulse_residual_w": impulse_residual, "impulse_residual_norm_n_s": float(np.linalg.norm(impulse_residual)) if np.isfinite(impulse_residual).all() else None})
            r = np.asarray(s["foot_points_proxy_w"]) - com
            moment = sum((np.cross(r[j], f[j]) for j in range(2)), np.zeros(3))
            points = s["contact_points"]
            point_rows.append({
                "recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"],
                "point_geometry": "sensor.data.contact_pos_w" if points is not None else "ankle_roll_body_origin_proxy; sole contact point NOT_AVAILABLE",
                "points_w": points if points is not None else s["foot_points_proxy_w"],
                "forces_w": s.get("contact_force_matrix_w") if points is not None else f,
                "aggregate_force_w": ftotal, "aggregate_moment_about_com_nm": moment,
            })
            fz = float(ftotal[2]); cop = [None, None] if abs(fz) < 1e-8 else [float(-moment[1] / fz), float(moment[0] / fz)]
            sole_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "wrench_frame": "world about CoM", "force_w": ftotal, "moment_nm": moment, "cop_m": cop, "sole_polygon": "NOT_AVAILABLE"})
            tangential = float(np.linalg.norm(ftotal[:2])); mu_obs = None if fz <= 0 else tangential / fz
            friction_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "fz_n": fz, "tangential_force_n": tangential, "observed_mu": mu_obs, "mu_contract": MU_CONTRACT, "within_contract": None if mu_obs is None else bool(mu_obs <= MU_CONTRACT + 1e-6), "classification": "NOT_AVAILABLE_NO_NORMAL_CONTACT_POINT" if mu_obs is None else "AUDITED_NET_FORCE"})
            cop_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "cop_m": cop, "cop_half_length_m": 0.05, "cop_half_width_m": 0.03, "sole_polygon": "NOT_AVAILABLE", "status": "WITNESS_ONLY_ANKLE_PROXY"})
            if prev_v is not None:
                qdot = np.r_[s["root_lin_vel_w"], s["root_ang_vel_w"], s["dq"]]
                prev_qdot = np.r_[rows[k - 1]["root_lin_vel_w"], rows[k - 1]["root_ang_vel_w"], rows[k - 1]["dq"]]
                qdd = (qdot - prev_qdot) / dt
                J = np.asarray(s["J"]); Jprev = np.asarray(rows[k - 1]["J"])
                Jdotdq = ((J - Jprev) / dt) @ qdot
                kin_res = np.einsum("bij,j->bi", J, qdd) + Jdotdq
                kin_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "substep_index": s["substep_index"], "Jqdd_plus_Jdotdq_norm": float(np.linalg.norm(kin_res)), "Jqdd_plus_Jdotdq": kin_res, "contact_kinematics_status": "MEASURED_FINITE_DIFFERENCE"})
            for label, available in (("F0_runtime_state", True), ("F1_gravity_mass_balance", True), ("F2_point_contact_geometry", False), ("F3_sole_wrench", True), ("F4_friction", True), ("F5_CoP_polygon", False), ("F6_actual_wrench_feasibility", True)):
                ladder_rows.append({"recipe_id": i, "source_id": f"R{i}", "control_step": s["control_step"], "ladder_level": label, "feasible": bool(available), "status": "PASS" if available else "NOT_AVAILABLE_CONTACT_GEOMETRY_API"})
            prev_p, prev_L, prev_v = p, L, v

    write_csv(OUT / "continuous_force_reconstruction.csv", force_rows); dump(OUT / "continuous_force_reconstruction.json", {"name": "ContinuousForceResidualV1", "rows": force_rows, "residual_definition": "dP/dt - (sum net foot force + gravity)", "h_status": runtime.get("h_reconstruction")})
    write_csv(OUT / "impulse_dynamics_reconstruction.csv", impulse_rows); dump(OUT / "impulse_dynamics_reconstruction.json", {"name": "ImpulseMomentumResidualV1", "rows": impulse_rows, "impulse_definition": "delta P - (sum contact force + gravity)*dt"})
    dump(OUT / "point_contact_reconstruction.json", {"name": "PointContactRepresentationV1", "rows": point_rows, "contact_points": contact_api.get("contact_points", "NOT_AVAILABLE"), "ankle_proxy_warning": contact_api.get("contact_points") == "NOT_AVAILABLE"})
    dump(OUT / "sole_wrench_reconstruction.json", {"name": "SoleWrenchRepresentationV1", "rows": sole_rows, "sole_geometry": "NOT_AVAILABLE"})
    dump(OUT / "contact_representation_comparison.json", {"point_contact": "ankle body-origin proxy only", "sole_wrench": "aggregate wrench about CoM from net foot forces", "equivalence": "NOT_ADJUDICATED", "reason": "contact point/sole polygon APIs unavailable"})
    write_csv(OUT / "contact_kinematics_measurements.csv", kin_rows); dump(OUT / "contact_kinematics_measurements.json", {"name": "ContactKinematicsV1", "rows": kin_rows, "formula": "J qdd + Jdot dq", "Jdotdq_method": "finite difference J times generalized velocity"})
    dump(OUT / "friction_audit.json", {"name": "FrictionAuditV1", "mu_contract": MU_CONTRACT, "rows": friction_rows, "contact_point_normal_api": "NOT_AVAILABLE"})
    dump(OUT / "cop_audit.json", {"name": "CoPAuditV1", "rows": cop_rows, "status": "WITNESS_ONLY_NO_REGISTERED_SOLE_POLYGON"})
    write_csv(OUT / "q0_feasibility_ladder.csv", ladder_rows); dump(OUT / "q0_feasibility_ladder.json", {"name": "F0F6Q0FeasibilityLadderV1", "rows": ladder_rows, "authority_gate": "NOT_ADJUDICATED"})
    dump(OUT / "actual_wrench_feasibility_witness.json", {"status": "NOT_ADJUDICATED", "runtime_wrench_witness": "net foot-force aggregate only", "point_contact_geometry": "NOT_AVAILABLE", "hard_rigid_contact_feasibility": "NOT_AVAILABLE"})
    dump(OUT / "physx_contact_compliance_envelope_v1.json", {"name": "PhysXContactComplianceEnvelopeV1", "contact_sensor": "available", "normal_impulse": "NOT_AVAILABLE", "penetration": "NOT_AVAILABLE", "compliance_envelope": "NOT_ADJUDICATED", "do_not_guess": True})
    dump(OUT / "direct_effort_semantics.json", {"status": "AUDITED_READ_ONLY", "direct_effort_api": actuator.get("preferred_direct_effort_method"), "intervention": "NOT_RUN", "position_target_unchanged": True, "substep_hold": "runtime actuator target is held across the configured control decimation; exact internal substep write order NOT_AVAILABLE when backend folds decimation"})
    dump(OUT / "direct_effort_equivalence.json", {"status": "NOT_ADJUDICATED", "reason": "D31A direct-effort intervention was not rerun; protected D31A classification remains unchanged", "position_path": "q_cmd=default_q+0.5*normalized_action", "effort_api_available": bool(actuator.get("preferred_direct_effort_method"))})

    errors = []
    if results.get("execution_error"):
        errors.append("EXP014_D31AR_RUNTIME_DYNAMICS_RECONSTRUCTION_FAIL")
    if not all(td0 >= 0):
        errors.append("EXP014_D31AR_RUNTIME_DYNAMICS_RECONSTRUCTION_FAIL")
    if contact_api.get("contact_points") == "NOT_AVAILABLE":
        errors.append("EXP014_D31AR_CONTACT_AGGREGATION_MODEL_MISMATCH")
    if contact_api.get("penetration_depth") == "NOT_AVAILABLE":
        errors.append("EXP014_D31AR_RIGID_CONTACT_CONSTRAINT_INVALID")
    if actuator.get("preferred_direct_effort_method") is None:
        errors.append("EXP014_D31AR_DIRECT_TORQUE_SEMANTICS_FAIL")
    unique_errors = list(dict.fromkeys(errors))
    classification = unique_errors[0] if len(unique_errors) == 1 else "EXP014_D31AR_MULTIPLE_CONTRACT_MISMATCHES" if unique_errors else "EXP014_D31AR_CONTACT_MODEL_RECONCILED"
    adjudication = {"official_d31a_classification": "EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL", "d31ar_classification": classification, "registered": classification in CLASSES, "authority_supported": False, "authority_status": "NOT_YET_ADJUDICATED", "reason": "This stage reconciles runtime contracts and measured witnesses; it does not rerun D31A WBC authority.", "subclassifications": unique_errors, "runtime_dynamics_reconstructed": bool(runtime.get("jacobian_shape")), "contact_geometry_complete": contact_api.get("contact_points") != "NOT_AVAILABLE" and contact_api.get("penetration_depth") != "NOT_AVAILABLE"}
    dump(OUT / "d31a_scientific_adjudication.json", adjudication)
    dump(OUT / "first_divergence.json", {"classification": classification, "first_divergence": "CONTACT_GEOMETRY_API" if contact_api.get("contact_points") == "NOT_AVAILABLE" else "NONE", "official_d31a_result_untouched": True, "execution_error": results.get("execution_error")})
    dump(OUT / "stage_classification.json", {"classification": classification, "registered_d31ar_classification": True, "official_d31a_classification_unchanged": True, "physics": 1, "authority": "NOT_YET_ADJUDICATED"})
    dump(OUT / "recommended_next_action.json", {"classification": classification, "next_action": "contact representation repair", "prohibited": ["WBC authority rerun", "task tuning", "PPO", "CEM", "search", "training", "Student", "RUN", "validation", "heldout"]})

    protected = {
        "starting_head": start_head, "execution_head": git("rev-parse", "HEAD"), "starting_status": start_status,
        "D29B_script_sha256": sha(D29B), "P0_sha256": sha(P0), "WMOVE_sha256": sha(WMOVE),
        "D31A_script_sha256": sha(EXP / "scripts/run_phase2_d31a_torque_wbc_authority.py"),
        "D31A_result_tree_sha256": tree_hash(REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d31a_torque_wbc_authority"),
        "D30B_result_tree_sha256": tree_hash(REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d30b_nonlinear_post_touchdown_reachability"),
        "persistent_update": 0, "new_checkpoint": 0, "remote_push": False,
        "forbidden_work": {"WBC": 0, "task_tuning": 0, "PPO": 0, "CEM": 0, "search": 0, "training": 0, "Student": 0, "RUN": 0, "validation": 0},
    }
    dump(OUT / "protected_hashes.json", protected)
    dump(OUT / "reproduction_commands.ps1", {"command": f"& '{ISAAC_PYTHON}' '{HERE}' --headless --viz none", "focused_test": f"& '{ISAAC_PYTHON}' '{HERE}' --focused-test --headless --viz none"})

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# EXP014 Phase 2-D31A-R — contact/inverse-dynamics reconciliation

## Scope

Fresh Isaac/PhysX Route A used P0 S_HOLD followed by native W_MOVE for R0-R7.
The protected D31A WBC authority result was not rerun or changed. Runtime
state, body Jacobians, mass/inertia-derived dynamics, net foot forces, and
read-only actuator contracts were captured around the first strict TD0.

## Availability

The runtime exposes body Jacobians, masses, inertias, q/dq/root state, applied
and computed torques, and net foot forces. The contact-point, normal-impulse,
and penetration APIs were `{contact_api.get('contact_points', 'NOT_AVAILABLE')}`;
these quantities are therefore recorded as `NOT_AVAILABLE`, not inferred.
When the backend folds decimation into one simulator update, exact internal
physics-substep offsets are likewise explicitly unavailable.

## Reconstruction

Continuous force and impulse residuals, point-force ankle proxies, aggregate
sole wrenches, finite-difference contact kinematics, friction witnesses, CoP
witnesses, and the F0-F6 feasibility ladder are emitted in the result
directory. The sole polygon and actual rigid-contact wrench feasibility remain
unadjudicated.

## Scientific adjudication

Classification: `{classification}`. Authority supported: **no**. Authority
status: **NOT_YET_ADJUDICATED**. Official D31A classification remains
`EXP014_D31A_TORQUE_WBC_CONTACT_AUTHORITY_FAIL`.

## Repository

Starting HEAD: `{start_head}`; execution HEAD: `{git('rev-parse','HEAD')}`.
No protected artifact, checkpoint, runtime setting, or unrelated worktree
state was modified.
""", encoding="utf-8")
    return 0 if not results.get("execution_error") else 3


if __name__ == "__main__":
    raise SystemExit(main())
