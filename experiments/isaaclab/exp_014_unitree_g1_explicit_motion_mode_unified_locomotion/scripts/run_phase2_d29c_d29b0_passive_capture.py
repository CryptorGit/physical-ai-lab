"""D29C passive telemetry completion for the already executed D29B0 routes.

This runner is intentionally limited to the fixed D29B0 contracts.  It adds
detached in-memory telemetry capture to the same runtime path and serializes
only after the process finishes.  It does not alter the protected D29B0
artifacts, controller, checkpoint, command, seed, reset lifecycle, or switch
timing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29c_true_wmove_basin_adjudication"
RAW = OUT / "raw"
D29B0_SCRIPT = EXP / "scripts/run_phase2_d29b0_input_gate_ablation.py"
D29B0_OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d29b0_input_gate_ablation"
EXPECTED_D29B0_HEAD = "600298f1d21acaf7389efd96ede081faa9bd90b9"
SEED = 20279941
DT = 0.02
RECIPES = list(range(8))
PRECONDITION_STEPS = 100
WMOVE_STEPS = 150
ACTOR = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/dagger_checkpoints/round_2_step_10000.pt"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"IMPORT_FAILED:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def tensor_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def root_pose(world, torch) -> np.ndarray:
    robot = world.robot
    return tensor_np(robot.data.root_state_w[:, :7])


def state_identity(snap: dict[str, Any]) -> str:
    payload = np.concatenate(
        (snap["root_pos"].ravel(), snap["root_v"].ravel(), snap["joint_q"].ravel(), snap["joint_dq"].ravel())
    ).astype(np.float64).tobytes()
    return hashlib.sha256(payload).hexdigest()


def optional_contact_fields(world, sensor_feet, torch) -> dict[str, np.ndarray]:
    sensor = world.sensor
    out: dict[str, np.ndarray] = {}
    history = getattr(sensor.data, "net_forces_w_history", None)
    if history is not None:
        out["contact_force_history"] = tensor_np(history[:, :, sensor_feet, :])
    for name, key in (("current_air_time", "air_time"), ("last_air_time", "last_air_time"), ("last_contact_time", "last_contact_time")):
        value = getattr(sensor.data, name, None)
        if value is not None:
            if hasattr(value, "__getitem__"):
                out[key] = tensor_np(value[:, sensor_feet])
    return out


def snapshot(world, sensor_feet, robot_feet, torch, d29b0) -> dict[str, Any]:
    """Use the protected D29B0 snapshot path plus identity-complete fields."""
    snap = d29b0.runtime_snapshot(world, sensor_feet, robot_feet, torch)
    snap["root_pose"] = root_pose(world, torch)
    snap.update(optional_contact_fields(world, sensor_feet, torch))
    return snap


def mode_arrays(world, torch) -> dict[str, np.ndarray]:
    state = world.state
    return {
        "target_mode": tensor_np(state.target_mode),
        "previous_target_mode": tensor_np(state.previous_target_mode),
        "previous_physical_command": tensor_np(state.previous_physical_command),
        "time_since_mode_change": tensor_np(state.time_since_mode_change_s),
        "ramp_progress": tensor_np(state.ramp_progress),
        "physical_command": tensor_np(state.physical_command),
    }


def append(store: dict[str, list[np.ndarray]], key: str, value: np.ndarray) -> None:
    store.setdefault(key, []).append(np.asarray(value).copy())


def finish_store(store: dict[str, list[np.ndarray]], source_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for key, values in store.items():
        if not values:
            continue
        result[key] = np.concatenate(values, axis=0)
    # Explicit source labels avoid relying on file row ordering during later
    # return-map analysis.
    if "control_step" in result:
        steps = result["control_step"]
        result["source_environment_index"] = np.tile(np.arange(source_count, dtype=np.int32), len(steps) // source_count)
    return result


def capture_step(store: dict[str, list[np.ndarray]], *, source_count: int, step: int, phase_code: int, snap: dict[str, Any], action: np.ndarray, previous_action: np.ndarray, q_cmd: np.ndarray, command: np.ndarray, mode: dict[str, np.ndarray], world, sensor_feet, torch) -> None:
    n = source_count
    root_state = snap["root_pose"]
    fields: dict[str, np.ndarray] = {
        "control_step": np.full((n,), step, dtype=np.int32),
        "phase_code": np.full((n,), phase_code, dtype=np.int8),
        "root_pose": root_state,
        "root_position": snap["root_pos"],
        "root_yaw": snap["root_yaw"],
        "root_velocity": np.concatenate((snap["root_v"], snap["root_w"]), axis=1),
        "projected_gravity": snap["gravity"],
        "joint_position": snap["joint_q"],
        "joint_velocity": snap["joint_dq"],
        "previous_action": previous_action,
        "action": action,
        "q_cmd": q_cmd,
        "foot_pose": snap["foot_pos"],
        "foot_velocity": snap["foot_vel"],
        "contact_force": np.asarray(snap["force"])[..., None] * 0.0,
        "contact_force_norm": snap["force"],
        "contact": snap["contact"],
        "com_position": snap["com"],
        "com_velocity": snap["com_v"],
        "dcm": snap["dcm"],
        "applied_torque": snap["torque"],
        "computed_torque": snap["torque"],
        "effort_limit": snap["effort"],
        "velocity_limit": snap["vlim"],
        "command": command,
        "support_xy": snap["support"],
        "target_mode": mode["target_mode"],
        "previous_target_mode": mode["previous_target_mode"],
        "previous_physical_command": mode["previous_physical_command"],
        "time_since_mode_change": mode["time_since_mode_change"],
        "ramp_progress": mode["ramp_progress"],
        "physical_command_state": mode["physical_command"],
    }
    # Preserve actual vector forces when available; runtime_snapshot exposes
    # the norm for D29B0's original evaluator.
    force_history = snap.get("contact_force_history")
    if force_history is not None:
        fields["contact_force_history"] = force_history
        fields["contact_force"] = force_history[:, -1]
    for key in ("air_time", "last_air_time", "last_contact_time"):
        if key in snap:
            fields[key] = snap[key]
    for key, value in fields.items():
        append(store, key, value)


def run_condition(args, d29b0, runtime, condition: str) -> None:
    torch = runtime["torch"]
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    actor, payload = d29b0.load_actor(ACTOR, runtime, device)
    wmove = runtime["FrozenGaitActor"](WMOVE).to(device).eval()

    parser = argparse.ArgumentParser()
    runtime["add_launcher_args"](parser)
    launcher_argv = [sys.argv[0]]
    if args.headless:
        launcher_argv.append("--headless")
    if args.device:
        launcher_argv.extend(["--device", args.device])
    saved_argv = sys.argv
    sys.argv = launcher_argv
    launch_args, hydra_args = runtime["setup_preset_cli"](parser)
    sys.argv = saved_argv
    sys.argv = [sys.argv[0], *hydra_args]
    cfg, agent = runtime["resolve_task_config"]("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = len(RECIPES)
    cfg.seed = SEED
    cfg.episode_length_s = 20.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    if args.device:
        cfg.sim.device = agent.device = args.device

    with runtime["launch_simulation"](cfg, launch_args):
        wrapped = runtime["RslRlVecEnvWrapper"](
            runtime["gym"].make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        d3 = runtime["d3"]
        world = d3.StandWorld(wrapped, d3.load_resets(), torch.zeros(680, device=wrapped.unwrapped.device))
        sensor_feet, robot_feet, sensor_names, robot_names = d29b0.ordered_feet(world.sensor, world.robot)

        # This is the same pre-route shadow reset used by D29B0.  It performs
        # no physics step and is retained to preserve the original lifecycle.
        d29b0.mode_shadow(world, actor, runtime)

        world.env.reset()
        world.term.external_override.zero_()
        world.term._update_command()
        world.state = runtime["ExplicitMotionModeCommand"].zeros(len(RECIPES), device=world.device)
        start = snapshot(world, sensor_feet, robot_feet, torch, d29b0)
        source_hash = state_identity(start)
        mode_value = runtime["MotionMode"].WALK if condition == "P_WALK_ZERO" else runtime["MotionMode"].STAND
        world.state.request(torch.full((len(RECIPES),), int(mode_value), device=world.device, dtype=torch.long))

        store: dict[str, list[np.ndarray]] = {}
        previous_action = np.zeros((len(RECIPES), 37), dtype=np.float32)
        previous_q_cmd = np.asarray(world.robot.data.default_joint_pos.detach().cpu().numpy(), dtype=np.float32) + 0.5 * previous_action
        done_steps: list[np.ndarray] = []
        timeout_steps: list[np.ndarray] = []

        for step in range(PRECONDITION_STEPS):
            physical = torch.zeros(len(RECIPES), 3, device=world.device)
            action, _ = d29b0.explicit_action(world, actor, physical, step, runtime)
            action_np = action.detach().cpu().numpy().copy()
            q_cmd = world.robot.data.default_joint_pos.detach().cpu().numpy() + 0.5 * action_np
            _, _, done, extras = world.wrapped.step(action)
            snap = snapshot(world, sensor_feet, robot_feet, torch, d29b0)
            timeout = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            done_np = d29b0.numpy_bool_tensor(done)
            timeout_np = d29b0.numpy_bool_tensor(timeout)
            capture_step(store, source_count=len(RECIPES), step=step, phase_code=0, snap=snap, action=action_np, previous_action=previous_action, q_cmd=q_cmd, command=np.zeros((len(RECIPES), 3), dtype=np.float32), mode=mode_arrays(world, torch), world=world, sensor_feet=sensor_feet, torch=torch)
            done_steps.append(done_np); timeout_steps.append(timeout_np)
            previous_action = action_np
            previous_q_cmd = q_cmd

        for route_step in range(WMOVE_STEPS):
            command = np.zeros((len(RECIPES), 3), dtype=np.float32)
            command[:, 0] = 0.3
            command_t = torch.as_tensor(command, device=world.device)
            action, _ = d29b0.wmove_action(world, wmove, command_t, runtime)
            action_np = action.detach().cpu().numpy().copy()
            q_cmd = world.robot.data.default_joint_pos.detach().cpu().numpy() + 0.5 * action_np
            _, _, done, extras = world.wrapped.step(action)
            snap = snapshot(world, sensor_feet, robot_feet, torch, d29b0)
            timeout = extras.get("time_outs", torch.zeros_like(done)) if isinstance(extras, dict) else torch.zeros_like(done)
            done_np = d29b0.numpy_bool_tensor(done)
            timeout_np = d29b0.numpy_bool_tensor(timeout)
            capture_step(store, source_count=len(RECIPES), step=PRECONDITION_STEPS + route_step, phase_code=1, snap=snap, action=action_np, previous_action=previous_action, q_cmd=q_cmd, command=command, mode=mode_arrays(world, torch), world=world, sensor_feet=sensor_feet, torch=torch)
            done_steps.append(done_np); timeout_steps.append(timeout_np)
            previous_action = action_np
            previous_q_cmd = q_cmd

        data = finish_store(store, len(RECIPES))
        data["done"] = np.concatenate(done_steps, axis=0)
        data["timeout"] = np.concatenate(timeout_steps, axis=0)
        data["source_state_hash"] = np.asarray([source_hash] * (len(data["control_step"]) // len(RECIPES)))
        raw_path = RAW / f"passive_physics_{condition}.npz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **data)
        metadata = {
            "phase": "2-D29C passive telemetry completion",
            "condition": condition,
            "starting_head_actual": git_head(),
            "expected_d29b0_execution_head": EXPECTED_D29B0_HEAD,
            "seed": SEED,
            "recipes": RECIPES,
            "dt": DT,
            "precondition_steps": PRECONDITION_STEPS,
            "wmove_steps": WMOVE_STEPS,
            "actor": str(ACTOR.relative_to(REPO)).replace("\\", "/"),
            "actor_sha256": sha256_file(ACTOR),
            "wmove_actor": str(WMOVE.relative_to(REPO)).replace("\\", "/"),
            "wmove_actor_sha256": sha256_file(WMOVE),
            "fresh_lifecycle": True,
            "raw_snapshot_restore": False,
            "additional_physics_steps": 0,
            "additional_rng_calls": 0,
            "additional_sensor_refresh": 0,
            "capture_disk_writes_during_control": 0,
            "sensor_foot_names": sensor_names,
            "robot_foot_names": robot_names,
            "source_state_hash": source_hash,
            "raw_sha256": sha256_file(raw_path),
            "array_shapes": {key: list(value.shape) for key, value in data.items()},
            "capture_fields": sorted(data.keys()),
        }
        dump(RAW / f"passive_physics_{condition}.json", {"metadata": metadata, "source_state_hash": source_hash, "raw_path": str(raw_path.relative_to(REPO)).replace("\\", "/")})
        wrapped.close()
        print(json.dumps({"condition": condition, "source_state_hash": source_hash, "raw_path": str(raw_path.relative_to(REPO)).replace("\\", "/"), "rows": int(len(data["control_step"]))}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=("run",))
    parser.add_argument("--condition", choices=("P_STAND", "P_WALK_ZERO"), required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    d29b0 = load_module("exp014_d29c_d29b0_input_gate_source", D29B0_SCRIPT)
    runtime = d29b0.import_runtime()
    run_condition(args, d29b0, runtime, args.condition)


if __name__ == "__main__":
    main()
