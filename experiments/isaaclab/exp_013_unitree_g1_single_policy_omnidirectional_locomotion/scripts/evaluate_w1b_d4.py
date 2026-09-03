"""Fresh-process paired endpoint and negative-control rollouts for W1B-D4."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
EXP = HERE.parent.parent
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d4_dynamic_endpoint_window_parity_preflight"
)
CHECKPOINT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(ROOT / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("paired", "controls"), required=True)
parser.add_argument("--debug", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

DIRECTIONS = [None, 0, 45, 90, 135, 180, 225, 270, 315]
YAWS = [-0.3, 0.3]
EPISODES = 100
DT = 0.02


def minjerk(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value**3 * (10 - 15 * value + 6 * value**2)


def condition_name(direction, yaw) -> str:
    label = "PURE" if direction is None else f"D{direction:03d}"
    return f"{label}_Y{yaw:+.1f}"


def command(direction, final_yaw, time_s, mode, control_kind=None):
    speed = 0.0 if direction is None else 0.3
    angle = 0.0 if direction is None else math.radians(direction)
    vx, vy = speed * math.cos(angle), speed * math.sin(angle)
    if mode == "static":
        target = final_yaw
    else:
        initial = -final_yaw
        blend = minjerk((time_s - 4.0) / 2.0)
        target = initial + (final_yaw - initial) * blend
    if control_kind == "native_uncalibrated":
        actor = target
    elif control_kind == "wrong_sign":
        actor = -abs(target) if final_yaw > 0 else abs(target)
    else:
        actor = float(calibrate_yaw(target))
    return vx, vy, target, actor


def run_batch(wrapped, actor, batch, trajectory_mode, seed, control_kind=None):
    env, device = wrapped.unwrapped, wrapped.unwrapped.device
    robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
    command_term = env.command_manager.get_term("base_velocity")
    command_term.external_override_enabled = True
    feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
    episodes = 1 if args.debug else EPISODES
    count = len(batch) * episodes
    duration = 8.0 if trajectory_mode == "static" else 12.0
    steps = round(duration / env.step_dt)
    wrapped.seed(seed)
    obs, _ = wrapped.reset()
    obs = obs["policy"].to(device)
    condition_ids = np.repeat(np.arange(len(batch)), episodes)
    episode_ids = np.tile(np.arange(episodes), len(batch))
    traces = {
        key: np.zeros((count, steps), dtype=np.float32)
        for key in ("target", "actor_input", "actual_yaw", "actual_vx", "actual_vy", "vector_error")
    }
    traces["contact"] = np.zeros((count, steps), dtype=np.int8)
    traces["gait_cycle"] = np.zeros((count, steps), dtype=np.int16)
    traces["fall"] = np.zeros((count, steps), dtype=np.bool_)
    traces["slip"] = np.zeros((count, steps), dtype=np.bool_)
    final_obs = np.zeros((count, obs.shape[-1]), dtype=np.float32)
    final_action = np.zeros((count, 37), dtype=np.float32)
    previous_left = np.zeros(count, dtype=bool)
    gait_cycle = np.zeros(count, dtype=np.int16)
    slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    for step in range(steps):
        time_s = step * env.step_dt
        vx = torch.zeros(env.num_envs, device=device)
        vy = torch.zeros_like(vx)
        target = torch.zeros_like(vx)
        actor_yaw = torch.zeros_like(vx)
        for env_id in range(count):
            direction, final_yaw = batch[int(condition_ids[env_id])]
            values = command(direction, final_yaw, time_s, trajectory_mode, control_kind)
            vx[env_id], vy[env_id], target[env_id], actor_yaw[env_id] = values
        command_term.external_override[:, 0] = vx
        command_term.external_override[:, 1] = vy
        command_term.external_override[:, 2] = actor_yaw
        if step == 0:
            command_term._update_command()
            obs = wrapped.get_observations()["policy"].to(device)
        with torch.inference_mode():
            action = actor(obs, torch.zeros(env.num_envs, device=device))
        obs, _, done, extras = wrapped.step(action)
        obs = obs["policy"].to(device)
        actual_v = robot.data.root_lin_vel_b[:, :2]
        actual_yaw = robot.data.root_ang_vel_b[:, 2]
        force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
        contact = force > 5
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        sliding = ((foot_speed > .55) & contact).any(-1)
        slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
        timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
        fallen |= done.bool() & ~timeout
        vector_error = torch.linalg.vector_norm(actual_v - torch.stack((vx, vy), -1), dim=-1)
        contact_np = contact[:count].cpu().numpy()
        left_now = contact_np[:, 0]
        gait_cycle += (~previous_left & left_now).astype(np.int16)
        previous_left = left_now
        contact_code = np.where(
            contact_np.all(axis=1), 3,
            np.where(contact_np[:, 0], 1, np.where(contact_np[:, 1], 2, 0)),
        )
        for key, tensor in (
            ("target", target), ("actor_input", actor_yaw), ("actual_yaw", actual_yaw),
            ("actual_vx", actual_v[:, 0]), ("actual_vy", actual_v[:, 1]),
            ("vector_error", vector_error),
        ):
            traces[key][:, step] = tensor[:count].detach().cpu().numpy()
        traces["contact"][:, step] = contact_code
        traces["gait_cycle"][:, step] = gait_cycle
        traces["fall"][:, step] = fallen[:count].cpu().numpy()
        traces["slip"][:, step] = (slip_streak[:count] >= 5).cpu().numpy()
        if step == steps - 1:
            final_obs[:] = obs[:count].detach().cpu().numpy()
            final_action[:] = action[:count].detach().cpu().numpy()
    metadata = {
        "trajectory_mode": trajectory_mode,
        "control_kind": control_kind or "calibrated",
        "seed": seed,
        "step_dt": float(env.step_dt),
        "condition_ids": condition_ids,
        "episode_ids": episode_ids,
        "directions": np.array([-1 if item[0] is None else item[0] for item in batch], dtype=np.int16),
        "final_yaws": np.array([item[1] for item in batch], dtype=np.float32),
        "final_obs": final_obs,
        "final_action": final_action,
    }
    return {**traces, **metadata}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 13.0
    cfg.seed = acfg.seed = 20284021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    conditions = [(direction, yaw) for direction in DIRECTIONS for yaw in YAWS]
    batches = [conditions[:9], conditions[9:]]
    manifest = []
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=acfg.clip_actions,
        )
        actor = FrozenGaitActor(CHECKPOINT).to(wrapped.unwrapped.device).eval()
        if args.mode == "paired":
            for batch_id, batch in enumerate(batches):
                seed = 20284021 + batch_id
                for trajectory_mode in ("static", "dynamic"):
                    data = run_batch(wrapped, actor, batch, trajectory_mode, seed)
                    path = OUT / f"paired_trace_part{batch_id}_{trajectory_mode}.npz"
                    np.savez_compressed(path, **data)
                    manifest.append({
                        "file": path.name,
                        "batch": batch_id,
                        "trajectory_mode": trajectory_mode,
                        "conditions": [condition_name(*item) for item in batch],
                        "episodes_per_condition": 1 if args.debug else EPISODES,
                        "seed": seed,
                    })
        else:
            positive = [(direction, 0.3) for direction in DIRECTIONS]
            for control_id, control_kind in enumerate(("native_uncalibrated", "wrong_sign")):
                data = run_batch(wrapped, actor, positive, "static", 20284121, control_kind)
                path = OUT / f"negative_control_{control_kind}.npz"
                np.savez_compressed(path, **data)
                manifest.append({
                    "file": path.name,
                    "control_kind": control_kind,
                    "conditions": [condition_name(*item) for item in positive],
                    "episodes_per_condition": 1 if args.debug else EPISODES,
                    "seed": 20284121,
                })
        (OUT / f"{args.mode}_rollout_manifest.json").write_text(
            json.dumps({
                "mode": args.mode,
                "checkpoint_sha256": "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",
                "calibration": "MonotonicPositiveYawCalibrationV1",
                "training_updates": 0,
                "files": manifest,
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"mode": args.mode, "parts": len(manifest)}, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
