"""Fresh frozen-policy path and random-segment endpoint evaluation for W1B-C2."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c2_shared_yaw_endpoint_evaluator"
)
CHECKPOINT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa: E402
from g1_omnidirectional.yaw_endpoint_evaluator import Exp013YawEndpointEvaluator  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random-only", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

EPISODES = 100
PATHS = [
    "CIRCLE_POS", "CIRCLE_NEG", "S_CURVE", "STRAFE_LEFT_NEG",
    "STRAFE_RIGHT_POS", "BACKWARD_POS", "BACKWARD_NEG",
]
COUNT = len(PATHS) * EPISODES + 50
STEPS = 3000
DT = .02


def minjerk(value):
    value = max(0., min(1., value))
    return value**3 * (10 - 15 * value + 6 * value**2)


def random_specs(episode):
    generator = torch.Generator().manual_seed(20282021 + episode)
    durations = (3 + 2 * torch.rand(20, generator=generator)).numpy()
    angles = (torch.rand(20, generator=generator) * 2 * math.pi).numpy()
    speeds = (torch.rand(20, generator=generator) * .4).numpy()
    yaws = (torch.rand(20, generator=generator) * .8 - .4).numpy()
    ends = np.cumsum(durations)
    return durations, ends, angles, speeds, yaws


def path_command(name, time_s):
    if name == "CIRCLE_POS":
        return .4, 0., .3
    if name == "CIRCLE_NEG":
        return .4, 0., -.3
    if name == "STRAFE_LEFT_NEG":
        return 0., .3, -.3
    if name == "STRAFE_RIGHT_POS":
        return 0., -.3, .3
    if name == "BACKWARD_POS":
        return -.3, 0., .3
    if name == "BACKWARD_NEG":
        return -.3, 0., -.3
    targets = (.3, -.3, .3)
    segment = min(int(time_s // 6), 2)
    local = time_s - segment * 6
    previous = targets[max(segment - 1, 0)]
    target = previous + (targets[segment] - previous) * minjerk(local / 2)
    return .4, 0., target


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 64 if args.random_only else 1024
    cfg.episode_length_s = 65.
    cfg.seed = acfg.seed = 20285121
    if args.device:
        cfg.sim.device = acfg.device = args.device
    random = [random_specs(i) for i in range(50)]
    evaluator = Exp013YawEndpointEvaluator()
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=acfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        actor = FrozenGaitActor(CHECKPOINT).to(device).eval()
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        count = 50 if args.random_only else COUNT
        traces = {key: np.zeros((count, STEPS), dtype=np.float32) for key in (
            "target", "vx_target", "vy_target", "yaw", "vx", "vy", "vector_error",
            "direction_error", "speed", "flight",
        )}
        safety = {key: np.zeros((count, STEPS), dtype=bool) for key in (
            "fall", "dangerous_slip", "impact", "saturation",
        )}
        slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        sat_streak = torch.zeros_like(slip_streak)
        fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        wrapped.seed(20285121)
        obs, _ = wrapped.reset()
        obs = obs["policy"].to(device)
        for step in range(STEPS):
            time_s = step * env.step_dt
            vx_target = torch.zeros(env.num_envs, device=device)
            vy_target = torch.zeros_like(vx_target)
            yaw_target = torch.zeros_like(vx_target)
            for env_id in range(count):
                if not args.random_only and env_id < 700:
                    name = PATHS[env_id // EPISODES]
                    vx, vy, yaw = path_command(name, time_s)
                else:
                    episode = env_id if args.random_only else env_id - 700
                    durations, ends, angles, speeds, yaws = random[episode]
                    segment = min(int(np.searchsorted(ends, time_s)), 19)
                    vx = speeds[segment] * math.cos(angles[segment])
                    vy = speeds[segment] * math.sin(angles[segment])
                    yaw = yaws[segment]
                vx_target[env_id], vy_target[env_id], yaw_target[env_id] = (
                    float(vx), float(vy), float(yaw)
                )
            command.external_override[:, 0] = vx_target
            command.external_override[:, 1] = vy_target
            command.external_override[:, 2] = calibrate_yaw(yaw_target)
            if step == 0:
                command._update_command()
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
            limits = robot.data.joint_vel_limits
            limits = limits[..., 1].abs() if limits.ndim == 3 else limits
            ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
            sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak))
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
            fallen |= done.bool() & ~timeout
            vector_error = torch.linalg.vector_norm(
                actual_v - torch.stack((vx_target, vy_target), -1), dim=-1)
            target_angle = torch.atan2(vy_target, vx_target)
            actual_angle = torch.atan2(actual_v[:, 1], actual_v[:, 0])
            direction_error = torch.atan2(
                torch.sin(actual_angle - target_angle), torch.cos(actual_angle - target_angle)
            ).abs() * 180 / math.pi
            for name, tensor in (
                ("target", yaw_target), ("vx_target", vx_target), ("vy_target", vy_target),
                ("yaw", actual_yaw), ("vx", actual_v[:, 0]), ("vy", actual_v[:, 1]),
                ("vector_error", vector_error), ("direction_error", direction_error),
                ("speed", torch.linalg.vector_norm(actual_v, dim=-1)),
                ("flight", (~contact.any(-1)).float()),
            ):
                traces[name][:, step] = tensor[:count].detach().cpu().numpy()
            safety["fall"][:, step] = fallen[:count].cpu().numpy()
            safety["dangerous_slip"][:, step] = (slip_streak[:count] >= 5).cpu().numpy()
            safety["impact"][:, step] = (force[:count].amax(-1) > 3500).cpu().numpy()
            safety["saturation"][:, step] = (sat_streak[:count] >= 5).cpu().numpy()

        def segment_result(
            env_id, condition, segment, start_s, end_s, formal, acquisition_start_s=None
        ):
            start, end = round(start_s / DT), round(end_s / DT)
            yaw_target = float(np.mean(traces["target"][env_id, start:end]))
            vx_target = float(np.mean(traces["vx_target"][env_id, start:end]))
            vy_target = float(np.mean(traces["vy_target"][env_id, start:end]))
            speed = math.hypot(vx_target, vy_target)
            replay = evaluator.replay_summary(
                yaw_target=yaw_target,
                mean_yaw=float(np.mean(traces["yaw"][env_id, start:end])),
                yaw_mae=float(np.mean(np.abs(traces["yaw"][env_id, start:end] - yaw_target))),
                condition_type="pure" if speed <= .05 else "moving",
                vector_mae=float(np.mean(traces["vector_error"][env_id, start:end])),
                direction_error_deg=float(np.mean(traces["direction_error"][env_id, start:end])),
                translation_drift=float(np.mean(traces["speed"][env_id, start:end])),
                gait_success=speed <= .05 or float(np.mean(traces["flight"][env_id, start:end])) < .1,
                fall=bool(np.any(safety["fall"][env_id, start:end])),
                dangerous_slip=bool(np.any(safety["dangerous_slip"][env_id, start:end])),
                impact=bool(np.any(safety["impact"][env_id, start:end])),
                long_dwell_saturation=bool(np.any(safety["saturation"][env_id, start:end])),
            )
            acquisition_start_s = start_s if acquisition_start_s is None else acquisition_start_s
            acquisition_start = round(acquisition_start_s / DT)
            sign_mask = traces["yaw"][env_id, acquisition_start:end] * yaw_target > 0
            sign_hits = np.flatnonzero(sign_mask)
            sustained_hits = np.flatnonzero(
                np.convolve(sign_mask.astype(np.int32), np.ones(10, dtype=np.int32), mode="valid") == 10
            ) if sign_mask.size >= 10 else np.array([], dtype=int)
            return {
                "condition": condition, "episode": env_id % 100, "segment": segment,
                "window_start_s": start_s, "window_end_s": end_s,
                "yaw_target": yaw_target, "yaw_actor_input": float(calibrate_yaw(yaw_target)),
                "first_sign_acquisition_s": (
                    None if not sign_hits.size else float(sign_hits[0] * DT)
                ),
                "first_0p20_sustained_acquisition_s": (
                    None if not sustained_hits.size else float(sustained_hits[0] * DT)
                ),
                "formal_path_segment": formal, **replay.to_dict(),
            }

        path_rows = []
        for env_id in range(0 if args.random_only else 700):
            name = PATHS[env_id // EPISODES]
            if name == "S_CURVE":
                for segment in range(3):
                    path_rows.append(segment_result(
                        env_id, name, segment, segment * 6 + 2, segment * 6 + 6, True,
                        acquisition_start_s=segment * 6))
            else:
                duration = 16 if name.startswith("CIRCLE") else 12
                path_rows.append(segment_result(env_id, name, 0, 0, duration, True))
        random_rows = []
        for episode in range(50):
            env_id = episode if args.random_only else 700 + episode
            durations, ends, *_ = random[episode]
            starts = np.r_[0., ends[:-1]]
            for segment, (start, end) in enumerate(zip(starts, ends)):
                if end > 60:
                    end = 60.
                if end - start < 1:
                    evaluable = False
                    row = {"condition": "RANDOM_60S", "episode": episode,
                           "segment": segment, "evaluable": False}
                else:
                    evaluable = True
                    row = segment_result(
                        env_id, "RANDOM_60S", segment, end - 1, end, False,
                        acquisition_start_s=float(start),
                    )
                    row["evaluable"] = True
                    row["segment_start_s"] = float(start)
                    row["segment_end_s"] = float(end)
                random_rows.append(row)
                if end >= 60:
                    break
        if not args.random_only:
            write_csv(OUT / "dynamic_path_evaluation.csv", path_rows)
        path_groups = defaultdict(list)
        for row in path_rows:
            path_groups[row["condition"]].append(row)
        path_summary = [{
            "condition": name, "segments": len(rows),
            "endpoint_segment_success_rate": np.mean([r["endpoint_success"] for r in rows]),
            "fall_rate": np.mean([r["fall"] for r in rows]),
            "dangerous_slip_rate": np.mean([r["dangerous_slip"] for r in rows]),
            "gate_pass": np.mean([r["endpoint_success"] for r in rows]) >= .9,
        } for name, rows in sorted(path_groups.items())]
        if not args.random_only:
            (OUT / "dynamic_path_evaluation.json").write_text(json.dumps({
                "rows": path_summary, "segment_rows": path_rows,
                "shared_endpoint_evaluator": True, "acquisition_formal_gate_member": False,
            }, indent=2, sort_keys=True, default=lambda value: value.item()) + "\n", encoding="utf-8")
        write_csv(OUT / "random_command_segment_evaluation.csv", random_rows)
        evaluable = [r for r in random_rows if r.get("evaluable")]
        episodes = defaultdict(list)
        for row in evaluable:
            episodes[row["episode"]].append(row)
        random_summary = {
            "episodes": 50, "segment_count": len(random_rows),
            "evaluable_segment_count": len(evaluable),
            "successful_endpoint_segment_count": sum(r["endpoint_success"] for r in evaluable),
            "endpoint_segment_success_rate": np.mean([r["endpoint_success"] for r in evaluable]),
            "transition_acquisition_success_rate": np.mean([
                r["first_sign_acquisition_s"] is not None for r in evaluable]),
            "full_episode_all_segment_success_rate": np.mean([
                all(r["endpoint_success"] for r in rows) for rows in episodes.values()]),
            "fall_free_episode_rate": np.mean([
                not any(r["fall"] for r in rows) for rows in episodes.values()]),
            "formal_gate_member": False,
        }
        (OUT / "random_command_segment_evaluation.json").write_text(json.dumps({
            "summary": random_summary, "segment_rows": random_rows,
        }, indent=2, sort_keys=True, default=lambda value: value.item()) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
