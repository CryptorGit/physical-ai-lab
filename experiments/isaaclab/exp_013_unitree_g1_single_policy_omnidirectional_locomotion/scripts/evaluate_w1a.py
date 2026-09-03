"""Deterministic Phase W1A checkpoint evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_single_policy.phase_gated_heading import yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--suite", choices=("timeline", "selection", "formal", "envelope", "continuous", "run"), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", default="")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vec(speed, degrees):
    radians = math.radians(degrees)
    return speed * math.cos(radians), speed * math.sin(radians)


def static(name, speed, degrees, episodes, gait=0, duration=8.0):
    vx, vy = vec(speed, degrees)
    return {"name": name, "speed": speed, "direction_deg": degrees, "vx": vx, "vy": vy,
            "gait": gait, "episodes": episodes, "duration": duration, "kind": "static"}


def conditions():
    if args.suite == "timeline":
        values = ((.6, 0), (1.2, 0), (.6, 90), (.6, 270), (.3, 180), (.6, 180),
                  (.6, 45), (.6, 315), (.6, 135), (.6, 225))
        return [static(f"S{s:.1f}_D{d:05.1f}", s, d, 20) for s, d in values]
    if args.suite == "selection":
        return [static(f"S{s:.1f}_D{d:05.1f}", s, d, 20)
                for s in (.3, .6) for d in (x * 22.5 for x in range(16))]
    if args.suite == "formal":
        return [static(f"S{s:.1f}_D{d:05.1f}", s, d, 50)
                for s in (.3, .6) for d in (x * 22.5 for x in range(16))]
    if args.suite == "envelope":
        rows = []
        for d in (x * 22.5 for x in range(16)):
            a = abs(((d + 180) % 360) - 180)
            speeds = (.9, 1.2) if a <= 45 else ((.8,) if a <= 90 else (.6,))
            rows.extend(static(f"S{s:.1f}_D{d:05.1f}", s, d, 50) for s in speeds)
        return rows
    if args.suite == "continuous":
        return [{"name": "CONTINUOUS_DIRECTION_30S", "episodes": 30, "duration": 30.,
                 "kind": "continuous", "gait": 0}]
    return [
        static("RUN_1P2", 1.2, 0, 20, gait=1),
        static("RUN_2P4", 2.4, 0, 20, gait=1),
        {"name": "WALK_TO_RUN", "episodes": 20, "duration": 12., "kind": "walk_to_run"},
        {"name": "RUN_TO_WALK", "episodes": 20, "duration": 12., "kind": "run_to_walk"},
    ]


def command(condition, time_s, episode):
    kind = condition["kind"]
    if kind == "static":
        return condition["vx"], condition["vy"], 0., float(condition["gait"])
    if kind == "continuous":
        segment = min(int(time_s // 4), 7)
        generator = torch.Generator().manual_seed(20271021 + episode)
        angles = torch.rand(8, generator=generator) * 2 * math.pi
        speeds = .2 + torch.rand(8, generator=generator) * .4
        return float(speeds[segment] * torch.cos(angles[segment])), float(
            speeds[segment] * torch.sin(angles[segment])), 0., 0.
    target_run = kind == "walk_to_run"
    gait = (0. if time_s < 6 else 1.) if target_run else (1. if time_s < 6 else 0.)
    return 1.2, 0., 0., gait


def wrap(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def main():
    spec = conditions()
    env_condition, env_episode = [], []
    for index, item in enumerate(spec):
        env_condition += [index] * item["episodes"]
        env_episode += list(range(item["episodes"]))
    count = len(env_condition)
    duration = max(item["duration"] for item in spec)
    cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = count
    cfg.episode_length_s = duration + 1
    cfg.seed = 20271021
    agent_cfg.seed = cfg.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    checkpoint = Path(args.checkpoint).resolve()
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        actor = FrozenGaitActor(checkpoint).to(device).eval()
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        sensor_feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in sensor_feet]
        ids = torch.tensor(env_condition, device=device)
        episodes = torch.tensor(env_episode, device=device)
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        initial_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        active = torch.ones(count, dtype=torch.bool, device=device)
        fall = torch.zeros_like(active)
        tilt_bad = torch.zeros_like(active)
        slip_bad = torch.zeros_like(active)
        impact = torch.zeros_like(active)
        saturation = torch.zeros_like(active)
        slip_streak = torch.zeros(count, dtype=torch.long, device=device)
        sat_streak = torch.zeros_like(slip_streak)
        steps = torch.zeros(count, device=device)
        headings = [[] for _ in range(count)]
        sums = {key: torch.zeros(count, device=device) for key in (
            "vx", "vy", "yaw", "vector_error", "direction_error", "flight", "slip",
            "roll", "pitch", "height", "vertical", "impact_force", "joint_proximity",
            "action_saturation", "left_contact", "right_contact")}
        for step in range(round(duration / float(env.step_dt))):
            time_s = step * float(env.step_dt)
            vx, vy, gait = (torch.zeros(count, device=device) for _ in range(3))
            valid = torch.zeros(count, dtype=torch.bool, device=device)
            for index, item in enumerate(spec):
                mask = torch.where(ids == index)[0]
                if time_s >= item["duration"]:
                    continue
                valid[mask] = True
                for env_id in mask.cpu().tolist():
                    x, y, _, g = command(item, time_s, int(episodes[env_id]))
                    vx[env_id], vy[env_id], gait[env_id] = x, y, g
            term.external_override[:, 0] = vx
            term.external_override[:, 1] = vy
            term.external_override[:, 2] = 0
            if step == 0:
                term._update_command()
                obs = wrapped.get_observations().to(device)
            with torch.inference_mode():
                actions = actor(obs["policy"], gait)
                actions[~active | ~valid] = 0
            obs, _, dones, extras = wrapped.step(actions)
            obs = obs.to(device)
            measure = active & valid
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fall |= dones.bool() & ~timeout & measure
            actual = robot.data.root_lin_vel_b
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            forces = sensor.data.net_forces_w_history[:, -1, sensor_feet, :].norm(dim=-1)
            contacts = forces > 5
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            slip_bad |= (slip_streak >= 5) & measure
            impact |= (forces.amax(-1) > 3500) & measure
            gravity = robot.data.projected_gravity_b
            roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            tilt = torch.maximum(roll, pitch)
            tilt_bad |= (tilt > .8) & measure
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            proximity = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
            saturated = proximity > .95
            sat_streak = torch.where(saturated, sat_streak + 1, torch.zeros_like(sat_streak))
            saturation |= (sat_streak >= 5) & measure
            error = torch.linalg.vector_norm(actual[:, :2] - torch.stack((vx, vy), 1), dim=-1)
            cmd_angle, actual_angle = torch.atan2(vy, vx), torch.atan2(actual[:, 1], actual[:, 0])
            direction_error = wrap(actual_angle - cmd_angle).abs() * 180 / math.pi
            values = {
                "vx": actual[:, 0], "vy": actual[:, 1], "yaw": actual_yaw,
                "vector_error": error, "direction_error": direction_error,
                "flight": (contacts.sum(-1) == 0).float(), "slip": slipping.float(),
                "roll": roll, "pitch": pitch, "height": robot.data.root_pos_w[:, 2],
                "vertical": actual[:, 2].abs(), "impact_force": forces.amax(-1),
                "joint_proximity": proximity, "action_saturation": saturated.float(),
                "left_contact": contacts[:, 0].float(), "right_contact": contacts[:, 1].float()}
            for key, value in values.items():
                sums[key] += torch.where(measure, value, 0)
            current_heading = wrap(yaw_from_quat_wxyz(robot.data.root_quat_w) - initial_yaw).abs()
            for env_id in torch.where(measure)[0].cpu().tolist():
                headings[env_id].append(float(current_heading[env_id]))
            steps += measure.float()
            active &= ~(fall | (tilt > .8) | impact | saturation)
        rows = []
        for env_id in range(count):
            n = max(float(steps[env_id]), 1)
            mean = {key: float(value[env_id] / n) for key, value in sums.items()}
            item = spec[env_condition[env_id]]
            speed = math.hypot(mean["vx"], mean["vy"])
            heading_p95 = float(torch.tensor(headings[env_id]).quantile(.95)) if headings[env_id] else math.inf
            gait_class = "FALL" if fall[env_id] else ("WALK_LIKE" if mean["flight"] < .10 else
                ("ISOLATED_FLIGHT" if mean["flight"] < .20 else "PERIODIC_RUNNING"))
            success = (not bool(fall[env_id]) and gait_class == "WALK_LIKE"
                and mean["vector_error"] <= .20 and mean["direction_error"] <= 20
                and abs(mean["yaw"]) <= .20 and heading_p95 <= .25
                and not bool(slip_bad[env_id]) and not bool(impact[env_id]) and not bool(saturation[env_id]))
            rows.append({
                "condition": item["name"], "episode": env_episode[env_id],
                "direction_deg": item.get("direction_deg"), "commanded_speed_mps": item.get("speed"),
                "actual_vx_body": mean["vx"], "actual_vy_body": mean["vy"], "actual_speed_mps": speed,
                "vector_velocity_mae": mean["vector_error"], "direction_error_deg": mean["direction_error"],
                "actual_yaw_rate_abs_mean": abs(mean["yaw"]), "heading_drift_p95_rad": heading_p95,
                "gait_classification": gait_class, "success": success, "fall": bool(fall[env_id]),
                "excessive_tilt": bool(tilt_bad[env_id]), "dangerous_slip": bool(slip_bad[env_id]),
                "impact_failure": bool(impact[env_id]), "long_dwell_saturation": bool(saturation[env_id]),
                "foot_slip_fraction": mean["slip"], "base_roll_abs_mean": mean["roll"],
                "base_pitch_abs_mean": mean["pitch"], "base_height_mean": mean["height"],
                "vertical_velocity_abs_mean": mean["vertical"], "max_impact_force_mean_n": mean["impact_force"],
                "joint_limit_proximity": mean["joint_proximity"],
                "action_saturation_fraction": mean["action_saturation"],
                "left_contact_fraction": mean["left_contact"], "right_contact_fraction": mean["right_contact"]})
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["condition"]].append(row)
        summary = []
        for name, values in grouped.items():
            base = {key: values[0][key] for key in ("direction_deg", "commanded_speed_mps")}
            base.update({"condition": name, "episodes": len(values)})
            for key in ("actual_vx_body", "actual_vy_body", "actual_speed_mps", "vector_velocity_mae",
                        "direction_error_deg", "actual_yaw_rate_abs_mean", "heading_drift_p95_rad",
                        "foot_slip_fraction", "base_roll_abs_mean", "base_pitch_abs_mean",
                        "base_height_mean", "vertical_velocity_abs_mean", "joint_limit_proximity",
                        "action_saturation_fraction", "left_contact_fraction", "right_contact_fraction"):
                base[key] = sum(row[key] for row in values) / len(values)
            for key in ("success", "fall", "excessive_tilt", "dangerous_slip", "impact_failure",
                        "long_dwell_saturation"):
                base[f"{key}_rate"] = sum(row[key] for row in values) / len(values)
            base["gate_pass"] = base["success_rate"] >= .90 and base["fall_rate"] <= .05
            summary.append(base)
        stem = f"{args.suite}_{args.tag or checkpoint.stem}"
        payload = {"suite": args.suite, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                   "seed": 20271021, "deterministic": True, "training_updates": 0,
                   "rows": summary, "episode_rows": rows}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"_raw_{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (OUT / f"_raw_{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        wrapped.close()


if __name__ == "__main__":
    main()
