"""Batched fresh-process W1B-D2 rollout diagnostics; never updates parameters."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis"
)
R2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
)
SELECTED = R2 / "checkpoints/model_200.pt"
D1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d1_yaw_translation_interference_diagnosis"
)

sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("timeline", "surface", "prewarp", "unlock", "mirror", "local"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def spec(angle, speed, yaw, episodes, name=None, actor_input_yaw=None, target_yaw=None, **extra):
    rad = math.radians(angle)
    return {
        "name": name or f"D{angle:05.1f}_S{speed:.3f}_Y{yaw:+.2f}",
        "direction_deg": angle, "speed": speed, "yaw": yaw,
        "actor_input_yaw": yaw if actor_input_yaw is None else actor_input_yaw,
        "target_yaw": yaw if target_yaw is None else target_yaw,
        "vx": speed * math.cos(rad), "vy": speed * math.sin(rad),
        "episodes": episodes, **extra,
    }


def category(name):
    if "torso" in name or "waist" in name:
        return "waist"
    for token in ("hip", "knee", "ankle", "shoulder", "elbow"):
        if token in name:
            return token
    return "hand"


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary = []
    for name, values in grouped.items():
        base = {key: values[0].get(key) for key in (
            "checkpoint", "condition", "direction_deg", "commanded_speed",
            "target_yaw", "actor_input_yaw", "wrapper", "lambda", "branch_steps",
        )}
        base["episodes"] = len(values)
        for key in (
            "success", "yaw_sign_correct", "translation_correct", "yaw_correct",
            "fall", "dangerous_slip", "impact", "saturation", "excessive_tilt",
        ):
            base[key + "_rate"] = sum(bool(row[key]) for row in values) / len(values)
        for key in (
            "actual_vx", "actual_vy", "actual_speed", "actual_yaw", "actual_yaw_p95",
            "vector_mae", "direction_error", "yaw_mae", "slip_fraction", "tilt_mean",
            "left_contact_fraction", "right_contact_fraction", "flight_fraction",
            "double_support_fraction", "action_abs_p95", "action_abs_p99",
            "joint_limit_proximity", "joint_velocity_ratio", "torque_abs_mean",
            "contact_force_mean", "basin_yaw_after_branch",
        ):
            base[key] = sum(row[key] for row in values) / len(values)
        base["gate_pass"] = base["success_rate"] >= .9 and base["fall_rate"] <= .05
        summary.append(base)
    return summary


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 10
    cfg.seed = acfg.seed = 20276021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    mirror = json.loads((D1 / "robot_mirror_contract.json").read_text(encoding="utf-8"))
    mirror_index_cpu = torch.tensor(mirror["mirror_indices"], dtype=torch.long)
    mirror_sign_cpu = torch.tensor(mirror["mirror_signs"], dtype=torch.float32)
    manifest = json.loads((R2 / "checkpoint_manifest.json").read_text(encoding="utf-8"))["entries"]

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=acfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[i]) for i in feet]
        mirror_index = mirror_index_cpu.to(device)
        mirror_sign = mirror_sign_cpu.to(device)

        def mirror_action(action):
            return action[:, mirror_index] * mirror_sign

        def mirror_observation(observation):
            value = observation.clone()
            value[:, 1] *= -1
            value[:, 3] *= -1
            value[:, 5] *= -1
            value[:, 7] *= -1
            value[:, 10] *= -1
            value[:, 11] *= -1
            for start in (12, 49, 86):
                value[:, start:start + 37] = observation[:, start:start + 37][:, mirror_index] * mirror_sign
            return value

        def run(checkpoint, specs, seed, wrapper_mode="normal"):
            count = sum(item["episodes"] for item in specs)
            if count > env.num_envs:
                raise ValueError(count)
            actor = FrozenGaitActor(checkpoint).to(device).eval()
            wrapped.seed(seed)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            condition_ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            episode_ids = torch.zeros_like(condition_ids)
            valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            cursor = 0
            for index, item in enumerate(specs):
                amount = item["episodes"]
                condition_ids[cursor:cursor + amount] = index
                episode_ids[cursor:cursor + amount] = torch.arange(amount, device=device)
                valid[cursor:cursor + amount] = True
                cursor += amount
            vx = torch.tensor([item["vx"] for item in specs], device=device)[condition_ids]
            vy = torch.tensor([item["vy"] for item in specs], device=device)[condition_ids]
            target_yaw = torch.tensor([item["target_yaw"] for item in specs], device=device)[condition_ids]
            input_yaw = torch.tensor([item["actor_input_yaw"] for item in specs], device=device)[condition_ids]
            lambdas = torch.tensor([item.get("lambda", 0.0) for item in specs], device=device)[condition_ids]
            branch_steps = torch.tensor([item.get("branch_steps", 0) for item in specs], device=device)[condition_ids]
            command.external_override[:, 0] = vx
            command.external_override[:, 1] = vy
            command.external_override[:, 2] = input_yaw
            command._update_command()
            obs = wrapped.get_observations()["policy"].to(device)
            steps = round(8 / env.step_dt)
            sums = {key: torch.zeros(env.num_envs, device=device) for key in (
                "vx", "vy", "speed", "vec", "direction", "yaw", "yawerr", "slip",
                "tilt", "left", "right", "flight", "double", "act95", "act99",
                "joint_limit", "joint_velocity", "torque", "force", "post_yaw",
            )}
            yaw_abs_values = []
            fall = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            dangerous = fall.clone(); impact = fall.clone(); saturation = fall.clone(); tiltbad = fall.clone()
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            sat_streak = slip_streak.clone()
            for step in range(steps):
                command.external_override[:, 0] = vx
                command.external_override[:, 1] = vy
                command.external_override[:, 2] = input_yaw
                with torch.inference_mode():
                    normal = actor(obs, torch.zeros(env.num_envs, device=device))
                    if wrapper_mode in ("mirror", "local"):
                        mirrored_obs = mirror_observation(obs)
                        mirrored_obs[:, 9] = vx
                        mirrored_obs[:, 10] = -vy
                        mirrored_obs[:, 11] = -input_yaw
                        mirrored = mirror_action(actor(mirrored_obs, torch.zeros(env.num_envs, device=device)))
                        if wrapper_mode == "mirror":
                            action = mirrored
                        else:
                            active = (step < branch_steps).float()[:, None]
                            action = normal + active * lambdas[:, None] * (mirrored - normal)
                    else:
                        action = normal
                obs, _, done, extras = wrapped.step(action)
                obs = obs["policy"].to(device)
                actual = robot.data.root_lin_vel_b[:, :2]
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
                dangerous |= slip_streak >= 5
                impact |= force.amax(-1) > 3500
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                tilt = torch.maximum(roll, pitch)
                tiltbad |= tilt > .8
                limits = robot.data.joint_vel_limits
                limits = limits[..., 1].abs() if limits.ndim == 3 else limits
                velocity_ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
                sat = velocity_ratio > .95
                sat_streak = torch.where(sat, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= sat_streak >= 5
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout
                cmd_angle = torch.atan2(vy, vx)
                actual_angle = torch.atan2(actual[:, 1], actual[:, 0])
                direction = torch.atan2(
                    torch.sin(actual_angle - cmd_angle), torch.cos(actual_angle - cmd_angle)
                ).abs() * 180 / math.pi
                action_abs = action.abs()
                joint_limits = robot.data.soft_joint_pos_limits
                low, high = joint_limits[..., 0], joint_limits[..., 1]
                pos = robot.data.joint_pos
                proximity = torch.maximum(
                    1 - (pos - low).div((high - low).clamp_min(1e-6)),
                    1 - (high - pos).div((high - low).clamp_min(1e-6)),
                ).amax(-1)
                values = {
                    "vx": actual[:, 0], "vy": actual[:, 1],
                    "speed": torch.linalg.vector_norm(actual, dim=-1),
                    "vec": torch.linalg.vector_norm(actual - torch.stack((vx, vy), -1), dim=-1),
                    "direction": direction, "yaw": actual_yaw,
                    "yawerr": (actual_yaw - target_yaw).abs(), "slip": sliding.float(),
                    "tilt": tilt, "left": contact[:, 0].float(), "right": contact[:, 1].float(),
                    "flight": (contact.sum(-1) == 0).float(),
                    "double": (contact.sum(-1) == 2).float(),
                    "act95": torch.quantile(action_abs, .95, dim=-1),
                    "act99": torch.quantile(action_abs, .99, dim=-1),
                    "joint_limit": proximity, "joint_velocity": velocity_ratio,
                    "torque": robot.data.applied_torque.abs().mean(-1),
                    "force": force.mean(-1),
                    "post_yaw": torch.where(step >= branch_steps, actual_yaw, torch.zeros_like(actual_yaw)),
                }
                for key, value in values.items():
                    sums[key] += value
                yaw_abs_values.append(actual_yaw.abs())
            for key in sums:
                sums[key] /= steps
            yaw_stack = torch.stack(yaw_abs_values)
            rows = []
            checkpoint_label = Path(checkpoint).stem
            for env_id in range(count):
                item = specs[int(condition_ids[env_id])]
                yaw_sign = float(sums["yaw"][env_id]) * item["target_yaw"] > 0
                translation_ok = float(sums["vec"][env_id]) <= .25 and (
                    item["speed"] == 0 or float(sums["direction"][env_id]) <= 25
                )
                yaw_ok = float(sums["yawerr"][env_id]) <= .15 and yaw_sign
                safe = not bool(fall[env_id] or dangerous[env_id] or impact[env_id] or saturation[env_id])
                if item["speed"] == 0:
                    success = safe and yaw_ok and float(sums["speed"][env_id]) <= .12
                else:
                    success = safe and yaw_ok and translation_ok
                rows.append({
                    "checkpoint": checkpoint_label, "condition": item["name"],
                    "direction_deg": item["direction_deg"], "commanded_speed": item["speed"],
                    "target_yaw": item["target_yaw"], "actor_input_yaw": item["actor_input_yaw"],
                    "wrapper": wrapper_mode, "lambda": item.get("lambda", 0.0),
                    "branch_steps": item.get("branch_steps", 0),
                    "episode": int(episode_ids[env_id]), "success": bool(success),
                    "actual_vx": float(sums["vx"][env_id]), "actual_vy": float(sums["vy"][env_id]),
                    "actual_speed": float(sums["speed"][env_id]), "actual_yaw": float(sums["yaw"][env_id]),
                    "actual_yaw_p95": float(torch.quantile(yaw_stack[:, env_id], .95)),
                    "vector_mae": float(sums["vec"][env_id]),
                    "direction_error": float(sums["direction"][env_id]),
                    "yaw_mae": float(sums["yawerr"][env_id]), "yaw_sign_correct": bool(yaw_sign),
                    "translation_correct": bool(translation_ok), "yaw_correct": bool(yaw_ok),
                    "fall": bool(fall[env_id]), "dangerous_slip": bool(dangerous[env_id]),
                    "impact": bool(impact[env_id]), "saturation": bool(saturation[env_id]),
                    "excessive_tilt": bool(tiltbad[env_id]),
                    "slip_fraction": float(sums["slip"][env_id]), "tilt_mean": float(sums["tilt"][env_id]),
                    "left_contact_fraction": float(sums["left"][env_id]),
                    "right_contact_fraction": float(sums["right"][env_id]),
                    "flight_fraction": float(sums["flight"][env_id]),
                    "double_support_fraction": float(sums["double"][env_id]),
                    "action_abs_p95": float(sums["act95"][env_id]),
                    "action_abs_p99": float(sums["act99"][env_id]),
                    "joint_limit_proximity": float(sums["joint_limit"][env_id]),
                    "joint_velocity_ratio": float(sums["joint_velocity"][env_id]),
                    "torque_abs_mean": float(sums["torque"][env_id]),
                    "contact_force_mean": float(sums["force"][env_id]),
                    "basin_yaw_after_branch": float(sums["post_yaw"][env_id]),
                })
            return rows

        all_rows = []
        if args.mode == "timeline":
            conditions = [spec(0, 0, yaw, 50, f"PURE_Y{yaw:+.1f}") for yaw in (-.3, .3)]
            conditions += [
                spec(direction, .3, yaw, 50, f"D{direction:03.0f}_Y{yaw:+.1f}")
                for direction in (0, 90, 135, 180, 225, 270) for yaw in (-.3, .3)
            ]
            for entry in manifest:
                rows = run(REPO / entry["path"], conditions, 20276021)
                for row in rows:
                    row["iteration"] = entry["iteration"]
                all_rows.extend(rows)
                print(f"[timeline] iteration {entry['iteration']}", flush=True)
            stem = "yaw_capability_checkpoint_timeline"

        elif args.mode == "surface":
            conditions = [
                spec(direction, speed, yaw, 20)
                for direction in (0, 45, 90, 135, 180, 225, 270, 315)
                for speed in (0, .1, .2, .3, .4, .6)
                for yaw in (-1, -.8, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .8, 1)
            ]
            for checkpoint in (PARENT, SELECTED):
                for start in range(0, len(conditions), 51):
                    batch = conditions[start:start + 51]
                    all_rows.extend(run(checkpoint, batch, 20277021 + start))
                    print(f"[surface] {Path(checkpoint).stem} {min(start+51,len(conditions))}/{len(conditions)}", flush=True)
            stem = "detailed_yaw_command_response_surface"

        elif args.mode == "prewarp":
            conditions = []
            for target in (.15, .3, .45):
                for gain in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
                    conditions.append(spec(0, 0, target, 30, f"PURE_T{target:.2f}_K{gain:.2f}",
                                           actor_input_yaw=target * gain, target_yaw=target))
                    conditions += [
                        spec(direction, .3, target, 30, f"D{direction:03.0f}_T{target:.2f}_K{gain:.2f}",
                             actor_input_yaw=target * gain, target_yaw=target)
                        for direction in (0, 45, 90, 135, 180, 225, 270, 315)
                    ]
            for start in range(0, len(conditions), 34):
                all_rows.extend(run(SELECTED, conditions[start:start + 34], 20278021 + start))
            stem = "positive_yaw_command_prewarp"

        elif args.mode == "unlock":
            conditions = [
                spec(direction, speed, .3, 30, f"D{direction:03.0f}_S{speed:.3f}_I{input_yaw:.1f}",
                     actor_input_yaw=input_yaw, target_yaw=.3)
                for direction in (0, 45, 90, 135, 180, 225, 270, 315)
                for speed in (0, .025, .05, .075, .1, .15, .2, .3)
                for input_yaw in (.3, .5, .7)
            ]
            for start in range(0, len(conditions), 34):
                all_rows.extend(run(SELECTED, conditions[start:start + 34], 20279021 + start))
            stem = "positive_yaw_translation_unlock_map"

        elif args.mode == "mirror":
            conditions = [
                spec(0, 0, .3, 100, "PURE_POS"),
                spec(90, .3, .3, 100, "D090_POS"),
                spec(135, .3, .3, 100, "D135_POS"),
                spec(180, .3, .3, 100, "D180_POS"),
                spec(225, .3, -.3, 100, "D225_NEG"),
            ]
            all_rows.extend(run(SELECTED, conditions, 20280021, "normal"))
            all_rows.extend(run(SELECTED, conditions, 20280021, "mirror"))
            stem = "mirrored_policy_positive_control"

        else:
            conditions = [
                spec(direction, 0 if name == "PURE" else .3, .3, 100,
                     f"{name}_L{lam:.2f}_H{horizon}", **{"lambda": lam, "branch_steps": horizon})
                for direction, name in ((0, "PURE"), (90, "D090"), (135, "D135"), (180, "D180"))
                for lam in (0, .1, .25, .5, .75, 1)
                for horizon in (1, 2, 4, 8)
            ]
            for start in range(0, len(conditions), 10):
                all_rows.extend(run(SELECTED, conditions[start:start + 10], 20281021 + start, "local"))
            stem = "positive_yaw_local_action_controllability"

        summary = aggregate(all_rows)
        write_csv(stem + ".csv", summary)
        dump(stem + ".json", {
            "rows": summary, "episode_rows": all_rows,
            "deterministic": True, "training_updates": 0, "optimizer_steps": 0,
            "seed": 20276021, "duration_s": 8,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
