"""Read-only fresh rollout diagnostics for W1B-D1 (never performs an optimizer step)."""
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
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
ITER1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"

sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("surface", "focused", "variance", "order"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def category(name):
    if "torso" in name or "waist" in name:
        return "waist"
    for value in ("hip", "knee", "ankle", "shoulder", "elbow"):
        if value in name:
            return value
    return "hand"


def condition(angle, speed, yaw, episodes, name=None):
    rad = math.radians(angle)
    return {
        "name": name or f"D{angle:05.1f}_S{speed:.1f}_Y{yaw:+.2f}",
        "direction_deg": angle,
        "speed": speed,
        "yaw": yaw,
        "vx": speed * math.cos(rad),
        "vy": speed * math.sin(rad),
        "episodes": episodes,
    }


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 960
    cfg.episode_length_s = 10
    cfg.seed = acfg.seed = 20275021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
                                     clip_actions=acfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        foot_indices = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_foot_indices = [robot.body_names.index(sensor.body_names[i]) for i in foot_indices]
        actors = {
            "parent": FrozenGaitActor(PARENT).to(device).eval(),
            "iteration1": FrozenGaitActor(ITER1).to(device).eval(),
        }
        payloads = {
            "parent": torch.load(PARENT, map_location=device, weights_only=False),
            "iteration1": torch.load(ITER1, map_location=device, weights_only=False),
        }
        action_term = env.action_manager.get_term("joint_pos")
        dump("_raw_robot_contract.json", {
            "joint_names": list(robot.joint_names),
            "default_joint_position": robot.data.default_joint_pos[0].detach().cpu().tolist(),
            "soft_joint_position_limits": robot.data.soft_joint_pos_limits[0].detach().cpu().tolist(),
            "joint_velocity_limits": robot.data.joint_vel_limits[0].detach().cpu().tolist(),
            "stiffness": robot.data.default_joint_stiffness[0].detach().cpu().tolist(),
            "damping": robot.data.default_joint_damping[0].detach().cpu().tolist(),
            "action_scale": (action_term._scale[0].detach().cpu().tolist()
                             if torch.is_tensor(action_term._scale) and action_term._scale.ndim > 1
                             else action_term._scale.detach().cpu().tolist()
                             if torch.is_tensor(action_term._scale) else action_term._scale),
            "source": "live immutable Unitree G1 articulation and JointPositionAction term",
        })

        def run(actor_name, specs, seed, sampling="D0", prelude=None, collect_obs=False):
            count = sum(x["episodes"] for x in specs)
            if count > env.num_envs:
                raise ValueError(f"{count} episodes exceed {env.num_envs} envs")
            wrapped.seed(seed)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            valid_env = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            episodes = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            cursor = 0
            for index, spec in enumerate(specs):
                amount = spec["episodes"]
                ids[cursor:cursor + amount] = index
                episodes[cursor:cursor + amount] = torch.arange(amount, device=device)
                valid_env[cursor:cursor + amount] = True
                cursor += amount
            vx = torch.tensor([x["vx"] for x in specs], device=device)[ids]
            vy = torch.tensor([x["vy"] for x in specs], device=device)[ids]
            yaw = torch.tensor([x["yaw"] for x in specs], device=device)[ids]
            if prelude:
                pvx, pvy, pyaw = prelude
                for step in range(round(2 / env.step_dt)):
                    command.external_override[:, 0] = pvx
                    command.external_override[:, 1] = pvy
                    command.external_override[:, 2] = pyaw
                    if step == 0:
                        command._update_command()
                        obs = wrapped.get_observations()["policy"].to(device)
                    with torch.inference_mode():
                        action = actors[actor_name](obs, torch.zeros(env.num_envs, device=device))
                    obs, _, _, _ = wrapped.step(action)
                    obs = obs["policy"].to(device)
                # Deliberately use the normal environment reset. This tests whether a prelude leaks across reset.
                obs, _ = wrapped.reset()
                obs = obs["policy"].to(device)
            steps = round(8 / env.step_dt)
            sums = {key: torch.zeros(env.num_envs, device=device) for key in
                    ("vx", "vy", "speed", "vec", "direction", "yaw", "yawerr", "flight", "slip", "tilt", "left", "right")}
            fall = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            dangerous = fall.clone()
            impact = fall.clone()
            saturation = fall.clone()
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            sat_streak = slip_streak.clone()
            observations = []
            actions = []
            initial_contact = None
            model = actors[actor_name]
            std = payloads[actor_name]["actor_state_dict"]["distribution.log_std_walk"].exp()
            generator = torch.Generator(device=device).manual_seed(seed + 9173)
            for step in range(steps):
                command.external_override[:, 0] = vx
                command.external_override[:, 1] = vy
                command.external_override[:, 2] = yaw
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations()["policy"].to(device)
                with torch.inference_mode():
                    mean = model(obs, torch.zeros(env.num_envs, device=device))
                    if sampling == "D0":
                        action = mean
                    else:
                        scale = 1.0 if sampling == "S030" else 1 / 3
                        action = mean + torch.randn(mean.shape, generator=generator, device=device) * std * scale
                if collect_obs and step % 20 == 0:
                    keep = torch.where(valid_env)[0][::max(1, count // 160)]
                    observations.append(obs[keep].detach().cpu())
                    actions.append(action[keep].detach().cpu())
                obs, _, done, extra = wrapped.step(action)
                obs = obs["policy"].to(device)
                actual = robot.data.root_lin_vel_b[:, :2]
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, foot_indices, :].norm(dim=-1)
                contact = force > 5
                if initial_contact is None:
                    initial_contact = contact.clone()
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_foot_indices, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
                dangerous |= slip_streak >= 5
                impact |= force.amax(-1) > 3500
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                joint_limit = robot.data.joint_vel_limits
                joint_limit = joint_limit[..., 1].abs() if joint_limit.ndim == 3 else joint_limit
                sat = robot.data.joint_vel.abs().div(joint_limit.clamp_min(1e-6)).amax(-1) > .95
                sat_streak = torch.where(sat, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= sat_streak >= 5
                timeout = extra.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout
                cmd_angle = torch.atan2(vy, vx)
                actual_angle = torch.atan2(actual[:, 1], actual[:, 0])
                direction_error = torch.atan2(torch.sin(actual_angle - cmd_angle),
                                              torch.cos(actual_angle - cmd_angle)).abs() * 180 / math.pi
                values = {
                    "vx": actual[:, 0], "vy": actual[:, 1],
                    "speed": torch.linalg.vector_norm(actual, dim=-1),
                    "vec": torch.linalg.vector_norm(actual - torch.stack((vx, vy), -1), dim=-1),
                    "direction": direction_error, "yaw": actual_yaw, "yawerr": (actual_yaw - yaw).abs(),
                    "flight": (contact.sum(-1) == 0).float(), "slip": sliding.float(),
                    "tilt": torch.maximum(roll, pitch), "left": contact[:, 0].float(), "right": contact[:, 1].float(),
                }
                for key, value in values.items():
                    sums[key] += value
            for key in sums:
                sums[key] /= steps
            episode_rows = []
            for env_id in range(count):
                spec = specs[int(ids[env_id])]
                yaw_sign = abs(spec["yaw"]) < 1e-8 or float(sums["yaw"][env_id]) * spec["yaw"] > 0
                translation_ok = float(sums["vec"][env_id]) <= .25 and (
                    spec["speed"] == 0 or float(sums["direction"][env_id]) <= 25)
                yaw_ok = float(sums["yawerr"][env_id]) <= .20 and yaw_sign
                safe = not bool(fall[env_id] or dangerous[env_id] or impact[env_id] or saturation[env_id])
                gait = float(sums["flight"][env_id]) < .10
                if spec["speed"] == 0:
                    success = safe and gait and yaw_ok and float(sums["speed"][env_id]) <= .12
                elif abs(spec["yaw"]) < 1e-8:
                    success = safe and gait and float(sums["vec"][env_id]) <= .20 and \
                              float(sums["direction"][env_id]) <= 20 and abs(float(sums["yaw"][env_id])) <= .20
                else:
                    success = safe and gait and translation_ok and yaw_ok
                phase_contact = initial_contact[env_id].tolist()
                phase = "double_support" if sum(phase_contact) == 2 else (
                    "left_support" if phase_contact[0] else ("right_support" if phase_contact[1] else "flight"))
                episode_rows.append({
                    "checkpoint": actor_name, "condition": spec["name"], "direction_deg": spec["direction_deg"],
                    "commanded_speed": spec["speed"], "yaw_cmd": spec["yaw"], "episode": int(episodes[env_id]),
                    "sampling_mode": sampling, "success": bool(success),
                    "actual_vx": float(sums["vx"][env_id]), "actual_vy": float(sums["vy"][env_id]),
                    "actual_speed": float(sums["speed"][env_id]), "actual_yaw": float(sums["yaw"][env_id]),
                    "vector_mae": float(sums["vec"][env_id]), "direction_error": float(sums["direction"][env_id]),
                    "yaw_mae": float(sums["yawerr"][env_id]), "yaw_sign_correct": bool(yaw_sign),
                    "translation_correct": bool(translation_ok), "yaw_correct": bool(yaw_ok),
                    "both_correct": bool(translation_ok and yaw_ok), "walk_like": bool(gait),
                    "fall": bool(fall[env_id]), "dangerous_slip": bool(dangerous[env_id]),
                    "slip_fraction": float(sums["slip"][env_id]), "tilt_mean": float(sums["tilt"][env_id]),
                    "impact": bool(impact[env_id]), "saturation": bool(saturation[env_id]),
                    "left_contact_fraction": float(sums["left"][env_id]),
                    "right_contact_fraction": float(sums["right"][env_id]), "initial_support_phase": phase,
                })
            grouped = defaultdict(list)
            for row in episode_rows:
                grouped[row["condition"]].append(row)
            summary = []
            for name, rows in grouped.items():
                base = {key: rows[0][key] for key in
                        ("checkpoint", "condition", "direction_deg", "commanded_speed", "yaw_cmd", "sampling_mode")}
                base["episodes"] = len(rows)
                for key in ("success", "yaw_sign_correct", "translation_correct", "yaw_correct", "both_correct",
                            "walk_like", "fall", "dangerous_slip", "impact", "saturation"):
                    base[key + "_rate"] = sum(bool(row[key]) for row in rows) / len(rows)
                for key in ("actual_vx", "actual_vy", "actual_speed", "actual_yaw", "vector_mae",
                            "direction_error", "yaw_mae", "slip_fraction", "tilt_mean",
                            "left_contact_fraction", "right_contact_fraction"):
                    base[key] = sum(row[key] for row in rows) / len(rows)
                base["gate_pass"] = base["success_rate"] >= .9 and base["fall_rate"] <= .05
                summary.append(base)
            return summary, episode_rows, {
                "observations": torch.cat(observations) if observations else torch.empty(0),
                "actions": torch.cat(actions) if actions else torch.empty(0),
            }

        if args.mode == "surface":
            specs = [condition(angle, speed, yaw, 20) for angle in (i * 22.5 for i in range(16))
                     for speed in (0, .2, .3, .4, .6)
                     for yaw in (-.6, -.45, -.3, -.15, 0, .15, .3, .45, .6)]
            rows, episodes = [], []
            for start in range(0, len(specs), 48):
                a, b, _ = run("parent", specs[start:start + 48], 20275021 + start)
                rows.extend(a)
                episodes.extend(b)
                print(f"[surface] {min(start + 48, len(specs))}/{len(specs)}", flush=True)
            write_csv("parent_translation_yaw_response_surface.csv", rows)
            dump("parent_translation_yaw_response_surface.json", {"rows": rows, "episodes_per_condition": 20,
                 "deterministic": True, "early_termination_guards": ["fall", "impact", "saturation"]})
            dump("_raw_surface_episodes.json", {"rows": episodes})

        elif args.mode == "focused":
            zero = [condition(i * 22.5, .3, 0, 50) for i in range(16)]
            focus = [
                condition(0, 0, y, 50, f"PURE_Y{y:+.1f}") for y in (-.3, .3)
            ] + [
                condition(d, s, y, 50, f"D{d:05.1f}_S{s:.1f}_Y{y:+.1f}")
                for d, s in ((0, .3), (0, .6), (90, .3), (270, .3), (45, .3),
                             (315, .3), (135, .3), (225, .3), (180, .3))
                for y in (-.3, .3)
            ]
            focused_rows, focused_episodes, state_data = [], [], {}
            for actor_name in ("parent", "iteration1"):
                for start in range(0, len(focus), 18):
                    a, b, _ = run(actor_name, focus[start:start + 18], 20276021 + start)
                    focused_rows.extend(a); focused_episodes.extend(b)
                for sampling in ("D0", "S030", "S010"):
                    a, b, snap = run(actor_name, zero, 20277021 + {"D0": 0, "S030": 1, "S010": 2}[sampling],
                                     sampling=sampling, collect_obs=sampling == "D0")
                    for row in a:
                        row["path"] = "fresh_baseline"
                    focused_rows.extend(a); focused_episodes.extend(b)
                    if sampling == "D0":
                        state_data[actor_name] = snap
            write_csv("parent_vs_iteration1_yaw_comparison.csv", [
                row for row in focused_rows if row["condition"].startswith(("PURE", "D")) and row["sampling_mode"] == "D0"
            ])
            dump("parent_vs_iteration1_yaw_comparison.json", {"rows": focused_rows})
            sampling_rows = [row for row in focused_rows if row["condition"].startswith("D") and
                             row["commanded_speed"] == .3 and row["yaw_cmd"] == 0]
            write_csv("zero_yaw_sampling_mode_comparison.csv", sampling_rows)
            dump("zero_yaw_sampling_mode_comparison.json", {"rows": sampling_rows,
                 "note": "S010 is diagnostic-only; S030 uses the checkpoint WALK standard deviation."})
            torch.save(state_data, OUT / "_raw_zero_yaw_state_samples.pt")
            dump("_raw_focused_episodes.json", {"rows": focused_episodes})

        elif args.mode == "variance":
            specs = [condition(i * 22.5, .3, 47 if i < 12 else 46) for i in range(16)]
            batch_rows = []
            # 100 independent seed/reset batches per checkpoint, matching the online direction sample counts.
            for actor_name in ("parent", "iteration1"):
                for batch in range(100):
                    summary, _, _ = run(actor_name, specs, 20280021 + batch)
                    pass_count = sum(row["gate_pass"] for row in summary)
                    row = {"checkpoint": actor_name, "batch": batch, "pass_directions": pass_count}
                    for item in summary:
                        row[f"d{item['direction_deg']:05.1f}_success"] = item["success_rate"]
                    batch_rows.append(row)
                    if (batch + 1) % 10 == 0:
                        print(f"[variance] {actor_name} {batch + 1}/100", flush=True)
            write_csv("early_guard_sampling_variance.csv", batch_rows)
            summary = {}
            for actor_name in ("parent", "iteration1"):
                values = [row["pass_directions"] for row in batch_rows if row["checkpoint"] == actor_name]
                summary[actor_name] = {
                    "batches": 100,
                    "probability_11_or_less": sum(v <= 11 for v in values) / 100,
                    "probability_less_than_12": sum(v < 12 for v in values) / 100,
                    "probability_16_of_16": sum(v == 16 for v in values) / 100,
                    "mean_pass_directions": sum(values) / 100,
                    "min_pass_directions": min(values),
                    "max_pass_directions": max(values),
                }
            dump("early_guard_sampling_variance.json", {"online_matched_deterministic_batches": summary,
                 "episodes_per_direction": {"0_to_247p5": 47, "270_to_337p5": 46}})

        else:
            zero = [condition(i * 22.5, .3, 0, 50) for i in range(16)]
            sequences = {
                "A_fresh_reset": None,
                "A2_fresh_reset_repeat": None,
                "B_yaw_rollout": (0, 0, .3),
                "C_training_distribution_rollout": (.3, 0, .2),
                "D_pure_yaw_rollout": (0, 0, -.3),
                "E_moving_turn_rollout": (0, .3, -.3),
            }
            rows = []
            for actor_name in ("parent", "iteration1"):
                for label, prelude in sequences.items():
                    result, _, _ = run(actor_name, zero, 20281021, prelude=prelude)
                    for row in result:
                        row["sequence"] = label
                    rows.extend(result)
            write_csv("evaluation_order_effects.csv", rows)
            dump("evaluation_state_contamination.json", {
                "sequences": sequences,
                "rows": rows,
                "reset_contract": "Each prelude is followed by the same wrapper.reset used by fresh evaluation.",
            })
        wrapped.close()


if __name__ == "__main__":
    main()
