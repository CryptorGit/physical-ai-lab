"""Audit CROUCH action mapping and symmetric-pose reachability without PPO."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from importlib import metadata
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
sys.path.insert(0, str(EXPERIMENT_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

JOINT_GROUPS = {
    "hip_pitch": ("left_hip_pitch_joint", "right_hip_pitch_joint"),
    "knee": ("left_knee_joint", "right_knee_joint"),
    "ankle_pitch": ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20260722)
parser.add_argument("--search-mode", choices=("full", "finite", "refine", "deep"), default="full")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100.0), len(values) - 1)]


def candidates(mode: str) -> list[dict]:
    result = []
    if mode == "finite":
        finite_values = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
        for group in JOINT_GROUPS:
            for value in finite_values:
                offsets = {name: 0.0 for name in JOINT_GROUPS}
                offsets[group] = value
                result.append({"kind": "finite_difference", "label": f"{group}_{value:+.2f}", **offsets})
        return result
    if mode == "refine":
        for hip, knee, ankle in itertools.product(
            (-0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20),
            (0.40, 0.60, 0.80, 1.00, 1.20),
            (-0.80, -0.60, -0.40, -0.20, 0.0, 0.20),
        ):
            result.append({"kind": "reachability_refine", "label": "refine", "hip_pitch": hip, "knee": knee, "ankle_pitch": ankle})
        return result
    if mode == "deep":
        for hip, knee, ankle in itertools.product(
            (-0.50, -0.40, -0.30, -0.25, -0.20, -0.15),
            (0.80, 1.00, 1.20, 1.40),
            (-1.20, -1.00, -0.80, -0.60),
        ):
            result.append({"kind": "reachability_deep", "label": "deep", "hip_pitch": hip, "knee": knee, "ankle_pitch": ankle})
        return result
    finite_values = (-0.50, -0.25, -0.10, -0.05, 0.0, 0.05, 0.10, 0.25, 0.50)
    for group in JOINT_GROUPS:
        for value in finite_values:
            offsets = {name: 0.0 for name in JOINT_GROUPS}
            offsets[group] = value
            result.append({"kind": "finite_difference", "label": f"{group}_{value:+.2f}", **offsets})
    for hip, knee, ankle in itertools.product(
        (-0.25, -0.125, 0.0, 0.125, 0.25),
        (-0.25, -0.125, 0.0, 0.125, 0.25),
        (-0.15, -0.075, 0.0, 0.075, 0.15),
    ):
        result.append({"kind": "current_bound", "label": "current", "hip_pitch": hip, "knee": knee, "ankle_pitch": ankle})
    # Expanded reachability grid is stated in normalized action. With the
    # environment scale 0.5, target-position deltas are half these values.
    for hip, knee, ankle in itertools.product(
        (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2),
        (-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2),
        (-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6),
    ):
        result.append({"kind": "reachability", "label": "coarse", "hip_pitch": hip, "knee": knee, "ankle_pitch": ankle})
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = Path(args_cli.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    specs = candidates(args_cli.search_mode)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = len(specs)
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make("Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        env = raw_env.unwrapped
        agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        actor = runner.alg.actor
        robot = env.scene["robot"]
        contact = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        joint_ids = {}
        for group, names in JOINT_GROUPS.items():
            joint_ids[group], _ = robot.find_joints(list(names), preserve_order=True)
        all_joint_ids, _ = robot.find_joints(".*")
        action_scale = float(env_cfg.actions.joint_pos.scale)
        offsets = torch.zeros(len(specs), 37, device=env.device)
        for env_id, spec in enumerate(specs):
            for group, ids in joint_ids.items():
                offsets[env_id, ids] = float(spec[group])

        wrapped.reset()
        dt = float(env.step_dt)
        settle_steps, ramp_steps, hold_steps = round(1.0 / dt), round(1.5 / dt), round(1.2 / dt)
        total_steps = settle_steps + ramp_steps + hold_steps
        active = torch.ones(len(specs), dtype=torch.bool, device=env.device)
        data = [defaultdict(list) for _ in specs]
        entry_samples = [[] for _ in specs]
        for step in range(total_steps):
            observations = wrapped.get_observations()
            with torch.inference_mode():
                components = actor.diagnostic_components(observations)
                standing_action = components["standing_base_action"]
            if step < settle_steps:
                progress = 0.0
            elif step < settle_steps + ramp_steps:
                x = (step - settle_steps + 1) / ramp_steps
                progress = x * x * (3.0 - 2.0 * x)
            else:
                progress = 1.0
            actions = standing_action + progress * offsets
            _, _, dones, _ = wrapped.step(actions)
            active &= ~dones.bool()
            forces = contact.data.net_forces_w_history.torch[:, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            torque_ratio = robot.data.applied_torque.torch[:, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            velocity_ratio = robot.data.joint_vel.torch[:, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            hard = robot.data.joint_pos_limits.torch[:, all_joint_ids]
            pos = robot.data.joint_pos.torch[:, all_joint_ids]
            proximity = torch.maximum(
                (pos - robot.data.default_joint_pos.torch[:, all_joint_ids]) / (hard[..., 1] - robot.data.default_joint_pos.torch[:, all_joint_ids]).clamp_min(1.0e-6),
                (robot.data.default_joint_pos.torch[:, all_joint_ids] - pos) / (robot.data.default_joint_pos.torch[:, all_joint_ids] - hard[..., 0]).clamp_min(1.0e-6),
            ).abs().amax(dim=-1)
            for env_id in range(len(specs)):
                if step >= settle_steps - round(0.3 / dt) and step < settle_steps:
                    entry_samples[env_id].append(float(robot.data.root_pos_w.torch[env_id, 2].item()))
                if step < settle_steps + ramp_steps or not bool(active[env_id].item()):
                    continue
                item = data[env_id]
                item["height"].append(float(robot.data.root_pos_w.torch[env_id, 2].item()))
                item["tilt"].append(float(torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[env_id, :2]).item()))
                item["horizontal_speed"].append(float(robot.data.root_lin_vel_b.torch[env_id, :2].norm().item()))
                item["vertical_speed"].append(abs(float(robot.data.root_lin_vel_b.torch[env_id, 2].item())))
                item["double_support"].append(float(bool(contacts[env_id].all().item())))
                item["flight"].append(float(not bool(contacts[env_id].any().item())))
                item["slip"].append(float((foot_speed[env_id] * contacts[env_id]).sum().item() / max(int(contacts[env_id].sum().item()), 1)))
                item["torque_utilization"].append(float(torque_ratio[env_id].amax().item()))
                item["velocity_utilization"].append(float(velocity_ratio[env_id].amax().item()))
                item["joint_limit_proximity"].append(float(proximity[env_id].item()))
                for group, ids in joint_ids.items():
                    item[f"{group}_position"].append(float(robot.data.joint_pos.torch[env_id, ids].mean().item()))
                    item[f"{group}_base_action"].append(float(standing_action[env_id, ids].mean().item()))
                    item[f"{group}_final_action"].append(float(actions[env_id, ids].mean().item()))
                    for side, joint_id in zip(("left", "right"), ids):
                        item[f"{side}_{group}_position"].append(float(robot.data.joint_pos.torch[env_id, joint_id].item()))
                        item[f"{side}_{group}_base_action"].append(float(standing_action[env_id, joint_id].item()))
                        item[f"{side}_{group}_final_action"].append(float(actions[env_id, joint_id].item()))

        rows = []
        for env_id, spec in enumerate(specs):
            item = data[env_id]
            entry = mean(entry_samples[env_id])
            minimum = min(item["height"], default=entry)
            row = {
                **spec,
                "hip_pitch_target_delta_rad": action_scale * spec["hip_pitch"],
                "knee_target_delta_rad": action_scale * spec["knee"],
                "ankle_pitch_target_delta_rad": action_scale * spec["ankle_pitch"],
                "entry_pelvis_height_m": entry,
                "minimum_pelvis_height_m": minimum,
                "pelvis_drop_m": entry - minimum,
                "hold_height_drop_mean_m": entry - mean(item["height"]),
                "tilt_p95_rad": percentile(item["tilt"], 95),
                "horizontal_speed_p95_mps": percentile(item["horizontal_speed"], 95),
                "vertical_speed_p95_mps": percentile(item["vertical_speed"], 95),
                "double_support_fraction": mean(item["double_support"]),
                "flight_fraction": mean(item["flight"]),
                "foot_slip_max_mps": max(item["slip"], default=0.0),
                "torque_utilization_max": max(item["torque_utilization"], default=0.0),
                "joint_velocity_utilization_max": max(item["velocity_utilization"], default=0.0),
                "joint_limit_proximity_max": max(item["joint_limit_proximity"], default=0.0),
                "fall": not bool(active[env_id].item()),
            }
            for group in JOINT_GROUPS:
                row[f"{group}_actual_position_rad"] = mean(item[f"{group}_position"])
                row[f"{group}_actual_delta_rad"] = mean(item[f"{group}_position"]) - float(robot.data.default_joint_pos.torch[env_id, joint_ids[group]].mean().item())
                row[f"{group}_base_action"] = mean(item[f"{group}_base_action"])
                row[f"{group}_final_action"] = mean(item[f"{group}_final_action"])
                for side, joint_id in zip(("left", "right"), joint_ids[group]):
                    position = mean(item[f"{side}_{group}_position"])
                    base_action = mean(item[f"{side}_{group}_base_action"])
                    final_action = mean(item[f"{side}_{group}_final_action"])
                    default = float(robot.data.default_joint_pos.torch[env_id, joint_id].item())
                    row[f"{side}_{group}_actual_position_rad"] = position
                    row[f"{side}_{group}_actual_delta_rad"] = position - default
                    row[f"{side}_{group}_base_action"] = base_action
                    row[f"{side}_{group}_residual_action"] = float(spec[group])
                    row[f"{side}_{group}_final_action"] = final_action
                    row[f"{side}_{group}_base_joint_target_rad"] = default + action_scale * base_action
                    row[f"{side}_{group}_final_joint_target_rad"] = default + action_scale * final_action
            row["stable"] = bool(
                not row["fall"] and len(item["height"]) >= hold_steps - 1
                and row["tilt_p95_rad"] <= 0.15 and row["horizontal_speed_p95_mps"] <= 0.10
                and row["flight_fraction"] <= 0.01 and row["double_support_fraction"] >= 0.90
                and row["foot_slip_max_mps"] <= 0.10 and row["torque_utilization_max"] < 0.95
                and row["joint_velocity_utilization_max"] < 0.95 and row["joint_limit_proximity_max"] < 0.95
            )
            rows.append(row)

        mapping = []
        hard_limits = robot.data.joint_pos_limits.torch[0]
        soft_limits = robot.data.soft_joint_pos_limits.torch[0]
        for group, names in JOINT_GROUPS.items():
            for side, (name, joint_id) in zip(("left", "right"), zip(names, joint_ids[group])):
                mapping.append({
                    "group": group, "side": side, "action_index": int(joint_id), "joint_name": name,
                    "action_sign": "positive action increases joint target",
                    "environment_action_scale": action_scale,
                    "default_joint_position_rad": float(robot.data.default_joint_pos.torch[0, joint_id].item()),
                    "soft_limit_rad": [float(v) for v in soft_limits[joint_id].tolist()],
                    "hard_limit_rad": [float(v) for v in hard_limits[joint_id].tolist()],
                })
        finite = [row for row in rows if row["kind"] == "finite_difference"]
        current = [row for row in rows if row["kind"] == "current_bound"]
        reach = [row for row in rows if row["kind"].startswith("reachability")]
        sensitivity = {}
        for group in JOINT_GROUPS:
            selected = [row for row in finite if row["label"].startswith(group) and abs(row[group]) <= 0.10]
            x = [row[group] for row in selected]
            y = [row["pelvis_drop_m"] for row in selected]
            x_mean, y_mean = mean(x), mean(y)
            denom = sum((value - x_mean) ** 2 for value in x)
            slope = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / denom if denom else 0.0
            sensitivity[group] = {
                "d_pelvis_drop_d_normalized_action_m": slope,
                "d_pelvis_drop_d_target_rad_m_per_rad": slope / action_scale,
            }
        stable_current = [row for row in current if row["stable"]]
        stable_reach = [row for row in reach if row["stable"]]
        target_poses = {}
        for target in (0.08, 0.10, 0.15):
            best = min(stable_reach, key=lambda row: abs(row["pelvis_drop_m"] - target), default=None)
            target_poses[f"{target:.2f}"] = best
        summary = {
            "checkpoint": str(checkpoint), "seed": args_cli.seed, "candidate_count": len(specs),
            "action_mapping": mapping, "finite_difference_sensitivity": sensitivity,
            "current_bound": {
                "candidate_count": len(current), "stable_count": len(stable_current),
                "maximum_stable_drop_m": max((row["pelvis_drop_m"] for row in stable_current), default=0.0),
                "best_pose": max(stable_current, key=lambda row: row["pelvis_drop_m"], default=None),
            },
            "reachability": {"candidate_count": len(reach), "stable_count": len(stable_reach), "target_poses": target_poses},
            "stable_definition": {
                "hold_duration_s": 1.2, "tilt_p95_max_rad": 0.15, "horizontal_speed_p95_max_mps": 0.10,
                "double_support_fraction_min": 0.90, "flight_fraction_max": 0.01, "foot_slip_max_mps": 0.10,
                "torque_and_velocity_utilization_max": 0.95, "joint_limit_proximity_max": 0.95,
            },
        }
        if finite:
            write_csv(output / "finite_difference.csv", finite)
        if current:
            write_csv(output / "current_bound_search.csv", current)
        if reach:
            write_csv(output / "reachability_search.csv", reach)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        raw_env.close()


if __name__ == "__main__":
    main()
