"""Evaluate the calibrated scripted CROUCH primitive at fixed Stage-A depths."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
from g1_command_skills.scripted_crouch import phased_offset, pose_for_depth  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episodes-per-depth", type=int, default=10)
parser.add_argument("--seed", type=int, default=20260723)
parser.add_argument("--max-primitive-depth", type=float, default=0.1010949334)
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True)
    output = Path(args_cli.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    depths = [value for value in (0.08, 0.10, 0.15) for _ in range(args_cli.episodes_per_depth)]
    count = len(depths)
    env_cfg, agent_cfg = resolve_task_config("Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0", "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = count; env_cfg.seed = args_cli.seed
    if args_cli.device is not None: env_cfg.sim.device = args_cli.device
    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make("Isaac-Motion-Flat-G1-Command-Crouch-Eval-v0", cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        env = raw_env.unwrapped; agent_cfg.device = env.device
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(checkpoint), load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False})
        actor = runner.alg.actor; robot = env.scene["robot"]; contact = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        all_joint_ids, _ = robot.find_joints(".*")
        depth_tensor = torch.tensor(depths, device=env.device)
        control_depth_tensor = depth_tensor.clamp_max(args_cli.max_primitive_depth)
        wrapped.reset()
        dt = float(env.step_dt)
        phase_durations = (1.0, 1.5, 1.0, 1.5, 1.0)
        phase_steps = [round(value / dt) for value in phase_durations]
        phase_ends, running = [], 0
        for value in phase_steps:
            running += value; phase_ends.append(running)
        active = torch.ones(count, dtype=torch.bool, device=env.device)
        entry_samples = [[] for _ in depths]
        traces = [{"heights": [], "hold_heights": [], "stand_heights": [], "support": [], "slip": [],
                   "velocity_sat": [], "torque_sat": [], "limit": [], "tilt": [], "residual_norm": []} for _ in depths]
        curves = []
        for step in range(phase_ends[-1]):
            phase = next(index for index, end in enumerate(phase_ends) if step < end)
            start = 0 if phase == 0 else phase_ends[phase - 1]
            progress_value = (step - start + 1) / phase_steps[phase]
            phases = torch.full((count,), phase, device=env.device, dtype=torch.long)
            progress = torch.full((count,), progress_value, device=env.device)
            observations = wrapped.get_observations()
            with torch.inference_mode():
                standing_action = actor.diagnostic_components(observations)["standing_base_action"]
                primitive = phased_offset(control_depth_tensor, phases, progress)
            actions = standing_action + primitive
            _, _, dones, _ = wrapped.step(actions); active &= ~dones.bool()
            forces = contact.data.net_forces_w_history.torch[:, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            velocity_ratio = robot.data.joint_vel.torch[:, all_joint_ids].abs() / robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            torque_ratio = robot.data.applied_torque.torch[:, all_joint_ids].abs() / robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            hard = robot.data.joint_pos_limits.torch[:, all_joint_ids]; pos = robot.data.joint_pos.torch[:, all_joint_ids]
            default = robot.data.default_joint_pos.torch[:, all_joint_ids]
            proximity = torch.maximum(
                (pos - default) / (hard[..., 1] - default).clamp_min(1.0e-6),
                (default - pos) / (default - hard[..., 0]).clamp_min(1.0e-6),
            ).abs().amax(dim=-1)
            for env_id in range(count):
                height = float(robot.data.root_pos_w.torch[env_id, 2].item())
                if phase == 0 and step >= phase_ends[0] - round(0.3 / dt): entry_samples[env_id].append(height)
                if phase == 0 or not bool(active[env_id].item()): continue
                trace = traces[env_id]; trace["heights"].append(height)
                if phase == 2: trace["hold_heights"].append(height)
                if phase == 4: trace["stand_heights"].append(height)
                state = 3 if bool(contacts[env_id].all().item()) else 1 if bool(contacts[env_id, 0].item()) else 2 if bool(contacts[env_id, 1].item()) else 0
                trace["support"].append(state)
                trace["slip"].append(float((foot_speed[env_id] * contacts[env_id]).sum().item() / max(int(contacts[env_id].sum().item()), 1)))
                trace["velocity_sat"].append(float(bool((velocity_ratio[env_id] >= 0.95).any().item())))
                trace["torque_sat"].append(float(bool((torque_ratio[env_id] >= 0.95).any().item())))
                trace["limit"].append(float(proximity[env_id].item()))
                trace["tilt"].append(float(torch.linalg.vector_norm(robot.data.projected_gravity_b.torch[env_id, :2]).item()))
                trace["residual_norm"].append(float(torch.linalg.vector_norm(primitive[env_id]).item()))
                curves.append({"episode": env_id, "depth_m": depths[env_id], "time_s": (step + 1) * dt,
                               "phase": phase, "progress": progress_value, "entry_height_m": mean(entry_samples[env_id]),
                               "pelvis_height_m": height, "primitive_norm": trace["residual_norm"][-1],
                               "double_support": state == 3, "fall": not bool(active[env_id].item())})
        records = []
        for env_id, depth in enumerate(depths):
            trace = traces[env_id]; entry = mean(entry_samples[env_id]); minimum = min(trace["heights"], default=entry)
            actual = entry - minimum; depth_error = abs(depth - actual)
            hold_errors = [abs((entry - depth) - value) for value in trace["hold_heights"]]
            stand_errors = [abs(entry - value) for value in trace["stand_heights"]]
            down_reached = actual >= depth - 0.04
            hold_success = down_reached and bool(hold_errors) and percentile(hold_errors, 95) <= 0.04
            return_kinematic = bool(stand_errors) and percentile(stand_errors, 95) <= 0.05
            return_success = down_reached and return_kinematic
            states = trace["support"]; max_runs = {0: 0, 1: 0, 2: 0, 3: 0}; last = None; run = switches = 0
            for state in states:
                if state == last: run += 1
                else:
                    if last is not None: max_runs[last] = max(max_runs[last], run); switches += 1
                    last, run = state, 1
            if last is not None: max_runs[last] = max(max_runs[last], run)
            dangerous_contact = max_runs[0] * dt > 0.10 or max(max_runs[1], max_runs[2]) * dt > 0.50 or switches / max(len(states) * dt, 1e-6) > 4.0
            saturation = mean(trace["velocity_sat"]) > 0.05 or mean(trace["torque_sat"]) > 0.05
            fall = not bool(active[env_id].item())
            success = depth_error <= 0.04 and hold_success and return_success and not fall and not dangerous_contact and not saturation
            records.append({
                "episode": env_id, "commanded_depth_m": depth, "entry_pelvis_height_m": entry,
                "target_minimum_pelvis_height_m": entry - depth, "actual_minimum_pelvis_height_m": minimum,
                "actual_depth_m": actual, "depth_error_m": depth_error, "down_reached": down_reached,
                "hold_success": hold_success, "return_kinematic_success": return_kinematic,
                "return_success": return_success, "stand_hold_success": return_kinematic,
                "fall": fall, "dangerous_contact_failure": dangerous_contact, "saturation_failure": saturation,
                "foot_slip_max_mps": max(trace["slip"], default=0.0),
                "joint_limit_proximity_max": max(trace["limit"], default=0.0),
                "tilt_max_rad": max(trace["tilt"], default=0.0),
                "primitive_norm_max": max(trace["residual_norm"], default=0.0), "success": success,
            })
        by_depth = {}
        for depth in (0.08, 0.10, 0.15):
            selected = [row for row in records if row["commanded_depth_m"] == depth]
            by_depth[f"{depth:.2f}"] = {
                "count": len(selected), "success_rate": mean([float(row["success"]) for row in selected]),
                "actual_depth_m": mean([row["actual_depth_m"] for row in selected]),
                "depth_error_m": mean([row["depth_error_m"] for row in selected]),
                "hold_success_rate": mean([float(row["hold_success"]) for row in selected]),
                "return_success_rate": mean([float(row["return_success"]) for row in selected]),
                "stand_hold_success_rate": mean([float(row["stand_hold_success"]) for row in selected]),
                "fall_rate": mean([float(row["fall"]) for row in selected]),
                "dangerous_contact_failure_rate": mean([float(row["dangerous_contact_failure"]) for row in selected]),
                "saturation_failure_rate": mean([float(row["saturation_failure"]) for row in selected]),
                "foot_slip_max_mps": max(row["foot_slip_max_mps"] for row in selected),
                "joint_limit_proximity_max": max(row["joint_limit_proximity_max"] for row in selected),
            }
        summary = {"controller": "frozen standing base + calibrated scripted CROUCH primitive", "checkpoint": str(checkpoint),
                   "episodes_per_depth": args_cli.episodes_per_depth, "seed": args_cli.seed,
                   "max_primitive_depth_m": args_cli.max_primitive_depth,
                   "phase_durations_s": {"settle": 1.0, "down": 1.5, "hold": 1.0, "return": 1.5, "stand_hold": 1.0},
                   "interpolation": "piecewise calibrated depth lookup + minimum-jerk phase blend",
                   "pose_offsets_normalized_action": {f"{depth:.2f}": pose_for_depth(torch.tensor([min(depth, args_cli.max_primitive_depth)]))[0, [0, 11, 15]].tolist() for depth in (0.08, 0.10, 0.15)},
                   "by_depth": by_depth}
        write_csv(output / "episodes.csv", records); write_csv(output / "curve.csv", curves)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2)); raw_env.close()


if __name__ == "__main__": main()
