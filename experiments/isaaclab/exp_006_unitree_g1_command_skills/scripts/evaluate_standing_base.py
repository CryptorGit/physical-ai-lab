"""Evaluate a frozen G1 policy or default-pose primitive as a standing base."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", required=True)
parser.add_argument("--checkpoint", default="")
parser.add_argument("--primitive", choices=("none", "default_pose"), default="none")
parser.add_argument("--task", default="Isaac-Velocity-Flat-G1-Run-Eval-v0")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--settle-timeout-s", type=float, default=2.0)
parser.add_argument("--settle-hold-s", type=float, default=0.4)
parser.add_argument("--stand-hold-s", type=float, default=6.0)
parser.add_argument("--output", required=True)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100.0), len(ordered) - 1)]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if (args_cli.primitive == "none") == (not args_cli.checkpoint):
        raise ValueError("Specify exactly one of --checkpoint or --primitive default_pose")
    checkpoint = Path(args_cli.checkpoint).resolve(strict=True) if args_cli.checkpoint else None
    output = Path(args_cli.output)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.mkdir(parents=True, exist_ok=True)

    env_cfg, agent_cfg = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.episodes
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
        env = raw_env.unwrapped
        agent_cfg.device = env.device
        policy = None
        if checkpoint is not None:
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
            runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
            runner.load(str(checkpoint), load_cfg={
                "actor": True, "critic": False, "optimizer": False, "iteration": False, "rnd": False,
            })
            policy = runner.get_inference_policy(device=env.device)

        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        contact = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_foot_ids = [contact.body_names.index(name) for name in foot_names]
        ankle_joint_ids, _ = robot.find_joints(".*ankle.*")
        all_joint_ids, all_joint_names = robot.find_joints(".*")
        pair_ids = []
        for left, right in (
            ("left_hip_pitch_joint", "right_hip_pitch_joint"),
            ("left_knee_joint", "right_knee_joint"),
            ("left_ankle_pitch_joint", "right_ankle_pitch_joint"),
        ):
            left_id, _ = robot.find_joints(left)
            right_id, _ = robot.find_joints(right)
            pair_ids.append((left_id[0], right_id[0]))

        wrapped.reset()
        n = args_cli.episodes
        dt = float(env.step_dt)
        settle_steps_required = max(1, round(args_cli.settle_hold_s / dt))
        settle_timeout_steps = max(settle_steps_required, round(args_cli.settle_timeout_s / dt))
        hold_steps_required = max(1, round(args_cli.stand_hold_s / dt))
        active = torch.ones(n, dtype=torch.bool, device=env.device)
        settled = torch.zeros(n, dtype=torch.bool, device=env.device)
        failed_settle = torch.zeros(n, dtype=torch.bool, device=env.device)
        fallen = torch.zeros(n, dtype=torch.bool, device=env.device)
        settle_streak = torch.zeros(n, dtype=torch.long, device=env.device)
        hold_steps = torch.zeros(n, dtype=torch.long, device=env.device)
        settle_times = torch.zeros(n, device=env.device)
        traces = [{
            "speed": [], "yaw": [], "height": [], "vertical": [], "roll": [], "pitch": [],
            "support": [], "left_slip": [], "right_slip": [], "ankle_torque_sat": [],
            "velocity_sat": [], "action": [], "asymmetry": [], "flight_run": 0,
            "single_run": 0, "max_flight_run": 0, "max_single_run": 0, "switches": 0,
            "previous_support": None,
        } for _ in range(n)]

        max_steps = settle_timeout_steps + hold_steps_required + 2
        for step in range(max_steps):
            command.vel_command_b.zero_()
            observations = wrapped.get_observations()
            if policy is None:
                actions = torch.zeros((n, wrapped.num_actions), device=env.device)
            else:
                with torch.inference_mode():
                    actions = policy(observations)
            with torch.inference_mode():
                _, _, dones, _ = wrapped.step(actions)
            command.vel_command_b.zero_()

            horizontal_speed = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical_speed = robot.data.root_lin_vel_w.torch[:, 2].abs()
            yaw_rate = robot.data.root_ang_vel_b.torch[:, 2]
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square()))
            forces = contact.data.net_forces_w_history.torch[:, :, sensor_foot_ids, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            contact_count = contacts.sum(dim=1)
            safe_settle = (
                (horizontal_speed <= 0.08) & (vertical_speed <= 0.05)
                & (roll.abs() <= 0.10) & (pitch.abs() <= 0.10) & (contact_count == 2)
            )
            waiting = active & ~settled
            settle_streak[waiting] = torch.where(
                safe_settle[waiting], settle_streak[waiting] + 1, torch.zeros_like(settle_streak[waiting])
            )
            newly_settled = waiting & (settle_streak >= settle_steps_required)
            settled[newly_settled] = True
            settle_times[newly_settled] = (step + 1) * dt
            timeout = waiting & ~newly_settled & ((step + 1) >= settle_timeout_steps)
            failed_settle[timeout] = True
            # A failed candidate must never advance into CROUCH, but continue a
            # stand-only diagnostic window so its stepping/drift is measurable.
            settle_times[timeout] = (step + 1) * dt
            fallen |= active & dones.bool()
            active[dones.bool()] = False

            velocities = robot.data.joint_vel.torch[:, all_joint_ids].abs()
            velocity_limits = robot.data.joint_vel_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            torque = robot.data.applied_torque.torch[:, all_joint_ids].abs()
            effort_limits = robot.data.joint_effort_limits.torch[:, all_joint_ids].abs().clamp_min(1.0e-6)
            velocity_ratio = velocities / velocity_limits
            torque_ratio = torque / effort_limits
            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            joint_pos = robot.data.joint_pos.torch
            diagnostic_hold = settled | failed_settle
            collecting = active & diagnostic_hold & (hold_steps < hold_steps_required)
            for env_id in torch.nonzero(collecting, as_tuple=False).flatten().tolist():
                trace = traces[env_id]
                state = int(contact_count[env_id].item())
                trace["speed"].append(float(horizontal_speed[env_id].item()))
                trace["yaw"].append(float(yaw_rate[env_id].item()))
                trace["height"].append(float(robot.data.root_pos_w.torch[env_id, 2].item()))
                trace["vertical"].append(float(vertical_speed[env_id].item()))
                trace["roll"].append(float(roll[env_id].item()))
                trace["pitch"].append(float(pitch[env_id].item()))
                trace["support"].append(state)
                trace["left_slip"].append(float(foot_speed[env_id, 0].item()) if contacts[env_id, 0] else 0.0)
                trace["right_slip"].append(float(foot_speed[env_id, 1].item()) if contacts[env_id, 1] else 0.0)
                trace["ankle_torque_sat"].append(float(bool((torque_ratio[env_id, ankle_joint_ids] >= 0.95).any().item())))
                trace["velocity_sat"].append(float(bool((velocity_ratio[env_id] >= 0.95).any().item())))
                trace["action"].append(float(actions[env_id].abs().max().item()))
                trace["asymmetry"].append(mean([
                    abs(float(joint_pos[env_id, left].item() - joint_pos[env_id, right].item()))
                    for left, right in pair_ids
                ]))
                if trace["previous_support"] is not None and trace["previous_support"] != state:
                    trace["switches"] += 1
                trace["previous_support"] = state
                trace["flight_run"] = trace["flight_run"] + 1 if state == 0 else 0
                trace["single_run"] = trace["single_run"] + 1 if state == 1 else 0
                trace["max_flight_run"] = max(trace["max_flight_run"], trace["flight_run"])
                trace["max_single_run"] = max(trace["max_single_run"], trace["single_run"])
            hold_steps[collecting] += 1
            active[diagnostic_hold & (hold_steps >= hold_steps_required)] = False
            if not bool(active.any().item()):
                break

        records = []
        for env_id, trace in enumerate(traces):
            support = trace["support"]
            speed = trace["speed"]
            height = trace["height"]
            ankle_sat = mean(trace["ankle_torque_sat"])
            velocity_sat = mean(trace["velocity_sat"])
            complete = len(speed) >= hold_steps_required
            hold_success = bool(
                settled[env_id] and complete and not fallen[env_id]
                and mean(speed) <= 0.05 and percentile(speed, 95) <= 0.10
                and (max(height) - min(height) if height else math.inf) <= 0.04
                and mean([float(state == 0) for state in support]) <= 0.001
                and trace["max_single_run"] * dt <= 0.50
                and ankle_sat <= 0.05 and velocity_sat <= 0.05
            )
            records.append({
                "episode": env_id, "seed": args_cli.seed, "candidate": args_cli.candidate,
                "checkpoint": str(checkpoint) if checkpoint else "", "primitive": args_cli.primitive,
                "settle_success": bool(settled[env_id].item()),
                "settle_time_s": float(settle_times[env_id].item()),
                "standing_hold_success": hold_success, "fall": bool(fallen[env_id].item()),
                "hold_duration_s": len(speed) * dt,
                "actual_speed_mean_mps": mean(speed), "actual_speed_p95_mps": percentile(speed, 95),
                "actual_speed_max_mps": max(speed, default=0.0),
                "yaw_rate_signed_mean_rps": mean(trace["yaw"]),
                "yaw_rate_abs_mean_rps": mean([abs(value) for value in trace["yaw"]]),
                "yaw_rate_p95_rps": percentile([abs(value) for value in trace["yaw"]], 95),
                "yaw_rate_max_rps": max([abs(value) for value in trace["yaw"]], default=0.0),
                "standing_height_m": mean(height),
                "pelvis_height_range_m": max(height) - min(height) if height else 0.0,
                "vertical_velocity_p95_mps": percentile(trace["vertical"], 95),
                "vertical_velocity_max_mps": max(trace["vertical"], default=0.0),
                "roll_abs_mean_rad": mean([abs(value) for value in trace["roll"]]),
                "roll_abs_p95_rad": percentile([abs(value) for value in trace["roll"]], 95),
                "roll_abs_max_rad": max([abs(value) for value in trace["roll"]], default=0.0),
                "pitch_abs_mean_rad": mean([abs(value) for value in trace["pitch"]]),
                "pitch_abs_p95_rad": percentile([abs(value) for value in trace["pitch"]], 95),
                "pitch_abs_max_rad": max([abs(value) for value in trace["pitch"]], default=0.0),
                "double_support_fraction": mean([float(state == 2) for state in support]),
                "single_support_fraction": mean([float(state == 1) for state in support]),
                "flight_fraction": mean([float(state == 0) for state in support]),
                "final_double_support": bool(support and support[-1] == 2),
                "support_switch_count": trace["switches"],
                "maximum_both_feet_airborne_s": trace["max_flight_run"] * dt,
                "maximum_single_support_s": trace["max_single_run"] * dt,
                "prolonged_single_support": trace["max_single_run"] * dt > 0.50,
                "left_foot_slip_mean_mps": mean(trace["left_slip"]),
                "right_foot_slip_mean_mps": mean(trace["right_slip"]),
                "ankle_torque_saturation_fraction": ankle_sat,
                "joint_velocity_saturation_fraction": velocity_sat,
                "action_magnitude_mean": mean(trace["action"]),
                "action_magnitude_p95": percentile(trace["action"], 95),
                "action_magnitude_max": max(trace["action"], default=0.0),
                "left_right_sagittal_asymmetry_rad": mean(trace["asymmetry"]),
            })

        write_csv(output / "episodes.csv", records)
        summary = {
            "candidate": args_cli.candidate, "checkpoint": str(checkpoint) if checkpoint else None,
            "primitive": args_cli.primitive, "episodes": n, "seed": args_cli.seed,
            "settle_success_rate": mean([float(row["settle_success"]) for row in records]),
            "standing_hold_success_rate": mean([float(row["standing_hold_success"]) for row in records]),
            "fall_rate": mean([float(row["fall"]) for row in records]),
            "actual_speed_mean_mps": mean([row["actual_speed_mean_mps"] for row in records]),
            "actual_speed_p95_mps": mean([row["actual_speed_p95_mps"] for row in records]),
            "actual_speed_max_mps": max(row["actual_speed_max_mps"] for row in records),
            "yaw_rate_abs_mean_rps": mean([row["yaw_rate_abs_mean_rps"] for row in records]),
            "yaw_rate_p95_rps": mean([row["yaw_rate_p95_rps"] for row in records]),
            "yaw_rate_max_rps": max(row["yaw_rate_max_rps"] for row in records),
            "pelvis_height_range_mean_m": mean([row["pelvis_height_range_m"] for row in records]),
            "vertical_velocity_p95_mps": mean([row["vertical_velocity_p95_mps"] for row in records]),
            "vertical_velocity_max_mps": max(row["vertical_velocity_max_mps"] for row in records),
            "roll_abs_p95_rad": mean([row["roll_abs_p95_rad"] for row in records]),
            "pitch_abs_p95_rad": mean([row["pitch_abs_p95_rad"] for row in records]),
            "double_support_fraction": mean([row["double_support_fraction"] for row in records]),
            "single_support_fraction": mean([row["single_support_fraction"] for row in records]),
            "flight_fraction": mean([row["flight_fraction"] for row in records]),
            "support_switch_count_mean": mean([row["support_switch_count"] for row in records]),
            "prolonged_single_support_rate": mean([float(row["prolonged_single_support"]) for row in records]),
            "dangerous_support_failure_rate": mean([
                float(row["maximum_both_feet_airborne_s"] > 0.10 or row["maximum_single_support_s"] > 0.50)
                for row in records
            ]),
            "final_double_support_rate": mean([float(row["final_double_support"]) for row in records]),
            "ankle_torque_saturation_failure_rate": mean([
                float(row["ankle_torque_saturation_fraction"] > 0.05) for row in records
            ]),
            "joint_velocity_saturation_failure_rate": mean([
                float(row["joint_velocity_saturation_fraction"] > 0.05) for row in records
            ]),
            "standing_height_m": mean([row["standing_height_m"] for row in records]),
            "action_magnitude_mean": mean([row["action_magnitude_mean"] for row in records]),
            "action_magnitude_p95": mean([row["action_magnitude_p95"] for row in records]),
            "action_magnitude_max": max(row["action_magnitude_max"] for row in records),
            "left_right_sagittal_asymmetry_rad": mean([
                row["left_right_sagittal_asymmetry_rad"] for row in records
            ]),
            "gate_thresholds": {
                "standing_hold_success_rate_min": 0.95, "fall_rate_max": 0.05,
                "actual_speed_mean_mps_max": 0.05, "actual_speed_p95_mps_max": 0.10,
                "pelvis_height_range_mean_m_max": 0.04, "flight_fraction_max": 0.001,
                "prolonged_single_support_rate_max": 0.05,
                "dangerous_support_failure_rate_max": 0.05,
                "final_double_support_rate_min": 0.95,
                "ankle_torque_saturation_failure_rate_max": 0.05,
                "joint_velocity_saturation_failure_rate_max": 0.05,
            },
        }
        thresholds = summary["gate_thresholds"]
        summary["gate_pass"] = (
            summary["standing_hold_success_rate"] >= thresholds["standing_hold_success_rate_min"]
            and summary["fall_rate"] <= thresholds["fall_rate_max"]
            and summary["actual_speed_mean_mps"] <= thresholds["actual_speed_mean_mps_max"]
            and summary["actual_speed_p95_mps"] <= thresholds["actual_speed_p95_mps_max"]
            and summary["pelvis_height_range_mean_m"] <= thresholds["pelvis_height_range_mean_m_max"]
            and summary["flight_fraction"] <= thresholds["flight_fraction_max"]
            and summary["prolonged_single_support_rate"] <= thresholds["prolonged_single_support_rate_max"]
            and summary["dangerous_support_failure_rate"] <= thresholds["dangerous_support_failure_rate_max"]
            and summary["final_double_support_rate"] >= thresholds["final_double_support_rate_min"]
            and summary["ankle_torque_saturation_failure_rate"] <= thresholds["ankle_torque_saturation_failure_rate_max"]
            and summary["joint_velocity_saturation_failure_rate"] <= thresholds["joint_velocity_saturation_failure_rate_max"]
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
