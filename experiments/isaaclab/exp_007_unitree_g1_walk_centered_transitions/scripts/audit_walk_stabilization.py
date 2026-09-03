"""Stage 2W-B heading diagnosis and fixed-policy controller evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

SPEEDS = (0.6, 0.8, 1.0, 1.2)
STAGE2W_FAILURE_EPISODES = (6, 22, 26, 34, 35, 43, 49)
PATH_MAX_M = 0.30
PATH_RATE_MPS = 0.08
HOLD_START_S = 2.5
HOLD_DURATION_S = 5.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", choices=("timeline", "controller", "formal-full", "formal-low"), required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--heading-mode", choices=("ZeroYaw", "FixedTarget"), required=True)
parser.add_argument("--k-heading", type=float, default=0.8)
parser.add_argument("--k-yaw-rate", type=float, default=0.10)
parser.add_argument("--yaw-rate-limit", type=float, default=0.30)
parser.add_argument("--low-pass-alpha", type=float, default=0.15)
parser.add_argument("--slew-limit", type=float, default=0.01)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def avg(values):
    return sum(values) / len(values) if values else 0.0


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def assignments() -> list[float]:
    if args.mode == "controller":
        return [speed for speed in SPEEDS for _ in range(8)]
    if args.mode == "formal-low":
        return [0.6] * 25 + [0.8] * 25
    return [0.6] * 13 + [0.8] * 13 + [1.0] * 12 + [1.2] * 12


def runs(values: list[bool], dt: float) -> list[float]:
    result, current = [], 0
    for active in values:
        if active:
            current += 1
        elif current:
            result.append(current * dt)
            current = 0
    if current:
        result.append(current * dt)
    return result


def reversal_frequency(values: list[float], dt: float) -> float:
    signs = [1 if value > 0.01 else -1 if value < -0.01 else 0 for value in values]
    filtered = [value for value in signs if value]
    reversals = sum(a != b for a, b in zip(filtered, filtered[1:]))
    return reversals / max(len(values) * dt, dt)


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    speeds = assignments()
    n = len(speeds)
    task = "Isaac-Velocity-Flat-G1-IndependentWalk-Eval-v0"
    cfg, agent = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 10.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        term.heading_mode = args.heading_mode
        term.k_heading = args.k_heading
        term.k_yaw_rate = args.k_yaw_rate
        term.yaw_limit = args.yaw_rate_limit
        term.low_pass_alpha = args.low_pass_alpha
        term.slew_limit = args.slew_limit
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        action_indices = {
            name: robot.joint_names.index(name)
            for name in (
                "left_hip_yaw_joint",
                "right_hip_yaw_joint",
                "left_hip_roll_joint",
                "right_hip_roll_joint",
            )
        }
        wrapped.reset()
        target = torch.tensor(speeds, device=env.device)
        term.target_speed[:] = target
        term.target_heading_w[:] = robot.data.heading_w.torch
        term.path_origin_xy[:] = robot.data.root_pos_w.torch[:, :2]
        target_heading = term.target_heading_w.clone()
        failed = torch.zeros(n, dtype=torch.bool, device=env.device)
        hold_origin = torch.zeros(n, 2, device=env.device)
        previous_action = torch.zeros(n, 37, device=env.device)
        traces = [
            {
                "vx": [], "heading": [], "yaw": [], "yaw_rate": [], "raw_yaw_cmd": [],
                "filtered_yaw_cmd": [], "controller_saturated": [], "cross": [], "lateral_v": [],
                "slip_left": [], "slip_right": [], "left_contact": [], "right_contact": [],
                "ankle_left": [], "ankle_right": [], "action_rate": [], "hip_yaw_asym": [],
                "hip_roll_asym": [], "flight": [],
            }
            for _ in speeds
        ]
        timeline_rows = []
        stance_steps = torch.zeros(n, 2, dtype=torch.long, device=env.device)
        dt = float(env.step_dt)
        hold_start_step = round(HOLD_START_S / dt)
        total_steps = round((HOLD_START_S + HOLD_DURATION_S) / dt)
        for step in range(total_steps):
            if step == hold_start_step:
                hold_origin[:] = robot.data.root_pos_w.torch[:, :2]
            obs = wrapped.get_observations()["policy"]
            command = term.vel_command_b.clone()
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(
                    state,
                    MotionCommand(command[:, 0], target_heading, target_yaw_rate_radps=command[:, 2]),
                )
                _, _, dones, _ = wrapped.step(action)
            failed |= dones.bool()
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            stance_steps = torch.where(contacts, stance_steps + 1, torch.zeros_like(stance_steps))
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            heading_error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            displacement = robot.data.root_pos_w.torch[:, :2] - (
                hold_origin if step >= hold_start_step else term.path_origin_xy
            )
            normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            cross_track = (displacement * normal).sum(dim=1)
            action_rate = torch.linalg.vector_norm(action - previous_action, dim=1) / dt
            previous_action = action.clone()
            if args.mode == "timeline":
                for i in STAGE2W_FAILURE_EPISODES:
                    support = (
                        "DOUBLE" if bool(contacts[i].all()) else
                        "LEFT" if bool(contacts[i, 0]) else
                        "RIGHT" if bool(contacts[i, 1]) else "FLIGHT"
                    )
                    timeline_rows.append(
                        {
                            "episode": i,
                            "time_s": step * dt,
                            "target_speed_mps": speeds[i],
                            "actual_forward_speed_mps": float(robot.data.root_lin_vel_b.torch[i, 0]),
                            "heading_error_rad": float(heading_error[i]),
                            "yaw_rad": float(robot.data.heading_w.torch[i]),
                            "yaw_rate_radps": float(robot.data.root_ang_vel_b.torch[i, 2]),
                            "generated_yaw_rate_command_radps": float(term.raw_yaw_command[i]),
                            "filtered_yaw_rate_command_radps": float(term.filtered_yaw_command[i]),
                            "heading_controller_saturated": bool(term.heading_controller_saturated[i]),
                            "yaw_rate_command_reversal": (
                                len(traces[i]["filtered_yaw_cmd"]) > 0
                                and abs(float(term.filtered_yaw_command[i])) > 0.01
                                and abs(traces[i]["filtered_yaw_cmd"][-1]) > 0.01
                                and float(term.filtered_yaw_command[i]) * traces[i]["filtered_yaw_cmd"][-1] < 0
                            ),
                            "support_foot": support,
                            "left_contact": bool(contacts[i, 0]),
                            "right_contact": bool(contacts[i, 1]),
                            "left_stance_duration_s": float(stance_steps[i, 0]) * dt,
                            "right_stance_duration_s": float(stance_steps[i, 1]) * dt,
                            "left_foot_slip_mps": float(slip[i, 0]) if bool(contacts[i, 0]) else 0.0,
                            "right_foot_slip_mps": float(slip[i, 1]) if bool(contacts[i, 1]) else 0.0,
                            "left_ankle_pitch_effort_ratio": float(ankle_ratio[i, 0]),
                            "right_ankle_pitch_effort_ratio": float(ankle_ratio[i, 1]),
                            "left_hip_yaw_action": float(action[i, action_indices["left_hip_yaw_joint"]]),
                            "right_hip_yaw_action": float(action[i, action_indices["right_hip_yaw_joint"]]),
                            "left_hip_roll_action": float(action[i, action_indices["left_hip_roll_joint"]]),
                            "right_hip_roll_action": float(action[i, action_indices["right_hip_roll_joint"]]),
                            "lateral_body_velocity_mps": float(robot.data.root_lin_vel_b.torch[i, 1]),
                            "action_rate_l2_per_s": float(action_rate[i]),
                            "path_lateral_error_m": float(cross_track[i]),
                        }
                    )
            if step < hold_start_step:
                continue
            for i, trace in enumerate(traces):
                if failed[i]:
                    continue
                trace["vx"].append(float(robot.data.root_lin_vel_b.torch[i, 0]))
                trace["heading"].append(abs(float(heading_error[i])))
                trace["yaw"].append(float(robot.data.heading_w.torch[i]))
                trace["yaw_rate"].append(float(robot.data.root_ang_vel_b.torch[i, 2]))
                trace["raw_yaw_cmd"].append(float(term.raw_yaw_command[i]))
                trace["filtered_yaw_cmd"].append(float(term.filtered_yaw_command[i]))
                trace["controller_saturated"].append(bool(term.heading_controller_saturated[i]))
                trace["cross"].append(abs(float(cross_track[i])))
                trace["lateral_v"].append(float(robot.data.root_lin_vel_b.torch[i, 1]))
                trace["left_contact"].append(bool(contacts[i, 0]))
                trace["right_contact"].append(bool(contacts[i, 1]))
                trace["slip_left"].append(float(slip[i, 0]) if bool(contacts[i, 0]) else 0.0)
                trace["slip_right"].append(float(slip[i, 1]) if bool(contacts[i, 1]) else 0.0)
                trace["ankle_left"].append(float(ankle_ratio[i, 0]))
                trace["ankle_right"].append(float(ankle_ratio[i, 1]))
                trace["action_rate"].append(float(action_rate[i]))
                trace["hip_yaw_asym"].append(abs(
                    float(action[i, action_indices["left_hip_yaw_joint"]])
                    + float(action[i, action_indices["right_hip_yaw_joint"]])
                ))
                trace["hip_roll_asym"].append(abs(
                    float(action[i, action_indices["left_hip_roll_joint"]])
                    + float(action[i, action_indices["right_hip_roll_joint"]])
                ))
                trace["flight"].append(not bool(contacts[i].any()))

        records = []
        for i, trace in enumerate(traces):
            saturation = [
                max(left, right) >= 0.95
                for left, right in zip(trace["ankle_left"], trace["ankle_right"])
            ]
            saturation_dwell = max(runs(saturation, dt), default=0.0)
            left_stances = runs(trace["left_contact"], dt)
            right_stances = runs(trace["right_contact"], dt)
            speed_errors = [abs(value - speeds[i]) for value in trace["vx"]]
            heading_p95 = pct(trace["heading"], 95)
            path_max = max(trace["cross"], default=999.0)
            path_pass = path_max <= PATH_MAX_M and path_max / HOLD_DURATION_S <= PATH_RATE_MPS
            support_switches = sum(
                (left_a, right_a) != (left_b, right_b)
                for left_a, right_a, left_b, right_b in zip(
                    trace["left_contact"], trace["right_contact"],
                    trace["left_contact"][1:], trace["right_contact"][1:],
                )
            )
            sustained = support_switches >= 2 and any(
                not (left and right) for left, right in zip(trace["left_contact"], trace["right_contact"])
            )
            slip_values = [max(left, right) for left, right in zip(trace["slip_left"], trace["slip_right"])]
            slip_failure = avg(slip_values) > 0.55
            flight_failure = avg(trace["flight"]) > 0.20
            saturation_failure = saturation_dwell >= 0.20
            flags = {
                "fall": bool(failed[i]),
                "walk_not_sustained": not sustained,
                "speed_tracking_failure": avg(speed_errors) > 0.20,
                "heading_failure": heading_p95 > 0.12,
                "path_drift_failure": not path_pass,
                "ankle_torque_saturation": saturation_failure,
                "dangerous_slip_failure": slip_failure,
                "excessive_flight_failure": flight_failure,
            }
            success = not any(flags.values())
            records.append(
                {
                    "episode": i,
                    "target_speed_mps": speeds[i],
                    "walk_success": success,
                    "fall": bool(failed[i]),
                    "actual_speed_mean_mps": avg(trace["vx"]),
                    "speed_error_mean_mps": avg(speed_errors),
                    "heading_error_p95_rad": heading_p95,
                    "path_drift_max_m": path_max,
                    "path_drift_failure": not path_pass,
                    "yaw_reversal_frequency_hz": reversal_frequency(trace["filtered_yaw_cmd"], dt),
                    "yaw_command_saturation_fraction": avg(trace["controller_saturated"]),
                    "yaw_command_abs_p95_radps": pct([abs(value) for value in trace["filtered_yaw_cmd"]], 95),
                    "action_rate_p95": pct(trace["action_rate"], 95),
                    "left_contact_fraction": avg(trace["left_contact"]),
                    "right_contact_fraction": avg(trace["right_contact"]),
                    "stance_duration_asymmetry_s": abs(avg(left_stances) - avg(right_stances)),
                    "left_slip_mean_mps": avg(trace["slip_left"]),
                    "right_slip_mean_mps": avg(trace["slip_right"]),
                    "slip_asymmetry_mps": abs(avg(trace["slip_left"]) - avg(trace["slip_right"])),
                    "dangerous_slip_failure": slip_failure,
                    "flight_fraction": avg(trace["flight"]),
                    "excessive_flight_failure": flight_failure,
                    "ankle_effort_p95_left": pct(trace["ankle_left"], 95),
                    "ankle_effort_p95_right": pct(trace["ankle_right"], 95),
                    "ankle_saturation_fraction": avg(saturation),
                    "ankle_saturation_max_dwell_s": saturation_dwell,
                    "long_dwell_saturation_failure": saturation_failure,
                    "lateral_velocity_abs_mean_mps": avg([abs(value) for value in trace["lateral_v"]]),
                    "hip_yaw_action_asymmetry_mean": avg(trace["hip_yaw_asym"]),
                    "hip_roll_action_asymmetry_mean": avg(trace["hip_roll_asym"]),
                    "primary_failure": next((name for name, active in flags.items() if active), ""),
                    "failure_flags": json.dumps(flags, sort_keys=True),
                }
            )

        per_speed = {}
        for speed in sorted(set(speeds)):
            group = [record for record in records if record["target_speed_mps"] == speed]
            per_speed[str(speed)] = {
                "episodes": len(group),
                "success_rate": avg([record["walk_success"] for record in group]),
                "heading_error_p95_rad": pct([record["heading_error_p95_rad"] for record in group], 95),
                "speed_error_mean_mps": avg([record["speed_error_mean_mps"] for record in group]),
                "path_drift_failure_rate": avg([record["path_drift_failure"] for record in group]),
                "fall_rate": avg([record["fall"] for record in group]),
                "slip_failure_rate": avg([record["dangerous_slip_failure"] for record in group]),
                "flight_failure_rate": avg([record["excessive_flight_failure"] for record in group]),
                "saturation_failure_rate": avg([record["long_dwell_saturation_failure"] for record in group]),
                "yaw_reversal_frequency_mean_hz": avg([record["yaw_reversal_frequency_hz"] for record in group]),
                "action_rate_p95": pct([record["action_rate_p95"] for record in group], 95),
            }
        overall = {
            "episodes": n,
            "success_rate": avg([record["walk_success"] for record in records]),
            "fall_rate": avg([record["fall"] for record in records]),
            "heading_error_p95_rad": pct([record["heading_error_p95_rad"] for record in records], 95),
            "speed_error_mean_mps": avg([record["speed_error_mean_mps"] for record in records]),
            "path_drift_failure_rate": avg([record["path_drift_failure"] for record in records]),
            "slip_failure_rate": avg([record["dangerous_slip_failure"] for record in records]),
            "flight_failure_rate": avg([record["excessive_flight_failure"] for record in records]),
            "saturation_failure_rate": avg([record["long_dwell_saturation_failure"] for record in records]),
            "yaw_reversal_frequency_mean_hz": avg([record["yaw_reversal_frequency_hz"] for record in records]),
            "yaw_command_saturation_fraction_mean": avg(
                [record["yaw_command_saturation_fraction"] for record in records]
            ),
            "action_rate_p95": pct([record["action_rate_p95"] for record in records], 95),
        }
        gate_checks = {
            "overall_success_ge_0_95": overall["success_rate"] >= 0.95,
            "each_speed_success_ge_0_90": all(value["success_rate"] >= 0.90 for value in per_speed.values()),
            "fall_le_0_02": overall["fall_rate"] <= 0.02,
            "heading_p95_le_0_12": overall["heading_error_p95_rad"] <= 0.12,
            "speed_error_mean_le_0_20": overall["speed_error_mean_mps"] <= 0.20,
            "path_drift_failure_le_0_05": overall["path_drift_failure_rate"] <= 0.05,
            "saturation_failure_le_0_05": overall["saturation_failure_rate"] <= 0.05,
            "slip_failure_le_0_05": overall["slip_failure_rate"] <= 0.05,
            "flight_failure_le_0_05": overall["flight_failure_rate"] <= 0.05,
        }
        summary = {
            "stage": "Stage 2W-B",
            "mode": args.mode,
            "label": args.label,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": sha(checkpoint),
            "seed": args.seed,
            "controller": {
                "mode": args.heading_mode,
                "k_heading": args.k_heading,
                "k_yaw_rate": args.k_yaw_rate,
                "yaw_rate_limit": args.yaw_rate_limit,
                "low_pass_alpha": args.low_pass_alpha,
                "slew_limit": args.slew_limit,
            },
            "overall": overall,
            "per_speed": per_speed,
            "gate_checks": gate_checks,
            "gate_pass": all(gate_checks.values()),
            "failure_counts": dict(Counter(record["primary_failure"] or "none" for record in records)),
        }
        with (output / f"{args.label}_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if args.mode == "timeline":
            with (output / "heading_failure_timelines.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(timeline_rows[0]))
                writer.writeheader()
                writer.writerows(timeline_rows)
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
