"""Evaluate only steady WALK maintenance for Stage 2W."""

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
PATH_MAX_M = 0.30
PATH_RATE_MPS = 0.08
HOLD_START_S = 2.5
HOLD_DURATION_S = 5.0

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", choices=("preflight", "pilot", "formal"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--seed", type=int, default=20260728)
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
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100), len(values) - 1)]


def assignments() -> list[float]:
    if args.mode == "preflight":
        return [speed for speed in SPEEDS for _ in range(5)]
    if args.mode == "pilot":
        return [speed for speed in SPEEDS for _ in range(10)]
    return [0.6] * 13 + [0.8] * 13 + [1.0] * 12 + [1.2] * 12


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
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        wrapped.reset()
        target = torch.tensor(speeds, device=env.device)
        term.target_speed[:] = target
        term.target_heading_w[:] = robot.data.heading_w.torch
        term.path_origin_xy[:] = robot.data.root_pos_w.torch[:, :2]
        target_heading = term.target_heading_w.clone()
        failed = torch.zeros(n, dtype=torch.bool, device=env.device)
        hold_origin = torch.zeros(n, 2, device=env.device)
        previous = torch.zeros(n, 37, device=env.device)
        traces = [
            {
                "vx": [], "heading": [], "cross": [], "slip": [], "support": [],
                "left_contact": [], "right_contact": [], "flight": [], "ankle_left": [],
                "ankle_right": [], "knee": [], "action_rate": [], "pelvis_z": [],
            }
            for _ in speeds
        ]
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
            if step < hold_start_step:
                previous = action.clone()
                continue
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            knee_ratio = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1.0e-6)
            )
            heading_error = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            displacement = robot.data.root_pos_w.torch[:, :2] - hold_origin
            normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            cross_track = (displacement * normal).sum(dim=1).abs()
            action_rate = torch.linalg.vector_norm(action - previous, dim=1) / dt
            previous = action.clone()
            for i, trace in enumerate(traces):
                if failed[i]:
                    continue
                trace["vx"].append(float(robot.data.root_lin_vel_b.torch[i, 0]))
                trace["heading"].append(abs(float(heading_error[i])))
                trace["cross"].append(float(cross_track[i]))
                trace["support"].append(int(contacts[i].sum()))
                trace["left_contact"].append(bool(contacts[i, 0]))
                trace["right_contact"].append(bool(contacts[i, 1]))
                trace["flight"].append(int(contacts[i].sum()) == 0)
                trace["slip"].append(max([float(slip[i, j]) for j in range(2) if contacts[i, j]] or [0.0]))
                trace["ankle_left"].append(float(ankle_ratio[i, 0]))
                trace["ankle_right"].append(float(ankle_ratio[i, 1]))
                trace["knee"].append(float(knee_ratio[i].max()))
                trace["action_rate"].append(float(action_rate[i]))
                trace["pelvis_z"].append(float(robot.data.root_pos_w.torch[i, 2]))
        records = []
        for i, trace in enumerate(traces):
            saturation = [
                max(left, right) >= 0.95
                for left, right in zip(trace["ankle_left"], trace["ankle_right"])
            ]
            dwell = max_dwell = 0
            for active in saturation:
                dwell = dwell + 1 if active else 0
                max_dwell = max(max_dwell, dwell)
            saturation_failure = max_dwell * dt >= 0.20
            support_switches = sum(a != b for a, b in zip(trace["support"], trace["support"][1:]))
            speed_errors = [abs(value - speeds[i]) for value in trace["vx"]]
            path_max = max(trace["cross"], default=999.0)
            path_rate = path_max / HOLD_DURATION_S
            heading_p95 = pct(trace["heading"], 95)
            slip_failure = avg(trace["slip"]) > 0.55
            flight_failure = avg(trace["flight"]) > 0.20
            path_pass = path_max <= PATH_MAX_M and path_rate <= PATH_RATE_MPS
            sustained = support_switches >= 2 and any(support < 2 for support in trace["support"])
            success = (
                not bool(failed[i])
                and sustained
                and avg(speed_errors) <= 0.20
                and heading_p95 <= 0.12
                and path_pass
                and not saturation_failure
                and not slip_failure
                and not flight_failure
            )
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
            primary = next((name for name, active in flags.items() if active), "")
            records.append(
                {
                    "episode": i,
                    "target_speed_mps": speeds[i],
                    "walk_success": success,
                    "fall": bool(failed[i]),
                    "sustained_walking": sustained,
                    "actual_speed_mean_mps": avg(trace["vx"]),
                    "speed_error_mean_mps": avg(speed_errors),
                    "speed_error_p95_mps": pct(speed_errors, 95),
                    "heading_error_p95_rad": heading_p95,
                    "path_drift_max_m": path_max,
                    "path_drift_rate_mps": path_rate,
                    "path_drift_pass": path_pass,
                    "foot_slip_mean_mps": avg(trace["slip"]),
                    "foot_slip_p95_mps": pct(trace["slip"], 95),
                    "dangerous_slip_failure": slip_failure,
                    "flight_fraction": avg(trace["flight"]),
                    "excessive_flight_failure": flight_failure,
                    "support_switches": support_switches,
                    "left_contact_fraction": avg(trace["left_contact"]),
                    "right_contact_fraction": avg(trace["right_contact"]),
                    "ankle_effort_p95_left": pct(trace["ankle_left"], 95),
                    "ankle_effort_p95_right": pct(trace["ankle_right"], 95),
                    "ankle_saturation_fraction": avg(saturation),
                    "ankle_saturation_max_dwell_s": max_dwell * dt,
                    "long_dwell_saturation_failure": saturation_failure,
                    "knee_velocity_utilization_max": max(trace["knee"], default=0.0),
                    "action_rate_p95": pct(trace["action_rate"], 95),
                    "pelvis_vertical_range_m": (
                        max(trace["pelvis_z"]) - min(trace["pelvis_z"]) if trace["pelvis_z"] else 0.0
                    ),
                    "primary_failure": primary,
                    "failure_flags": json.dumps(flags, sort_keys=True),
                }
            )
        per_speed = {}
        for speed in SPEEDS:
            group = [record for record in records if record["target_speed_mps"] == speed]
            per_speed[str(speed)] = {
                "episodes": len(group),
                "success_rate": avg([record["walk_success"] for record in group]),
                "fall_rate": avg([record["fall"] for record in group]),
                "speed_error_mean_mps": avg([record["speed_error_mean_mps"] for record in group]),
                "heading_error_p95_rad": pct([record["heading_error_p95_rad"] for record in group], 95),
                "path_drift_max_p95_m": pct([record["path_drift_max_m"] for record in group], 95),
                "path_drift_failure_rate": avg([not record["path_drift_pass"] for record in group]),
                "saturation_failure_rate": avg([record["long_dwell_saturation_failure"] for record in group]),
                "slip_failure_rate": avg([record["dangerous_slip_failure"] for record in group]),
                "flight_failure_rate": avg([record["excessive_flight_failure"] for record in group]),
            }
        overall = {
            "episodes": len(records),
            "success_rate": avg([record["walk_success"] for record in records]),
            "fall_rate": avg([record["fall"] for record in records]),
            "speed_error_mean_mps": avg([record["speed_error_mean_mps"] for record in records]),
            "heading_error_p95_rad": pct([record["heading_error_p95_rad"] for record in records], 95),
            "path_drift_failure_rate": avg([not record["path_drift_pass"] for record in records]),
            "saturation_failure_rate": avg([record["long_dwell_saturation_failure"] for record in records]),
            "slip_failure_rate": avg([record["dangerous_slip_failure"] for record in records]),
            "flight_failure_rate": avg([record["excessive_flight_failure"] for record in records]),
        }
        gate_checks = {
            "overall_success_ge_0_95": overall["success_rate"] >= 0.95,
            "each_speed_success_ge_0_90": all(value["success_rate"] >= 0.90 for value in per_speed.values()),
            "fall_le_0_02": overall["fall_rate"] <= 0.02,
            "heading_p95_le_0_12": overall["heading_error_p95_rad"] <= 0.12,
            "speed_error_mean_le_0_20": overall["speed_error_mean_mps"] <= 0.20,
            "saturation_failure_le_0_05": overall["saturation_failure_rate"] <= 0.05,
            "slip_failure_le_0_05": overall["slip_failure_rate"] <= 0.05,
            "flight_failure_le_0_05": overall["flight_failure_rate"] <= 0.05,
            "path_drift_gate": overall["path_drift_failure_rate"] <= 0.05,
        }
        summary = {
            "stage": "Stage 2W",
            "mode": args.mode,
            "label": args.label,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": sha(checkpoint),
            "seed": args.seed,
            "scope": "STEADY_WALK_ONLY",
            "stand_metrics_in_gate": False,
            "transition_metrics_in_gate": False,
            "path_gate_frozen": {"max_cross_track_m": PATH_MAX_M, "max_drift_rate_mps": PATH_RATE_MPS},
            "overall": overall,
            "per_speed": per_speed,
            "gate_checks": gate_checks,
            "gate_pass": all(gate_checks.values()),
            "failure_counts": dict(Counter(record["primary_failure"] or "none" for record in records)),
        }
        csv_path = output / f"{args.label}_episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (output / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
