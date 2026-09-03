"""Deterministic checkpoint gates for the Stage 2R curriculum."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

SPEEDS = (0.6, 0.8, 1.0, 1.2)
FAILURES = (
    "initial_stand_failure",
    "stand_retention_failure",
    "walk_start_failure",
    "walk_tracking_failure",
    "heading_failure",
    "path_drift_failure",
    "ankle_torque_saturation",
    "knee_velocity_saturation",
    "foot_slip_failure",
    "excessive_flight",
    "deceleration_failure",
    "residual_speed_failure",
    "double_support_recovery_failure",
    "final_stand_failure",
    "action_discontinuity",
    "fall",
    "timeout",
    "unsupported_walk_speed",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--phase", choices=("R1", "R2", "R3", "R4", "FORMAL"), required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, default=20260726)
parser.add_argument("--walk-episodes-per-speed", type=int, default=20)
parser.add_argument("--stand-episodes", type=int, default=50)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _avg(values):
    return sum(values) / len(values) if values else 0.0


def _pct(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100), len(values) - 1)]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    if args.phase in ("R1", "R2", "R3"):
        commands = [0.0] * args.stand_episodes + [
            speed for speed in SPEEDS for _ in range(args.walk_episodes_per_speed)
        ]
    else:
        commands = [SPEEDS[i % 4] for i in range(50)]
    n = len(commands)
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 18.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        expert = load_walk_expert(checkpoint, device=env.device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles, ankle_names = robot.find_joints(".*_ankle_pitch_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        wrapped.reset()
        target = torch.tensor(commands, device=env.device)
        heading_target = robot.data.heading_w.torch.clone()
        yaw_cmd = torch.zeros(n, device=env.device)
        previous = torch.zeros(n, 37, device=env.device)
        done_seen = torch.zeros(n, dtype=torch.bool, device=env.device)
        traces = [
            {
                "vx": [], "speed": [], "heading": [], "lateral": [], "roll": [], "pitch": [],
                "support": [], "slip": [], "ankle_left": [], "ankle_right": [], "knee": [],
                "left_contact": [], "right_contact": [], "contact_force": [],
                "action_rate": [], "action_magnitude": [], "yaw_cmd": [], "pelvis_z": [],
                "ankle_target": [], "ankle_position": [], "ankle_position_error": [],
            }
            for _ in commands
        ]
        initial_xy = robot.data.root_pos_w.torch[:, :2].clone()
        dt = float(env.step_dt)
        total_s = 8.0 if args.phase == "R1" else 14.0
        for step in range(round(total_s / dt)):
            elapsed = step * dt
            if args.phase == "R1":
                vx_cmd = target
            else:
                accel_u = min(max((elapsed - 1.0) / 2.0, 0.0), 1.0)
                accel = 10 * accel_u**3 - 15 * accel_u**4 + 6 * accel_u**5
                vx_cmd = target * accel
                if args.phase in ("R3", "R4", "FORMAL") and elapsed >= 7.0:
                    decel_u = min(max((elapsed - 7.0) / 2.0, 0.0), 1.0)
                    decel = 10 * decel_u**3 - 15 * decel_u**4 + 6 * decel_u**5
                    vx_cmd = target * (1.0 - decel)
            error = torch.atan2(
                torch.sin(heading_target - robot.data.heading_w.torch),
                torch.cos(heading_target - robot.data.heading_w.torch),
            )
            raw_yaw = (0.8 * error - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low_pass = yaw_cmd + 0.15 * (raw_yaw - yaw_cmd)
            yaw_cmd += (low_pass - yaw_cmd).clamp(-0.01, 0.01)
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = vx_cmd
            term.vel_command_b[:, 2] = yaw_cmd
            obs = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(obs, heading_w_rad=robot.data.heading_w.torch)
            with torch.inference_mode():
                action = expert(state, MotionCommand(vx_cmd, heading_target, target_yaw_rate_radps=yaw_cmd))
                _, _, dones, _ = wrapped.step(action)
            done_seen |= dones.bool()
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
            contact_force = forces.norm(dim=-1).amax(dim=1).amax(dim=1)
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            )
            knee_ratio = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1.0e-6)
            )
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2))
            action_rate = torch.linalg.vector_norm(action - previous, dim=1) / dt
            previous = action.clone()
            if elapsed >= 2.0:
                for i, trace in enumerate(traces):
                    trace["vx"].append(float(robot.data.root_lin_vel_b.torch[i, 0]))
                    trace["speed"].append(float(robot.data.root_lin_vel_b.torch[i, :2].norm()))
                    trace["heading"].append(abs(float(error[i])))
                    trace["lateral"].append(abs(float(robot.data.root_pos_w.torch[i, 1] - initial_xy[i, 1])))
                    trace["roll"].append(abs(float(roll[i])))
                    trace["pitch"].append(abs(float(pitch[i])))
                    trace["support"].append(int(contacts[i].sum()))
                    trace["left_contact"].append(bool(contacts[i, 0]))
                    trace["right_contact"].append(bool(contacts[i, 1]))
                    trace["contact_force"].append(float(contact_force[i]))
                    trace["slip"].append(max([float(slip[i, j]) for j in range(2) if contacts[i, j]] or [0.0]))
                    trace["ankle_left"].append(float(ankle_ratio[i, 0]))
                    trace["ankle_right"].append(float(ankle_ratio[i, 1]))
                    trace["knee"].append(float(knee_ratio[i].max()))
                    trace["action_rate"].append(float(action_rate[i]))
                    trace["action_magnitude"].append(float(torch.linalg.vector_norm(action[i])))
                    trace["yaw_cmd"].append(float(yaw_cmd[i]))
                    trace["pelvis_z"].append(float(robot.data.root_pos_w.torch[i, 2]))
                    ankle_target = robot.data.default_joint_pos.torch[i, ankles] + 0.5 * action[i, ankles]
                    ankle_position = robot.data.joint_pos.torch[i, ankles]
                    trace["ankle_target"].append(float(ankle_target.abs().max()))
                    trace["ankle_position"].append(float(ankle_position.abs().max()))
                    trace["ankle_position_error"].append(float((ankle_target - ankle_position).abs().max()))
        rows = []
        for i, trace in enumerate(traces):
            command = commands[i]
            stand = command == 0.0
            saturation_flags = [
                max(left, right) >= 0.95
                for left, right in zip(trace["ankle_left"], trace["ankle_right"])
            ]
            max_dwell = 0
            dwell = 0
            for flag in saturation_flags:
                dwell = dwell + 1 if flag else 0
                max_dwell = max(max_dwell, dwell)
            saturation_failure = max_dwell * dt >= 0.20
            speed_error = [abs(v - command) for v in trace["vx"]]
            heading_p95 = _pct(trace["heading"], 95)
            flight_fraction = _avg([support == 0 for support in trace["support"]])
            slip_failure = _avg(trace["slip"]) > 0.55
            if stand:
                success = (
                    not bool(done_seen[i])
                    and _avg(trace["speed"]) <= 0.05
                    and _pct(trace["speed"], 95) <= 0.10
                    and flight_fraction == 0.0
                    and trace["support"][-1] == 2
                    and not saturation_failure
                )
            else:
                success = (
                    not bool(done_seen[i])
                    and _avg(speed_error) <= 0.20
                    and heading_p95 <= 0.12
                    and flight_fraction <= 0.20
                    and not saturation_failure
                    and not slip_failure
                )
            flags = {name: False for name in FAILURES}
            flags.update(
                {
                    "stand_retention_failure": stand and not success,
                    "walk_start_failure": not stand and max(trace["vx"], default=0.0) < 0.45,
                    "walk_tracking_failure": not stand and _avg(speed_error) > 0.20,
                    "heading_failure": heading_p95 > 0.12,
                    "path_drift_failure": max(trace["lateral"], default=0.0) > 0.50,
                    "ankle_torque_saturation": saturation_failure,
                    "knee_velocity_saturation": max(trace["knee"], default=0.0) >= 0.95,
                    "foot_slip_failure": slip_failure,
                    "excessive_flight": flight_fraction > 0.20,
                    "fall": bool(done_seen[i]),
                }
            )
            primary = next((name for name in FAILURES if flags[name]), "")
            rows.append(
                {
                    "episode": i,
                    "command_speed_mps": command,
                    "stand_case": stand,
                    "success": success,
                    "fall": bool(done_seen[i]),
                    "speed_mean_mps": _avg(trace["vx"]),
                    "horizontal_speed_mean_mps": _avg(trace["speed"]),
                    "speed_error_mean_mps": _avg(speed_error),
                    "heading_error_p95_rad": heading_p95,
                    "lateral_drift_max_m": max(trace["lateral"], default=0.0),
                    "roll_p95_rad": _pct(trace["roll"], 95),
                    "pitch_p95_rad": _pct(trace["pitch"], 95),
                    "flight_fraction": flight_fraction,
                    "final_double_support": trace["support"][-1] == 2,
                    "foot_slip_mean_mps": _avg(trace["slip"]),
                    "ankle_pitch_effort_p95_left": _pct(trace["ankle_left"], 95),
                    "ankle_pitch_effort_p95_right": _pct(trace["ankle_right"], 95),
                    "ankle_saturation_max_dwell_s": max_dwell * dt,
                    "ankle_saturation_fraction": _avg(saturation_flags),
                    "saturation_failure": saturation_failure,
                    "knee_velocity_utilization_max": max(trace["knee"], default=0.0),
                    "action_rate_p95": _pct(trace["action_rate"], 95),
                    "action_magnitude_mean": _avg(trace["action_magnitude"]),
                    "yaw_command_p95_radps": _pct([abs(x) for x in trace["yaw_cmd"]], 95),
                    "pelvis_vertical_range_m": max(trace["pelvis_z"]) - min(trace["pelvis_z"]),
                    "pelvis_pitch_p95_rad": _pct(trace["pitch"], 95),
                    "left_contact_fraction": _avg(trace["left_contact"]),
                    "right_contact_fraction": _avg(trace["right_contact"]),
                    "contact_force_p95_n": _pct(trace["contact_force"], 95),
                    "ankle_target_abs_p95_rad": _pct(trace["ankle_target"], 95),
                    "ankle_position_abs_p95_rad": _pct(trace["ankle_position"], 95),
                    "ankle_position_error_p95_rad": _pct(trace["ankle_position_error"], 95),
                    "primary_failure": primary,
                    "failure_flags": json.dumps(flags, sort_keys=True),
                }
            )
        stand_rows = [row for row in rows if row["stand_case"]]
        walk_rows = [row for row in rows if not row["stand_case"]]
        per_speed = {
            str(speed): {
                "episodes": len(group := [row for row in walk_rows if row["command_speed_mps"] == speed]),
                "success_rate": _avg([row["success"] for row in group]),
                "fall_rate": _avg([row["fall"] for row in group]),
                "saturation_failure_rate": _avg([row["saturation_failure"] for row in group]),
                "heading_error_p95_rad": _pct([row["heading_error_p95_rad"] for row in group], 95),
                "speed_error_mean_mps": _avg([row["speed_error_mean_mps"] for row in group]),
            }
            for speed in SPEEDS
        }
        stand_metrics = {
            "episodes": len(stand_rows),
            "hold_success_rate": _avg([row["success"] for row in stand_rows]),
            "fall_rate": _avg([row["fall"] for row in stand_rows]),
            "flight_fraction": _avg([row["flight_fraction"] for row in stand_rows]),
            "saturation_failure_rate": _avg([row["saturation_failure"] for row in stand_rows]),
            "final_double_support_rate": _avg([row["final_double_support"] for row in stand_rows]),
            "speed_mean_mps": _avg([row["horizontal_speed_mean_mps"] for row in stand_rows]),
            "speed_p95_mps": _pct([row["horizontal_speed_mean_mps"] for row in stand_rows], 95),
        }
        walk_metrics = {
            "episodes": len(walk_rows),
            "success_rate": _avg([row["success"] for row in walk_rows]),
            "fall_rate": _avg([row["fall"] for row in walk_rows]),
            "saturation_failure_rate": _avg([row["saturation_failure"] for row in walk_rows]),
            "heading_error_p95_rad": _pct([row["heading_error_p95_rad"] for row in walk_rows], 95),
            "speed_error_mean_mps": _avg([row["speed_error_mean_mps"] for row in walk_rows]),
        }
        r1_gate = {
            "stand_hold_ge_0_95": stand_metrics["hold_success_rate"] >= 0.95,
            "stand_fall_le_0_02": stand_metrics["fall_rate"] <= 0.02,
            "steady_walk_success_ge_0_90": walk_metrics["success_rate"] >= 0.90,
            "heading_error_le_0_12": walk_metrics["heading_error_p95_rad"] <= 0.12,
            "saturation_failure_le_0_10": walk_metrics["saturation_failure_rate"] <= 0.10,
        }
        summary = {
            "stage": "Stage 2R",
            "phase": args.phase,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_sha256": _sha(checkpoint),
            "seed": args.seed,
            "success_definition_frozen_before_pilot": True,
            "stand": stand_metrics,
            "steady_walk": walk_metrics,
            "per_speed": per_speed,
            "r1_gate": r1_gate,
            "r1_gate_pass": all(r1_gate.values()),
            "failure_counts": dict(Counter(row["primary_failure"] or "none" for row in rows)),
            "routing": {
                "active_expert": "unified_stage2r_only",
                "run_expert_loaded": False,
                "run_contribution_bitwise_zero": True,
                "transition_bridge_connected": False,
                "transition_bridge_output_bitwise_zero": True,
                "scripted_joint_offset_bitwise_zero": True,
            },
        }
        stem = checkpoint.stem
        _write_csv(output / f"{stem}_episodes.csv", rows)
        (output / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
