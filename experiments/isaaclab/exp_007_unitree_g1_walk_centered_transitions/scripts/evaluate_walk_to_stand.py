"""Evaluate the Stage 4 WALK_TO_STAND parameter-free direct hard switch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
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
FAILURES = (
    "source_walk_failure",
    "transition_start_failure",
    "deceleration_failure",
    "gait_termination_failure",
    "residual_speed_failure",
    "reverse_motion_failure",
    "heading_failure",
    "path_drift_failure",
    "double_support_failure",
    "false_completion",
    "action_discontinuity_at_entry",
    "action_discontinuity_at_exit",
    "ankle_saturation_failure",
    "knee_saturation_failure",
    "dangerous_slip",
    "excessive_flight",
    "transition_timeout",
    "stand_takeover_failure",
    "stand_hold_failure",
    "fall",
    "torso_contact",
    "routing_failure",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stand-checkpoint", required=True)
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--stand-to-walk-checkpoint", required=True)
parser.add_argument("--transition-checkpoint")
parser.add_argument("--mode", choices=("baseline", "pilot", "formal", "smoke"), required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, required=True)
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


def runs(values, dt):
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


def minimum_jerk(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def candidate_assignments():
    if args.mode == "smoke":
        return list(SPEEDS) * 2, {speed: 1 for speed in SPEEDS}
    if args.mode in ("baseline", "pilot"):
        return [speed for speed in SPEEDS for _ in range(10)], {speed: 8 for speed in SPEEDS}
    return [speed for speed in SPEEDS for _ in range(16)], {0.6: 13, 0.8: 13, 1.0: 12, 1.2: 12}


def main() -> None:
    stand_path = Path(args.stand_checkpoint).resolve(strict=True)
    walk_path = Path(args.walk_checkpoint).resolve(strict=True)
    start_path = Path(args.stand_to_walk_checkpoint).resolve(strict=True)
    transition_path = (
        Path(args.transition_checkpoint).resolve(strict=True)
        if args.transition_checkpoint
        else None
    )
    if args.mode == "pilot" and transition_path is None:
        raise ValueError("pilot mode requires --transition-checkpoint")
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "direct_switch_baseline.json"
    speeds, desired = candidate_assignments()
    n = len(speeds)
    rng = random.Random(args.seed)
    stand_holds = [rng.uniform(0.8, 1.8) for _ in speeds]
    walk_holds = [rng.uniform(2.0, 3.5) for _ in speeds]
    ramp_durations = [rng.uniform(1.3, 1.7) for _ in speeds]

    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 22.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        stand = load_walk_expert(stand_path, device=env.device)
        walk = load_walk_expert(walk_path, device=env.device)
        stand_to_walk = load_walk_expert(start_path, device=env.device)
        transition = (
            load_walk_expert(transition_path, device=env.device)
            if transition_path
            else None
        )
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        torso_ids, _ = robot.find_bodies("torso_link")
        sensor_torso = [sensor.body_names.index(robot.body_names[index]) for index in torso_ids]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        wrapped.reset()
        device = env.device
        dt = float(env.step_dt)
        target_speed = torch.tensor(speeds, device=device)
        target_heading = robot.data.heading_w.torch.clone()
        stop_origin = robot.data.root_pos_w.torch[:, :2].clone()
        final_origin = stop_origin.clone()
        # 0 initial STAND, 1 STAND_TO_WALK, 2 WALK hold, 3 direct switch,
        # 4 STAND takeover hold, 5 terminal.
        phase = torch.zeros(n, dtype=torch.long, device=device)
        phase_elapsed = torch.zeros(n, device=device)
        settle_streak = torch.zeros(n, device=device)
        source_completion_streak = torch.zeros(n, device=device)
        source_walk_streak = torch.zeros(n, device=device)
        stop_completion_streak = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        no_switch_elapsed = torch.zeros(n, device=device)
        filtered_yaw = torch.zeros(n, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        fallen = torch.zeros(n, dtype=torch.bool, device=device)
        source_fallen = torch.zeros(n, dtype=torch.bool, device=device)
        torso_seen = torch.zeros(n, dtype=torch.bool, device=device)
        source_generation_ok = torch.zeros(n, dtype=torch.bool, device=device)
        walk_hold_ok = torch.zeros(n, dtype=torch.bool, device=device)
        completed = torch.zeros(n, dtype=torch.bool, device=device)
        takeover_ok = torch.zeros(n, dtype=torch.bool, device=device)
        previous_action = torch.zeros(n, 37, device=device)
        last_walk_action = torch.zeros_like(previous_action)
        last_edge_action = torch.zeros_like(previous_action)
        entry_jump = torch.zeros(n, device=device)
        exit_jump = torch.zeros(n, device=device)
        entry_joint_jump = torch.zeros(n, device=device)
        exit_joint_jump = torch.zeros(n, device=device)
        traces = [
            {key: [] for key in (
                "phase", "vx", "horizontal", "vertical", "heading", "cross", "longitudinal",
                "roll", "pitch", "support", "slip", "ankle", "knee", "flight",
                "action_rate", "action_step", "torque_spike",
            )}
            for _ in speeds
        ]
        stop_support = torch.zeros(n, dtype=torch.long, device=device)
        stop_time = torch.zeros(n, device=device)
        stop_vx = torch.zeros(n, device=device)
        timeline_rows = []
        max_steps = round(21.5 / dt)
        for step in range(max_steps):
            active = ~finished
            controller_phase = phase.clone()
            command_vx = torch.zeros(n, device=device)
            source_edge = controller_phase == 1
            source_walk = controller_phase == 2
            stop_edge = controller_phase == 3
            ramp = torch.tensor(ramp_durations, device=device)
            command_vx[source_edge] = target_speed[source_edge] * minimum_jerk(
                phase_elapsed[source_edge] / ramp[source_edge]
            )
            command_vx[source_walk] = target_speed[source_walk]
            if transition is not None:
                command_vx[stop_edge] = target_speed[stop_edge] * (
                    1.0 - minimum_jerk(phase_elapsed[stop_edge] / 1.6)
                )
            heading_error_signed = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw_yaw = (
                0.8 * heading_error_signed - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]
            ).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw_yaw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            filtered_yaw[controller_phase == 0] = 0.0
            if transition is None:
                filtered_yaw[controller_phase >= 3] = 0.0
            else:
                filtered_yaw[controller_phase >= 4] = 0.0
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = command_vx
            term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(
                legacy, heading_w_rad=robot.data.heading_w.torch
            )
            zero = torch.zeros(n, device=device)
            with torch.inference_mode():
                stand_action = stand(
                    state, MotionCommand(zero, target_heading, target_yaw_rate_radps=zero)
                )
                start_action = stand_to_walk(
                    state,
                    MotionCommand(
                        command_vx,
                        target_heading,
                        target_yaw_rate_radps=filtered_yaw,
                    ),
                )
                walk_action = walk(
                    state,
                    MotionCommand(
                        target_speed,
                        target_heading,
                        target_yaw_rate_radps=filtered_yaw,
                    ),
                )
                transition_action = (
                    transition(
                        state,
                        MotionCommand(
                            command_vx,
                            target_heading,
                            target_yaw_rate_radps=filtered_yaw,
                        ),
                    )
                    if transition is not None
                    else stand_action
                )
                action = torch.where(
                    (controller_phase == 0).unsqueeze(1),
                    stand_action,
                    torch.where(
                        (controller_phase == 1).unsqueeze(1),
                        start_action,
                        torch.where(
                            (controller_phase == 2).unsqueeze(1),
                            walk_action,
                            torch.where(
                                (controller_phase == 3).unsqueeze(1),
                                transition_action,
                                stand_action,
                            ),
                        ),
                    ),
                )
                _, _, dones, _ = wrapped.step(action)

            forces = sensor.data.net_forces_w_history.torch
            contacts = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            torso_contact = forces[:, :, sensor_torso, :].norm(dim=-1).amax(dim=(1, 2)) > 5.0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            contact_slip = torch.where(contacts, slip, torch.zeros_like(slip)).amax(dim=1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            knee_ratio = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            torque_ratio = (
                robot.data.applied_torque.torch.abs()
                / robot.data.joint_effort_limits.torch.abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2])
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            heading = heading_error_signed.abs()
            displacement = robot.data.root_pos_w.torch[:, :2] - stop_origin
            forward = torch.stack((torch.cos(target_heading), torch.sin(target_heading)), dim=1)
            normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            longitudinal = (displacement * forward).sum(dim=1)
            cross = (displacement * normal).sum(dim=1).abs()
            action_delta = action - previous_action
            action_step = torch.linalg.vector_norm(action_delta, dim=1)
            action_rate = action_step / dt
            just_entry = (controller_phase == 3) & (phase_elapsed <= dt * 1.5)
            entry_jump[just_entry] = torch.linalg.vector_norm(
                action[just_entry] - last_walk_action[just_entry], dim=1
            )
            entry_joint_jump[just_entry] = (
                action[just_entry] - last_walk_action[just_entry]
            ).abs().amax(dim=1)
            just_exit = (controller_phase == 4) & (phase_elapsed <= dt * 1.5)
            exit_jump[just_exit] = torch.linalg.vector_norm(
                action[just_exit] - last_edge_action[just_exit], dim=1
            )
            exit_joint_jump[just_exit] = (
                action[just_exit] - last_edge_action[just_exit]
            ).abs().amax(dim=1)
            last_walk_action[:] = torch.where(
                (controller_phase == 2).unsqueeze(1), action, last_walk_action
            )
            last_edge_action[:] = torch.where(
                (controller_phase == 3).unsqueeze(1), action, last_edge_action
            )
            previous_action[:] = action

            stand_safe = (
                (horizontal <= 0.08)
                & (vertical <= 0.05)
                & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10)
                & contacts.all(dim=1)
            )
            settle_streak[:] = torch.where(
                (controller_phase == 0) & stand_safe,
                settle_streak + dt,
                torch.zeros_like(settle_streak),
            )
            to_source_edge = (controller_phase == 0) & (settle_streak >= 0.4) & (
                phase_elapsed >= torch.tensor(stand_holds, device=device)
            )
            phase[to_source_edge] = 1
            phase_elapsed[to_source_edge] = 0.0
            target_heading[to_source_edge] = robot.data.heading_w.torch[to_source_edge]
            support_switches[to_source_edge] = 0
            previous_support[to_source_edge] = support[to_source_edge]

            switched_source = (
                (controller_phase == 1) & (support != previous_support) & (support != 0)
            )
            support_switches += switched_source.long()
            previous_support[:] = torch.where(
                controller_phase == 1, support, previous_support
            )
            source_acquired = (
                (controller_phase == 1)
                & (vx >= 0.75 * target_speed)
                & ((vx - target_speed).abs() <= 0.20)
                & (heading <= 0.12)
                & (roll.abs() <= 0.20)
                & (pitch.abs() <= 0.20)
                & (support_switches >= 2)
            )
            source_completion_streak[:] = torch.where(
                source_acquired,
                source_completion_streak + dt,
                torch.zeros_like(source_completion_streak),
            )
            to_walk = (controller_phase == 1) & (source_completion_streak >= 0.4)
            phase[to_walk] = 2
            phase_elapsed[to_walk] = 0.0
            source_generation_ok[to_walk] = True
            source_walk_streak[to_walk] = 0.0

            walk_good = (
                (controller_phase == 2)
                & ((vx - target_speed).abs() <= 0.20)
                & (heading <= 0.12)
                & (~dones.bool())
                & (~torso_contact)
            )
            source_walk_streak[:] = torch.where(
                walk_good,
                source_walk_streak + dt,
                torch.zeros_like(source_walk_streak),
            )
            to_stop = (controller_phase == 2) & (
                source_walk_streak >= torch.tensor(walk_holds, device=device)
            )
            phase[to_stop] = 3
            phase_elapsed[to_stop] = 0.0
            walk_hold_ok[to_stop] = True
            stop_origin[to_stop] = robot.data.root_pos_w.torch[to_stop, :2]
            stop_support[to_stop] = support[to_stop]
            stop_time[to_stop] = step * dt
            stop_vx[to_stop] = vx[to_stop]
            no_switch_elapsed[to_stop] = 0.0
            previous_support[to_stop] = support[to_stop]

            switched_stop = (
                (controller_phase == 3) & (support != previous_support) & (support != 0)
            )
            no_switch_elapsed[:] = torch.where(
                controller_phase == 3,
                torch.where(
                    switched_stop,
                    torch.zeros_like(no_switch_elapsed),
                    no_switch_elapsed + dt,
                ),
                no_switch_elapsed,
            )
            previous_support[:] = torch.where(
                controller_phase == 3, support, previous_support
            )
            completion_good = (
                (controller_phase == 3)
                & (horizontal <= 0.08)
                & (vertical <= 0.05)
                & (heading <= 0.12)
                & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10)
                & contacts.all(dim=1)
                & (no_switch_elapsed >= 0.4)
                & (~dones.bool())
                & (~torso_contact)
            )
            stop_completion_streak[:] = torch.where(
                completion_good,
                stop_completion_streak + dt,
                torch.zeros_like(stop_completion_streak),
            )
            to_takeover = (controller_phase == 3) & (stop_completion_streak >= 0.4)
            phase[to_takeover] = 4
            phase_elapsed[to_takeover] = 0.0
            completed[to_takeover] = True
            final_origin[to_takeover] = robot.data.root_pos_w.torch[to_takeover, :2]

            takeover_done = (controller_phase == 4) & (phase_elapsed >= 5.0)
            takeover_ok[takeover_done] = True
            phase[takeover_done] = 5
            finished[takeover_done] = True
            source_settle_timeout = (controller_phase == 0) & (phase_elapsed >= 2.0)
            source_edge_timeout = (controller_phase == 1) & (phase_elapsed >= 4.0)
            source_walk_timeout = (controller_phase == 2) & (phase_elapsed >= 6.0)
            transition_timeout = (controller_phase == 3) & (phase_elapsed >= 4.0)
            unsafe_done = dones.bool() | torso_contact
            fallen |= dones.bool()
            source_fallen |= dones.bool() & (controller_phase <= 2)
            torso_seen |= torso_contact
            newly_failed = (
                source_settle_timeout
                | source_edge_timeout
                | source_walk_timeout
                | transition_timeout
                | unsafe_done
            )
            finished |= newly_failed
            phase[newly_failed] = 5

            for i, trace in enumerate(traces):
                if not active[i]:
                    continue
                for key, value in (
                    ("phase", int(controller_phase[i])),
                    ("vx", float(vx[i])),
                    ("horizontal", float(horizontal[i])),
                    ("vertical", float(vertical[i])),
                    ("heading", float(heading[i])),
                    ("cross", float(cross[i])),
                    ("longitudinal", float(longitudinal[i])),
                    ("roll", float(roll[i])),
                    ("pitch", float(pitch[i])),
                    ("support", int(support[i])),
                    ("slip", float(contact_slip[i])),
                    ("ankle", float(ankle_ratio[i])),
                    ("knee", float(knee_ratio[i])),
                    ("flight", not bool(contacts[i].any())),
                    ("action_rate", float(action_rate[i])),
                    ("action_step", float(action_step[i])),
                    ("torque_spike", float(torque_ratio[i])),
                ):
                    trace[key].append(value)
                timeline_rows.append({
                    "candidate_episode": i,
                    "time_s": step * dt,
                    "controller_state": int(controller_phase[i]),
                    "source_speed_mps": speeds[i],
                    "command_vx_mps": float(command_vx[i]),
                    "actual_vx_mps": float(vx[i]),
                    "horizontal_speed_mps": float(horizontal[i]),
                    "heading_error_rad": float(heading[i]),
                    "support_state": int(support[i]),
                    "stop_requested": bool(controller_phase[i] >= 3),
                    "completion_streak_s": float(stop_completion_streak[i]),
                    "stopping_distance_m": float(longitudinal[i]),
                    "lateral_stop_displacement_m": float(cross[i]),
                    "action_rate_l2_per_s": float(action_rate[i]),
                })
            phase_elapsed += dt
            if bool(finished.all()):
                break

        if args.mode in ("baseline", "smoke"):
            direct = [float(entry_jump[i]) for i in range(n) if walk_hold_ok[i]]
            steady_steps = [
                value
                for trace in traces
                for value, p in zip(trace["action_step"], trace["phase"])
                if p in (2, 4)
            ]
            threshold = 1.5 * max(pct(direct, 99), pct(steady_steps, 99))
        else:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            threshold = float(baseline["frozen_action_jump_l2_threshold"])

        raw_records = []
        for i, trace in enumerate(traces):
            ids = lambda phase_id: [k for k, p in enumerate(trace["phase"]) if p == phase_id]
            source_ids, edge_ids, take_ids = ids(2), ids(3), ids(4)
            values = lambda key, selected: [trace[key][k] for k in selected]
            edge_ankle_dwell = max(
                runs([value >= 0.95 for value in values("ankle", edge_ids)], dt),
                default=0.0,
            )
            source_ankle_dwell = max(
                runs([value >= 0.95 for value in values("ankle", source_ids)], dt),
                default=0.0,
            )
            source_knee_dwell = max(
                runs([value >= 0.95 for value in values("knee", source_ids)], dt),
                default=0.0,
            )
            source_slip_failure = avg(values("slip", source_ids)) > 0.55
            source_flight_failure = avg(values("flight", source_ids)) > 0.20
            source_safe = bool(
                walk_hold_ok[i]
                and source_ankle_dwell < 0.20
                and source_knee_dwell < 0.05
                and not source_slip_failure
                and not source_flight_failure
            )
            edge_knee_dwell = max(
                runs([value >= 0.95 for value in values("knee", edge_ids)], dt),
                default=0.0,
            )
            take_ankle_dwell = max(
                runs([value >= 0.95 for value in values("ankle", take_ids)], dt),
                default=0.0,
            )
            reverse_dwell = max(
                runs([value < -0.10 for value in values("vx", edge_ids)], dt),
                default=0.0,
            )
            slip_failure = avg(values("slip", edge_ids)) > 0.55
            flight_failure = avg(values("flight", edge_ids)) > 0.05
            take_horizontal = values("horizontal", take_ids)
            take_support = values("support", take_ids)
            take_flight = values("flight", take_ids)
            take_safe_fraction = avg([
                speed <= 0.10 and abs(r) <= 0.10 and abs(p) <= 0.10 and support == 3
                for speed, r, p, support in zip(
                    take_horizontal,
                    values("roll", take_ids),
                    values("pitch", take_ids),
                    take_support,
                )
            ])
            stand_hold = (
                bool(takeover_ok[i])
                and pct(take_horizontal, 95) <= 0.10
                and take_safe_fraction >= 0.95
                and not any(take_flight)
                and take_ankle_dwell < 0.20
            )
            path_drift = max(values("cross", edge_ids) + values("cross", take_ids), default=0.0)
            flags = {name: False for name in FAILURES}
            flags.update({
                "source_walk_failure": not source_safe,
                "transition_start_failure": not source_safe,
                "deceleration_failure": source_safe and not bool(completed[i]),
                "gait_termination_failure": source_safe and not bool(completed[i]),
                "residual_speed_failure": source_safe and not bool(completed[i]),
                "reverse_motion_failure": reverse_dwell >= 0.20,
                "heading_failure": pct(values("heading", edge_ids), 95) > 0.12,
                "path_drift_failure": path_drift > 0.30,
                "double_support_failure": source_safe and not bool(completed[i]),
                "false_completion": bool(completed[i]) and not stand_hold,
                "action_discontinuity_at_entry": float(entry_jump[i]) > threshold,
                "action_discontinuity_at_exit": bool(completed[i]) and float(exit_jump[i]) > threshold,
                "ankle_saturation_failure": edge_ankle_dwell >= 0.20,
                "knee_saturation_failure": edge_knee_dwell >= 0.05,
                "dangerous_slip": slip_failure,
                "excessive_flight": flight_failure,
                "transition_timeout": source_safe and not bool(completed[i]),
                "stand_takeover_failure": bool(completed[i]) and not stand_hold,
                "stand_hold_failure": bool(completed[i]) and not stand_hold,
                "fall": bool(fallen[i]),
                "torso_contact": bool(torso_seen[i]),
            })
            full_success = bool(
                source_safe and completed[i] and stand_hold and not any(flags.values())
            )
            primary = next((name for name in FAILURES if flags[name]), "")
            stop_distance = (
                values("longitudinal", edge_ids)[-1] if edge_ids else 0.0
            )
            final_distance = (
                values("longitudinal", take_ids)[-1] if take_ids else stop_distance
            )
            raw_records.append({
                "candidate_episode": i,
                "source_speed_mps": speeds[i],
                "source_generation_success": bool(source_generation_ok[i]),
                "source_walk_hold_success": source_safe,
                "source_ankle_saturation_max_dwell_s": source_ankle_dwell,
                "source_knee_saturation_max_dwell_s": source_knee_dwell,
                "source_dangerous_slip": source_slip_failure,
                "source_excessive_flight": source_flight_failure,
                "source_heading_p95_rad": pct(values("heading", source_ids), 95),
                "pre_transition_fall": bool(source_fallen[i]),
                "stop_request_time_s": float(stop_time[i]),
                "stop_request_support_state": int(stop_support[i]),
                "stop_request_vx_mps": float(stop_vx[i]),
                "transition_completion": bool(completed[i]),
                "stand_takeover_success": stand_hold,
                "full_edge_success": full_success,
                "transition_duration_s": len(edge_ids) * dt,
                "transition_heading_p95_rad": pct(values("heading", edge_ids), 95),
                "final_speed_mean_mps": avg(take_horizontal),
                "final_speed_p95_mps": pct(take_horizontal, 95),
                "final_double_support": bool(take_support and take_support[-1] == 3),
                "double_support_acquisition": bool(completed[i]),
                "stand_safe_fraction": take_safe_fraction,
                "minimum_forward_velocity_mps": min(values("vx", edge_ids), default=0.0),
                "reverse_motion_max_dwell_s": reverse_dwell,
                "reverse_displacement_m": abs(min(values("longitudinal", edge_ids), default=0.0)),
                "stopping_distance_m": stop_distance,
                "final_longitudinal_displacement_m": final_distance,
                "lateral_stopping_displacement_m": path_drift,
                "fall": bool(fallen[i]),
                "torso_contact": bool(torso_seen[i]),
                "dangerous_slip": slip_failure,
                "excessive_flight": flight_failure,
                "ankle_saturation_max_dwell_s": edge_ankle_dwell,
                "knee_saturation_max_dwell_s": edge_knee_dwell,
                "takeover_ankle_saturation_max_dwell_s": take_ankle_dwell,
                "entry_action_jump_l2": float(entry_jump[i]),
                "exit_action_jump_l2": float(exit_jump[i]),
                "entry_joint_max_jump": float(entry_joint_jump[i]),
                "exit_joint_max_jump": float(exit_joint_jump[i]),
                "transition_action_rate_p95": pct(values("action_rate", edge_ids), 95),
                "transition_action_rate_p99": pct(values("action_rate", edge_ids), 99),
                "transition_action_rate_max": max(values("action_rate", edge_ids), default=0.0),
                "torque_spike_max_ratio": max(values("torque_spike", edge_ids), default=0.0),
                "primary_failure": primary,
                "failure_flags": json.dumps(flags, sort_keys=True),
            })

        selected = []
        for speed in SPEEDS:
            candidates = [
                record
                for record in raw_records
                if record["source_speed_mps"] == speed and record["source_walk_hold_success"]
            ]
            selected.extend(candidates[:desired[speed]])
        for episode, record in enumerate(selected):
            record["episode"] = episode

        def summarize(rows):
            return {
                "episodes": len(rows),
                "source_walk_hold_success_rate": avg([r["source_walk_hold_success"] for r in rows]),
                "pre_transition_heading_p95_rad": pct([
                    r["source_heading_p95_rad"] for r in rows
                ], 95),
                "pre_transition_saturation_failure_rate": avg([
                    r["source_ankle_saturation_max_dwell_s"] >= 0.20
                    or r["source_knee_saturation_max_dwell_s"] >= 0.05
                    for r in rows
                ]),
                "pre_transition_fall_rate": avg([r["pre_transition_fall"] for r in rows]),
                "transition_completion_rate": avg([r["transition_completion"] for r in rows]),
                "stand_takeover_success_rate": avg([r["stand_takeover_success"] for r in rows]),
                "stand_hold_success_rate": avg([r["stand_takeover_success"] for r in rows]),
                "full_edge_success_rate": avg([r["full_edge_success"] for r in rows]),
                "transition_duration_mean_s": avg([
                    r["transition_duration_s"] for r in rows if r["transition_completion"]
                ]),
                "stopping_distance_mean_m": avg([r["stopping_distance_m"] for r in rows]),
                "stopping_distance_p95_m": pct([r["stopping_distance_m"] for r in rows], 95),
                "final_speed_mean_mps": avg([r["final_speed_mean_mps"] for r in rows]),
                "final_speed_p95_mps": pct([r["final_speed_p95_mps"] for r in rows], 95),
                "double_support_acquisition_rate": avg([
                    r["double_support_acquisition"] for r in rows
                ]),
                "final_double_support_rate": avg([r["final_double_support"] for r in rows]),
                "heading_error_p95_rad": pct([r["transition_heading_p95_rad"] for r in rows], 95),
                "reverse_motion_failure_rate": avg([
                    r["reverse_motion_max_dwell_s"] >= 0.20 for r in rows
                ]),
                "fall_rate": avg([r["fall"] for r in rows]),
                "slip_failure_rate": avg([r["dangerous_slip"] for r in rows]),
                "saturation_failure_rate": avg([
                    r["ankle_saturation_max_dwell_s"] >= 0.20
                    or r["knee_saturation_max_dwell_s"] >= 0.05
                    for r in rows
                ]),
                "takeover_saturation_failure_rate": avg([
                    r["takeover_ankle_saturation_max_dwell_s"] >= 0.20 for r in rows
                ]),
                "flight_failure_rate": avg([r["excessive_flight"] for r in rows]),
                "entry_discontinuity_failure_rate": avg([
                    r["entry_action_jump_l2"] > threshold for r in rows
                ]),
                "exit_discontinuity_failure_rate": avg([
                    r["transition_completion"] and r["exit_action_jump_l2"] > threshold
                    for r in rows
                ]),
                "entry_action_jump_l2_p95": pct([r["entry_action_jump_l2"] for r in rows], 95),
                "exit_action_jump_l2_p95": pct([
                    r["exit_action_jump_l2"] for r in rows if r["transition_completion"]
                ], 95),
            }

        overall = summarize(selected)
        per_speed = {
            str(speed): summarize([
                record for record in selected if record["source_speed_mps"] == speed
            ])
            for speed in SPEEDS
        }
        source_stats = {
            "candidates": n,
            "source_generation_successes": sum(r["source_generation_success"] for r in raw_records),
            "walk_hold_successes": sum(r["source_walk_hold_success"] for r in raw_records),
            "edge_denominator": len(selected),
            "required_edge_denominator": sum(desired.values()),
            "excluded_source_failures": sum(not r["source_walk_hold_success"] for r in raw_records),
            "per_speed_walk_hold_successes": {
                str(speed): sum(
                    r["source_walk_hold_success"]
                    for r in raw_records
                    if r["source_speed_mps"] == speed
                )
                for speed in SPEEDS
            },
        }
        strict_checks = {
            "edge_denominator_complete": len(selected) == sum(desired.values()),
            "source_walk_hold_ge_0_95": overall["source_walk_hold_success_rate"] >= 0.95,
            "pre_transition_heading_le_0_12": overall["pre_transition_heading_p95_rad"] <= 0.12,
            "pre_transition_saturation_le_0_05": overall["pre_transition_saturation_failure_rate"] <= 0.05,
            "pre_transition_fall_le_0_02": overall["pre_transition_fall_rate"] <= 0.02,
            "transition_completion_ge_0_95": overall["transition_completion_rate"] >= 0.95,
            "stand_takeover_ge_0_95": overall["stand_takeover_success_rate"] >= 0.95,
            "stand_hold_ge_0_95": overall["stand_hold_success_rate"] >= 0.95,
            "full_edge_ge_0_90": overall["full_edge_success_rate"] >= 0.90,
            "each_speed_ge_0_90": all(
                value["episodes"] == desired[float(speed)]
                and value["full_edge_success_rate"] >= 0.90
                for speed, value in per_speed.items()
            ),
            "fall_le_0_02": overall["fall_rate"] <= 0.02,
            "heading_le_0_12": overall["heading_error_p95_rad"] <= 0.12,
            "final_speed_mean_le_0_08": overall["final_speed_mean_mps"] <= 0.08,
            "final_speed_p95_le_0_10": overall["final_speed_p95_mps"] <= 0.10,
            "double_support_ge_0_95": overall["double_support_acquisition_rate"] >= 0.95,
            "final_double_support_ge_0_95": overall["final_double_support_rate"] >= 0.95,
            "saturation_le_0_05": overall["saturation_failure_rate"] <= 0.05,
            "takeover_saturation_le_0_05": overall["takeover_saturation_failure_rate"] <= 0.05,
            "slip_le_0_05": overall["slip_failure_rate"] <= 0.05,
            "reverse_le_0_05": overall["reverse_motion_failure_rate"] <= 0.05,
            "flight_eq_0": overall["flight_failure_rate"] == 0.0,
            "entry_discontinuity_le_0_05": overall["entry_discontinuity_failure_rate"] <= 0.05,
            "exit_discontinuity_le_0_05": overall["exit_discontinuity_failure_rate"] <= 0.05,
        }
        pilot_checks = {
            "edge_denominator_complete": strict_checks["edge_denominator_complete"],
            "full_edge_ge_0_90": overall["full_edge_success_rate"] >= 0.90,
            "fall_le_0_05": overall["fall_rate"] <= 0.05,
            "final_speed_mean_le_0_08": overall["final_speed_mean_mps"] <= 0.08,
            "double_support_ge_0_90": overall["double_support_acquisition_rate"] >= 0.90,
            "stand_takeover_ge_0_90": overall["stand_takeover_success_rate"] >= 0.90,
            "heading_le_0_12": overall["heading_error_p95_rad"] <= 0.12,
            "saturation_le_0_05": overall["saturation_failure_rate"] <= 0.05,
            "action_discontinuity_le_0_05": (
                overall["entry_discontinuity_failure_rate"] <= 0.05
                and overall["exit_discontinuity_failure_rate"] <= 0.05
            ),
        }
        summary = {
            "stage": "Stage 4",
            "mode": args.mode,
            "label": args.label,
            "seed": args.seed,
            "controller_type": (
                "LEARNED_TRANSITION_EXPERT"
                if transition_path
                else "PARAMETER_FREE_DIRECT_SWITCH"
            ),
            "training_performed": transition_path is not None,
            "walk_checkpoint": str(walk_path.relative_to(REPO)),
            "walk_sha256": sha(walk_path),
            "stand_checkpoint": str(stand_path.relative_to(REPO)),
            "stand_sha256": sha(stand_path),
            "source_generator_checkpoint": str(start_path.relative_to(REPO)),
            "source_generator_sha256": sha(start_path),
            "transition_checkpoint": (
                str(transition_path.relative_to(REPO)) if transition_path else None
            ),
            "transition_sha256": sha(transition_path) if transition_path else None,
            "controller_routing": (
                "WALK frozen -> independent WALK_TO_STAND -> STAND frozen"
                if transition_path
                else "WALK frozen -> zero command direct hard switch -> STAND frozen"
            ),
            "runtime_action_blend": False,
            "completion_hold_s": 0.4,
            "no_support_switch_before_completion_s": 0.4,
            "transition_timeout_s": 4.0,
            "stand_takeover_hold_s": 5.0,
            "reverse_threshold": {"velocity_mps": -0.10, "dwell_s": 0.20},
            "frozen_action_jump_l2_threshold": threshold,
            "source_generation": source_stats,
            "overall": overall,
            "per_speed": per_speed,
            "pilot_checks": pilot_checks,
            "formal_checks": strict_checks,
            "pilot_gate_pass": all(pilot_checks.values()),
            "formal_gate_pass": all(strict_checks.values()),
            "failure_counts": dict(Counter(
                record["primary_failure"] or "none" for record in selected
            )),
        }
        if args.mode == "baseline":
            summary["diagnostic_only"] = True
            baseline_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        else:
            (output / f"{args.label}_summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
        if selected:
            with (output / f"{args.label}_episodes.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                fields = ["episode"] + [key for key in selected[0] if key != "episode"]
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
        selected_candidates = {record["candidate_episode"] for record in selected}
        selected_timelines = [
            row for row in timeline_rows if row["candidate_episode"] in selected_candidates
        ]
        with (output / f"{args.label}_timelines.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected_timelines[0]))
            writer.writeheader()
            writer.writerows(selected_timelines)
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
