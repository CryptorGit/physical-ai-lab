"""Stage 5D controlled diagnostics for frozen STAND/WALK integration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
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

PHASES = ("INITIAL_STAND", "STAND_TO_WALK_ACTIVE", "WALK_HOLD", "WALK_TO_STAND_ACTIVE", "FINAL_STAND", "DONE")
CONTROLLERS = (
    "stage2_model_4246", "stand_to_walk_transition_v1",
    "walk_steady_state_expert_v1", "walk_to_stand_transition_v1",
)
FORMAL_SPEEDS = [0.6] * 13 + [0.8] * 13 + [1.0] * 12 + [1.2] * 12

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--test", choices=("reproduce", "stand", "forward", "reverse", "full"), required=True)
parser.add_argument("--speed", type=float, choices=(0.6, 0.8))
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--manifest", default=str(EXP / "integration_manifest.json"))
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def minimum_jerk(value):
    value = value.clamp(0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def contiguous_events(values, threshold, dt):
    events, start = [], None
    for index, value in enumerate(values):
        if value >= threshold and start is None:
            start = index
        if value < threshold and start is not None:
            events.append((start, index - 1, (index - start) * dt))
            start = None
    if start is not None:
        events.append((start, len(values) - 1, (len(values) - start) * dt))
    return events


def main():
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    paths = {}
    for name in CONTROLLERS:
        spec = manifest["controllers"][name]
        path = (REPO / spec["checkpoint"]).resolve(strict=True)
        if sha(path) != spec["sha256"]:
            raise RuntimeError(f"protected hash mismatch: {name}")
        paths[name] = path
    if args.test == "reproduce":
        if args.episodes != 50 or args.seed != 20260901:
            raise ValueError("reproduction is frozen to 50 episodes and seed 20260901")
        speeds = FORMAL_SPEEDS
    elif args.test == "stand":
        speeds = [0.0] * args.episodes
    else:
        if args.speed is None:
            raise ValueError("--speed is required")
        speeds = [args.speed] * args.episodes
    n = len(speeds)
    rng = random.Random(args.seed)
    stand_holds = [rng.uniform(0.8, 1.8) for _ in speeds]
    walk_holds = [rng.uniform(2.0, 3.5) for _ in speeds]
    ramp_up = [rng.uniform(1.3, 1.7) for _ in speeds]
    ramp_down = [rng.uniform(1.4, 1.8) for _ in speeds]
    yaw_offsets = [rng.uniform(-0.03, 0.03) for _ in speeds]
    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 14.0 if args.test == "stand" else 24.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        models = {name: load_walk_expert(path, device=env.device) for name, path in paths.items()}
        robot = env.scene["robot"]
        velocity_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        torso_ids, _ = robot.find_bodies("torso_link")
        sensor_torso = [sensor.body_names.index(robot.body_names[index]) for index in torso_ids]
        ankle_pitch, ankle_pitch_names = robot.find_joints(".*_ankle_pitch_joint")
        ankle_roll, ankle_roll_names = robot.find_joints(".*_ankle_roll_joint")
        knees, knee_names = robot.find_joints(".*_knee_joint")
        diagnostic_joint_ids = ankle_pitch + ankle_roll + knees
        diagnostic_joint_names = ankle_pitch_names + ankle_roll_names + knee_names
        wrapped.reset()
        device = env.device
        dt = float(env.step_dt)
        target_speed = torch.tensor(speeds, device=device)
        target_heading = robot.data.heading_w.torch.clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)
        phase_elapsed = torch.zeros(n, device=device)
        settle_streak = torch.zeros(n, device=device)
        acquire_streak = torch.zeros(n, device=device)
        walk_streak = torch.zeros(n, device=device)
        stop_streak = torch.zeros(n, device=device)
        no_switch = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        filtered_yaw = torch.zeros(n, device=device)
        previous_action = torch.zeros(n, 37, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        fallen = torch.zeros(n, dtype=torch.bool, device=device)
        first_fall_step = torch.full((n,), -1, dtype=torch.long, device=device)
        torso_seen = torch.zeros(n, dtype=torch.bool, device=device)
        initial_ok = torch.zeros(n, dtype=torch.bool, device=device)
        up_ok = torch.zeros(n, dtype=torch.bool, device=device)
        walk_ok = torch.zeros(n, dtype=torch.bool, device=device)
        down_ok = torch.zeros(n, dtype=torch.bool, device=device)
        final_ok = torch.zeros(n, dtype=torch.bool, device=device)
        previous_mismatch = torch.zeros(n, dtype=torch.long, device=device)
        traces = [defaultdict(list) for _ in speeds]
        entry_rows, boundary_rows = [], []
        previous_controller = ["stage2_model_4246"] * n
        stop_origin = robot.data.root_pos_w.torch[:, :2].clone()
        max_steps = round((13.5 if args.test == "stand" else 23.5) / dt)

        for step in range(max_steps):
            current_phase = phase.clone()
            active = ~finished
            command_vx = torch.zeros(n, device=device)
            up_mask, walk_mask, down_mask = current_phase == 1, current_phase == 2, current_phase == 3
            up_duration = torch.tensor(ramp_up, device=device)
            down_duration = torch.tensor(ramp_down, device=device)
            command_vx[up_mask] = target_speed[up_mask] * minimum_jerk(
                phase_elapsed[up_mask] / up_duration[up_mask]
            )
            command_vx[walk_mask] = target_speed[walk_mask]
            command_vx[down_mask] = target_speed[down_mask] * (
                1.0 - minimum_jerk(phase_elapsed[down_mask] / down_duration[down_mask])
            )
            heading_signed = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw_yaw = (0.8 * heading_signed - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw_yaw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            filtered_yaw[(current_phase == 0) | (current_phase == 4)] = 0.0
            velocity_term.vel_command_b.zero_()
            velocity_term.vel_command_b[:, 0] = command_vx
            velocity_term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            previous_match = (legacy[:, 86:123] == previous_action).all(dim=1)
            previous_mismatch += (~previous_match & active).long()
            state = canonical_state_from_legacy_observation(
                legacy, heading_w_rad=robot.data.heading_w.torch
            )
            command = MotionCommand(command_vx, target_heading, target_yaw_rate_radps=filtered_yaw)
            selected = torch.zeros_like(previous_action)
            controller_names = []
            with torch.inference_mode():
                for i in range(n):
                    controller_names.append(
                        "stage2_model_4246" if current_phase[i] in (0, 4, 5)
                        else "stand_to_walk_transition_v1" if current_phase[i] == 1
                        else "walk_steady_state_expert_v1" if current_phase[i] == 2
                        else "walk_to_stand_transition_v1"
                    )
                for name, model in models.items():
                    mask = torch.tensor(
                        [active[i] and controller_names[i] == name for i in range(n)], device=device
                    )
                    if bool(mask.any()):
                        actions = model(state, command)
                        selected[mask] = actions[mask]
                selected[finished] = previous_action[finished]
                for i in range(n):
                    if finished[i] or controller_names[i] == previous_controller[i]:
                        continue
                    delta = selected[i] - previous_action[i]
                    boundary_rows.append({
                        "test": args.label, "episode": i, "step": step,
                        "time_s": step * dt, "target_speed_mps": speeds[i],
                        "from_controller": previous_controller[i], "to_controller": controller_names[i],
                        "action_l2_jump": float(torch.linalg.vector_norm(delta)),
                        "joint_max_jump": float(delta.abs().amax()),
                        "joint_max_name": robot.joint_names[int(delta.abs().argmax())],
                        "previous_action_bitwise_match": bool(previous_match[i]),
                    })
                    previous_controller[i] = controller_names[i]
                _, _, dones, _ = wrapped.step(selected)

            forces = sensor.data.net_forces_w_history.torch
            contacts = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            foot_forces = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1)
            torso_contact = forces[:, :, sensor_torso, :].norm(dim=-1).amax(dim=(1, 2)) > 5.0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            slip = torch.where(
                contacts,
                robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1),
                torch.zeros_like(contacts, dtype=torch.float32),
            ).amax(dim=1)
            effort_ratio = (
                robot.data.applied_torque.torch.abs()
                / robot.data.joint_effort_limits.torch.abs().clamp_min(1e-6)
            )
            velocity_ratio = (
                robot.data.joint_vel.torch.abs()
                / robot.data.joint_vel_limits.torch.abs().clamp_min(1e-6)
            )
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2])
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            heading = heading_signed.abs()
            action_delta = selected - previous_action
            action_jump = torch.linalg.vector_norm(action_delta, dim=1)
            previous_action[:] = selected
            newly_fallen = dones.bool() & active & (~fallen)
            first_fall_step[newly_fallen] = step
            fallen |= dones.bool() & active
            torso_seen |= torso_contact & active

            for i in range(n):
                if finished[i]:
                    continue
                trace = traces[i]
                trace["step"].append(step)
                trace["phase"].append(int(current_phase[i]))
                trace["controller"].append(controller_names[i])
                trace["phase_elapsed"].append(float(phase_elapsed[i]))
                trace["command_vx"].append(float(command_vx[i]))
                trace["vx"].append(float(vx[i]))
                trace["horizontal"].append(float(horizontal[i]))
                trace["vertical"].append(float(vertical[i]))
                trace["heading"].append(float(heading[i]))
                trace["roll"].append(float(roll[i]))
                trace["pitch"].append(float(pitch[i]))
                trace["support"].append(int(support[i]))
                trace["left_contact"].append(bool(contacts[i, 0]))
                trace["right_contact"].append(bool(contacts[i, 1]))
                trace["left_contact_force"].append(float(foot_forces[i, 0]))
                trace["right_contact_force"].append(float(foot_forces[i, 1]))
                trace["slip"].append(float(slip[i]))
                trace["action_jump"].append(float(action_jump[i]))
                trace["action"].append(selected[i].detach().cpu().tolist())
                trace["previous_action"].append((selected[i] - action_delta[i]).detach().cpu().tolist())
                trace["previous_match"].append(bool(previous_match[i]))
                trace["completion_streak"].append(
                    float(acquire_streak[i] if current_phase[i] == 1 else stop_streak[i])
                )
                trace["aggregate_ankle_pitch_effort"].append(float(effort_ratio[i, ankle_pitch].amax()))
                trace["aggregate_knee_velocity"].append(float(velocity_ratio[i, knees].amax()))
                trace["dominant_ankle_pitch"].append(
                    ankle_pitch_names[int(effort_ratio[i, ankle_pitch].argmax())]
                )
                trace["dominant_knee"].append(
                    knee_names[int(velocity_ratio[i, knees].argmax())]
                )
                for joint_id, joint_name in zip(diagnostic_joint_ids, diagnostic_joint_names):
                    kind = "velocity" if "knee" in joint_name else "effort"
                    ratio = velocity_ratio[i, joint_id] if kind == "velocity" else effort_ratio[i, joint_id]
                    key = f"joint::{joint_name}"
                    trace[key].append(float(ratio))
                    trace[f"position::{joint_name}"].append(float(robot.data.joint_pos.torch[i, joint_id]))
                    trace[f"position_error::{joint_name}"].append(float(
                        robot.data.joint_pos_target.torch[i, joint_id] - robot.data.joint_pos.torch[i, joint_id]
                    ))
                    trace[f"target::{joint_name}"].append(float(robot.data.joint_pos_target.torch[i, joint_id]))

            stand_good = (
                (horizontal <= 0.08) & (vertical <= 0.05) & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10) & contacts.all(dim=1) & (~dones.bool())
            )
            settle_streak = torch.where(
                ((current_phase == 0) | (current_phase == 4)) & stand_good,
                settle_streak + dt, torch.zeros_like(settle_streak)
            )
            for i in range(n):
                if finished[i]:
                    continue
                if current_phase[i] == 0:
                    if args.test == "stand":
                        if settle_streak[i] >= 0.4:
                            initial_ok[i] = True
                        if phase_elapsed[i] >= 10.0:
                            final_ok[i] = bool(settle_streak[i] >= 8.0)
                            finished[i] = True
                            phase[i] = 5
                    elif settle_streak[i] >= 0.4 and phase_elapsed[i] >= stand_holds[i]:
                        initial_ok[i] = True
                        entry_rows.append({
                            "test": args.label, "episode": i, "entry": "STAND_TO_WALK",
                            "target_speed_mps": speeds[i], "root_vx": float(vx[i]),
                            "root_horizontal_speed": float(horizontal[i]), "roll": float(roll[i]),
                            "pitch": float(pitch[i]), "joint_position_norm": float(robot.data.joint_pos.torch[i].norm()),
                            "joint_velocity_norm": float(robot.data.joint_vel.torch[i].norm()),
                            "previous_action_norm": float(previous_action[i].norm()),
                            "double_support": int(support[i] == 3), "left_contact_force": float(foot_forces[i, 0]),
                            "right_contact_force": float(foot_forces[i, 1]), "heading_error": float(heading[i]),
                            "ankle_effort_max": float(effort_ratio[i, ankle_pitch + ankle_roll].amax()),
                            "request_time_s": float(phase_elapsed[i]),
                        })
                        phase[i], phase_elapsed[i] = 1, 0.0
                        target_heading[i] = robot.data.heading_w.torch[i] + yaw_offsets[i]
                        support_switches[i] = 0
                        previous_support[i] = support[i]
                elif current_phase[i] == 1:
                    if support[i] and support[i] != previous_support[i]:
                        support_switches[i] += 1
                    previous_support[i] = support[i]
                    good = (
                        vx[i] >= 0.75 * target_speed[i] and abs(vx[i] - target_speed[i]) <= 0.20
                        and heading[i] <= 0.12 and abs(roll[i]) <= 0.20 and abs(pitch[i]) <= 0.20
                        and support_switches[i] >= 2 and not dones[i] and not torso_contact[i]
                    )
                    acquire_streak[i] = acquire_streak[i] + dt if good else 0.0
                    if acquire_streak[i] >= 0.4:
                        up_ok[i] = True
                        phase[i], phase_elapsed[i], walk_streak[i] = 2, 0.0, 0.0
                elif current_phase[i] == 2:
                    good = abs(vx[i] - target_speed[i]) <= 0.20 and heading[i] <= 0.12 and not dones[i]
                    walk_streak[i] = walk_streak[i] + dt if good else 0.0
                    hold = 3.0 if args.test == "forward" else walk_holds[i]
                    if walk_streak[i] >= hold:
                        walk_ok[i] = True
                        if args.test == "forward":
                            finished[i] = True
                            phase[i] = 5
                        else:
                            entry_rows.append({
                                "test": args.label, "episode": i, "entry": "WALK_TO_STAND",
                                "target_speed_mps": speeds[i], "actual_speed": float(vx[i]),
                                "speed_error": float(abs(vx[i] - target_speed[i])),
                                "heading_error": float(heading[i]), "support_foot": int(support[i]),
                                "joint_velocity_norm": float(robot.data.joint_vel.torch[i].norm()),
                                "previous_action_norm": float(previous_action[i].norm()),
                                "ankle_effort_max": float(effort_ratio[i, ankle_pitch + ankle_roll].amax()),
                                "path_drift": 0.0, "stop_request_time_s": float(phase_elapsed[i]),
                            })
                            phase[i], phase_elapsed[i], no_switch[i] = 3, 0.0, 0.0
                            stop_origin[i] = robot.data.root_pos_w.torch[i, :2]
                            previous_support[i] = support[i]
                elif current_phase[i] == 3:
                    switched = support[i] and support[i] != previous_support[i]
                    no_switch[i] = 0.0 if switched else no_switch[i] + dt
                    previous_support[i] = support[i]
                    good = (
                        horizontal[i] <= 0.08 and vertical[i] <= 0.05 and heading[i] <= 0.12
                        and abs(roll[i]) <= 0.10 and abs(pitch[i]) <= 0.10
                        and support[i] == 3 and no_switch[i] >= 0.4 and not dones[i]
                    )
                    stop_streak[i] = stop_streak[i] + dt if good else 0.0
                    if stop_streak[i] >= 0.4:
                        down_ok[i] = True
                        phase[i], phase_elapsed[i], settle_streak[i] = 4, 0.0, 0.0
                elif current_phase[i] == 4 and settle_streak[i] >= 5.0:
                    final_ok[i] = True
                    finished[i] = True
                    phase[i] = 5
                if current_phase[i] in (1, 3) and phase_elapsed[i] > 5.0:
                    finished[i] = True
                if dones[i] or torso_contact[i]:
                    finished[i] = True
            phase_elapsed += dt
            if bool(finished.all()):
                break

        episode_rows, saturation_rows = [], []
        for i, trace in enumerate(traces):
            joint_events = []
            for joint_name in diagnostic_joint_names:
                kind = "velocity" if "knee" in joint_name else "effort"
                values = trace[f"joint::{joint_name}"]
                for start, end, dwell in contiguous_events(values, 0.90, dt):
                    high95 = contiguous_events(values[start:end + 1], 0.95, dt)
                    dwell95 = max([event[2] for event in high95], default=0.0)
                    event = {
                        "test": args.label, "episode": i, "target_speed_mps": speeds[i],
                        "joint_name": joint_name, "side": "left" if joint_name.startswith("left") else "right",
                        "quantity": kind, "start_step": trace["step"][start], "end_step": trace["step"][end],
                        "start_time_s": trace["step"][start] * dt, "end_time_s": trace["step"][end] * dt,
                        "dwell_above_90_s": dwell, "dwell_above_95_s": dwell95,
                        "longest_continuous_dwell_s": dwell,
                        "max_utilization": max(values[start:end + 1]),
                        "active_controller": trace["controller"][start],
                        "phase": PHASES[trace["phase"][start]],
                        "switch_boundary_elapsed_s": trace["phase_elapsed"][start],
                        "support_foot": trace["support"][start],
                        "action_target": trace[f"target::{joint_name}"][start],
                        "joint_position": trace[f"position::{joint_name}"][start],
                        "position_error": trace[f"position_error::{joint_name}"][start],
                        "left_contact_force": trace["left_contact_force"][start],
                        "right_contact_force": trace["right_contact_force"][start],
                    }
                    saturation_rows.append(event)
                    if (kind == "effort" and dwell95 >= 0.20) or (kind == "velocity" and dwell95 >= 0.05):
                        joint_events.append(event)
            for aggregate_key, kind, dwell_threshold, dominant_key in (
                ("aggregate_ankle_pitch_effort", "effort", 0.20, "dominant_ankle_pitch"),
                ("aggregate_knee_velocity", "velocity", 0.05, "dominant_knee"),
            ):
                values = trace[aggregate_key]
                for start, end, dwell in contiguous_events(values, 0.95, dt):
                    if dwell < dwell_threshold:
                        continue
                    peak = max(range(start, end + 1), key=lambda index: values[index])
                    joint_name = trace[dominant_key][peak]
                    joint_id = robot.joint_names.index(joint_name)
                    event = {
                        "test": args.label, "episode": i, "target_speed_mps": speeds[i],
                        "joint_name": joint_name, "side": "left" if joint_name.startswith("left") else "right",
                        "quantity": f"aggregate_{kind}", "start_step": trace["step"][start],
                        "end_step": trace["step"][end], "start_time_s": trace["step"][start] * dt,
                        "end_time_s": trace["step"][end] * dt, "dwell_above_90_s": dwell,
                        "dwell_above_95_s": dwell, "longest_continuous_dwell_s": dwell,
                        "max_utilization": values[peak], "active_controller": trace["controller"][start],
                        "phase": PHASES[trace["phase"][start]],
                        "switch_boundary_elapsed_s": trace["phase_elapsed"][start],
                        "support_foot": trace["support"][start],
                        "action_target": trace[f"target::{joint_name}"][peak],
                        "joint_position": trace[f"position::{joint_name}"][peak],
                        "position_error": trace[f"position_error::{joint_name}"][peak],
                        "left_contact_force": trace["left_contact_force"][start],
                        "right_contact_force": trace["right_contact_force"][start],
                        "dominant_joint_sequence": json.dumps(trace[dominant_key][start:end + 1]),
                    }
                    saturation_rows.append(event)
                    joint_events.append(event)
            first_sat = min(joint_events, key=lambda row: row["start_step"]) if joint_events else None
            initial = bool(initial_ok[i])
            if args.test == "stand":
                success = bool(initial_ok[i] and final_ok[i] and not fallen[i] and not joint_events)
            elif args.test == "forward":
                success = bool(initial and up_ok[i] and walk_ok[i] and not fallen[i] and not joint_events)
            else:
                success = bool(initial and up_ok[i] and walk_ok[i] and down_ok[i] and final_ok[i] and not fallen[i] and not joint_events)
            primary = (
                "initial_stand_failure" if not initial
                else "fall" if fallen[i]
                else "saturation_failure" if joint_events
                else "" if args.test == "stand" and success
                else "stand_to_walk_failure" if not up_ok[i]
                else "walk_hold_failure" if not walk_ok[i]
                else "walk_to_stand_failure" if args.test in ("reverse", "full", "reproduce") and not down_ok[i]
                else "final_stand_failure" if args.test in ("reverse", "full", "reproduce") and not final_ok[i]
                else ""
            )
            first_abnormal_step = (
                trace["step"][0] if not initial and trace["step"]
                else first_sat["start_step"] if first_sat
                else int(first_fall_step[i]) if fallen[i]
                else -1
            )
            first_index = trace["step"].index(first_abnormal_step) if first_abnormal_step in trace["step"] else max(len(trace["step"]) - 1, 0)
            episode_rows.append({
                "test": args.label, "episode": i, "target_speed_mps": speeds[i], "success": success,
                "initial_stand_success": initial, "stand_to_walk_completion": bool(up_ok[i]),
                "walk_hold_success": bool(walk_ok[i]), "walk_to_stand_completion": bool(down_ok[i]),
                "final_stand_success": bool(final_ok[i]), "fall": bool(fallen[i]),
                "torso_contact": bool(torso_seen[i]), "saturation_failure": bool(joint_events),
                "first_failure_phase": PHASES[trace["phase"][first_index]] if trace["phase"] else "",
                "first_failure_controller": trace["controller"][first_index] if trace["controller"] else "",
                "first_failure_step": first_abnormal_step, "first_failure_time_s": first_abnormal_step * dt,
                "first_saturation_joint": first_sat["joint_name"] if first_sat else "",
                "first_saturation_dwell_s": first_sat["dwell_above_95_s"] if first_sat else 0.0,
                "previous_action_mismatch_steps": int(previous_mismatch[i]),
                "primary_failure": primary,
            })

        summary = {
            "test": args.label, "diagnostic_only": True, "seed": args.seed,
            "episodes": n, "speed_mps": args.speed,
            "success_rate": mean([row["success"] for row in episode_rows]),
            "initial_stand_success_rate": mean([row["initial_stand_success"] for row in episode_rows]),
            "stand_to_walk_completion_rate": mean([row["stand_to_walk_completion"] for row in episode_rows]),
            "walk_hold_success_rate": mean([row["walk_hold_success"] for row in episode_rows]),
            "walk_to_stand_completion_rate": mean([row["walk_to_stand_completion"] for row in episode_rows]),
            "final_stand_success_rate": mean([row["final_stand_success"] for row in episode_rows]),
            "fall_rate": mean([row["fall"] for row in episode_rows]),
            "saturation_failure_rate": mean([row["saturation_failure"] for row in episode_rows]),
            "previous_action_mismatch_rate": mean([row["previous_action_mismatch_steps"] > 0 for row in episode_rows]),
            "failure_counts": dict(Counter(row["primary_failure"] or "none" for row in episode_rows)),
            "protected_hashes": {name: sha(path) for name, path in paths.items()},
        }
        write_json(output / f"{args.label}_summary.json", summary)
        write_csv(output / f"{args.label}_episodes.csv", episode_rows)
        write_csv(output / f"{args.label}_entries.csv", entry_rows)
        write_csv(output / f"{args.label}_boundaries.csv", boundary_rows)
        write_csv(output / f"{args.label}_saturation_events.csv", saturation_rows)

        if args.test == "reproduce":
            failure_ids = [row["episode"] for row in episode_rows if not row["success"]]
            timeline_rows = []
            for i in failure_ids:
                trace = traces[i]
                for k, step in enumerate(trace["step"]):
                    timeline_rows.append({
                        "episode": i, "step": step, "time_s": step * dt,
                        "phase": PHASES[trace["phase"][k]], "active_controller": trace["controller"][k],
                        "target_speed_mps": speeds[i], "command_vx": trace["command_vx"][k],
                        "actual_forward_speed": trace["vx"][k], "horizontal_speed": trace["horizontal"][k],
                        "heading_error": trace["heading"][k], "roll": trace["roll"][k], "pitch": trace["pitch"][k],
                        "support_foot": trace["support"][k], "left_contact": trace["left_contact"][k],
                        "right_contact": trace["right_contact"][k], "slip": trace["slip"][k],
                        "action_l2_jump": trace["action_jump"][k],
                        "transition_elapsed": trace["phase_elapsed"][k],
                        "completion_streak": trace["completion_streak"][k],
                        "previous_action_match": trace["previous_match"][k],
                        "action": json.dumps(trace["action"][k]),
                        "previous_global_action": json.dumps(trace["previous_action"][k]),
                    })
            write_csv(output / "failure_timelines.csv", timeline_rows)
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
