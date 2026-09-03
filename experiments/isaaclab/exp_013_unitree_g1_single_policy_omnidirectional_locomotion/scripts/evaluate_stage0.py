"""Frozen deterministic Stage 0 directional baseline evaluator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_single_policy.phase_gated_heading import yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--suite", choices=(
    "candidate", "anchor", "translation_walk", "translation_run", "yaw",
    "translation_yaw", "independence", "transitions", "random",
), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--tag", default="")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

DIRECTIONS = tuple(index * 22.5 for index in range(16))
CARDINAL8 = tuple(range(0, 360, 45))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_jerk(value):
    value = max(0.0, min(1.0, value))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def vector(speed, degrees):
    radians = math.radians(degrees)
    return speed * math.cos(radians), speed * math.sin(radians)


def static(name, vx, vy, yaw, gait, episodes, duration=8.0, **extra):
    return {
        "name": name, "vx": vx, "vy": vy, "yaw": yaw, "gait": gait,
        "episodes": episodes, "duration": duration, "kind": "static", **extra,
    }


def conditions(suite):
    rows = []
    if suite in ("candidate", "anchor"):
        n = 30 if suite == "candidate" else 100
        rows.extend([
            static("WALK_0P6", .6, 0, 0, 0, n),
            static("WALK_1P2", 1.2, 0, 0, 0, n),
            static("RUN_1P2", 1.2, 0, 0, 1, n),
            static("RUN_2P4", 2.4, 0, 0, 1, n),
            {"name": "WALK_TO_RUN", "episodes": n, "duration": 12., "kind": "walk_to_run"},
            {"name": "RUN_TO_WALK", "episodes": n, "duration": 12., "kind": "run_to_walk"},
            {"name": "PRACTICAL_STOP", "episodes": n, "duration": 12., "kind": "stop"},
        ])
    elif suite == "translation_walk":
        for speed in (.3, .6, .9, 1.2):
            for degrees in DIRECTIONS:
                vx, vy = vector(speed, degrees)
                rows.append(static(f"WALK_S{speed:.1f}_D{degrees:05.1f}", vx, vy, 0, 0, 20,
                                   speed=speed, direction_deg=degrees))
    elif suite == "translation_run":
        for speed in (1.2, 1.6, 2.0, 2.4):
            for degrees in DIRECTIONS:
                vx, vy = vector(speed, degrees)
                rows.append(static(f"RUN_S{speed:.1f}_D{degrees:05.1f}", vx, vy, 0, 1, 10,
                                   speed=speed, direction_deg=degrees))
    elif suite == "yaw":
        for gait in (0, 1):
            for yaw in (-1., -.6, -.3, .3, .6, 1.):
                rows.append(static(f"{'RUN' if gait else 'WALK'}_YAW_{yaw:+.1f}", 0, 0, yaw, gait, 30))
    elif suite == "translation_yaw":
        for gait, speeds in ((0, (.6, 1.0)), (1, (1.2, 2.0))):
            for speed in speeds:
                for degrees in CARDINAL8:
                    for yaw in (-.6, -.3, 0., .3, .6):
                        vx, vy = vector(speed, degrees)
                        rows.append(static(
                            f"{'RUN' if gait else 'WALK'}_S{speed:.1f}_D{degrees:03d}_Y{yaw:+.1f}",
                            vx, vy, yaw, gait, 10, speed=speed, direction_deg=degrees,
                        ))
    elif suite == "independence":
        cases = ((270, .6), (90, -.6), (45, -.6), (315, .6), (180, .6), (180, -.6))
        for gait, speed in ((0, .6), (1, 1.2)):
            for degrees, yaw in cases:
                vx, vy = vector(speed, degrees)
                rows.append(static(
                    f"{'RUN' if gait else 'WALK'}_D{degrees:03d}_Y{yaw:+.1f}",
                    vx, vy, yaw, gait, 20, speed=speed, direction_deg=degrees,
                ))
    elif suite == "transitions":
        rows.extend([
            {"name": "WALK_DIRECTION_SEQUENCE", "episodes": 50, "duration": 24., "kind": "walk_sequence"},
            {"name": "RUN_DIRECTION_GAIT_SEQUENCE", "episodes": 50, "duration": 18., "kind": "run_sequence"},
        ])
    elif suite == "random":
        rows.extend([
            {"name": "WALK_RANDOM_60S", "episodes": 20, "duration": 60., "kind": "random_walk"},
            {"name": "RUN_CAPABLE_RANDOM_60S", "episodes": 20, "duration": 60., "kind": "random_run"},
        ])
    return rows


def command_for(condition, t, episode, random_schedules):
    kind = condition["kind"]
    if kind == "static":
        return condition["vx"], condition["vy"], condition["yaw"], condition["gait"], 0
    if kind in ("walk_to_run", "run_to_walk"):
        gait = 0. if kind == "walk_to_run" else 1.
        if 5 <= t < 7:
            blend = minimum_jerk((t - 5) / 2)
            gait = blend if kind == "walk_to_run" else 1 - blend
        elif t >= 7:
            gait = 1. if kind == "walk_to_run" else 0.
        return 1.2, 0., 0., gait, int(t >= 7)
    if kind == "stop":
        speed = 1.2 if t < 5 else 1.2 * (1 - minimum_jerk(t - 5)) if t < 6 else 0.
        return speed, 0., 0., 0., int(t >= 6)
    if kind in ("walk_sequence", "run_sequence"):
        if kind == "walk_sequence":
            targets = [
                (0., 0., 0., 0.), (.6, 0., 0., 0.), (0., .6, 0., 0.),
                (*vector(.6, 225), 0., 0.), (.6, 0., 0., 0.),
                (-.6, 0., 0., 0.), (0., 0., 0., 0.),
            ]
            segment = min(int(t // 3.0), len(targets) - 1)
            local = t - segment * 3.0
            ramp_duration = 1.5
        else:
            targets = [
                (1.2, 0., 0., 0.), (1.2, 0., 0., 1.),
                (*vector(1.6, 45), 0., 1.), (*vector(1.6, 315), 0., 1.),
                (1.2, 0., 0., 0.),
            ]
            segment = min(int(t // 3.5), len(targets) - 1)
            local = t - segment * 3.5
            ramp_duration = 2.0
        previous = targets[max(0, segment - 1)]
        target = targets[segment]
        blend = minimum_jerk(local / ramp_duration)
        value = tuple(a + (b - a) * blend for a, b in zip(previous, target))
        return *value, segment
    if kind.startswith("random"):
        schedule = random_schedules[(kind, episode)]
        selected = schedule[-1]
        for item in schedule:
            if t < item["end"]:
                selected = item
                break
        return selected["vx"], selected["vy"], selected["yaw"], selected["gait"], selected["segment"]
    raise ValueError(kind)


def make_random_schedules(rows):
    import random
    schedules = {}
    for condition in rows:
        if not condition["kind"].startswith("random"):
            continue
        for episode in range(condition["episodes"]):
            rng = random.Random(20261380 + episode + (1000 if condition["kind"] == "random_run" else 0))
            t, segment, values = 0., 0, []
            while t < 60:
                interval = rng.uniform(2, 4)
                speed = rng.uniform(0, 1.2 if condition["kind"] == "random_walk" else 2.4)
                direction = rng.uniform(0, 2 * math.pi)
                gait = 0.
                if condition["kind"] == "random_run" and speed >= 1.0:
                    gait = float(rng.random() < .5)
                values.append({
                    "end": min(60., t + interval), "vx": speed * math.cos(direction),
                    "vy": speed * math.sin(direction),
                    "yaw": rng.uniform(-.8, .8) if gait == 0 else rng.uniform(-.6, .6),
                    "gait": gait, "segment": segment,
                })
                t += interval
                segment += 1
            schedules[(condition["kind"], episode)] = values
    return schedules


def wrap_angle(value):
    return torch.atan2(torch.sin(value), torch.cos(value))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    spec = conditions(args.suite)
    random_schedules = make_random_schedules(spec)
    env_condition, env_episode = [], []
    for index, condition in enumerate(spec):
        env_condition.extend([index] * condition["episodes"])
        env_episode.extend(range(condition["episodes"]))
    count = len(env_condition)
    duration = max(row["duration"] for row in spec)
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = count
    cfg.episode_length_s = duration + 1.
    cfg.seed = 20261320 + list(parser._option_string_actions["--suite"].choices).index(args.suite)
    agent_cfg.seed = cfg.seed
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        device = env.device
        actor = FrozenGaitActor(checkpoint).to(device).eval()
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        foot_sensor = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        foot_bodies = [next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[j])
                       for j in foot_sensor]
        condition_ids = torch.tensor(env_condition, dtype=torch.long, device=device)
        episode_ids = torch.tensor(env_episode, dtype=torch.long, device=device)
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        initial_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        initial_pos = robot.data.root_pos_w[:, :2].clone()
        final_yaw_snapshot = initial_yaw.clone()
        final_pos_snapshot = initial_pos.clone()
        active = torch.ones(count, dtype=torch.bool, device=device)
        fallen = torch.zeros_like(active)
        excessive_tilt = torch.zeros_like(active)
        impact = torch.zeros_like(active)
        dangerous_slip = torch.zeros_like(active)
        saturation = torch.zeros_like(active)
        slip_streak = torch.zeros(count, dtype=torch.long, device=device)
        saturation_streak = torch.zeros_like(slip_streak)
        flight_streak = torch.zeros_like(slip_streak)
        flight_events = torch.zeros_like(slip_streak)
        flight_duration_sum = torch.zeros(count, device=device)
        alternating = torch.zeros_like(slip_streak)
        last_landing = torch.full_like(slip_streak, -1)
        steps_active = torch.zeros(count, device=device)
        sums = {key: torch.zeros(count, device=device) for key in (
            "vx", "vy", "yaw", "cmd_vx", "cmd_vy", "cmd_yaw", "cmd_gait", "vector_error",
            "yaw_error", "cross_axis", "flight", "single", "double", "left_contact",
            "right_contact", "slip", "roll_abs", "pitch_abs", "height", "vertical_velocity_abs",
            "joint_limit_proximity", "action_saturation", "speed_overshoot",
        )}
        max_force = torch.zeros(count, device=device)
        max_tilt = torch.zeros(count, device=device)
        acquisition_step = torch.full((count,), -1, dtype=torch.long, device=device)
        previous_segment = torch.full_like(acquisition_step, -1)
        segment_start = torch.zeros_like(acquisition_step)
        direction_overshoot = torch.zeros(count, device=device)
        max_steps = round(duration / float(env.step_dt))
        for step in range(max_steps):
            t = step * float(env.step_dt)
            vx = torch.zeros(count, device=device)
            vy = torch.zeros_like(vx)
            yaw_cmd = torch.zeros_like(vx)
            gait = torch.zeros_like(vx)
            segment = torch.zeros(count, dtype=torch.long, device=device)
            valid_time = torch.zeros(count, dtype=torch.bool, device=device)
            for index, condition in enumerate(spec):
                mask_ids = torch.where(condition_ids == index)[0]
                valid = t < condition["duration"]
                if not valid:
                    continue
                valid_time[mask_ids] = True
                for env_id in mask_ids.cpu().tolist():
                    values = command_for(condition, t, int(episode_ids[env_id]), random_schedules)
                    vx[env_id], vy[env_id], yaw_cmd[env_id], gait[env_id], segment[env_id] = values
            changed = segment != previous_segment
            segment_start[changed] = step
            acquisition_step[changed] = -1
            previous_segment.copy_(segment)
            command.external_override[:, 0] = vx
            command.external_override[:, 1] = vy
            command.external_override[:, 2] = yaw_cmd
            if step == 0:
                command._update_command()
                obs = wrapped.get_observations().to(device)
            with torch.inference_mode():
                action = actor(obs["policy"], gait)
                action[~active | ~valid_time] = 0
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            just_fell = dones.bool() & ~timeout & active & valid_time
            fallen |= just_fell
            actual = robot.data.root_lin_vel_b
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            forces = sensor.data.net_forces_w_history[:, -1, foot_sensor, :].norm(dim=-1)
            contacts = forces > 5
            in_flight = contacts.sum(-1) == 0
            previous_flight = flight_streak.clone()
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            landed = (~in_flight) & (previous_flight > 0)
            safety_measure = active & valid_time
            flight_events += (in_flight & (previous_flight == 0) & safety_measure).long()
            flight_duration_sum += torch.where(landed & safety_measure, previous_flight.float() * float(env.step_dt), 0)
            landing_foot = contacts.long().argmax(-1)
            alternate_now = landed & (contacts.sum(-1) == 1) & (last_landing >= 0) & (landing_foot != last_landing)
            alternating += alternate_now.long()
            last_landing[landed & (contacts.sum(-1) == 1)] = landing_foot[landed & (contacts.sum(-1) == 1)]
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, foot_bodies, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            dangerous_slip |= (slip_streak >= 5) & safety_measure
            impact |= (forces.amax(-1) > 3500) & safety_measure
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            velocity_ratio = robot.data.joint_vel.abs() / limits.clamp_min(1e-6)
            saturated_now = (velocity_ratio > .95).any(-1)
            saturation_streak = torch.where(saturated_now, saturation_streak + 1, torch.zeros_like(saturation_streak))
            saturation |= (saturation_streak >= 5) & safety_measure
            gravity = robot.data.projected_gravity_b
            roll_abs = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            pitch_abs = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
            tilt = torch.maximum(roll_abs, pitch_abs)
            excessive_tilt |= (tilt > .8) & safety_measure
            final_yaw_snapshot[safety_measure] = yaw_from_quat_wxyz(robot.data.root_quat_w)[safety_measure]
            final_pos_snapshot[safety_measure] = robot.data.root_pos_w[safety_measure, :2]
            measure = safety_measure.clone()
            for index, condition in enumerate(spec):
                if condition["kind"] in ("walk_to_run", "run_to_walk", "stop"):
                    measure[(condition_ids == index) & (t < 7.)] = False
            cmd_speed = torch.sqrt(vx**2 + vy**2)
            actual_speed = torch.linalg.vector_norm(actual[:, :2], dim=-1)
            vector_error = torch.linalg.vector_norm(actual[:, :2] - torch.stack((vx, vy), dim=1), dim=-1)
            cmd_unit = torch.stack((vx, vy), dim=1) / cmd_speed[:, None].clamp_min(1e-6)
            cross = torch.abs(actual[:, 0] * cmd_unit[:, 1] - actual[:, 1] * cmd_unit[:, 0])
            acquired = (vector_error <= .35) & ((actual_yaw - yaw_cmd).abs() <= .25) & (acquisition_step < 0)
            acquisition_step[acquired & measure] = step - segment_start[acquired & measure]
            steps_active += measure.float()
            values = {
                "vx": actual[:, 0], "vy": actual[:, 1], "yaw": actual_yaw,
                "cmd_vx": vx, "cmd_vy": vy, "cmd_yaw": yaw_cmd, "cmd_gait": gait,
                "vector_error": vector_error, "yaw_error": (actual_yaw - yaw_cmd).abs(),
                "cross_axis": cross, "flight": in_flight.float(),
                "single": (contacts.sum(-1) == 1).float(), "double": (contacts.sum(-1) == 2).float(),
                "left_contact": contacts[:, 0].float(), "right_contact": contacts[:, 1].float(),
                "slip": slipping.float(), "roll_abs": roll_abs, "pitch_abs": pitch_abs,
                "height": robot.data.root_pos_w[:, 2],
                "vertical_velocity_abs": actual[:, 2].abs(),
                "joint_limit_proximity": velocity_ratio.amax(-1),
                "action_saturation": saturated_now.float(),
                "speed_overshoot": torch.clamp(actual_speed - cmd_speed, min=0),
            }
            for key, value in values.items():
                sums[key] += torch.where(measure, value, 0)
            max_force = torch.maximum(max_force, torch.where(measure, forces.amax(-1), 0))
            max_tilt = torch.maximum(max_tilt, torch.where(measure, tilt, 0))
            # Safe early termination guard. Metrics after the first hazard are excluded.
            guard = (just_fell | (tilt > .8) | impact | saturation) & valid_time
            active &= ~guard
        displacement = torch.linalg.vector_norm(final_pos_snapshot - initial_pos, dim=-1)
        heading_drift = wrap_angle(final_yaw_snapshot - initial_yaw).abs()
        rows = []
        for env_id in range(count):
            n = max(float(steps_active[env_id]), 1.)
            c = spec[env_condition[env_id]]
            mean = {key: float(value[env_id] / n) for key, value in sums.items()}
            cmd_mag = math.hypot(mean["cmd_vx"], mean["cmd_vy"])
            actual_mag = math.hypot(mean["vx"], mean["vy"])
            if cmd_mag > .05 and actual_mag > .02:
                dot = mean["cmd_vx"] * mean["vx"] + mean["cmd_vy"] * mean["vy"]
                cosine = max(-1., min(1., dot / (cmd_mag * actual_mag)))
                direction_error = math.degrees(math.acos(cosine))
            else:
                direction_error = None
            periodic = int(flight_events[env_id]) >= 4 and int(alternating[env_id]) >= 3
            if bool(fallen[env_id]):
                gait_label = "FALL"
            elif cmd_mag < .05 and actual_mag <= .10 and mean["flight"] < .02:
                gait_label = "STAND_OR_NEAR_STAND"
            elif mean["flight"] < .10:
                gait_label = "WALK_LIKE"
            elif int(flight_events[env_id]) in (1, 2):
                gait_label = "ISOLATED_FLIGHT"
            elif periodic:
                gait_label = "PERIODIC_RUNNING"
            else:
                gait_label = "IRREGULAR"
            target_run_fraction = mean["cmd_gait"]
            gait_success = (
                gait_label == "STAND_OR_NEAR_STAND" if cmd_mag < .05
                else gait_label == ("PERIODIC_RUNNING" if target_run_fraction >= .5 else "WALK_LIKE")
            )
            rows.append({
                "condition": c["name"], "episode": env_episode[env_id],
                "commanded_speed_mps": cmd_mag, "actual_speed_mps": actual_mag,
                "cmd_vx": mean["cmd_vx"], "cmd_vy": mean["cmd_vy"], "cmd_yaw_rate": mean["cmd_yaw"],
                "gait_cmd_mean": mean["cmd_gait"],
                "actual_vx_body": mean["vx"], "actual_vy_body": mean["vy"], "actual_yaw_rate": mean["yaw"],
                "vx_error": abs(mean["vx"] - mean["cmd_vx"]), "vy_error": abs(mean["vy"] - mean["cmd_vy"]),
                "vector_velocity_mae": mean["vector_error"], "direction_error_deg": direction_error,
                "cross_axis_velocity": mean["cross_axis"], "yaw_rate_mae": mean["yaw_error"],
                "heading_drift_rad": float(heading_drift[env_id]),
                "position_drift_m": float(displacement[env_id]),
                "path_curvature_radpm": float(heading_drift[env_id] / displacement[env_id].clamp_min(.05)),
                "fall": bool(fallen[env_id]), "excessive_tilt": bool(excessive_tilt[env_id]),
                "base_roll_abs_mean": mean["roll_abs"], "base_pitch_abs_mean": mean["pitch_abs"],
                "base_height_mean": mean["height"], "vertical_velocity_abs_mean": mean["vertical_velocity_abs"],
                "flight_fraction": mean["flight"], "flight_events": int(flight_events[env_id]),
                "flight_event_duration_mean": float(flight_duration_sum[env_id] / max(int(flight_events[env_id]), 1)),
                "stride_frequency_hz": float(flight_events[env_id] / max(n * float(env.step_dt), 1e-6)),
                "single_support_fraction": mean["single"], "double_support_fraction": mean["double"],
                "alternating_landing_events": int(alternating[env_id]),
                "left_contact_fraction": mean["left_contact"], "right_contact_fraction": mean["right_contact"],
                "contact_symmetry_error": abs(mean["left_contact"] - mean["right_contact"]),
                "foot_slip_fraction": mean["slip"], "dangerous_slip": bool(dangerous_slip[env_id]),
                "max_impact_force_n": float(max_force[env_id]), "impact_failure": bool(impact[env_id]),
                "joint_limit_proximity": mean["joint_limit_proximity"],
                "action_saturation_fraction": mean["action_saturation"],
                "long_dwell_saturation": bool(saturation[env_id]),
                "gait_classification": gait_label, "target_gait_success": gait_success,
                "transition_time_s": None if int(acquisition_step[env_id]) < 0 else int(acquisition_step[env_id]) * float(env.step_dt),
                "speed_overshoot": mean["speed_overshoot"],
                "early_termination": not bool(active[env_id]),
            })
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["condition"]].append(row)
        summary_rows = []
        numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != "episode"]
        booleans = [key for key, value in rows[0].items() if isinstance(value, bool)]
        for name, values in grouped.items():
            record = {"condition": name, "episodes": len(values)}
            for key in numeric:
                present = [row[key] for row in values if row[key] is not None]
                record[key] = sum(present) / len(present) if present else None
            for key in booleans:
                record[f"{key}_rate"] = sum(row[key] for row in values) / len(values)
            labels = defaultdict(int)
            for row in values:
                labels[row["gait_classification"]] += 1
            record["gait_classification_counts"] = dict(labels)
            summary_rows.append(record)
        stem = {
            "candidate": f"_candidate_{args.tag or checkpoint.stem}",
            "anchor": "anchor_baseline", "translation_walk": "pure_translation_walk",
            "translation_run": "pure_translation_run", "yaw": "pure_yaw_results",
            "translation_yaw": "translation_yaw_matrix", "independence": "direction_heading_independence",
            "transitions": "direction_transition_results", "random": "random_command_results",
        }[args.suite]
        payload = {
            "suite": args.suite, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "deterministic": True, "training_updates": 0, "rows": summary_rows,
            "episode_rows": rows, "early_termination_guard": {
                "fall": True, "excessive_tilt_rad": .8, "impact_force_n": 3500,
                "long_dwell_joint_velocity_saturation": True,
            },
        }
        (OUT / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(json.dumps({"suite": args.suite, "environments": count, "output": stem,
                          "checkpoint_sha256": payload["checkpoint_sha256"]}, sort_keys=True))
        wrapped.close()


if __name__ == "__main__":
    main()
