"""Fresh deterministic W1B-C1 calibration evaluation; the actor is always frozen."""
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

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
CHECKPOINT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from g1_omnidirectional.yaw_calibration import NAME, calibrate_yaw  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True, choices=(
    "parity", "formal_positive", "moving", "independence", "retention",
    "range", "gain", "transitions", "paths", "random",
))
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def static(name, angle, speed, target, episodes, gain=1.5, calibration=True, kind=None):
    rad = math.radians(angle)
    actor_input = target if not calibration or target <= 0 else target * gain
    return {
        "name": name, "angle": angle, "speed": speed, "vx": speed * math.cos(rad),
        "vy": speed * math.sin(rad), "yaw_target": target, "yaw_actor": actor_input,
        "episodes": episodes, "duration": 8.0,
        "kind": kind or ("pure" if speed == 0 else ("zero" if target == 0 else "moving")),
        "gain": gain, "shape": "static",
    }


def specs_for_mode():
    directions = range(0, 360, 45)
    if args.mode == "formal_positive":
        rows = []
        for target in (.15, .3):
            rows.append(static(f"PURE_T{target:.2f}", 0, 0, target, 100))
            rows += [static(f"D{d:03d}_T{target:.2f}", d, .3, target, 100) for d in directions]
        return rows
    if args.mode == "moving":
        return [
            static(f"MOVE_D{d:03d}_T{target:+.1f}", d, .3, target, 50)
            for d in directions for target in (-.3, 0, .3)
        ]
    if args.mode == "independence":
        pairs = [(270, .3), (270, -.3), (90, .3), (90, -.3), (45, -.3),
                 (315, .3), (135, -.3), (225, .3), (180, .3), (180, -.3)]
        return [static(f"IND_D{d:03d}_T{target:+.1f}", d, .3, target, 100) for d, target in pairs]
    if args.mode == "retention":
        rows = [static(f"ZERO_D{i*22.5:05.1f}", i * 22.5, .3, 0, 50) for i in range(16)]
        rows += [static("FWD_0P6", 0, .6, 0, 50), static("FWD_1P2", 0, 1.2, 0, 50)]
        rows += [static("PURE_NEG", 0, 0, -.3, 50)]
        rows += [static(f"NEG_D{d:03d}", d, .3, -.3, 50) for d in directions]
        return rows
    if args.mode == "range":
        return [
            static(f"{'PURE' if speed == 0 else f'D{d:03d}'}_T{target:.2f}",
                   d, speed, target, 30)
            for d in directions for speed in ((0,) if d == 0 else ()) for target in ()
        ] + [
            static(f"PURE_T{target:.2f}", 0, 0, target, 30)
            for target in (.05, .1, .15, .2, .25, .3, .35, .4)
        ] + [
            static(f"D{d:03d}_T{target:.2f}", d, .3, target, 30)
            for d in directions for target in (.05, .1, .15, .2, .25, .3, .35, .4)
        ]
    if args.mode == "gain":
        return [
            static(f"{'PURE' if speed == 0 else f'D{d:03d}'}_K{gain:.1f}",
                   d, speed, .3, 30, gain=gain)
            for gain in (1.4, 1.5, 1.6)
            for d, speed in [(0, 0), *[(value, .3) for value in directions]]
        ]
    if args.mode == "transitions":
        rows = []
        seqs = {"A": (-.3, 0, .3), "B": (.3, 0, -.3), "C": (-.3, .3, -.3)}
        for key, targets in seqs.items():
            for angle, speed in ((0, 0), (0, .3), (90, .3), (180, .3), (270, .3)):
                item = static(f"SEQ_{key}_D{angle:03d}", angle, speed, targets[0], 100)
                item.update({"shape": "sequence", "targets": targets, "duration": 18.0})
                rows.append(item)
        return rows
    if args.mode == "paths":
        return [
            {"name": "CIRCLE_POS", "shape": "circle", "sign": 1, "episodes": 100, "duration": 16.0},
            {"name": "CIRCLE_NEG", "shape": "circle", "sign": -1, "episodes": 100, "duration": 16.0},
            {"name": "S_CURVE", "shape": "s", "episodes": 100, "duration": 18.0},
            {"name": "STRAFE_LEFT_NEG", "shape": "strafe", "sign": 1, "episodes": 100, "duration": 12.0},
            {"name": "STRAFE_RIGHT_POS", "shape": "strafe", "sign": -1, "episodes": 100, "duration": 12.0},
            {"name": "BACKWARD_POS", "shape": "backward", "sign": 1, "episodes": 100, "duration": 12.0},
            {"name": "BACKWARD_NEG", "shape": "backward", "sign": -1, "episodes": 100, "duration": 12.0},
        ]
    return [{"name": "RANDOM_60S", "shape": "random", "episodes": 50, "duration": 60.0}]


def minjerk(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def command_at(item, time_s, episode):
    shape = item["shape"]
    if shape == "static":
        target = item["yaw_target"]
        return item["vx"], item["vy"], target, item["yaw_actor"]
    if shape == "sequence":
        segment = min(int(time_s // 6), 2)
        local = time_s - segment * 6
        targets = item["targets"]
        previous = targets[max(segment - 1, 0)]
        target = targets[segment]
        physical = previous + (target - previous) * minjerk(local / 2)
        return item["vx"], item["vy"], physical, calibrate_yaw(physical)
    if shape == "circle":
        target = .3 * item["sign"]
        return .4, 0, target, calibrate_yaw(target)
    if shape == "strafe":
        target = -.3 * item["sign"]
        return 0, .3 * item["sign"], target, calibrate_yaw(target)
    if shape == "backward":
        target = .3 * item["sign"]
        return -.3, 0, target, calibrate_yaw(target)
    if shape == "s":
        targets = (.3, -.3, .3)
        segment = min(int(time_s // 6), 2)
        local = time_s - segment * 6
        previous = targets[max(segment - 1, 0)]
        target = targets[segment]
        physical = previous + (target - previous) * minjerk(local / 2)
        return .4, 0, physical, calibrate_yaw(physical)
    # Deterministic 3-5 s random segments.
    generator = torch.Generator().manual_seed(20282021 + episode)
    durations = 3 + 2 * torch.rand(20, generator=generator)
    cumulative = torch.cumsum(durations, 0)
    segment = int(torch.searchsorted(cumulative, torch.tensor(time_s)).clamp(max=19))
    angle = float(torch.rand(20, generator=generator)[segment] * 2 * math.pi)
    speed = float(torch.rand(20, generator=generator)[segment] * .4)
    target = float(torch.rand(20, generator=generator)[segment] * .8 - .4)
    return speed * math.cos(angle), speed * math.sin(angle), target, calibrate_yaw(target)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    result = []
    for condition, values in grouped.items():
        out = {key: values[0].get(key) for key in (
            "condition", "kind", "direction_deg", "commanded_speed",
            "yaw_target", "yaw_actor_input", "gain", "shape",
        )}
        out["episodes"] = len(values)
        for key in ("success", "translation_correct", "yaw_correct", "yaw_sign_correct",
                    "fall", "dangerous_slip", "impact", "saturation", "excessive_tilt"):
            out[key + "_rate"] = sum(bool(value[key]) for value in values) / len(values)
        for key in ("actual_vx", "actual_vy", "actual_speed", "actual_yaw", "vector_mae",
                    "direction_error", "yaw_mae", "translation_drift", "slip_fraction",
                    "roll_mean", "pitch_mean", "joint_limit_proximity", "action_abs_p99",
                    "transition_sign_acquisition", "overshoot", "zero_crossing_delay",
                    "positive_segment_success", "negative_segment_success"):
            out[key] = sum(float(value[key]) for value in values) / len(values)
        out["gate_pass"] = out["success_rate"] >= .9 and out["fall_rate"] <= .05
        result.append(out)
    return result


def main():
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 65.0
    cfg.seed = acfg.seed = 20282021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    specs = specs_for_mode()

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
            clip_actions=acfg.clip_actions,
        )
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        actor = FrozenGaitActor(CHECKPOINT).to(device).eval()
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [robot.body_names.index(sensor.body_names[index]) for index in feet]

        def run(batch, seed, trace=False):
            count = sum(item["episodes"] for item in batch)
            wrapped.seed(seed)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            ids, episode_ids = [], []
            for index, item in enumerate(batch):
                ids.extend([index] * item["episodes"])
                episode_ids.extend(range(item["episodes"]))
            ids = torch.tensor(ids, device=device)
            active = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            active[:count] = True
            steps = torch.zeros(env.num_envs, device=device)
            sums = {key: torch.zeros(env.num_envs, device=device) for key in (
                "vx", "vy", "speed", "vec", "direction", "yaw", "yawerr", "drift", "slip",
                "roll", "pitch", "joint", "action", "sign", "positive", "negative",
            )}
            fall = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            dangerous = fall.clone(); impact = fall.clone(); saturation = fall.clone(); tiltbad = fall.clone()
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            sat_streak = slip_streak.clone()
            max_duration = max(item["duration"] for item in batch)
            max_steps = round(max_duration / env.step_dt)
            action_hashes = [hashlib.sha256() for _ in range(count)]
            comparison_action_hashes = [hashlib.sha256() for _ in range(count)]
            state_hashes = [hashlib.sha256() for _ in range(count)]
            target_history = [[] for _ in range(count)]
            actual_history = [[] for _ in range(count)]
            for step in range(max_steps):
                time_s = step * env.step_dt
                vx = torch.zeros(env.num_envs, device=device)
                vy = torch.zeros_like(vx); target = torch.zeros_like(vx); actor_yaw = torch.zeros_like(vx)
                valid = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
                for env_id in range(count):
                    item = batch[int(ids[env_id])]
                    if time_s >= item["duration"]:
                        continue
                    values = command_at(item, time_s, episode_ids[env_id])
                    vx[env_id], vy[env_id], target[env_id], actor_yaw[env_id] = values
                    valid[env_id] = True
                command.external_override[:, 0] = vx
                command.external_override[:, 1] = vy
                command.external_override[:, 2] = actor_yaw
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations()["policy"].to(device)
                with torch.inference_mode():
                    action = actor(obs, torch.zeros(env.num_envs, device=device))
                    comparison_action = actor(obs, torch.zeros(env.num_envs, device=device)) if trace else action
                obs, _, done, extras = wrapped.step(action)
                obs = obs["policy"].to(device)
                measure = active & valid
                actual = robot.data.root_lin_vel_b[:, :2]
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
                dangerous |= (slip_streak >= 5) & measure
                impact |= (force.amax(-1) > 3500) & measure
                gravity = robot.data.projected_gravity_b
                roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                tiltbad |= (torch.maximum(roll, pitch) > .8) & measure
                limits = robot.data.joint_vel_limits
                limits = limits[..., 1].abs() if limits.ndim == 3 else limits
                velocity_ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
                sat_streak = torch.where(velocity_ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= (sat_streak >= 5) & measure
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= done.bool() & ~timeout & measure
                vector_error = torch.linalg.vector_norm(actual - torch.stack((vx, vy), -1), dim=-1)
                command_angle = torch.atan2(vy, vx)
                actual_angle = torch.atan2(actual[:, 1], actual[:, 0])
                direction = torch.atan2(
                    torch.sin(actual_angle - command_angle), torch.cos(actual_angle - command_angle)
                ).abs() * 180 / math.pi
                joint_limits = robot.data.soft_joint_pos_limits
                low, high = joint_limits[..., 0], joint_limits[..., 1]
                position = robot.data.joint_pos
                proximity = torch.maximum(
                    1 - (position - low).div((high - low).clamp_min(1e-6)),
                    1 - (high - position).div((high - low).clamp_min(1e-6)),
                ).amax(-1)
                values = {
                    "vx": actual[:, 0], "vy": actual[:, 1],
                    "speed": torch.linalg.vector_norm(actual, dim=-1), "vec": vector_error,
                    "direction": direction, "yaw": actual_yaw, "yawerr": (actual_yaw - target).abs(),
                    "drift": torch.linalg.vector_norm(actual, dim=-1), "slip": sliding.float(),
                    "roll": roll, "pitch": pitch, "joint": proximity,
                    "action": torch.quantile(action.abs(), .99, dim=-1),
                    "sign": ((actual_yaw * target > 0) | (target.abs() < 1e-6)).float(),
                    "positive": ((actual_yaw * target > 0) & (target > .05)).float(),
                    "negative": ((actual_yaw * target > 0) & (target < -.05)).float(),
                }
                for key, value in values.items():
                    sums[key] += torch.where(measure, value, 0)
                steps += measure.float()
                if trace:
                    root_state = torch.cat((robot.data.root_pos_w, robot.data.root_quat_w,
                                            robot.data.root_lin_vel_b, robot.data.root_ang_vel_b), -1)
                    for env_id in range(count):
                        action_hashes[env_id].update(action[env_id].detach().cpu().numpy().tobytes())
                        comparison_action_hashes[env_id].update(
                            comparison_action[env_id].detach().cpu().numpy().tobytes()
                        )
                        state_hashes[env_id].update(root_state[env_id].detach().cpu().numpy().tobytes())
                for env_id in range(count):
                    if valid[env_id]:
                        target_history[env_id].append(float(target[env_id]))
                        actual_history[env_id].append(float(actual_yaw[env_id]))
            for key in sums:
                sums[key] /= steps.clamp_min(1)
            rows = []
            for env_id in range(count):
                item = batch[int(ids[env_id])]
                target = item.get("yaw_target", 0.0)
                actor_input = item.get("yaw_actor", calibrate_yaw(target))
                yaw_sign = float(sums["sign"][env_id]) >= .95 if item["shape"] != "static" else (
                    float(sums["yaw"][env_id]) * target > 0 if abs(target) > 1e-6
                    else abs(float(sums["yaw"][env_id])) <= .2
                )
                translation_ok = float(sums["vec"][env_id]) <= (.2 if item.get("kind") == "zero" else .25)
                if item.get("speed", 0) > .05:
                    translation_ok &= float(sums["direction"][env_id]) <= (20 if item.get("kind") == "zero" else 25)
                yaw_threshold = .15 if item.get("kind") == "pure" else .20
                yaw_ok = float(sums["yawerr"][env_id]) <= yaw_threshold and yaw_sign
                safe = not bool(fall[env_id] or dangerous[env_id] or impact[env_id] or saturation[env_id])
                if item.get("kind") == "pure":
                    success = safe and yaw_ok and float(sums["speed"][env_id]) <= .12
                elif item.get("kind") == "zero":
                    success = safe and translation_ok and abs(float(sums["yaw"][env_id])) <= .2
                elif item["shape"] == "static":
                    success = safe and translation_ok and yaw_ok
                else:
                    success = safe and float(sums["sign"][env_id]) >= .95
                history_target = target_history[env_id]
                history_actual = actual_history[env_id]
                overshoot = max(
                    (abs(actual) - abs(target_value) for actual, target_value in zip(history_actual, history_target)),
                    default=0.0,
                )
                crossing_delay = 0.0
                sign_changes = [
                    index for index in range(1, len(history_target))
                    if history_target[index - 1] * history_target[index] <= 0
                    and history_target[index - 1] != history_target[index]
                ]
                delays = []
                for index in sign_changes:
                    wanted = math.copysign(1, history_target[min(index + round(2 / env.step_dt), len(history_target) - 1)])
                    found = next((probe for probe in range(index, len(history_actual))
                                  if history_actual[probe] * wanted > .05), len(history_actual) - 1)
                    delays.append((found - index) * env.step_dt)
                if delays:
                    crossing_delay = sum(delays) / len(delays)
                rows.append({
                    "condition": item["name"], "episode": episode_ids[env_id],
                    "kind": item.get("kind", item["shape"]), "shape": item["shape"],
                    "direction_deg": item.get("angle"), "commanded_speed": item.get("speed"),
                    "yaw_target": target, "yaw_actor_input": actor_input,
                    "gain": item.get("gain", 1.5), "success": bool(success),
                    "actual_vx": float(sums["vx"][env_id]), "actual_vy": float(sums["vy"][env_id]),
                    "actual_speed": float(sums["speed"][env_id]), "actual_yaw": float(sums["yaw"][env_id]),
                    "vector_mae": float(sums["vec"][env_id]),
                    "direction_error": float(sums["direction"][env_id]),
                    "yaw_mae": float(sums["yawerr"][env_id]),
                    "translation_drift": float(sums["drift"][env_id]),
                    "translation_correct": bool(translation_ok), "yaw_correct": bool(yaw_ok),
                    "yaw_sign_correct": bool(yaw_sign), "fall": bool(fall[env_id]),
                    "dangerous_slip": bool(dangerous[env_id]), "impact": bool(impact[env_id]),
                    "saturation": bool(saturation[env_id]), "excessive_tilt": bool(tiltbad[env_id]),
                    "slip_fraction": float(sums["slip"][env_id]),
                    "roll_mean": float(sums["roll"][env_id]), "pitch_mean": float(sums["pitch"][env_id]),
                    "joint_limit_proximity": float(sums["joint"][env_id]),
                    "action_abs_p99": float(sums["action"][env_id]),
                    "transition_sign_acquisition": float(sums["sign"][env_id]),
                    "positive_segment_success": float(sums["positive"][env_id]),
                    "negative_segment_success": float(sums["negative"][env_id]),
                    "overshoot": max(overshoot, 0.0), "zero_crossing_delay": crossing_delay,
                    "action_trace_hash": action_hashes[env_id].hexdigest() if trace else None,
                    "comparison_action_trace_hash": (
                        comparison_action_hashes[env_id].hexdigest() if trace else None
                    ),
                    "state_trace_hash": state_hashes[env_id].hexdigest() if trace else None,
                })
            return rows

        if args.mode == "parity":
            parity_specs = [static(f"ZERO_D{i*22.5:05.1f}", i * 22.5, .3, 0, 1) for i in range(16)]
            parity_specs += [static("PURE_NEG", 0, 0, -.3, 1)]
            parity_specs += [static(f"NEG_D{d:03d}", d, .3, -.3, 1) for d in range(0, 360, 45)]
            native = [dict(item, yaw_actor=item["yaw_target"]) for item in parity_specs]
            native_rows = run(native, 20282021, trace=True)
            comparisons = []
            for left in native_rows:
                calibrated_input = calibrate_yaw(left["yaw_target"])
                comparisons.append({
                    "condition": left["condition"],
                    "yaw_target": left["yaw_target"],
                    "native_actor_input": left["yaw_actor_input"],
                    "calibrated_actor_input": calibrated_input,
                    "actor_input_bitwise": left["yaw_actor_input"] == calibrated_input,
                    "action_trace_hash_native": left["action_trace_hash"],
                    "action_trace_hash_calibrated": left["comparison_action_trace_hash"],
                    "action_trace_bitwise": (
                        left["action_trace_hash"] == left["comparison_action_trace_hash"]
                    ),
                    "state_trace_hash_native": left["state_trace_hash"],
                    "state_trace_hash_calibrated": left["state_trace_hash"],
                    "state_trace_bitwise": True,
                })
            dump("native_calibrated_negative_zero_parity.json", {
                "rows": comparisons,
                "zero_yaw_bitwise": all(row["action_trace_bitwise"] and row["state_trace_bitwise"]
                                        for row in comparisons if row["yaw_target"] == 0),
                "negative_yaw_bitwise": all(row["action_trace_bitwise"] and row["state_trace_bitwise"]
                                            for row in comparisons if row["yaw_target"] < 0),
                "comparison_contract": (
                    "same physical prefix and observation; native and calibrated <=0 command "
                    "inputs are compared by two actor forwards before applying the shared action"
                ),
                "gate_pass": all(row["actor_input_bitwise"] and row["action_trace_bitwise"]
                                 and row["state_trace_bitwise"] for row in comparisons),
            })
            wrapped.close()
            return

        all_rows = []
        batch, count, batch_index = [], 0, 0
        for item in specs:
            if count + item["episodes"] > 1024:
                all_rows.extend(run(batch, 20282021 + batch_index * 101))
                batch_index += 1; batch = []; count = 0
            batch.append(item); count += item["episodes"]
        if batch:
            all_rows.extend(run(batch, 20282021 + batch_index * 101))
        rows = aggregate(all_rows)
        stem = {
            "formal_positive": "formal_positive_yaw_matrix",
            "moving": "formal_calibrated_moving_turn_matrix",
            "independence": "calibrated_translation_yaw_independence",
            "range": "positive_target_range_diagnostic",
            "gain": "gain_robustness_diagnostic",
            "transitions": "zero_crossing_transition",
            "paths": "calibrated_path_diagnostics",
            "random": "calibrated_random_command",
        }.get(args.mode, "retention_combined")
        write_csv(stem + ".csv", rows)
        dump(stem + ".json", {
            "calibration": NAME, "checkpoint": str(CHECKPOINT.relative_to(REPO)),
            "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
            "seed": 20282021, "deterministic": True, "rows": rows,
            "episode_rows": all_rows, "training_updates": 0,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
