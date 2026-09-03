"""Frozen-actor formal dynamic endpoint evaluation for W1B-C2."""
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
import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c2_shared_yaw_endpoint_evaluator"
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
from g1_omnidirectional.yaw_calibration import calibrate_yaw  # noqa: E402
from g1_omnidirectional.yaw_endpoint_evaluator import Exp013YawEndpointEvaluator  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

DT = .02
EPISODES = 100
DIRECTIONS = [None, 0, 45, 90, 135, 180, 225, 270, 315]
TRANSITIONS = {
    "NEG_ZERO_POS": (-.3, .3, "via_zero"),
    "POS_ZERO_NEG": (.3, -.3, "via_zero"),
    "NEG_POS": (-.3, .3, "direct"),
    "POS_NEG": (.3, -.3, "direct"),
}


def minjerk(value):
    value = max(0., min(1., value))
    return value**3 * (10 - 15 * value + 6 * value**2)


def physical_target(initial, final, profile, time_s):
    if time_s < 4:
        return initial
    if time_s >= 6:
        return final
    if profile == "direct":
        return initial + (final - initial) * minjerk((time_s - 4) / 2)
    if time_s < 5:
        return initial * (1 - minjerk(time_s - 4))
    return final * minjerk(time_s - 5)


def write_csv(path, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(episode_rows, acquisition_rows):
    grouped = defaultdict(list)
    for row in episode_rows:
        grouped[row["condition"]].append(row)
    condition_rows = []
    for condition, rows in sorted(grouped.items()):
        condition_rows.append({
            "condition": condition, "transition": rows[0]["transition"],
            "direction_deg": rows[0]["direction_deg"], "episodes": len(rows),
            "endpoint_success_rate": sum(r["endpoint_success"] for r in rows) / len(rows),
            "condition_pass": (
                sum(r["endpoint_success"] for r in rows) / len(rows) >= .9
                and sum(r["fall"] for r in rows) / len(rows) <= .05
            ),
            "endpoint_mean_yaw": np.mean([r["endpoint_mean_yaw"] for r in rows]),
            "endpoint_yaw_mae": np.mean([r["endpoint_yaw_mae"] for r in rows]),
            "translation_vector_mae": np.mean([r["translation_vector_mae"] for r in rows]),
            "translation_direction_error_deg": np.mean(
                [r["translation_direction_error_deg"] for r in rows]),
            "fall_rate": np.mean([r["fall"] for r in rows]),
            "dangerous_slip_rate": np.mean([r["dangerous_slip"] for r in rows]),
            "impact_rate": np.mean([r["impact"] for r in rows]),
            "saturation_rate": np.mean([r["long_dwell_saturation"] for r in rows]),
        })
    acquisition_summary = []
    acquisition_groups = defaultdict(list)
    for row in acquisition_rows:
        acquisition_groups[row["condition"]].append(row)
    for condition, rows in sorted(acquisition_groups.items()):
        item = {"condition": condition, "episodes": len(rows)}
        for key in (
            "first_instantaneous_correct_sign_s", "first_static_mae_pass_s",
            "first_0p10_sustained_endpoint_like_pass_s",
            "first_0p20_sustained_endpoint_like_pass_s",
            "first_complete_gait_cycle_mean_pass_s",
        ):
            values = [r[key] for r in rows if not math.isnan(r[key])]
            item[key + "_median"] = float(np.median(values)) if values else None
            item[key + "_p95"] = float(np.quantile(values, .95)) if values else None
        item["never_acquired_rate"] = np.mean([r["never_acquired_target_sign"] for r in rows])
        acquisition_summary.append(item)
    write_csv(OUT / "formal_dynamic_yaw_transitions.csv", condition_rows)
    (OUT / "formal_dynamic_yaw_transitions.json").write_text(json.dumps({
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "calibration": "MonotonicPositiveYawCalibrationV1",
        "endpoint_window_s": [6, 12], "rows": condition_rows,
        "episode_rows": episode_rows, "conditions_pass": sum(r["condition_pass"] for r in condition_rows),
        "conditions_total": len(condition_rows), "training_updates": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "dynamic_yaw_acquisition_diagnostics.csv", acquisition_summary)
    (OUT / "dynamic_yaw_acquisition_diagnostics.json").write_text(json.dumps({
        "formal_gate_member": False, "rows": acquisition_summary,
        "episode_rows": acquisition_rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 64 if args.debug else 1024
    cfg.episode_length_s = 13.0
    cfg.seed = acfg.seed = 20285021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    evaluator = Exp013YawEndpointEvaluator()
    episode_rows = []
    acquisition_rows = []
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
        episodes = 1 if args.debug else EPISODES
        for transition_index, (transition, (initial, final, profile)) in enumerate(TRANSITIONS.items()):
            count = len(DIRECTIONS) * episodes
            condition_id = np.repeat(np.arange(len(DIRECTIONS)), episodes)
            episode_id = np.tile(np.arange(episodes), len(DIRECTIONS))
            wrapped.seed(20285021 + transition_index * 1009)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            endpoint_steps = 300
            sums = {name: np.zeros(count, dtype=np.float64) for name in (
                "yaw", "yaw_mae", "vx", "vy", "vector_mae", "direction_error",
                "speed", "flight", "slip", "roll", "pitch",
            )}
            fall = np.zeros(count, dtype=bool)
            dangerous = np.zeros(count, dtype=bool)
            impact = np.zeros(count, dtype=bool)
            saturation = np.zeros(count, dtype=bool)
            slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=device)
            sat_streak = torch.zeros_like(slip_streak)
            first_sign = np.full(count, np.nan)
            first_mae = np.full(count, np.nan)
            first_010 = np.full(count, np.nan)
            first_020 = np.full(count, np.nan)
            sustained = np.zeros(count, dtype=np.int32)
            overshoot = np.zeros(count, dtype=np.float64)
            zero_crossings = np.zeros(count, dtype=np.int32)
            previous_yaw = np.zeros(count, dtype=np.float64)
            previous_left = np.zeros(count, dtype=bool)
            cycle_id = np.zeros(count, dtype=np.int32)
            cycle_sum = np.zeros(count, dtype=np.float64)
            cycle_count = np.zeros(count, dtype=np.int32)
            first_cycle = np.full(count, np.nan)
            for step in range(600):
                time_s = step * env.step_dt
                target_value = physical_target(initial, final, profile, time_s)
                actor_value = float(calibrate_yaw(target_value))
                vx_target = torch.zeros(env.num_envs, device=device)
                vy_target = torch.zeros_like(vx_target)
                for env_id in range(count):
                    direction = DIRECTIONS[int(condition_id[env_id])]
                    if direction is not None:
                        angle = math.radians(direction)
                        vx_target[env_id] = .3 * math.cos(angle)
                        vy_target[env_id] = .3 * math.sin(angle)
                command.external_override[:, 0] = vx_target
                command.external_override[:, 1] = vy_target
                command.external_override[:, 2] = actor_value
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations()["policy"].to(device)
                with torch.inference_mode():
                    action = actor(obs, torch.zeros(env.num_envs, device=device))
                obs, _, done, extras = wrapped.step(action)
                obs = obs["policy"].to(device)
                actual_v = robot.data.root_lin_vel_b[:, :2]
                actual_yaw = robot.data.root_ang_vel_b[:, 2]
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contact = force > 5
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                sliding = ((foot_speed > .55) & contact).any(-1)
                slip_streak = torch.where(sliding, slip_streak + 1, torch.zeros_like(slip_streak))
                dangerous |= (slip_streak[:count] >= 5).cpu().numpy()
                impact |= (force[:count].amax(-1) > 3500).cpu().numpy()
                limits = robot.data.joint_vel_limits
                limits = limits[..., 1].abs() if limits.ndim == 3 else limits
                ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(-1)
                sat_streak = torch.where(ratio > .95, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= (sat_streak[:count] >= 5).cpu().numpy()
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool()
                fall |= (done[:count].bool() & ~timeout[:count]).cpu().numpy()
                yaw_np = actual_yaw[:count].detach().cpu().numpy()
                contact_np = contact[:count].cpu().numpy()
                left_now = contact_np[:, 0]
                new_cycle = ~previous_left & left_now
                for env_id in np.flatnonzero(new_cycle):
                    if cycle_count[env_id] and np.isnan(first_cycle[env_id]):
                        if np.sign(cycle_sum[env_id] / cycle_count[env_id]) == np.sign(final):
                            first_cycle[env_id] = max(0, (step - 200) * env.step_dt)
                    cycle_id[env_id] += 1
                    cycle_sum[env_id] = 0
                    cycle_count[env_id] = 0
                previous_left = left_now
                if step >= 200:
                    cycle_sum += yaw_np
                    cycle_count += 1
                    correct = yaw_np * final > 0
                    pure_mask = np.array(
                        [DIRECTIONS[int(i)] is None for i in condition_id], dtype=bool
                    )
                    mae_limit = np.where(pure_mask, .15, .20)
                    mae_ok = np.abs(yaw_np - final) <= mae_limit
                    elapsed = (step - 200) * env.step_dt
                    first_sign[np.isnan(first_sign) & correct] = elapsed
                    first_mae[np.isnan(first_mae) & mae_ok] = elapsed
                    sustained = np.where(correct & mae_ok, sustained + 1, 0)
                    first_010[np.isnan(first_010) & (sustained >= 5)] = elapsed - .08
                    first_020[np.isnan(first_020) & (sustained >= 10)] = elapsed - .18
                zero_crossings += ((previous_yaw * yaw_np < 0) & (step > 0)).astype(np.int32)
                previous_yaw = yaw_np.copy()
                if step >= 300:
                    actual_v_np = actual_v[:count].detach().cpu().numpy()
                    vx_np = vx_target[:count].detach().cpu().numpy()
                    vy_np = vy_target[:count].detach().cpu().numpy()
                    vector_error = np.hypot(actual_v_np[:, 0] - vx_np, actual_v_np[:, 1] - vy_np)
                    target_angle = np.arctan2(vy_np, vx_np)
                    actual_angle = np.arctan2(actual_v_np[:, 1], actual_v_np[:, 0])
                    direction_error = np.abs(np.arctan2(
                        np.sin(actual_angle - target_angle), np.cos(actual_angle - target_angle)
                    )) * 180 / np.pi
                    flight = ~contact_np.any(axis=1)
                    gravity = robot.data.projected_gravity_b[:count]
                    roll = torch.atan2(gravity[:, 1].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                    pitch = torch.atan2(gravity[:, 0].abs(), gravity[:, 2].abs().clamp_min(1e-6))
                    for name, values in (
                        ("yaw", yaw_np), ("yaw_mae", np.abs(yaw_np - final)),
                        ("vx", actual_v_np[:, 0]), ("vy", actual_v_np[:, 1]),
                        ("vector_mae", vector_error), ("direction_error", direction_error),
                        ("speed", np.linalg.norm(actual_v_np, axis=1)),
                        ("flight", flight.astype(float)), ("slip", sliding[:count].float().cpu().numpy()),
                        ("roll", roll.cpu().numpy()), ("pitch", pitch.cpu().numpy()),
                    ):
                        sums[name] += values
                    overshoot = np.maximum(overshoot, np.maximum(0, np.abs(yaw_np) - abs(final)))
            for name in sums:
                sums[name] /= endpoint_steps
            for env_id in range(count):
                direction = DIRECTIONS[int(condition_id[env_id])]
                pure = direction is None
                replay = evaluator.replay_summary(
                    yaw_target=final, mean_yaw=sums["yaw"][env_id],
                    yaw_mae=sums["yaw_mae"][env_id],
                    condition_type="pure" if pure else "moving",
                    vector_mae=sums["vector_mae"][env_id],
                    direction_error_deg=0 if pure else sums["direction_error"][env_id],
                    translation_drift=sums["speed"][env_id],
                    gait_success=pure or sums["flight"][env_id] < .1,
                    fall=fall[env_id], dangerous_slip=dangerous[env_id],
                    impact=impact[env_id], long_dwell_saturation=saturation[env_id],
                )
                condition = f"{transition}_{'PURE' if pure else f'D{direction:03d}'}"
                row = {
                    "condition": condition, "transition": transition, "profile": profile,
                    "direction_deg": -1 if pure else direction, "episode": int(episode_id[env_id]),
                    "yaw_target": final, "yaw_actor_input": float(calibrate_yaw(final)),
                    **replay.to_dict(),
                    "endpoint_window_start_s": 6.0, "endpoint_window_end_s": 12.0,
                    "flight_fraction": sums["flight"][env_id],
                    "slip_fraction": sums["slip"][env_id],
                    "roll_mean": sums["roll"][env_id], "pitch_mean": sums["pitch"][env_id],
                }
                episode_rows.append(row)
                acquisition_rows.append({
                    "condition": condition, "transition": transition,
                    "direction_deg": -1 if pure else direction, "episode": int(episode_id[env_id]),
                    "first_instantaneous_correct_sign_s": first_sign[env_id],
                    "first_static_mae_pass_s": first_mae[env_id],
                    "first_0p10_sustained_endpoint_like_pass_s": first_010[env_id],
                    "first_0p20_sustained_endpoint_like_pass_s": first_020[env_id],
                    "first_complete_gait_cycle_mean_pass_s": first_cycle[env_id],
                    "overshoot_rad_s": overshoot[env_id],
                    "zero_crossing_count": int(zero_crossings[env_id]),
                    "never_acquired_target_sign": bool(np.isnan(first_sign[env_id])),
                    "formal_gate_member": False,
                })
        write_outputs(episode_rows, acquisition_rows)
        wrapped.close()


if __name__ == "__main__":
    main()
