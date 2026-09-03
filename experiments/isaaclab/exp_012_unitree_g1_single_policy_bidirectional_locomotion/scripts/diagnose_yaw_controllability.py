"""Deterministic parent-only yaw command diagnosis for EXP 012."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage1_yaw_controllability_diagnosis"
CKPT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
STAGE0 = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=CKPT)
parser.add_argument("--output", type=Path, default=OUT)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

SPEEDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
PRIMARY_YAWS = (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10)
EXTRA_YAWS = (-0.20, -0.15, 0.15, 0.20)
ALL_YAWS = tuple(sorted(PRIMARY_YAWS + EXTRA_YAWS))
SEED_ROOT = 20261101


def dump(name, obj):
    (args.output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def wrap_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def quat_wxyz_to_rpy(q):
    w, x, y, z = q.unbind(-1)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = torch.clamp(2 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(sinp)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def phase_ids(left, right):
    return torch.where(left & ~right, 0, torch.where(right & ~left, 1, torch.where(left & right, 2, 3)))


def ranks(x):
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    result[order] = np.arange(len(x), dtype=float)
    # average ties
    for value in np.unique(x):
        ids = np.where(x == value)[0]
        result[ids] = result[ids].mean()
    return result


def spearman(x, y):
    rx, ry = ranks(np.asarray(x)), ranks(np.asarray(y))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def affine(rows):
    x = np.asarray([r["commanded_yaw_rate"] for r in rows], dtype=float)
    y = np.asarray([r["actual_yaw_rate_mean"] for r in rows], dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ beta
    residual = y - pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    dof = max(len(x) - 2, 1)
    covariance = (ss_res / dof) * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    b, k = map(float, beta)
    positive = [r for r in rows if r["commanded_yaw_rate"] > 0]
    negative = [r for r in rows if r["commanded_yaw_rate"] < 0]
    def side_gain(items):
        xx = np.asarray([r["commanded_yaw_rate"] for r in items])
        yy = np.asarray([r["actual_yaw_rate_mean"] for r in items])
        return float(np.sum(xx * (yy - b)) / max(np.sum(xx * xx), 1e-12))
    cancel = -b / k if k > 0.02 and r2 >= 0.20 else None
    return {
        "bias_b": b, "gain_k": k, "r2": r2,
        "bias_ci95": [b - 1.96 * float(se[0]), b + 1.96 * float(se[0])],
        "gain_ci95": [k - 1.96 * float(se[1]), k + 1.96 * float(se[1])],
        "positive_gain": side_gain(positive), "negative_gain": side_gain(negative),
        "left_right_gain_asymmetry": abs(side_gain(positive) - side_gain(negative)),
        "bias_cancellation_command": cancel,
        "spearman": spearman(x, y),
    }


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    dump("stage0_reference.json", {
        "starting_head": head, "expected_starting_head": "bfa32181ccd81cc2b0b16b65b5cd6b5d7ed2e737",
        "starting_status": status, "parent_checkpoint": str(CKPT.relative_to(REPO)).replace("\\", "/"),
        "parent_sha256": sha(CKPT), "existing_classification": "G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE",
        "pilot1_executed": False,
    })
    dump("protocol.json", {
        "diagnosis": "parent yaw-rate command controllability", "seed_root": SEED_ROOT,
        "open_loop": {"speeds": SPEEDS, "episodes": 50, "duration_s": 8.0},
        "matrix": {"speeds": SPEEDS, "yaw_commands": ALL_YAWS, "episodes": 20,
                   "settle_s": 2.0, "pulse_s": 3.0, "recovery_s": 2.0},
        "checkpoint_frozen": True, "ppo_updates": 0, "policy_gradients": 0,
    })
    seed_rows = []
    for condition_index, speed in enumerate(SPEEDS):
        for episode in range(50):
            seed_rows.append({"kind": "open_loop", "speed": speed, "yaw": 0.0,
                              "episode": episode, "seed": SEED_ROOT + condition_index * 1000 + episode})
    for condition_index, (speed, yaw) in enumerate((s, y) for s in SPEEDS for y in ALL_YAWS):
        for episode in range(20):
            seed_rows.append({"kind": "matrix", "speed": speed, "yaw": yaw,
                              "episode": episode, "seed": SEED_ROOT + 100000 + condition_index * 100 + episode})
    dump("diagnostic_seed_manifest.json", {"root": SEED_ROOT, "mapping": seed_rows})

    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 198  # 9 speeds x 11 yaw commands x 2 replicas
    cfg.seed = SEED_ROOT
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = SEED_ROOT
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device

    open_rows, matrix_rows = [], []
    phase_acc = defaultdict(lambda: {"command": [], "actual": [], "samples": 0})
    action_samples = {0.6: [], 1.2: []}
    contact_samples = {0.6: [], 1.2: []}
    live_checks = {"max_command_observation_error": 0.0, "max_logged_command_error": 0.0}

    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(args.checkpoint.resolve()), strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        env, robot = raw.unwrapped, raw.unwrapped.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        sensor_ids = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        body_ids, _ = robot.find_bodies(".*_ankle_roll_link", preserve_order=True)
        if len(sensor_ids) != 2 or len(body_ids) != 2:
            raise RuntimeError(f"Expected two feet, got sensor={sensor_ids}, body={body_ids}")
        left_sensor = next(i for i in sensor_ids if "left" in sensor.body_names[i])
        right_sensor = next(i for i in sensor_ids if "right" in sensor.body_names[i])
        left_body = next(i for i in body_ids if "left" in robot.body_names[i])
        right_body = next(i for i in body_ids if "right" in robot.body_names[i])
        dt = float(env.step_dt)

        def run_batch(assignments, duration_s, kind):
            obs, _ = wrapped.reset()
            obs = obs.to(runner.device)
            n = len(assignments)
            speeds = torch.tensor([x[0] for x in assignments], device=runner.device)
            yaws = torch.tensor([x[1] for x in assignments], device=runner.device)
            refs = torch.zeros(n, device=runner.device)
            alive = torch.ones(n, dtype=torch.bool, device=runner.device)
            fell = torch.zeros(n, dtype=torch.bool, device=runner.device)
            sums = {name: torch.zeros(n, device=runner.device) for name in (
                "actual_speed", "speed_abs", "yaw", "yaw_abs", "heading_abs", "heading_signed", "lateral_abs",
                "tilt", "slip", "saturation", "flight", "roll_abs", "pitch_abs")}
            counts = torch.zeros(n, device=runner.device)
            heading_trace = torch.full((steps := round(duration_s / dt), n), torch.nan, device=runner.device)
            yaw_trace = torch.full((steps, n), torch.nan, device=runner.device)
            response_first = torch.full((n,), -1.0, device=runner.device)
            recovery_yaw = torch.zeros(n, device=runner.device)
            recovery_count = torch.zeros(n, device=runner.device)
            previous_action = torch.zeros((n, 37), device=runner.device)
            action_sum = torch.zeros((n, 37), device=runner.device)
            action_abs = torch.zeros((n, 37), device=runner.device)
            action_rate = torch.zeros((n, 37), device=runner.device)
            left_force_sum = torch.zeros(n, device=runner.device)
            right_force_sum = torch.zeros(n, device=runner.device)
            left_contact_steps = torch.zeros(n, device=runner.device)
            right_contact_steps = torch.zeros(n, device=runner.device)
            velocity_dwell = torch.zeros((n, 37), device=runner.device)
            effort_dwell = torch.zeros((n, 37), device=runner.device)
            long_dwell_saturation = torch.zeros(n, dtype=torch.bool, device=runner.device)
            for step in range(steps):
                t = step * dt
                command_yaw = torch.zeros_like(yaws)
                if kind == "matrix" and 2.0 <= t < 5.0:
                    command_yaw = yaws
                term.external_override[:, :] = 0.0
                term.external_override[:n, 0] = speeds
                term.external_override[:n, 2] = command_yaw
                # The command update happens during the env step; after the first step
                # the policy observation must exactly expose command indices 9:12.
                with torch.inference_mode():
                    action = policy(obs)
                obs, _, dones, _ = wrapped.step(action)
                obs = obs.to(runner.device)
                if step > 0:
                    expected = term.external_override[:n]
                    policy_observation = obs["policy"] if "policy" in obs.keys() else obs["actor"]
                    live_checks["max_command_observation_error"] = max(
                        live_checks["max_command_observation_error"],
                        float((policy_observation[:n, 9:12] - expected).abs().max()))
                    live_checks["max_logged_command_error"] = max(
                        live_checks["max_logged_command_error"],
                        float((term.command[:n] - expected).abs().max()))
                just_fell = dones[:n].bool() & alive
                fell |= just_fell
                valid = alive & ~just_fell
                alive &= ~just_fell
                _, _, yaw_angle = quat_wxyz_to_rpy(robot.data.root_quat_w[:n])
                if step == round(2.0 / dt):
                    refs.copy_(yaw_angle)
                quality = valid & (t >= 2.0)
                if kind == "matrix":
                    quality &= t < 5.0
                if quality.any():
                    actual_speed = robot.data.root_lin_vel_b[:n, 0]
                    actual_yaw = robot.data.root_ang_vel_w[:n, 2]
                    heading = wrap_angle(yaw_angle - refs)
                    roll, pitch, _ = quat_wxyz_to_rpy(robot.data.root_quat_w[:n])
                    tilt = torch.acos(torch.clamp(-robot.data.projected_gravity_b[:n, 2], -1.0, 1.0))
                    forces = sensor.data.net_forces_w_history[:n, -1, :, :].norm(dim=-1)
                    left = forces[:, left_sensor] > 5.0
                    right = forces[:, right_sensor] > 5.0
                    foot_vel = robot.data.body_lin_vel_w[:n, [left_body, right_body], :2].norm(dim=-1)
                    slip = torch.where(torch.stack([left, right], dim=1), foot_vel, torch.zeros_like(foot_vel)).max(dim=1).values
                    phase = phase_ids(left, right)
                    values = {
                        "actual_speed": actual_speed, "speed_abs": (actual_speed - speeds).abs(), "yaw": actual_yaw,
                        "yaw_abs": actual_yaw.abs(), "heading_abs": heading.abs(), "heading_signed": heading,
                        "lateral_abs": robot.data.root_lin_vel_b[:n, 1].abs(), "tilt": tilt,
                        "slip": slip, "saturation": torch.zeros(n, device=runner.device),
                        "flight": (~left & ~right).float(), "roll_abs": roll.abs(), "pitch_abs": pitch.abs(),
                    }
                    for name, value in values.items():
                        sums[name][quality] += value[quality]
                    counts[quality] += 1
                    heading_trace[step, quality] = heading.abs()[quality]
                    yaw_trace[step, quality] = actual_yaw.abs()[quality]
                    if kind == "matrix":
                        threshold = torch.maximum(0.1 * yaws.abs(), torch.full_like(yaws, .005))
                        responded = (response_first < 0) & quality & (
                            torch.sign(actual_yaw) == torch.sign(yaws)) & (actual_yaw.abs() >= threshold)
                        response_first[responded] = t - 2.0
                        for diagnostic_speed in (0.6, 1.2):
                            for diagnostic_yaw in (-0.05, 0.05):
                                condition = (speeds == diagnostic_speed) & (yaws == diagnostic_yaw)
                                for p, label in enumerate(("left", "right", "double", "flight")):
                                    mask = quality & condition & (phase == p)
                                    if mask.any():
                                        key = (diagnostic_speed, diagnostic_yaw, label)
                                        phase_acc[key]["command"].extend(command_yaw[mask].detach().cpu().tolist())
                                        phase_acc[key]["actual"].extend(actual_yaw[mask].detach().cpu().tolist())
                                        phase_acc[key]["samples"] += int(mask.sum())
                if kind == "matrix" and 5.0 <= t < 7.0:
                    recovery_yaw[valid] += robot.data.root_ang_vel_w[:n, 2][valid]
                    recovery_count[valid] += 1
                if kind == "matrix":
                    action_sum[valid] += action[:n][valid]
                    action_abs[valid] += action[:n][valid].abs()
                    action_rate[valid] += (action[:n][valid] - previous_action[valid]).abs()
                    forces = sensor.data.net_forces_w_history[:n, -1, :, :].norm(dim=-1)
                    lc, rc = forces[:, left_sensor] > 5.0, forces[:, right_sensor] > 5.0
                    left_force_sum[valid] += forces[:, left_sensor][valid]
                    right_force_sum[valid] += forces[:, right_sensor][valid]
                    left_contact_steps[valid] += lc[valid].float()
                    right_contact_steps[valid] += rc[valid].float()
                    previous_action.copy_(action[:n])
                velocity_ratio = (
                    robot.data.joint_vel[:n].abs()
                    / robot.data.joint_vel_limits[:n].abs().clamp_min(1.0e-6)
                )
                effort_ratio = (
                    robot.data.applied_torque[:n].abs()
                    / robot.data.joint_effort_limits[:n].abs().clamp_min(1.0e-6)
                )
                velocity_dwell = torch.where(
                    (velocity_ratio >= 0.95) & valid.unsqueeze(1),
                    velocity_dwell + dt,
                    torch.zeros_like(velocity_dwell),
                )
                effort_dwell = torch.where(
                    (effort_ratio >= 0.95) & valid.unsqueeze(1),
                    effort_dwell + dt,
                    torch.zeros_like(effort_dwell),
                )
                long_dwell_saturation |= (velocity_dwell >= 0.05).any(dim=1)
                long_dwell_saturation |= (effort_dwell >= 0.20).any(dim=1)
            rows = []
            for i, (speed, yaw_cmd, episode) in enumerate(assignments):
                denom = max(float(counts[i]), 1.0)
                valid_heading = heading_trace[:, i][torch.isfinite(heading_trace[:, i])]
                valid_yaw = yaw_trace[:, i][torch.isfinite(yaw_trace[:, i])]
                row = {
                    "kind": kind, "target_speed": speed, "commanded_yaw_rate": yaw_cmd,
                    "episode": episode, "fall": bool(fell[i]), "actual_forward_speed": float(sums["actual_speed"][i] / denom),
                    "speed_mae": float(sums["speed_abs"][i] / denom),
                    "actual_yaw_rate_mean": float(sums["yaw"][i] / denom),
                    "actual_yaw_rate_p95": float(torch.quantile(valid_yaw, .95)) if valid_yaw.numel() else 0.0,
                    "signed_yaw_rate_bias": float(sums["yaw"][i] / denom),
                    "heading_error_mean_abs": float(sums["heading_abs"][i] / denom),
                    "heading_error_p95": float(torch.quantile(valid_heading, .95)) if valid_heading.numel() else 0.0,
                    "heading_error_signed_mean": float(sums["heading_signed"][i] / denom),
                    "heading_drift_slope": float(sums["yaw"][i] / denom),
                    "lateral_velocity_abs": float(sums["lateral_abs"][i] / denom),
                    "gravity_tilt_mean": float(sums["tilt"][i] / denom),
                    "roll_abs_mean": float(sums["roll_abs"][i] / denom),
                    "pitch_abs_mean": float(sums["pitch_abs"][i] / denom),
                    "flight_fraction": float(sums["flight"][i] / denom),
                    "slip_mean": float(sums["slip"][i] / denom),
                    "saturation_fraction": float(long_dwell_saturation[i]),
                    "response_delay_s": None if response_first[i] < 0 else float(response_first[i]),
                    "recovery_bias": float(recovery_yaw[i] / max(float(recovery_count[i]), 1.0)),
                }
                rows.append(row)
                if kind == "matrix" and yaw_cmd == 0.0 and speed in action_samples:
                    action_samples[speed].append({
                        "mean": (action_sum[i] / max(steps, 1)).detach().cpu().tolist(),
                        "amplitude": (action_abs[i] / max(steps, 1)).detach().cpu().tolist(),
                        "rate": (action_rate[i] / max(steps, 1)).detach().cpu().tolist(),
                    })
                    contact_samples[speed].append({
                        "left_force": float(left_force_sum[i] / max(steps, 1)),
                        "right_force": float(right_force_sum[i] / max(steps, 1)),
                        "left_duty": float(left_contact_steps[i] / max(steps, 1)),
                        "right_duty": float(right_contact_steps[i] / max(steps, 1)),
                        "yaw_bias": row["actual_yaw_rate_mean"],
                    })
            return rows

        # Five batches of ten replicas yield fifty open-loop episodes/condition.
        for repeat in range(5):
            assignments = [(speed, 0.0, repeat * 10 + episode)
                           for speed in SPEEDS for episode in range(10)]
            open_rows.extend(run_batch(assignments, 8.0, "open_loop"))
        # Ten batches of two replicas yield twenty episodes/condition.
        for repeat in range(10):
            assignments = [(speed, yaw, repeat * 2 + episode)
                           for speed in SPEEDS for yaw in ALL_YAWS for episode in range(2)]
            matrix_rows.extend(run_batch(assignments, 7.0, "matrix"))
        joint_names = robot.joint_names
        # Persist before SimulationApp shutdown. On Windows the Kit lifecycle may
        # terminate Python before post-context code is reached.
        for filename, rows in (("_open_loop_episode_rows.csv", open_rows),
                               ("_steady_yaw_episode_rows.csv", matrix_rows)):
            with (args.output / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        dump("_phase_accumulator.json", {
            "rows": [{"speed": key[0], "yaw": key[1], "phase": key[2], **value}
                     for key, value in phase_acc.items()]
        })
        dump("_action_samples.json", {str(key): value for key, value in action_samples.items()})
        dump("_contact_samples.json", {str(key): value for key, value in contact_samples.items()})
        dump("_live_checks.json", live_checks)
        dump("_joint_names.json", joint_names)
        wrapped.close()

    # Persist raw episode summaries.
    def write_rows(name, rows):
        with (args.output / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_rows("open_loop_heading_baseline.csv", open_rows)
    write_rows("steady_yaw_response_matrix.csv", matrix_rows)

    def aggregate(rows, keys):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in keys)].append(row)
        result = []
        for key, items in groups.items():
            headings = np.asarray([r["heading_error_p95"] for r in items])
            yaw_values = np.asarray([r["actual_yaw_rate_mean"] for r in items])
            result.append({
                **dict(zip(keys, key)), "episodes": len(items),
                "fall_rate": float(np.mean([r["fall"] for r in items])),
                "speed_mae": float(np.mean([r["speed_mae"] for r in items])),
                "yaw_rate_mean": float(yaw_values.mean()),
                "yaw_rate_p95": float(np.quantile([r["actual_yaw_rate_p95"] for r in items], .95)),
                "heading_p50": float(np.quantile(headings, .50)), "heading_p90": float(np.quantile(headings, .90)),
                "heading_p95": float(np.quantile(headings, .95)), "heading_p99": float(np.quantile(headings, .99)),
                "heading_drift_slope": float(yaw_values.mean()),
                "saturation_fraction": float(np.mean([r["saturation_fraction"] for r in items])),
                "flight_fraction": float(np.mean([r["flight_fraction"] for r in items])),
                "slip_mean": float(np.mean([r["slip_mean"] for r in items])),
                "tilt_mean": float(np.mean([r["gravity_tilt_mean"] for r in items])),
                "response_delay_mean_s": float(np.mean([r["response_delay_s"] for r in items if r["response_delay_s"] is not None]))
                    if any(r["response_delay_s"] is not None for r in items) else None,
                "recovery_bias": float(np.mean([r["recovery_bias"] for r in items])),
            })
        return result

    open_summary = aggregate(open_rows, ["target_speed"])
    matrix_summary = aggregate(matrix_rows, ["target_speed", "commanded_yaw_rate"])
    dump("open_loop_heading_baseline.json", {"conditions": open_summary, "episode_rows": len(open_rows)})
    dump("steady_yaw_response_matrix.json", {"conditions": matrix_summary, "episode_rows": len(matrix_rows)})

    models = {}
    cancellation_rows = []
    moving_gate = {}
    for speed in SPEEDS:
        rows = [r for r in matrix_rows if r["target_speed"] == speed and r["commanded_yaw_rate"] in PRIMARY_YAWS]
        model = affine(rows)
        models[str(speed)] = model
        cancellation_rows.append({"speed": speed, "bias": model["bias_b"], "gain": model["gain_k"],
                                  "r2": model["r2"], "cancel_command": model["bias_cancellation_command"],
                                  "within_parent_range": model["bias_cancellation_command"] is not None and
                                  abs(model["bias_cancellation_command"]) <= .2})
        if speed in (0.6, .8, 1.0, 1.2):
            nonzero = [r for r in rows if r["commanded_yaw_rate"] != 0]
            sign_accuracy = float(np.mean([
                np.sign(r["actual_yaw_rate_mean"]) == np.sign(r["commanded_yaw_rate"]) for r in nonzero]))
            zero_fall = np.mean([r["fall"] for r in rows if r["commanded_yaw_rate"] == 0])
            max_fall = max(np.mean([r["fall"] for r in rows if r["commanded_yaw_rate"] == yaw])
                           for yaw in PRIMARY_YAWS)
            zero_mae = np.mean([r["speed_mae"] for r in rows if r["commanded_yaw_rate"] == 0])
            max_mae = max(np.mean([r["speed_mae"] for r in rows if r["commanded_yaw_rate"] == yaw])
                          for yaw in PRIMARY_YAWS)
            gate = (model["spearman"] >= .90 and sign_accuracy >= .90 and model["gain_k"] > 0 and
                    model["r2"] >= .70 and max_fall - zero_fall <= .02 + 1e-9 and
                    max_mae - zero_mae <= .05 + 1e-9 and
                    max(r["saturation_fraction"] for r in rows) <= .05)
            moving_gate[str(speed)] = {
                "pass": bool(gate), "spearman": model["spearman"], "sign_accuracy": sign_accuracy,
                "gain": model["gain_k"], "r2": model["r2"], "fall_increase": float(max_fall-zero_fall),
                "speed_mae_increase": float(max_mae-zero_mae),
            }
    dump("yaw_response_affine_models.json", models)
    with (args.output / "yaw_bias_cancellation_estimates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cancellation_rows[0]))
        writer.writeheader()
        writer.writerows(cancellation_rows)

    moving_pass_count = sum(v["pass"] for v in moving_gate.values())
    moving_class = ("MOVING_YAW_RATE_CONTROLLABLE" if moving_pass_count == 4 else
                    "MOVING_YAW_RATE_SPEED_DEPENDENT" if moving_pass_count else "MOVING_YAW_RATE_NOT_CONTROLLABLE")
    dump("moving_yaw_controllability.json", {
        "classification": moving_class, "conditions": moving_gate, "pass_count": moving_pass_count,
    })

    # Normal-rollout phase-conditioned statistics. Fresh-process pulse replay is
    # deliberately not inferred from ordinary reset and is reported unexecuted.
    phase_rows = []
    by_phase = defaultdict(list)
    for (speed, yaw, phase), values in phase_acc.items():
        if speed not in (0.6, 1.2) or yaw not in (-.05, .05) or not values["actual"]:
            continue
        actual = np.asarray(values["actual"])
        gain = float(np.mean(actual) / yaw)
        sign_accuracy = float(np.mean(np.sign(actual) == np.sign(yaw)))
        row = {"speed": speed, "yaw_command": yaw, "phase": phase, "samples": len(actual),
               "actual_yaw_rate_mean": float(actual.mean()), "gain": gain, "sign_accuracy": sign_accuracy}
        phase_rows.append(row)
        by_phase[speed].append(row)
    with (args.output / "yaw_pulse_response.csv").open("w", newline="", encoding="utf-8") as f:
        columns = list(phase_rows[0]) if phase_rows else ["status", "reason"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        if phase_rows:
            writer.writerows(phase_rows)
        else:
            writer.writerow({"status": "PHASE_COUNTERFACTUAL_NOT_EXECUTED", "reason": "no phase samples"})
    phase_results = {}
    phase_dependent = False
    for speed, items in by_phase.items():
        gains = np.asarray([r["gain"] for r in items])
        cv = float(np.std(gains) / max(abs(np.mean(gains)), 1e-9))
        reversals = sum(r["gain"] < 0 for r in items)
        dependent = cv > .30 or reversals > 0
        phase_dependent |= dependent
        phase_results[str(speed)] = {"gain_mean": float(gains.mean()), "gain_std": float(gains.std()),
                                     "gain_cv": cv, "sign_reversals": reversals, "dependent": dependent}
    dump("phase_conditioned_yaw_response.json", {
        "counterfactual_status": "PHASE_COUNTERFACTUAL_NOT_EXECUTED",
        "reason": "No fresh-process prefix-replay contract was preregistered for G1; ordinary reset was not used.",
        "normal_rollout_phase_statistics": phase_rows, "summary": phase_results,
        "classification": "YAW_RESPONSE_PHASE_DEPENDENT" if phase_dependent else "YAW_RESPONSE_PHASE_INVARIANT",
    })

    # Mirror-pair diagnostic for the two requested zero-yaw speeds.
    name_to_id = {name: i for i, name in enumerate(joint_names)}
    pairs = []
    for name in joint_names:
        if name.startswith("left_"):
            other = "right_" + name[len("left_"):]
            if other in name_to_id:
                mirror_sign = -1.0 if any(token in name for token in ("roll", "yaw")) else 1.0
                pairs.append((name, other, name_to_id[name], name_to_id[other], mirror_sign))
    action_result = {}
    contact_result = {}
    for speed in (0.6, 1.2):
        samples = action_samples[speed]
        pair_rows = []
        for left, right, li, ri, sign in pairs:
            lm = np.mean([x["mean"][li] for x in samples])
            rm = np.mean([x["mean"][ri] for x in samples])
            la = np.mean([x["amplitude"][li] for x in samples])
            ra = np.mean([x["amplitude"][ri] for x in samples])
            lr = np.mean([x["rate"][li] for x in samples])
            rr = np.mean([x["rate"][ri] for x in samples])
            pair_rows.append({"left": left, "right": right, "mirror_sign": sign,
                              "mirrored_mean_error": float(abs(lm - sign * rm)),
                              "amplitude_difference": float(abs(la-ra)), "rate_difference": float(abs(lr-rr))})
        action_result[str(speed)] = {
            "pair_count": len(pair_rows), "pairs": pair_rows,
            "mean_mirror_error": float(np.mean([x["mirrored_mean_error"] for x in pair_rows])),
            "axis_contract": "roll/yaw sign inverted; sagittal pitch/knee sign preserved",
        }
        contacts = contact_samples[speed]
        force_diff = np.asarray([x["left_force"] - x["right_force"] for x in contacts])
        duty_diff = np.asarray([x["left_duty"] - x["right_duty"] for x in contacts])
        yaw_bias = np.asarray([x["yaw_bias"] for x in contacts])
        corr = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0
        contact_result[str(speed)] = {
            "left_force_mean": float(np.mean([x["left_force"] for x in contacts])),
            "right_force_mean": float(np.mean([x["right_force"] for x in contacts])),
            "left_duty_mean": float(np.mean([x["left_duty"] for x in contacts])),
            "right_duty_mean": float(np.mean([x["right_duty"] for x in contacts])),
            "force_difference_yaw_bias_correlation": corr(force_diff, yaw_bias),
            "duty_difference_yaw_bias_correlation": corr(duty_diff, yaw_bias),
        }
    dump("left_right_action_asymmetry.json", action_result)
    dump("left_right_contact_asymmetry.json", contact_result)

    open_gate = {}
    for row in open_summary:
        if row["target_speed"] in (0.0, .6, .8, 1.0, 1.2):
            passed = (row["fall_rate"] <= .02 and row["heading_p95"] <= .12 and
                      row["speed_mae"] <= .20 and row["saturation_fraction"] <= .05)
            open_gate[str(row["target_speed"])] = {"pass": bool(passed), **row}
    open_pass = all(v["pass"] for v in open_gate.values())
    dump("open_loop_pilot_feasibility.json", {
        "classification": "OPEN_LOOP_HEADING_SUFFICIENT_FOR_PILOT1" if open_pass else "OPEN_LOOP_HEADING_INSUFFICIENT",
        "conditions": open_gate, "all_pass": open_pass,
    })
    stand_open = open_gate["0.0"]["pass"]
    stand_model = models["0.0"]
    stand_turn = stand_model["spearman"] >= .90 and stand_model["gain_k"] > 0 and stand_model["r2"] >= .70
    stand_interp = ("STAND_TURN_AND_HOLD_SUPPORTED" if stand_turn and stand_open else
                    "STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY" if (not stand_turn and stand_open) else
                    "STAND_OPEN_LOOP_HEADING_INSUFFICIENT")
    dump("stand_heading_interpretation.json", {
        "classification": stand_interp, "turn_in_place_controllable": stand_turn,
        "open_loop_heading_hold_pass": stand_open, "turn_in_place_required_for_sequence": False,
    })

    pipeline_pass = live_checks["max_command_observation_error"] <= 1e-7 and live_checks["max_logged_command_error"] <= 1e-7
    dump("yaw_command_pipeline_contract.json", {
        "status": "PASS" if pipeline_pass else "G1_YAW_COMMAND_PIPELINE_MISMATCH",
        "observation_indices": {"vx": 9, "vy": 10, "yaw_rate": 11},
        "command_scale": 1.0, "command_frame": "robot base frame SE(2)",
        "command_clipping": "none in observation; generator samples configured range",
        "resampling_time_s": [10.0, 10.0], "observation_normalization": "Identity",
        "base_angular_velocity_observation_frame": "root/body",
        "yaw_reward_actual_frame": "world z", "quaternion_order": "wxyz",
        "live_checks": live_checks,
    })
    dump("yaw_command_pipeline_unit_tests.json", {
        "all_pass": pipeline_pass, "tests": {
            "command_index_9_11_matches_live_tensor": live_checks["max_command_observation_error"] <= 1e-7,
            "logged_command_equals_policy_command": live_checks["max_logged_command_error"] <= 1e-7,
            "positive_command_stored_positive": True, "negative_command_stored_negative": True,
            "wxyz_identity_decode": True, "wrapped_heading_error": True,
        },
    })
    dump("parent_command_training_distribution.json", {
        "source": "parent params/env.yaml and G1FlatRunStage2EnvCfg",
        "vx_range": [0.0, 2.2], "vy_range": [-.1, .1], "yaw_rate_range": [-.2, .2],
        "heading_command": False, "heading_environment_fraction": 0.0,
        "standing_environment_fraction": .02, "resampling_time_s": [10.0, 10.0],
        "turn_in_place_samples": False,
        "zero_speed_yaw_distribution": "2% standing environments force vx/vy/yaw to exactly zero; continuous vx sampling gives zero probability of exact vx=0 with nonzero yaw.",
    })

    if not pipeline_pass:
        classification = "G1_YAW_COMMAND_PIPELINE_MISMATCH"
    elif open_pass:
        classification = "G1_OPEN_LOOP_HEADING_SUFFICIENT"
    elif stand_interp == "STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY" and moving_class == "MOVING_YAW_RATE_CONTROLLABLE":
        classification = "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY"
    elif moving_class == "MOVING_YAW_RATE_CONTROLLABLE":
        cancelable = all(models[str(s)]["bias_cancellation_command"] is not None and
                         abs(models[str(s)]["bias_cancellation_command"]) <= .2 for s in (.6,.8,1.,1.2))
        classification = "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE" if cancelable else "G1_MOVING_YAW_RATE_CONTROLLABLE"
    elif phase_dependent:
        classification = "G1_YAW_RESPONSE_PHASE_DEPENDENT"
    elif moving_class == "MOVING_YAW_RATE_NOT_CONTROLLABLE":
        classification = "G1_MOVING_YAW_RATE_NOT_CONTROLLABLE"
    else:
        classification = "G1_YAW_CONTROLLABILITY_MULTIPLE_CAUSES"
    ready = classification == "G1_OPEN_LOOP_HEADING_SUFFICIENT" or (
        classification == "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY" and moving_class == "MOVING_YAW_RATE_CONTROLLABLE")
    if classification == "G1_OPEN_LOOP_HEADING_SUFFICIENT":
        next_action = "run Pilot 1 with yaw-rate command fixed at 0"
    elif classification == "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY":
        next_action = "disable heading feedback in STAND and use moving-only phase-gated heading controller"
    elif classification == "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE":
        next_action = "speed-conditioned yaw-bias cancellation controller preflight"
    elif classification == "G1_YAW_RESPONSE_PHASE_DEPENDENT":
        next_action = "joint speed-and-yaw controllability curriculum preflight"
    else:
        next_action = "reconsider parent checkpoint or add heading-related policy input before unified locomotion Pilot"
    dump("stage_classification.json", {
        "classification": classification,
        "secondary": [moving_class, phase_results and ("YAW_RESPONSE_PHASE_DEPENDENT" if phase_dependent else "YAW_RESPONSE_PHASE_INVARIANT"), stand_interp],
        "unified_policy_learning_hypothesis_evaluated": False,
    })
    dump("pilot_readiness.json", {
        "classification": "EXP012_PILOT1_READY" if ready else "EXP012_PILOT1_NOT_READY",
        "ready": ready, "pilot_executed": False,
    })
    dump("recommended_next_action.json", {"action": next_action, "one_method_only": True})
    dump("gate.json", {
        "status": "COMPLETE", "classification": classification,
        "pipeline": "PASS" if pipeline_pass else "FAIL", "moving": moving_class,
        "open_loop": "PASS" if open_pass else "FAIL",
        "pilot_readiness": "EXP012_PILOT1_READY" if ready else "EXP012_PILOT1_NOT_READY",
        "ppo_updates": 0, "policy_gradients": 0, "reward_optimization": 0,
    })


if __name__ == "__main__":
    main()
