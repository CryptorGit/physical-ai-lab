"""Evaluate disposable Stage-2G shadow actors; no checkpoint is written."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2g_event_stratified_on_policy_preflight"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"
CONDITIONS = ("M0_UNIFORM", "M4_EVENT_STRATIFIED", "M8_EVENT_STRATIFIED", "M16_EVENT_STRATIFIED")
RUN_SPEEDS = (2.35, 2.40, 2.45, 2.50, 2.60)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--seed-root", type=int, default=20268221)
parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
parser.add_argument("--output-prefix", default="")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / f"{args.output_prefix}{name}").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runner_state(shadow):
    state = {}
    for key, value in shadow["actor_state_dict"].items():
        state["distribution.std_param" if key == "std" else key] = value
    return state


def q95(values):
    return float(torch.quantile(torch.tensor(values), .95)) if values else 0.


def main():
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 380
    cfg.seed = args.seed_root
    cfg.episode_length_s = 20.
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = args.seed_root
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    import importlib.metadata
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-PhaseA-RunAcquisition-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        command_term = env.command_manager.get_term("base_velocity")
        command_term.external_override_enabled = True
        reward_term = env.reward_manager.get_term_cfg("safe_periodic_flight").func
        sensor_feet = list(reward_term.foot_ids)
        robot_feet = [
            next(i for i, body in enumerate(robot.body_names) if body == sensor.body_names[int(sensor_id)])
            for sensor_id in sensor_feet
        ]
        behavior_rows, retention_rows = [], []
        for condition_index, condition in enumerate(args.conditions):
            runner.load(str(CHECKPOINT), load_cfg={
                "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
            }, strict=True, map_location=runner.device)
            shadow = torch.load(
                RAW / f"{args.output_prefix}shadow_{condition}.pt",
                map_location=runner.device,
                weights_only=False,
            )
            runner.alg.actor.load_state_dict(runner_state(shadow), strict=True)
            torch.manual_seed(args.seed_root + condition_index)
            torch.cuda.manual_seed_all(args.seed_root + condition_index)
            obs, _ = wrapped.reset()
            obs = obs.to(runner.device)
            reference = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
            speed_targets = torch.zeros(380, device=runner.device)
            action_stochastic = torch.zeros(380, dtype=torch.bool, device=runner.device)
            specs = []
            cursor = 0
            for mode in ("D0", "S100"):
                for speed in RUN_SPEEDS:
                    speed_targets[cursor:cursor + 30] = speed
                    action_stochastic[cursor:cursor + 30] = mode == "S100"
                    specs.extend([("run", mode, speed)] * 30)
                    cursor += 30
            for name, speed in (("stand", 0.), ("walk_0p6", .6), ("walk_1p2", 1.2), ("walk_to_stand", 0.)):
                speed_targets[cursor:cursor + 20] = speed
                specs.extend([(name, "D0", speed)] * 20)
                cursor += 20
            fallen = torch.zeros(380, dtype=torch.bool, device=runner.device)
            quality_steps = torch.zeros(380, device=runner.device)
            speed_error = torch.zeros(380, device=runner.device)
            heading_values = [[] for _ in range(380)]
            flight_events = torch.zeros(380, dtype=torch.long, device=runner.device)
            safe_flights = torch.zeros_like(flight_events)
            alternating = torch.zeros_like(flight_events)
            completions = torch.zeros_like(flight_events)
            was_flight = torch.zeros(380, dtype=torch.bool, device=runner.device)
            flight_duration = torch.zeros(380, device=runner.device)
            previous_landing = torch.full((380,), -1, dtype=torch.long, device=runner.device)
            slip_streak = torch.zeros(380, dtype=torch.long, device=runner.device)
            slip = torch.zeros(380, dtype=torch.bool, device=runner.device)
            impact = torch.zeros(380, dtype=torch.bool, device=runner.device)
            sat_streak = torch.zeros(380, dtype=torch.long, device=runner.device)
            saturation = torch.zeros(380, dtype=torch.bool, device=runner.device)
            dt = float(env.step_dt)
            for step in range(500):
                command = speed_targets.clone()
                # Registered retention stop profile: .6 hold, 1.5 s minimum-jerk, then zero.
                stop_slice = slice(360, 380)
                time_value = step * dt
                if time_value < 2.0:
                    command[stop_slice] = .6
                elif time_value < 3.5:
                    command[stop_slice] = .6 * (
                        1 - float(minimum_jerk((time_value - 2.0) / 1.5))
                    )
                else:
                    command[stop_slice] = 0.
                command_term.external_override[:, 0] = command
                command_term.external_override[:, 1:] = 0.
                with torch.inference_mode():
                    mean_action = runner.alg.actor(obs, stochastic_output=False)
                    sampled_action = runner.alg.actor(obs, stochastic_output=True)
                    action = torch.where(action_stochastic[:, None], sampled_action, mean_action)
                obs, _, dones, extras = wrapped.step(action)
                obs = obs.to(runner.device)
                dones = dones.to(runner.device).bool()
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).to(runner.device).bool()
                fallen |= dones & ~timeouts
                quality = torch.full((380,), step >= 50, device=runner.device, dtype=torch.bool)
                quality_steps += quality.float()
                actual = robot.data.root_lin_vel_b[:, 0]
                speed_error += (actual - command).abs() * quality.float()
                heading = wrapped_heading_error(reference, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
                for env_id in torch.where(quality)[0].tolist():
                    heading_values[env_id].append(float(heading[env_id]))
                forces = sensor.data.net_forces_w_history[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1)
                contacts = forces > 1.
                flight = contacts.sum(-1) == 0
                started = flight & ~was_flight & quality
                flight_events += started.long()
                flight_duration = torch.where(flight, flight_duration + dt, flight_duration)
                landing = was_flight & ~flight
                single = landing & (contacts.sum(-1) == 1)
                foot = contacts.long().argmax(-1)
                safe = single & (flight_duration >= .04 - 1e-6) & (flight_duration <= .16 + 1e-6)
                alternate = safe & (previous_landing >= 0) & (foot != previous_landing)
                safe_flights += safe.long()
                alternating += alternate.long()
                previous_landing[single] = foot[single]
                completions += (reward_term.last_raw_reward >= 1.).long()
                flight_duration[landing] = 0
                was_flight.copy_(flight)
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                slip_now = ((foot_speed > .55) & contacts).any(-1) & quality
                slip_streak = torch.where(slip_now, slip_streak + 1, torch.zeros_like(slip_streak))
                slip |= slip_streak >= 5
                impact |= (forces.amax(-1) > 3500.) & quality
                limits = robot.data.joint_vel_limits
                if limits.ndim == 3:
                    limits = limits[..., 1].abs()
                sat_now = (robot.data.joint_vel.abs() / torch.clamp(limits, min=1e-6) > .95).any(-1) & quality
                sat_streak = torch.where(sat_now, sat_streak + 1, torch.zeros_like(sat_streak))
                saturation |= sat_streak >= 5
            for env_id, (name, mode, target) in enumerate(specs):
                periodic = (
                    not bool(fallen[env_id]) and int(flight_events[env_id]) >= 4
                    and int(safe_flights[env_id]) >= 3 and int(alternating[env_id]) >= 3
                )
                row = {
                    "condition": condition, "evaluation_seed_root": args.seed_root,
                    "episode": env_id, "mode": mode, "target_speed": target,
                    "completion_events": int(completions[env_id]),
                    "periodic_running": int(periodic), "fall": int(fallen[env_id]),
                    "speed_mae": float(speed_error[env_id] / quality_steps[env_id]),
                    "heading_p95": q95(heading_values[env_id]),
                    "dangerous_slip": int(slip[env_id]), "impact_failure": int(impact[env_id]),
                    "long_dwell_saturation": int(saturation[env_id]),
                }
                if name == "run":
                    behavior_rows.append(row)
                else:
                    if name == "stand":
                        success = not bool(fallen[env_id]) and row["speed_mae"] <= .08 and int(flight_events[env_id]) == 0
                    elif name.startswith("walk_"):
                        success = not bool(fallen[env_id]) and row["speed_mae"] <= .20
                    else:
                        success = (
                            not bool(fallen[env_id]) and float(robot.data.root_lin_vel_b[env_id, 0].abs()) <= .08
                            and int(flight_events[env_id]) == 0
                        )
                    row.update({"retention_condition": name, "success": int(success)})
                    retention_rows.append(row)
            print(f"[Stage2G eval] {condition} complete", flush=True)
        write_csv("temporary_behavioral_evaluation.csv", behavior_rows)
        write_csv("retention_evaluation.csv", retention_rows)
        raw.close()


if __name__ == "__main__":
    main()
