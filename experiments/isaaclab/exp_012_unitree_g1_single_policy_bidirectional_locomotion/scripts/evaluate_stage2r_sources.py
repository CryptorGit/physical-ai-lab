"""Positive-control true STAND and WALK_TO_STAND sources in the Stage 2Q environment."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2r_true_stand_stop_integration"
STAND = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
TRANSITION = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_stand_transition_v1/model_0.pt"
EXPECTED = {
    "stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "transition": "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd",
}
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_jerk(x):
    x = torch.clamp(x, 0., 1.)
    return 10*x**3 - 15*x**4 + 6*x**5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in (("stand", STAND), ("walk", WALK), ("transition", TRANSITION)):
        if sha(path) != EXPECTED[name]:
            raise RuntimeError(f"STAGE2R_{name.upper()}_SOURCE_PROVENANCE_FAIL")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 200
    cfg.episode_length_s = 12.
    # Reuse the formal exp_007 integrated-source evaluation seed so that the
    # positive control changes only the destination environment contract.
    cfg.seed = 20260901
    agent_cfg.seed = 20260901
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        actors = {}
        for name, path in (("stand", STAND), ("walk", WALK), ("transition", TRANSITION)):
            actor = copy.deepcopy(runner.alg.actor)
            actor.load_state_dict(torch.load(path, map_location=runner.device, weights_only=False)["actor_state_dict"], strict=True)
            actors[name] = actor.eval()
        env = wrapped.unwrapped
        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        robot_feet = [next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[j]) for j in feet]
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        fallen = torch.zeros(200, dtype=torch.bool, device=runner.device)
        flight_seen = torch.zeros_like(fallen)
        final_flight_seen = torch.zeros_like(fallen)
        final_double_last = torch.zeros(200, dtype=torch.bool, device=runner.device)
        speed_sum = torch.zeros(200, device=runner.device)
        speed_steps = torch.zeros(200, device=runner.device)
        heading_trace = []
        slip_streak = torch.zeros(200, dtype=torch.long, device=runner.device)
        slip = torch.zeros_like(fallen)
        impact = torch.zeros_like(fallen)
        sat_streak = torch.zeros_like(slip_streak)
        saturation = torch.zeros_like(fallen)
        transition_complete = torch.zeros_like(fallen)
        completion_streak = torch.zeros(200, device=runner.device)
        settle_streak = torch.zeros(200, device=runner.device)
        stand_settled = torch.zeros_like(fallen)
        stand_hold_time = torch.zeros(200, device=runner.device)
        stop_takeover = torch.zeros_like(fallen)
        stop_hold_time = torch.zeros(200, device=runner.device)
        previous_support = torch.zeros(200, dtype=torch.long, device=runner.device)
        no_switch_elapsed = torch.zeros(200, device=runner.device)
        reference_yaw = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        steps = round(12 / float(env.step_dt))
        for step in range(steps):
            t = step * float(env.step_dt)
            speed = torch.zeros(200, device=runner.device)
            stop = torch.arange(200, device=runner.device) >= 100
            if t < 3:
                speed[stop] = 1.2
            else:
                speed[stop & ~stop_takeover] = 1.2 * (
                    1 - minimum_jerk(torch.tensor((t - 3) / 1.6, device=runner.device))
                )
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            action = torch.empty((200, 37), device=runner.device)
            with torch.inference_mode():
                action[:100] = actors["stand"](obs[:100], stochastic_output=False)
                if t < 3:
                    action[100:] = actors["walk"](obs[100:], stochastic_output=False)
                else:
                    transition_action = actors["transition"](obs[100:], stochastic_output=False)
                    stand_action = actors["stand"](obs[100:], stochastic_output=False)
                    action[100:] = torch.where(
                        stop_takeover[100:].unsqueeze(1), stand_action, transition_action
                    )
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(runner.device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fallen |= dones.bool() & ~timeout
            force_history = sensor.data.net_forces_w_history
            forces = force_history[:, :, feet, :].norm(dim=-1).amax(dim=1)
            contacts = forces > 5
            airborne = contacts.sum(-1) == 0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            actual = robot.data.root_lin_vel_b[:, 0].abs()
            horizontal = robot.data.root_lin_vel_b[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w[:, 2].abs()
            gravity = robot.data.projected_gravity_b
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
            pitch = torch.atan2(
                -gravity[:, 0], torch.sqrt(gravity[:, 1].square() + gravity[:, 2].square())
            )
            stand_safe = (
                (horizontal <= .08) & (vertical <= .05)
                & (roll.abs() <= .10) & (pitch.abs() <= .10) & contacts.all(dim=1)
            )
            stand_waiting = (~stop) & ~stand_settled
            settle_streak = torch.where(
                stand_waiting & stand_safe,
                settle_streak + float(env.step_dt),
                torch.where(stand_waiting, torch.zeros_like(settle_streak), settle_streak),
            )
            stand_settled |= stand_waiting & (settle_streak >= .4)
            stand_hold_time += stand_settled.float() * float(env.step_dt)

            stop_edge = stop & (t >= 3) & ~stop_takeover
            switched = stop_edge & (support != previous_support) & (support != 0)
            no_switch_elapsed = torch.where(
                stop_edge,
                torch.where(
                    switched, torch.zeros_like(no_switch_elapsed),
                    no_switch_elapsed + float(env.step_dt),
                ),
                no_switch_elapsed,
            )
            previous_support = torch.where(stop_edge, support, previous_support)
            completion_good = (
                stop_edge & (horizontal <= .08) & (vertical <= .05)
                & (roll.abs() <= .10) & (pitch.abs() <= .10)
                & contacts.all(dim=1) & (no_switch_elapsed >= .4)
            )
            completion_streak = torch.where(
                completion_good,
                completion_streak + float(env.step_dt),
                torch.zeros_like(completion_streak),
            )
            newly_complete = stop_edge & (completion_streak >= .4)
            stop_takeover |= newly_complete
            transition_complete |= newly_complete
            stop_hold_time += stop_takeover.float() * float(env.step_dt)

            stand_evaluate = (~stop) & stand_settled & (stand_hold_time <= 8)
            stop_final_evaluate = stop & stop_takeover & (stop_hold_time <= 5)
            evaluate = stand_evaluate | stop_final_evaluate
            flight_seen |= airborne & evaluate
            final_flight_seen |= airborne & stop_final_evaluate
            double_support_evaluate = stand_evaluate | stop_final_evaluate
            final_double_last = torch.where(
                double_support_evaluate, contacts.all(dim=-1), final_double_last
            )
            speed_sum += actual * evaluate.float()
            speed_steps += evaluate.float()
            heading_trace.append(wrapped_heading_error(reference_yaw, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs().cpu())
            foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
            slipping = ((foot_speed > .55) & contacts).any(-1)
            slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
            slip |= slip_streak >= 5
            impact |= forces.amax(-1) > 3500
            limits = robot.data.joint_vel_limits
            if limits.ndim == 3:
                limits = limits[..., 1].abs()
            saturated = (robot.data.joint_vel.abs() / limits.clamp_min(1e-6) > .95).any(-1)
            sat_streak = torch.where(saturated, sat_streak + 1, torch.zeros_like(sat_streak))
            saturation |= sat_streak >= 5
        heading = torch.stack(heading_trace)
        rows = []
        for env_id in range(200):
            is_stand = env_id < 100
            final_double = bool(final_double_last[env_id])
            passed = (
                not bool(fallen[env_id]) and not bool(flight_seen[env_id]) and
                float(speed_sum[env_id] / speed_steps[env_id].clamp_min(1)) <= .05 and
                final_double and
                (
                    bool(stand_settled[env_id]) and float(stand_hold_time[env_id]) >= 8
                    if is_stand
                    else bool(transition_complete[env_id]) and float(stop_hold_time[env_id]) >= 5
                )
            )
            rows.append({
                "condition": "TRUE_STAND" if is_stand else "WALK_TO_STAND",
                "episode": env_id % 100, "success": passed, "fall": bool(fallen[env_id]),
                "speed_mean": float(speed_sum[env_id] / speed_steps[env_id].clamp_min(1)),
                "flight_zero": not bool(flight_seen[env_id]),
                "final_flight_zero": not bool(final_flight_seen[env_id]) if not is_stand else not bool(flight_seen[env_id]),
                "final_double_support": final_double,
                "transition_completion": True if is_stand else bool(transition_complete[env_id]),
                "heading_p95": float(torch.quantile(heading[:, env_id], .95)),
                "dangerous_slip": bool(slip[env_id]), "impact_failure": bool(impact[env_id]),
                "saturation_failure": bool(saturation[env_id]),
            })
        summary = {}
        for condition in ("TRUE_STAND", "WALK_TO_STAND"):
            values = [row for row in rows if row["condition"] == condition]
            summary[condition] = {
                "episodes": 100, "success_rate": sum(row["success"] for row in values) / 100,
                "fall_rate": sum(row["fall"] for row in values) / 100,
                "speed_mean": sum(row["speed_mean"] for row in values) / 100,
                "flight_zero_rate": sum(row["flight_zero"] for row in values) / 100,
                "final_flight_zero_rate": sum(row["final_flight_zero"] for row in values) / 100,
                "final_double_support_rate": sum(row["final_double_support"] for row in values) / 100,
                "completion_rate": sum(row["transition_completion"] for row in values) / 100,
                "heading_p95_mean": sum(row["heading_p95"] for row in values) / 100,
                "dangerous_slip_rate": sum(row["dangerous_slip"] for row in values) / 100,
                "impact_failure_rate": sum(row["impact_failure"] for row in values) / 100,
                "saturation_failure_rate": sum(row["saturation_failure"] for row in values) / 100,
            }
        with (OUT / "source_positive_control_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (OUT / "source_positive_control_results.json").write_text(json.dumps({
            "environment": "Isaac-Exp012-G1-Reverse-PhaseR1-v0", "seed": 20260901,
            "summary": summary,
            "source_sha256": EXPECTED, "teacher_calls_runtime_final_artifact": 0,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
