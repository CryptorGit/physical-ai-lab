"""Detailed 50-episode integrated-sequence evaluation for the frozen selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.phase_gated_heading import minimum_jerk, wrapped_heading_error, yaw_from_quat_wxyz  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

POINTS = (
    (0.0, 0.0), (2.0, 0.0), (3.0, 0.6), (5.0, 0.6), (6.0, 1.2), (8.0, 1.2),
    (9.5, 2.4), (12.5, 2.4), (14.0, 2.6), (17.0, 2.6), (18.5, 2.4),
    (21.5, 2.4), (23.0, 1.2), (25.0, 1.2), (26.0, 0.6), (28.0, 0.6),
    (29.0, 0.0), (32.0, 0.0),
)
HOLDS = (
    ("initial_stand", 0.5, 2.0, 0.0), ("walk_0p6_up", 3.0, 5.0, 0.6),
    ("walk_1p2_up", 6.0, 8.0, 1.2), ("run_2p4_up", 9.5, 12.5, 2.4),
    ("run_2p6", 14.0, 17.0, 2.6), ("run_2p4_down", 18.5, 21.5, 2.4),
    ("walk_1p2_down", 23.0, 25.0, 1.2), ("walk_0p6_down", 26.0, 28.0, 0.6),
    ("final_stand", 29.0, 32.0, 0.0),
)


def command(t: float) -> float:
    for (ta, va), (tb, vb) in zip(POINTS, POINTS[1:]):
        if ta <= t < tb:
            return va if va == vb else va + (vb - va) * float(minimum_jerk((t - ta) / (tb - ta)))
    return 0.0


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    selected = json.loads((args.output / "selected_checkpoint.json").read_text(encoding="utf-8"))
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 50
    cfg.seed = 20263021
    cfg.episode_length_s = 35.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    agent_cfg.seed = 20263021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    import importlib.metadata
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(Path(selected["checkpoint"]).resolve()), strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        env = raw.unwrapped
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        term.external_override_enabled = True
        obs, _ = wrapped.reset()
        ref = yaw_from_quat_wxyz(robot.data.root_quat_w).clone()
        dt = float(env.step_dt)
        fallen = torch.zeros(50, dtype=torch.bool, device=runner.device)
        stats = {
            name: {
                "speed_error": torch.zeros(50, device=runner.device),
                "heading": [[] for _ in range(50)], "count": torch.zeros(50, device=runner.device),
                "flight": torch.zeros(50, device=runner.device),
                "double": torch.zeros(50, device=runner.device),
                "events": torch.zeros(50, dtype=torch.long, device=runner.device),
                "safe_events": torch.zeros(50, dtype=torch.long, device=runner.device),
            } for name, *_ in HOLDS
        }
        flight_streak = torch.zeros(50, dtype=torch.long, device=runner.device)
        previous_flight = torch.zeros(50, dtype=torch.bool, device=runner.device)
        for step in range(round(32.0 / dt)):
            t = step * dt
            target = command(t)
            term.external_override[:, 0] = target
            term.external_override[:, 1:] = 0.0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            with torch.inference_mode():
                action = policy(obs)
            obs, _, dones, _ = wrapped.step(action)
            obs = obs.to(runner.device)
            fallen |= dones.bool()
            contacts = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1) > 5.0
            in_flight = contacts.sum(dim=1) == 0
            landed = (~in_flight) & previous_flight
            prior = flight_streak.clone()
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            yaw_error = wrapped_heading_error(ref, yaw_from_quat_wxyz(robot.data.root_quat_w)).abs()
            for name, begin, end, hold_target in HOLDS:
                if begin <= t < end:
                    item = stats[name]
                    item["count"] += 1
                    item["speed_error"] += (robot.data.root_lin_vel_b[:, 0] - hold_target).abs()
                    item["flight"] += in_flight.float()
                    item["double"] += (contacts.sum(dim=1) >= 2).float()
                    item["events"] += (in_flight & ~previous_flight).long()
                    item["safe_events"] += (landed & (prior >= 2) & (prior <= 8)).long()
                    for env_id in range(50):
                        item["heading"][env_id].append(float(yaw_error[env_id]))
            previous_flight = in_flight
        segments = {}
        episode_complete = torch.ones(50, dtype=torch.bool, device=runner.device)
        for name, _, _, target in HOLDS:
            item = stats[name]
            count = torch.clamp(item["count"], min=1)
            speed_mae = item["speed_error"] / count
            flight_fraction = item["flight"] / count
            double = item["double"] / count
            heading_p95 = torch.tensor([
                float(torch.quantile(torch.tensor(values), .95)) for values in item["heading"]
            ], device=runner.device)
            if target == 0.0:
                success = (~fallen) & (speed_mae <= .05) & (flight_fraction == 0) & (double >= .95)
            elif target >= 2.3:
                success = (~fallen) & (speed_mae <= .25) & (item["events"] >= 2) & (item["safe_events"] >= 2)
            else:
                success = (~fallen) & (speed_mae <= .20) & (flight_fraction <= .05)
            success &= heading_p95 <= .12
            episode_complete &= success
            segments[name] = {
                "target_speed": target, "success_rate": float(success.float().mean()),
                "speed_mae": float(speed_mae.mean()), "heading_p95": float(torch.quantile(heading_p95, .95)),
                "flight_fraction": float(flight_fraction.mean()),
                "double_support_rate": float((double >= .95).float().mean()),
                "flight_event_mean": float(item["events"].float().mean()),
                "safe_flight_event_mean": float(item["safe_events"].float().mean()),
            }
        dump(args.output / "formal_integrated_sequence_detail.json", {
            "episodes": 50, "seed_root": 20263021, "checkpoint": selected["checkpoint"],
            "checkpoint_sha256": selected["sha256"], "sequence_completion_rate": float(episode_complete.float().mean()),
            "fall_rate": float(fallen.float().mean()), "segments": segments,
            "checkpoint_switch": 0, "expert_action_calls": 0,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
