"""Detailed 1.2 m/s endpoint hysteresis telemetry for the selected policy."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from g1_single_policy.phase_gated_heading import minimum_jerk  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def profile(name: str, t: float) -> float:
    if name == "reset_steady_1p2":
        return 1.2
    source = 0.6 if name == "upward_0p6_to_1p2" else (2.4 if "2p4" in name else 2.6)
    if t < 2.0:
        return source
    if t < 3.5:
        tau = (t - 2.0) / 1.5
        return source + (1.2 - source) * float(minimum_jerk(tau))
    return 1.2


def main() -> None:
    selected = json.loads((args.output / "selected_checkpoint.json").read_text(encoding="utf-8"))
    names = ("reset_steady_1p2", "upward_0p6_to_1p2", "after_2p4_to_1p2", "after_2p6_to_1p2")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 200
    cfg.seed = 20263021
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
        env, robot = raw.unwrapped, raw.unwrapped.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        term.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        spec = torch.arange(200, device=runner.device) // 50
        obs, _ = wrapped.reset()
        previous = torch.zeros((200, 37), device=runner.device)
        fields = {
            key: torch.zeros(200, device=runner.device)
            for key in ("count", "speed", "flight", "pitch", "height", "action_norm", "action_rate", "slip")
        }
        for step in range(round(8.0 / float(env.step_dt))):
            t = step * float(env.step_dt)
            command = torch.tensor([profile(names[int(i)], t) for i in spec.cpu()], device=runner.device)
            term.external_override[:, 0] = command
            term.external_override[:, 1:] = 0.0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            with torch.inference_mode():
                action = policy(obs)
            obs, _, _, _ = wrapped.step(action)
            obs = obs.to(runner.device)
            if t >= 4.0:
                contacts = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1) > 5.0
                quat = robot.data.root_quat_w
                pitch = torch.asin(torch.clamp(2 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1]), -1, 1))
                body_ids = [i for i, body in enumerate(robot.body_names) if "ankle_roll" in body]
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, body_ids, :2], dim=-1)
                fields["count"] += 1
                fields["speed"] += robot.data.root_lin_vel_b[:, 0]
                fields["flight"] += (contacts.sum(dim=1) == 0).float()
                fields["pitch"] += pitch
                fields["height"] += robot.data.root_pos_w[:, 2]
                fields["action_norm"] += torch.linalg.vector_norm(action, dim=1)
                fields["action_rate"] += torch.linalg.vector_norm(action - previous, dim=1)
                fields["slip"] += ((foot_speed > .55) & contacts).any(dim=1).float()
            previous.copy_(action)
        rows = []
        for index, name in enumerate(names):
            mask = spec == index
            count = torch.clamp(fields["count"][mask], min=1)
            rows.append({
                "endpoint": name,
                "actual_speed": float((fields["speed"][mask] / count).mean()),
                "flight_fraction": float((fields["flight"][mask] / count).mean()),
                "base_pitch": float((fields["pitch"][mask] / count).mean()),
                "base_height": float((fields["height"][mask] / count).mean()),
                "action_norm": float((fields["action_norm"][mask] / count).mean()),
                "action_rate": float((fields["action_rate"][mask] / count).mean()),
                "slip_fraction": float((fields["slip"][mask] / count).mean()),
            })
        with (args.output / "endpoint_state_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        wrapped.close()


if __name__ == "__main__":
    main()
