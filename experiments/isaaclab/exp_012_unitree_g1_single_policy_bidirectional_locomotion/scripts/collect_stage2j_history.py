"""Collect R1 observations at current command 1.2 after four distinct command histories."""

from __future__ import annotations

import argparse
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
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2j_low_speed_action_manifold_reachability"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1/checkpoints/model_1.pt"
EXPECTED = "707bd50a8a168f2b247965ff6977e41da1d560094a1d5328737eaa76963f3ecd"
NAMES = ("steady_reset_1p2", "zero_to_1p2", "2p4_to_1p2", "2p6_to_1p2")
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(value):
    value = max(0.0, min(1.0, float(value)))
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def command_at(group, t):
    if group == 0:
        return 1.2
    if group == 1:
        return 1.2 * float(minimum_jerk(t / 1.0)) if t < 1.0 else 1.2
    high = 2.4 if group == 2 else 2.6
    if t < 3.0:
        return high
    if t < 4.5:
        blend = float(minimum_jerk((t - 3.0) / 1.5))
        return high + (1.2 - high) * blend
    return 1.2


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != EXPECTED:
        raise RuntimeError("STAGE2J_R1_PROVENANCE_FAIL")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 400
    cfg.seed = 20268051
    cfg.episode_length_s = 20
    agent_cfg.seed = 20268051
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(CHECKPOINT), load_cfg={
            "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
        }, strict=True, map_location=runner.device)
        policy = runner.get_inference_policy(device=runner.device)
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        group_ids = torch.arange(400, device=runner.device) // 100
        obs, _ = wrapped.reset()
        traces = {"observation": [], "contact": [], "actual_speed": [], "action": []}
        for step in range(500):
            t = step * float(env.step_dt)
            for group_id in range(4):
                mask = group_ids == group_id
                command.external_override[mask, 0] = command_at(group_id, t)
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            with torch.inference_mode():
                action = policy(obs)
            if step >= 300:
                forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                traces["observation"].append(obs["policy"].detach().cpu())
                traces["contact"].append((forces > 5).detach().cpu())
                traces["actual_speed"].append(env.scene["robot"].data.root_lin_vel_b[:, 0].detach().cpu())
                traces["action"].append(action.detach().cpu())
            obs, _, _, _ = wrapped.step(action)
            obs = obs.to(runner.device)
        output = {key: torch.stack(value) for key, value in traces.items()}
        output["group_names"] = NAMES
        output["group_ids"] = group_ids.cpu()
        output["seed"] = 20268051
        torch.save(output, RAW / "command_history_trajectories.pt")
        (RAW / "command_history_manifest.json").write_text(json.dumps({
            "checkpoint_sha256": EXPECTED, "seed": 20268051, "episodes_per_history": 100,
            "current_command_in_analysis_window": 1.2, "history_names": NAMES,
            "parameter_update": False, "controller": "OFF",
        }, indent=2) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
