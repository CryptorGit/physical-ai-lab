"""Collect fail-closed Stage 2Q DAgger labels on STAND/STOP states."""

from __future__ import annotations

import argparse
import copy
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
EXPECTED_WALK = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, choices=(1, 2), required=True)
parser.add_argument("--student", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def minimum_jerk(x):
    x = torch.clamp(x, 0., 1.)
    return 10*x**3 - 15*x**4 + 6*x**5


class Student(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37))
        self.hidden.load_state_dict(OrderedDict((k.removeprefix("hidden."), v) for k, v in state.items() if k.startswith("hidden.")))

    def forward(self, obs, gait):
        return self.hidden(nn.functional.linear(obs, self.first_base_weight, self.first_bias) + gait[:, None] * self.first_gait_column.T)


def main():
    if hashlib.sha256(WALK.read_bytes()).hexdigest() != EXPECTED_WALK:
        raise RuntimeError("STAGE2Q_DAGGER_TEACHER_PROVENANCE_FAIL")
    student_path = Path(args.student).resolve()
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 500
    cfg.episode_length_s = 10.
    cfg.seed = 20269040 + args.round
    agent_cfg.seed = cfg.seed
    if args.device:
        cfg.sim.device, agent_cfg.device = args.device, args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        teacher = copy.deepcopy(runner.alg.actor)
        teacher.load_state_dict(torch.load(WALK, map_location=runner.device, weights_only=False)["actor_state_dict"], strict=True)
        teacher.eval()
        student = Student(torch.load(student_path, map_location=runner.device, weights_only=False)["actor_state_dict"]).to(runner.device).eval()
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        fields = {key: [] for key in ("observation", "target_action", "gait_cmd")}
        group_id = torch.cat((torch.zeros(250, dtype=torch.long), torch.ones(250, dtype=torch.long))).to(runner.device)
        for step in range(500):
            t = step * float(env.step_dt)
            speed = torch.zeros(500, device=runner.device)
            stop = group_id == 1
            if t < 2:
                speed[stop] = 1.2
            elif t < 3:
                speed[stop] = 1.2 + (.6 - 1.2) * minimum_jerk(torch.tensor(t - 2, device=runner.device))
            elif t < 4:
                speed[stop] = .6 * (1 - minimum_jerk(torch.tensor(t - 3, device=runner.device)))
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            gait = torch.zeros(500, device=runner.device)
            with torch.inference_mode():
                action = student(obs["policy"], gait)
                label = teacher(obs, stochastic_output=False)
            fields["observation"].append(obs["policy"].cpu())
            fields["target_action"].append(label.cpu())
            fields["gait_cmd"].append(gait.cpu())
            obs, _, _, _ = wrapped.step(action)
            obs = obs.to(runner.device)
        payload = {key: torch.stack(value) for key, value in fields.items()}
        payload.update({"group_id": group_id.cpu(), "group_names": ["STAND", "WALK_TO_STAND"], "source_student_sha": hashlib.sha256(student_path.read_bytes()).hexdigest()})
        torch.save(payload, OUT / f"raw/dagger_round_{args.round}.pt")
        wrapped.close()


if __name__ == "__main__":
    main()
