"""Collect the Stage 2Q supplemental endpoint and toggle-retention datasets."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
RAW = OUT / "raw"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
EXPECTED = {
    "WALK": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "RUN": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
    "PARENT": "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121",
}
SUPPLEMENT = (
    ("STAND_0P0", 0, 500, 0.0, 0.0),
    ("WALK_0P6", 0, 300, 0.6, 0.0),
    ("WALK_0P8", 0, 300, 0.8, 0.0),
    ("WALK_1P0", 0, 300, 1.0, 0.0),
    ("RUN_1P2_EXTRA", 1, 50, 1.2, 1.0),
    ("RUN_2P4_EXTRA", 1, 50, 2.4, 1.0),
    ("RUN_2P6_EXTRA", 1, 50, 2.6, 1.0),
)
TOGGLES = (
    ("WALK_TO_RUN", 300, 0.0, 1.0),
    ("RUN_TO_WALK", 300, 1.0, 0.0),
)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("supplement", "toggle"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minimum_jerk(x):
    x = torch.clamp(x, 0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


class IntegratedActor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"], requires_grad=False)
        self.first_gait_column = nn.Parameter(state["first_gait_column"], requires_grad=False)
        self.first_bias = nn.Parameter(state["first_bias"], requires_grad=False)
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value)
            for key, value in state.items() if key.startswith("hidden.")
        ))

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait[:, None] * self.first_gait_column.T)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    for name, path in (("WALK", WALK), ("RUN", RUN), ("PARENT", PARENT)):
        if sha(path) != EXPECTED[name]:
            raise RuntimeError(f"STAGE2Q_{name}_PROVENANCE_FAIL")
    groups = SUPPLEMENT if args.mode == "supplement" else TOGGLES
    count = sum(group[2] if args.mode == "supplement" else group[1] for group in groups)
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = count
    cfg.episode_length_s = 10.0
    cfg.seed = 20269021
    agent_cfg.seed = 20269021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        device = runner.device
        teacher_payloads = {
            "WALK": torch.load(WALK, map_location=device, weights_only=False),
            "RUN": torch.load(RUN, map_location=device, weights_only=False),
        }
        teachers = {}
        for name in ("WALK", "RUN"):
            actor = copy.deepcopy(runner.alg.actor)
            actor.load_state_dict(teacher_payloads[name]["actor_state_dict"], strict=True)
            teachers[name] = actor.eval()
        parent_payload = torch.load(PARENT, map_location=device, weights_only=False)
        parent = IntegratedActor(parent_payload["actor_state_dict"]).to(device).eval()
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        group_id = torch.empty(count, dtype=torch.long, device=device)
        speed = torch.empty(count, device=device)
        gait_start = torch.empty(count, device=device)
        gait_end = torch.empty(count, device=device)
        teacher_id = torch.full((count,), -1, dtype=torch.long, device=device)
        cursor = 0
        group_names = []
        for index, group in enumerate(groups):
            if args.mode == "supplement":
                name, teacher, episodes, target, gait0 = group
                gait1 = gait0
                teacher_id[cursor:cursor + episodes] = teacher
            else:
                name, episodes, gait0, gait1 = group
                target = 1.2
            group_names.append(name)
            right = cursor + episodes
            group_id[cursor:right] = index
            speed[cursor:right] = target
            gait_start[cursor:right] = gait0
            gait_end[cursor:right] = gait1
            cursor = right
        command.external_override[:, 0] = speed
        command.external_override[:, 1:] = 0
        obs, _ = wrapped.reset()
        obs = obs.to(device)
        stored = {key: [] for key in ("observation", "target_action", "contact", "gait_cmd")}
        for step in range(500):
            t = step * float(env.step_dt)
            if args.mode == "toggle":
                phase = minimum_jerk(torch.full((count,), (t - 4.0) / 2.0, device=device))
                gait = gait_start + (gait_end - gait_start) * phase
            else:
                gait = gait_start
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(device)
            action = torch.zeros((count, 37), device=device)
            with torch.inference_mode():
                if args.mode == "supplement":
                    for teacher, name in enumerate(("WALK", "RUN")):
                        mask = teacher_id == teacher
                        action[mask] = teachers[name](obs[mask], stochastic_output=False)
                else:
                    action = parent(obs["policy"], gait)
            contacts = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1) > 5
            stored["observation"].append(obs["policy"].detach().cpu())
            stored["target_action"].append(action.detach().cpu())
            stored["contact"].append(contacts.detach().cpu())
            stored["gait_cmd"].append(gait.detach().cpu())
            obs, _, _, _ = wrapped.step(action)
            obs = obs.to(device)
        payload = {key: torch.stack(value) for key, value in stored.items()}
        payload.update({
            "group_id": group_id.cpu(), "group_names": group_names, "speed": speed.cpu(),
            "teacher_id": teacher_id.cpu(), "episode_count": count, "steps_per_episode": 500,
            "mode": args.mode,
        })
        path = RAW / f"stage2q_{args.mode}_dataset.pt"
        torch.save(payload, path)
        manifest = {
            "mode": args.mode, "path": str(path.relative_to(REPO)), "sha256": sha(path),
            "episodes": count, "samples": count * 500, "group_names": group_names,
            "teacher_action_runtime": False, "deterministic_labels": True,
        }
        (OUT / f"{args.mode}_collection_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        wrapped.close()


if __name__ == "__main__":
    main()
