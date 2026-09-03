"""Collect frozen WALK/RUN teacher endpoint data for Stage 2K."""

from __future__ import annotations

import argparse
import copy
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
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight"
RAW = OUT / "raw"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
EXPECTED = {
    "WALK": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "RUN": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
}
GROUPS = (
    ("WALK_1P2", 0, 500, 1.2, 0.0),
    ("RUN_1P2", 1, 250, 1.2, 1.0),
    ("RUN_2P4", 1, 250, 2.4, 1.0),
    ("RUN_2P6", 1, 250, 2.6, 1.0),
)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    if sha(WALK) != EXPECTED["WALK"] or sha(RUN) != EXPECTED["RUN"]:
        raise RuntimeError("GAIT_LATENT_TEACHER_PROVENANCE_FAIL")
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = sum(group[2] for group in GROUPS)
    cfg.episode_length_s = 10.0
    cfg.seed = 20267021
    agent_cfg.seed = 20267021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        payloads = {
            "WALK": torch.load(WALK, map_location=runner.device, weights_only=False),
            "RUN": torch.load(RUN, map_location=runner.device, weights_only=False),
        }
        actors, critics = {}, {}
        for name in ("WALK", "RUN"):
            actor = copy.deepcopy(runner.alg.actor)
            critic = copy.deepcopy(runner.alg.critic)
            actor.load_state_dict(payloads[name]["actor_state_dict"], strict=True)
            critic.load_state_dict(payloads[name]["critic_state_dict"], strict=True)
            actor.eval()
            critic.eval()
            actors[name], critics[name] = actor, critic
        env = wrapped.unwrapped
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        sensor = env.scene.sensors["contact_forces"]
        feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        count = cfg.scene.num_envs
        group_id = torch.empty(count, dtype=torch.long, device=runner.device)
        teacher_id = torch.empty_like(group_id)
        speed = torch.empty(count, device=runner.device)
        gait = torch.empty(count, device=runner.device)
        starts = []
        cursor = 0
        for index, (_, teacher, episodes, target, gait_value) in enumerate(GROUPS):
            starts.append((cursor, cursor + episodes))
            group_id[cursor:cursor + episodes] = index
            teacher_id[cursor:cursor + episodes] = teacher
            speed[cursor:cursor + episodes] = target
            gait[cursor:cursor + episodes] = gait_value
            cursor += episodes
        command.external_override[:, 0] = speed
        command.external_override[:, 1:] = 0
        obs, _ = wrapped.reset()
        obs = obs.to(runner.device)
        fields = {key: [] for key in ("observation", "teacher_action", "teacher_value", "contact")}
        flight_streak = torch.zeros(count, dtype=torch.long, device=runner.device)
        flight_events = torch.zeros_like(flight_streak)
        safe_flights = torch.zeros_like(flight_streak)
        alternating = torch.zeros_like(flight_streak)
        last_landing = torch.full_like(flight_streak, -1)
        fallen = torch.zeros(count, dtype=torch.bool, device=runner.device)
        for step in range(500):
            command.external_override[:, 0] = speed
            command.external_override[:, 1:] = 0
            if step == 0:
                obs = wrapped.get_observations().to(runner.device)
            action = torch.zeros((count, 37), device=runner.device)
            value = torch.zeros(count, device=runner.device)
            with torch.inference_mode():
                for teacher, name in enumerate(("WALK", "RUN")):
                    mask = teacher_id == teacher
                    action[mask] = actors[name](obs[mask], stochastic_output=False)
                    value[mask] = critics[name](obs[mask]).squeeze(-1)
            forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contacts = forces > 5.0
            fields["observation"].append(obs["policy"].detach().cpu())
            fields["teacher_action"].append(action.detach().cpu())
            fields["teacher_value"].append(value.detach().cpu())
            fields["contact"].append(contacts.detach().cpu())
            obs, _, dones, extras = wrapped.step(action)
            obs = obs.to(runner.device)
            timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
            fallen |= dones.bool() & ~timeout
            forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contacts = forces > 5.0
            in_flight = contacts.sum(-1) == 0
            previous = flight_streak.clone()
            flight_events += (in_flight & (flight_streak == 0)).long()
            flight_streak = torch.where(in_flight, flight_streak + 1, torch.zeros_like(flight_streak))
            landing = ~in_flight & (previous > 0)
            single = landing & (contacts.sum(-1) == 1)
            foot = contacts.long().argmax(-1)
            safe = single & (previous >= 2) & (previous <= 8)
            alternation = safe & (last_landing >= 0) & (foot != last_landing)
            safe_flights += safe.long()
            alternating += alternation.long()
            last_landing[single] = foot[single]
        tensors = {key: torch.stack(value) for key, value in fields.items()}
        tensors.update({
            "group_id": group_id.cpu(), "teacher_id": teacher_id.cpu(), "speed": speed.cpu(),
            "gait_cmd": gait.cpu(), "teacher_std": torch.stack([
                payloads["WALK"]["actor_state_dict"]["distribution.std_param"].cpu(),
                payloads["RUN"]["actor_state_dict"]["distribution.std_param"].cpu(),
            ]),
            "group_names": [group[0] for group in GROUPS],
        })
        gait_labels = []
        group_summary = {}
        for group_index, (name, _, episodes, target, gait_value) in enumerate(GROUPS):
            left, right = starts[group_index]
            labels = []
            for env_id in range(left, right):
                periodic = (
                    int(flight_events[env_id]) >= 4
                    and int(safe_flights[env_id]) >= 3
                    and int(alternating[env_id]) >= 3
                )
                flight_fraction = float((~tensors["contact"][:, env_id].any(-1)).float().mean())
                label = (
                    "FALL" if bool(fallen[env_id]) else
                    "PERIODIC_RUNNING" if periodic else
                    "WALK_LIKE" if flight_fraction < .10 else
                    "ISOLATED_FLIGHT"
                )
                gait_labels.append(label)
                labels.append(label)
            group_summary[name] = {
                "episodes": episodes, "target_speed": target, "gait_cmd": gait_value,
                "walk_like_rate": labels.count("WALK_LIKE") / episodes,
                "periodic_running_rate": labels.count("PERIODIC_RUNNING") / episodes,
                "fall_rate": labels.count("FALL") / episodes,
            }
        tensors["episode_gait_labels"] = gait_labels
        raw_path = RAW / "gait_latent_endpoint_dataset.pt"
        torch.save(tensors, raw_path)
        # Split strictly by episode, stratified by the four collection groups.
        split_generator = torch.Generator().manual_seed(20267021)
        splits = {"train": [], "validation": [], "held_out": []}
        split_by_group = {}
        for group_index, (name, _, episodes, _, _) in enumerate(GROUPS):
            left, _ = starts[group_index]
            order = torch.randperm(episodes, generator=split_generator) + left
            train_end, validation_end = int(.8 * episodes), int(.9 * episodes)
            group_split = {
                "train": order[:train_end].tolist(),
                "validation": order[train_end:validation_end].tolist(),
                "held_out": order[validation_end:].tolist(),
            }
            split_by_group[name] = group_split
            for split, ids in group_split.items():
                splits[split].extend(ids)
        split_payload = {
            "seed": 20267021, "unit": "episode", "stratified_groups": split_by_group,
            "counts": {name: len(ids) for name, ids in splits.items()},
            "episode_overlap": 0,
        }
        split_text = json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode()
        split_payload["split_sha256"] = hashlib.sha256(split_text).hexdigest()
        dump("gait_latent_dataset_split.json", split_payload)
        dump("gait_latent_dataset_manifest.json", {
            "path": str(raw_path.relative_to(REPO)), "raw_git_tracked": False,
            "episodes": count, "steps_per_episode": 500, "samples": count * 500,
            "observation_dimension": 123, "action_dimension": 37,
            "walk_endpoint_episodes": 500, "run_endpoint_episodes": 750,
            "training_sampling": {"WALK": .5, "RUN": .5, "RUN_speed_conditional": {"1.2": 1/3, "2.4": 1/3, "2.6": 1/3}},
            "fields": [
                "observation", "gait_cmd", "teacher_action_mean", "teacher_std", "teacher_value",
                "contact_state", "gait_classification", "speed", "episode_id", "timestep",
            ],
            "teacher_deterministic": True, "group_summary": group_summary,
        })
        dump("gait_latent_dataset_hashes.json", {
            "raw_file_sha256": sha(raw_path), "split_sha256": split_payload["split_sha256"],
            "observation_sha256": hashlib.sha256(tensors["observation"].numpy().tobytes()).hexdigest(),
            "action_sha256": hashlib.sha256(tensors["teacher_action"].numpy().tobytes()).hexdigest(),
        })
        dump("walk_teacher_manifest.json", {
            "path": str(WALK.relative_to(REPO)), "sha256": EXPECTED["WALK"],
            "role": "diagnostic endpoint label at gait_cmd=0", "updated": False,
        })
        dump("run_teacher_manifest.json", {
            "path": str(RUN.relative_to(REPO)), "sha256": EXPECTED["RUN"],
            "role": "student initialization and diagnostic endpoint labels at gait_cmd=1", "updated": False,
        })
        wrapped.close()


if __name__ == "__main__":
    main()
