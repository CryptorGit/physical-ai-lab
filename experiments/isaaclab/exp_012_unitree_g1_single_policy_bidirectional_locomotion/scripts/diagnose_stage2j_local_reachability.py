"""Fresh-process, same-seed bounded one-step action reachability probe for Stage 2J."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
PHASES = ("left_support", "right_support", "double_support", "flight")
PERTURBATIONS = (-.04, -.02, -.01, .01, .02, .04)
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=("baseline", "variant", "sequence_baseline", "sequence_variant"), required=True
)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def phase_mask(contacts, phase_id):
    if phase_id == 0:
        return contacts[:, 0] & ~contacts[:, 1]
    if phase_id == 1:
        return ~contacts[:, 0] & contacts[:, 1]
    if phase_id == 2:
        return contacts[:, 0] & contacts[:, 1]
    return ~contacts[:, 0] & ~contacts[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(exist_ok=True)
    if sha(CHECKPOINT) != EXPECTED:
        raise RuntimeError("STAGE2J_R1_PROVENANCE_FAIL")
    is_sequence = args.mode.startswith("sequence")
    if is_sequence:
        generator = torch.Generator().manual_seed(20268042)
        leg_joints = torch.tensor([0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20])
        sequences = torch.zeros((1024, 4, 37))
        for env_id in range(1024):
            selected = leg_joints[torch.randperm(len(leg_joints), generator=generator)[:4]]
            sequences[env_id, :, selected] = (
                torch.randint(0, 3, (4, len(selected)), generator=generator).float() - 1.0
            ) * .04
        specs = [(env_id // 256, -1, float(sequences[env_id].norm()), env_id % 256) for env_id in range(1024)]
    else:
        # Screening covers every phase/joint/signed magnitude and at least 20 distinct
        # reset states per phase. Full 20x replication is intentionally reserved for
        # any localized follow-up because it would add no information before screening.
        specs = [
            (phase, joint, perturbation, 0)
            for phase in range(4) for joint in range(37) for perturbation in PERTURBATIONS
        ]
        while len(specs) < 1024:
            phase, joint, perturbation, _ = specs[len(specs) % (4 * 37 * len(PERTURBATIONS))]
            specs.append((phase, joint, perturbation, 1))
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-Reverse-PhaseR1-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.seed = 20268041
    cfg.episode_length_s = 20.0
    agent_cfg.seed = 20268041
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw_env = gym.make("Isaac-Exp012-G1-Reverse-PhaseR1-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
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
        robot = env.scene["robot"]
        sensor = env.scene.sensors["contact_forces"]
        feet = [index for index, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        batches = math.ceil(len(specs) / 1024)
        all_results = []
        prefix_hashes = []
        for batch in range(batches):
            batch_specs = specs[batch * 1024:(batch + 1) * 1024]
            count = len(batch_specs)
            obs, _ = wrapped.reset()
            obs = obs.to(runner.device)
            command.external_override[:, 0] = 1.2
            command.external_override[:, 1:] = 0
            spec_phase = torch.zeros(1024, dtype=torch.long, device=runner.device)
            spec_joint = torch.zeros_like(spec_phase)
            spec_delta = torch.zeros(1024, device=runner.device)
            for env_id, (phase, joint, delta, _) in enumerate(batch_specs):
                spec_phase[env_id] = phase
                spec_joint[env_id] = joint
                spec_delta[env_id] = delta
            triggered = torch.zeros(1024, dtype=torch.bool, device=runner.device)
            trigger_step = torch.full((1024,), -1, dtype=torch.long, device=runner.device)
            prefix_obs = torch.zeros((1024, 123), device=runner.device)
            speed_trace = torch.full((32, 1024), torch.nan, device=runner.device)
            contact_trace = torch.zeros((32, 1024, 2), dtype=torch.bool, device=runner.device)
            fallen = torch.zeros(1024, dtype=torch.bool, device=runner.device)
            slip = torch.zeros_like(fallen)
            impact = torch.zeros_like(fallen)
            for step in range(170):
                command.external_override[:, 0] = 1.2
                command.external_override[:, 1:] = 0
                if step == 0:
                    obs = wrapped.get_observations().to(runner.device)
                with torch.inference_mode():
                    policy_action = policy(obs)
                action = torch.empty(policy_action.shape, device=policy_action.device, dtype=policy_action.dtype)
                action.copy_(policy_action)
                forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contacts = forces > 5.0
                if step >= 100:
                    match = torch.zeros(1024, dtype=torch.bool, device=runner.device)
                    for phase_id in range(4):
                        match |= (spec_phase == phase_id) & phase_mask(contacts, phase_id)
                    new = ~triggered & match
                    prefix_obs[new] = obs["policy"][new]
                    trigger_step[new] = step
                    if args.mode == "variant":
                        ids = torch.where(new)[0]
                        action[ids, spec_joint[ids]] += spec_delta[ids]
                    triggered |= new
                if args.mode == "sequence_variant":
                    elapsed_for_action = step - trigger_step
                    for sequence_step in range(4):
                        ids = torch.where(triggered & (elapsed_for_action == sequence_step))[0]
                        if ids.numel():
                            action[ids] += sequences[ids.cpu(), sequence_step].to(action.device)
                obs, _, dones, extras = wrapped.step(action)
                obs = obs.to(runner.device)
                timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
                fallen |= dones.bool() & ~timeout
                forces_after = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contacts_after = forces_after > 5.0
                foot_body = [
                    next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[foot])
                    for foot in feet
                ]
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, foot_body, :2], dim=-1)
                slip |= ((foot_speed > .55) & contacts_after).any(-1)
                impact |= forces_after.amax(-1) > 3500.0
                elapsed = step - trigger_step
                active = triggered & (elapsed >= 1) & (elapsed <= 32)
                for horizon_index in range(1, 33):
                    mask = active & (elapsed == horizon_index)
                    speed_trace[horizon_index - 1, mask] = robot.data.root_lin_vel_b[mask, 0]
                    contact_trace[horizon_index - 1, mask] = contacts_after[mask]
            for env_id, (phase, joint, delta, replicate) in enumerate(batch_specs):
                valid = torch.isfinite(speed_trace[:, env_id])
                final_valid = valid & (torch.arange(32, device=runner.device) >= 15)
                final_contacts = contact_trace[final_valid, env_id]
                flight_fraction = float((final_contacts.sum(-1) == 0).float().mean()) if final_contacts.numel() else 1.0
                double_fraction = float((final_contacts.sum(-1) == 2).float().mean()) if final_contacts.numel() else 0.0
                final_speed_mae = (
                    float((speed_trace[final_valid, env_id] - 1.2).abs().mean()) if final_valid.any() else 99.0
                )
                all_results.append({
                    "spec_index": batch * 1024 + env_id, "phase": PHASES[phase], "joint_index": joint,
                    "delta": delta, "replicate": replicate, "triggered": bool(triggered[env_id]),
                    "sequence_hash": (
                        hashlib.sha256(sequences[env_id].numpy().tobytes()).hexdigest() if is_sequence else ""
                    ),
                    "trigger_step": int(trigger_step[env_id]), "prefix_observation_sha256": hashlib.sha256(
                        prefix_obs[env_id].detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                    "final_flight_fraction": flight_fraction, "final_double_support_fraction": double_fraction,
                    "final_speed_mae": final_speed_mae, "fall": bool(fallen[env_id]),
                    "slip": bool(slip[env_id]), "impact": bool(impact[env_id]),
                    "walk_like": flight_fraction <= .10 and final_speed_mae <= .20 and not bool(fallen[env_id]),
                })
            prefix_hashes.append(hashlib.sha256(prefix_obs[:count].detach().cpu().numpy().tobytes()).hexdigest())
        output = RAW / f"local_action_{args.mode}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_results[0]))
            writer.writeheader()
            writer.writerows(all_results)
        (RAW / f"local_action_{args.mode}_manifest.json").write_text(json.dumps({
            "mode": args.mode, "seed": 20268041, "checkpoint_sha256": EXPECTED,
            "samples": len(all_results), "batches": batches, "prefix_batch_hashes": prefix_hashes,
            "screening_design": (
                "1024 bounded four-step random-shooting candidates across four phases"
                if is_sequence else "all phase/joint/delta combinations; >=20 independent reset states per phase"
            ),
            "state_injection": False, "parameter_update": False,
        }, indent=2) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
