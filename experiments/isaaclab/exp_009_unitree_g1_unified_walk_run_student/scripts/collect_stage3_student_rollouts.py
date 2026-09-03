"""Collect Stage 0/1/2 student WALK occupancy with actual applied actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision"
CFG_PATH = EXP / "configs/stage3_nonlinear_rollout_supervision.yaml"
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import g1_command_skills.tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
import isaaclab_tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_walk_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from unified_walk_run.command_profile import minimum_jerk  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=160)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def load_student(path: Path, device: torch.device) -> UnifiedWalkRunStudent123:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("student", payload.get("model"))
    if state and all(key.startswith("network.") for key in state):
        state = {key.removeprefix("network."): value for key, value in state.items()}
    model = UnifiedWalkRunStudent123().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def grouped_split(episode: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(episode.encode()).digest()[:4], "little") % 100
    return "train" if bucket < 70 else ("validation" if bucket < 85 else "test")


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = 1024
    task_cfg.seed = 20270405
    task_cfg.episode_length_s = 30.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    output_dir = OUT / "student_rollouts.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device, n = wrapped.unwrapped, wrapped.unwrapped.device, wrapped.num_envs
        dt = float(env.step_dt)
        walk = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
            device=device,
        )
        stand = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
            device=device,
        )
        stw = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
            device=device,
        )
        students = [
            load_student(REPO / cfg["sources"]["stage0_student"], device),
            load_student(REPO / cfg["sources"]["stage1_student"], device),
            load_student(REPO / cfg["sources"]["stage2_student"], device),
        ]
        labels = ["stage0_student", "stage1_student", "stage2_student"]
        for module in (walk.actor, stand.actor, stw.actor):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        robot, command_term = env.scene["robot"], env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        target_values = torch.tensor([0.6, 0.8, 1.0, 1.2], device=device)
        target_index = torch.arange(n, device=device) % 4
        model_index = (torch.arange(n, device=device) // 4) % 3
        targets = target_values[target_index]

        def state():
            legacy = wrapped.get_observations()["policy"]
            return canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)

        def command(speed, heading):
            error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
            yaw = (0.8 * error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed, yaw
            return MotionCommand(speed, heading, target_yaw_rate_radps=yaw), error

        def contact_state():
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(1) > 5
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            return contacts, support

        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)
        elapsed = torch.zeros(n, device=device)
        good = torch.zeros(n, device=device)
        ready = torch.zeros(n, dtype=torch.bool, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        switches = torch.zeros(n, dtype=torch.long, device=device)
        for _ in range(round(12 / dt)):
            canonical = state()
            speed = torch.where(phase < 2, torch.zeros_like(targets), torch.where(phase == 2, targets * minimum_jerk(elapsed / 1.5), targets))
            motion, error = command(speed, heading)
            with torch.no_grad():
                a0, a1, a2 = stand(canonical, motion), stw(canonical, motion), walk(canonical, motion)
                action = torch.where((phase < 2)[:, None], a0, torch.where((phase == 2)[:, None], a1, a2))
                _, _, dones, _ = wrapped.step(action)
            contacts, support = contact_state()
            settled = (robot.data.root_lin_vel_b.torch[:, 0].abs() < 0.1) & contacts.all(1) & ~dones.bool()
            good = torch.where((phase == 0) & settled, good + dt, torch.where(phase == 0, 0, good))
            advance = (phase == 0) & (good >= 0.4)
            phase[advance], elapsed[advance], good[advance] = 1, 0, 0
            advance = (phase == 1) & (elapsed >= 0.8)
            phase[advance], elapsed[advance] = 2, 0
            changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
            switches[changed] += 1
            acquire = (phase == 2) & ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.2) & (error.abs() <= 0.12) & (switches >= 2) & ~dones.bool()
            good = torch.where(acquire, good + dt, torch.where(phase == 2, 0, good))
            advance = (phase == 2) & (good >= 0.4)
            phase[advance], elapsed[advance], good[advance] = 3, 0, 0
            valid = (phase == 3) & ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.4) & (error.abs() <= 0.12) & ~dones.bool()
            ready |= valid
            previous_support = support
            elapsed += dt
            if int(ready.sum()) >= math.ceil(0.9 * n):
                break
        if int(ready.sum()) < math.ceil(0.9 * n):
            raise RuntimeError(f"student rollout WALK source preparation shortfall: {int(ready.sum())}/{n}")

        alive = ready.clone()
        buffer = []
        for step in range(args.steps):
            canonical = state()
            motion, _ = command(targets, heading)
            obs = to_walk_observation(canonical, motion)
            with torch.no_grad():
                action = students[0](obs)
                for index, student in enumerate(students):
                    mask = model_index == index
                    action[mask] = student(obs[mask])
                _, _, dones, _ = wrapped.step(action)
            contacts, support = contact_state()
            ids = torch.nonzero(alive & torch.isfinite(obs).all(1) & torch.isfinite(action).all(1)).flatten()
            if len(ids):
                obs_np, action_np = obs[ids].cpu().numpy().astype(np.float32), action[ids].cpu().numpy().astype(np.float32)
                id_np = ids.cpu().numpy()
                buffer.append(pd.DataFrame({
                    **{f"obs_{i:03d}": obs_np[:, i] for i in range(123)},
                    **{f"action_{i:03d}": action_np[:, i] for i in range(37)},
                    "teacher": np.asarray([labels[int(model_index[i])] for i in ids], dtype=object),
                    "regime": np.full(len(ids), "student_walk_rollout", dtype=object),
                    "source_speed_mps": targets[ids].cpu().numpy(),
                    "target_speed_mps": targets[ids].cpu().numpy(),
                    "support_phase": support[ids].cpu().numpy().astype(np.int8),
                    "left_contact": contacts[ids, 0].cpu().numpy().astype(np.int8),
                    "right_contact": contacts[ids, 1].cpu().numpy().astype(np.int8),
                    "episode_id": np.asarray([f"{labels[int(model_index[i])]}_{float(targets[i]):.1f}_env{int(i)}" for i in ids], dtype=object),
                    "sequence_step": np.full(len(ids), step, dtype=np.int16),
                    "split": np.full(len(ids), "", dtype=object),
                }))
                for row_index, env_id in enumerate(id_np):
                    episode = buffer[-1].iloc[row_index]["episode_id"]
                    buffer[-1].iat[row_index, buffer[-1].columns.get_loc("split")] = grouped_split(str(episode))
            alive &= ~dones.bool()
        frame = pd.concat(buffer, ignore_index=True)
        frame.to_parquet(output_dir / "part-0000.parquet", compression="zstd", index=False)
        summary = {
            "rows": len(frame), "episodes": int(frame["episode_id"].nunique()),
            "checkpoint_counts": frame["teacher"].value_counts().to_dict(),
            "speed_counts": {str(k): int(v) for k, v in frame["target_speed_mps"].value_counts().sort_index().items()},
            "source_ready": int(ready.sum()), "physical_envs": n, "steps": args.steps,
            "state_setter": 0, "teleport": 0, "snapshot_injection": 0, "teacher_gradients": 0,
        }
        (OUT / "student_rollout_collection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
