"""Collect 1.5M+ frozen multi-teacher steps in actual Isaac physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
import gymnasium as gym
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
DATA = OUT / "teacher_dataset.parquet"
CFG_PATH = EXP / "configs/stage0_multiteacher_distillation.yaml"
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
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation, to_walk_observation  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from unified_walk_run.dataset import action_columns, grouped_split, observation_columns  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mj(value):
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


parser = argparse.ArgumentParser()
parser.add_argument("--skip-walk", action="store_true")
parser.add_argument("--skip-run", action="store_true")
parser.add_argument("--skip-transition", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    for teacher in cfg["teachers"].values():
        path = REPO / teacher["path"]
        if sha(path) != teacher["sha256"]:
            raise RuntimeError(f"teacher SHA mismatch: {path}")
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["dataset"]["physical_envs"]
    task_cfg.seed = cfg["experiment"]["training_seed"]
    task_cfg.episode_length_s = 30.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    existing_parts = sorted(DATA.glob("part-*.parquet")) if DATA.exists() else []
    part_index = len(existing_parts)
    existing_rows = sum(len(pd.read_parquet(path, columns=["regime"])) for path in existing_parts)
    row_count = 0
    regime_counts = Counter()
    teacher_counts = Counter()
    speed_counts = Counter()
    phase_counts = Counter()
    split_counts = Counter()
    safety_counts = Counter()
    episode_count = 0
    buffer = []

    def flush():
        nonlocal part_index, buffer
        if not buffer:
            return
        packed = {}
        for key in buffer[0]:
            values = [item[key] for item in buffer]
            packed[key] = np.concatenate(values, axis=0)
        frame = pd.DataFrame({
            **{name: packed["obs"][:, index] for index, name in enumerate(observation_columns())},
            **{name: packed["action"][:, index] for index, name in enumerate(action_columns())},
            **{f"previous_action_{index:03d}": packed["previous_action"][:, index] for index in range(37)},
            "teacher": packed["teacher"],
            "regime": packed["regime"],
            "source_speed_mps": packed["source_speed"],
            "target_speed_mps": packed["target_speed"],
            "support_phase": packed["support_phase"],
            "left_contact": packed["contact"][:, 0],
            "right_contact": packed["contact"][:, 1],
            "left_foot_air_time": packed["air_time"][:, 0],
            "right_foot_air_time": packed["air_time"][:, 1],
            "sequence_id": packed["sequence_id"],
            "episode_id": packed["episode_id"],
            "sequence_step": packed["sequence_step"],
            "finite": packed["finite"],
            "fall": packed["fall"],
            "slip": packed["slip"],
            "saturation": packed["saturation"],
            "split": packed["split"],
        })
        frame.to_parquet(DATA / f"part-{part_index:04d}.parquet", compression="zstd", index=False)
        part_index += 1
        buffer = []

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device, n = wrapped.unwrapped, wrapped.unwrapped.device, wrapped.num_envs
        dt = float(env.step_dt)
        walk = load_walk_expert(REPO / cfg["teachers"]["walk"]["path"], device=device)
        run = load_run_expert(REPO / cfg["teachers"]["run"]["path"], device=device)
        stand = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
            device=device,
        )
        stw = load_walk_expert(
            REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
            device=device,
        )
        run_parent = run.actor
        wtr = WalkToRunTransitionActor152(run_parent).to(device)
        wtr.load_state_dict(
            torch.load(REPO / cfg["teachers"]["walk_to_run"]["path"], map_location=device, weights_only=False)["actor"],
            strict=True,
        )
        wtr.eval()
        robot, command_term = env.scene["robot"], env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, _ = robot.find_joints(".*")

        def contacts_and_safety(dones):
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(1) > 5
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1) > 0.8
            effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
            saturation = (effort >= 0.95).any(1)
            return contacts, support, dones.bool(), slip, saturation

        def prepare_walk(target_speed):
            wrapped.reset()
            phase = torch.zeros(n, dtype=torch.long, device=device)
            elapsed = torch.zeros(n, device=device)
            good = torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            heading = robot.data.heading_w.torch.clone()
            previous_action = torch.zeros(n, 37, device=device)
            targets = target_speed if torch.is_tensor(target_speed) else torch.full((n,), target_speed, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            for source_step in range(round(12.0 / dt)):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                command_speed = torch.where(phase < 2, torch.zeros_like(targets), torch.where(phase == 2, targets * mj(elapsed / 1.5), targets))
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    stand_action, stw_action, walk_action = stand(canonical, command), stw(canonical, command), walk(canonical, command)
                    action = torch.where((phase < 2)[:, None], stand_action, torch.where((phase == 2)[:, None], stw_action, walk_action))
                    _, _, dones, _ = wrapped.step(action)
                previous_action.copy_(action)
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                contacts, _, fall, slip, saturation = contacts_and_safety(dones)
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                reset = torch.nonzero(dones.bool()).flatten()
                if len(reset):
                    phase[reset], elapsed[reset], good[reset], switches[reset], previous_support[reset] = 0, 0, 0, 0, 0
                    heading[reset] = robot.data.heading_w.torch[reset]
                settled = (speed.abs() < 0.1) & contacts.all(1) & ~fall
                good = torch.where((phase == 0) & settled, good + dt, torch.where(phase == 0, 0, good))
                advance = (phase == 0) & (good >= 0.4)
                phase[advance], elapsed[advance], good[advance] = 1, 0, 0
                advance = (phase == 1) & (elapsed >= 0.8)
                phase[advance], elapsed[advance] = 2, 0
                changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquire = (phase == 2) & ((speed - targets).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (switches >= 2) & ~fall
                good = torch.where(acquire, good + dt, torch.where(phase == 2, 0, good))
                advance = (phase == 2) & (good >= 0.4)
                phase[advance], elapsed[advance], good[advance] = 3, 0, 0
                valid = (phase == 3) & ((speed - targets).abs() <= 0.2) & (heading_error.abs() <= 0.12) & ~fall & ~slip & ~saturation
                good = torch.where(valid, good + dt, torch.where(phase == 3, 0, good))
                ready_ever |= good >= 0.10
                previous_support = support
                if int(ready_ever.sum()) >= math.ceil(0.9 * n):
                    return heading, targets, previous_action, ready_ever
                elapsed += dt
                if source_step % 100 == 0:
                    print(
                        f"[walk-source] step={source_step} phases={torch.bincount(phase, minlength=4).tolist()} "
                        f"ready_ever={int(ready_ever.sum())} max_good={float(good.max()):.3f}",
                        flush=True,
                    )
            raise RuntimeError(
                f"WALK source preparation below 90%: phases={torch.bincount(phase, minlength=4).tolist()} "
                f"ready_ever={int(ready_ever.sum())}/{n}"
            )

        def prepare_run(targets):
            heading, _, previous_action, walk_ready = prepare_walk(torch.full((n,), 1.2, device=device))
            good = torch.zeros(n, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            air = torch.zeros(n, 2, device=device)
            for step in range(500):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                progress = mj(torch.full((n,), step * dt / 1.4, device=device))
                command_speed = 1.2 + (targets - 1.2) * progress
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (1.5 * heading_error).clamp(-1.5, 1.5)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    action = run(canonical, command)
                    _, _, dones, _ = wrapped.step(action)
                previous_action.copy_(action)
                contacts, _, fall, slip, saturation = contacts_and_safety(dones)
                air = torch.where(contacts, torch.zeros_like(air), air + dt)
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                valid = ((speed - targets).abs() <= 0.2) & (heading_error.abs() <= 0.12) & ~fall & (air.max(1).values <= 0.2)
                good = torch.where(valid, good + dt, torch.zeros_like(good))
                ready_ever |= good >= 0.30
            ready = ready_ever
            if int(ready.sum()) < int(0.3 * n):
                counts = [int(ready[targets == value].sum()) for value in (2.4, 2.6, 2.8)]
                raise RuntimeError(f"RUN source preparation below 30%: {counts}")
            return heading, targets, previous_action, ready

        def collect_steps(regime, teacher_name, heading, source_speed, target_speed, previous_action, ready, steps, episode_prefix, action_fn):
            nonlocal row_count, episode_count
            air_time = torch.zeros(n, 2, device=device)
            episode_ids = np.asarray([f"{episode_prefix}_env{index}" for index in range(n)], dtype=object)
            splits = np.asarray([grouped_split(cfg["experiment"]["training_seed"], value) for value in episode_ids], dtype=object)
            episode_count += n
            for step in range(steps):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                if torch.is_tensor(target_speed):
                    command_speed = target_speed
                else:
                    command_speed = torch.full((n,), float(target_speed), device=device)
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                obs = to_walk_observation(canonical, command)
                with torch.no_grad():
                    action = action_fn(canonical, command, step)
                    _, _, dones, _ = wrapped.step(action)
                contacts, support, fall, slip, saturation = contacts_and_safety(dones)
                air_time = torch.where(contacts, torch.zeros_like(air_time), air_time + dt)
                actual_speed = robot.data.root_lin_vel_b.torch[:, 0]
                mask = (
                    ready & torch.isfinite(obs).all(1) & torch.isfinite(action).all(1)
                    & ~fall & ((actual_speed - command_speed).abs() <= 0.4)
                )
                ids = torch.nonzero(mask).flatten()
                if len(ids):
                    count = len(ids)
                    values_source = source_speed[ids] if torch.is_tensor(source_speed) else torch.full((count,), float(source_speed), device=device)
                    values_target = command_speed[ids]
                    phase = support[ids]
                    teacher_array = np.full(count, teacher_name, dtype=object)
                    regime_array = np.full(count, regime, dtype=object)
                    episode_array = episode_ids[ids.cpu().numpy()]
                    split_array = splits[ids.cpu().numpy()]
                    buffer.append({
                        "obs": obs[ids].cpu().numpy().astype(np.float32),
                        "action": action[ids].cpu().numpy().astype(np.float32),
                        "previous_action": previous_action[ids].cpu().numpy().astype(np.float32),
                        "teacher": teacher_array,
                        "regime": regime_array,
                        "source_speed": values_source.cpu().numpy().astype(np.float32),
                        "target_speed": values_target.cpu().numpy().astype(np.float32),
                        "support_phase": phase.cpu().numpy().astype(np.int8),
                        "contact": contacts[ids].cpu().numpy().astype(np.int8),
                        "air_time": air_time[ids].cpu().numpy().astype(np.float32),
                        "sequence_id": episode_array,
                        "episode_id": episode_array,
                        "sequence_step": np.full(count, step, dtype=np.int16),
                        "finite": np.ones(count, dtype=np.int8),
                        "fall": fall[ids].cpu().numpy().astype(np.int8),
                        "slip": slip[ids].cpu().numpy().astype(np.int8),
                        "saturation": saturation[ids].cpu().numpy().astype(np.int8),
                        "split": split_array,
                    })
                    row_count += count
                    regime_counts[regime] += count
                    teacher_counts[teacher_name] += count
                    for value in values_target.cpu().numpy():
                        speed_counts[round(float(value), 1)] += 1
                    for value in phase.cpu().numpy():
                        phase_counts[int(value)] += 1
                    for value in split_array:
                        split_counts[str(value)] += 1
                    safety_counts["fall"] += int(fall[ids].sum())
                    safety_counts["slip"] += int(slip[ids].sum())
                    safety_counts["saturation"] += int(saturation[ids].sum())
                previous_action.copy_(action)
                if sum(len(item["obs"]) for item in buffer) >= 65536:
                    flush()

        if not args.skip_walk:
            walk_targets = torch.tensor([0.6, 0.8, 1.0, 1.2], device=device).repeat((n + 3) // 4)[:n]
            heading, _, previous, ready = prepare_walk(torch.full((n,), 1.2, device=device))
            # Settle the formal WALK teacher onto each requested steady command.
            good = torch.zeros(n, device=device)
            for step in range(150):
                canonical = canonical_state_from_legacy_observation(wrapped.get_observations()["policy"], heading_w_rad=robot.data.heading_w.torch)
                command_speed = 1.2 + (walk_targets - 1.2) * mj(torch.full((n,), step * dt / 1.0, device=device))
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                motion = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    action = walk(canonical, motion)
                    _, _, dones, _ = wrapped.step(action)
                contacts, _, fall, slip, saturation = contacts_and_safety(dones)
                valid = ((robot.data.root_lin_vel_b.torch[:, 0] - walk_targets).abs() <= 0.2) & (heading_error.abs() <= 0.12) & ~fall
                good = torch.where(valid, good + dt, torch.zeros_like(good))
                previous.copy_(action)
            ready = good >= 0.30
            if int(ready.sum()) < math.ceil(0.8 * n):
                raise RuntimeError(f"WALK command settling below 80%: {int(ready.sum())}/{n}")
            collect_steps("walk_steady", "walk", heading, walk_targets, walk_targets, previous, ready, cfg["dataset"]["walk_steps_per_env"], "walk0", lambda canonical, command, step: walk(canonical, command))

        if not args.skip_run:
            run_targets = torch.tensor([2.4, 2.6, 2.8], device=device).repeat((n + 2) // 3)[:n]
            heading, run_targets, previous, ready = prepare_run(run_targets)
            collect_steps("run_steady", "run", heading, run_targets, run_targets, previous, ready, cfg["dataset"]["run_steps_per_env"], "run0", lambda canonical, command, step: run(canonical, command))

        remaining = 0 if args.skip_transition else cfg["dataset"]["transition_steps_per_env"]
        transition_episode = 0
        while remaining > 0:
            transition_targets = torch.where(torch.arange(n, device=device) % 2 == 0, 2.6, 2.8)
            heading, _, previous, ready = prepare_walk(torch.full((n,), 1.2, device=device))
            steps = min(100, remaining)

            def transition_action(canonical, command, step):
                return wtr(to_run_observation(canonical, command, route="RUN"))

            # collect_steps receives a tensor command; build the ramp explicitly per step via closure wrapper.
            air_time = torch.zeros(n, 2, device=device)
            episode_ids = np.asarray([f"wtr{transition_episode}_env{index}" for index in range(n)], dtype=object)
            splits = np.asarray([grouped_split(cfg["experiment"]["training_seed"], value) for value in episode_ids], dtype=object)
            episode_count += n
            for step in range(steps):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                command_speed = 1.2 + (transition_targets - 1.2) * mj(torch.full((n,), step * dt / 1.4, device=device))
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = (1.5 * heading_error).clamp(-1.5, 1.5)
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                obs = to_walk_observation(canonical, command)
                with torch.no_grad():
                    action = transition_action(canonical, command, step)
                    _, _, dones, _ = wrapped.step(action)
                contacts, support, fall, slip, saturation = contacts_and_safety(dones)
                air_time = torch.where(contacts, torch.zeros_like(air_time), air_time + dt)
                mask = ready & torch.isfinite(obs).all(1) & torch.isfinite(action).all(1)
                ids = torch.nonzero(mask).flatten()
                count = len(ids)
                if count:
                    target_np = command_speed[ids].cpu().numpy().astype(np.float32)
                    episode_array = episode_ids[ids.cpu().numpy()]
                    split_array = splits[ids.cpu().numpy()]
                    buffer.append({
                        "obs": obs[ids].cpu().numpy().astype(np.float32),
                        "action": action[ids].cpu().numpy().astype(np.float32),
                        "previous_action": previous[ids].cpu().numpy().astype(np.float32),
                        "teacher": np.full(count, "walk_to_run", dtype=object),
                        "regime": np.full(count, "walk_to_run", dtype=object),
                        "source_speed": np.full(count, 1.2, dtype=np.float32),
                        "target_speed": target_np,
                        "support_phase": support[ids].cpu().numpy().astype(np.int8),
                        "contact": contacts[ids].cpu().numpy().astype(np.int8),
                        "air_time": air_time[ids].cpu().numpy().astype(np.float32),
                        "sequence_id": episode_array,
                        "episode_id": episode_array,
                        "sequence_step": np.full(count, step, dtype=np.int16),
                        "finite": np.ones(count, dtype=np.int8),
                        "fall": fall[ids].cpu().numpy().astype(np.int8),
                        "slip": slip[ids].cpu().numpy().astype(np.int8),
                        "saturation": saturation[ids].cpu().numpy().astype(np.int8),
                        "split": split_array,
                    })
                    row_count += count
                    regime_counts["walk_to_run"] += count
                    teacher_counts["walk_to_run"] += count
                    for value in target_np:
                        speed_counts[round(float(value), 1)] += 1
                    for value in support[ids].cpu().numpy():
                        phase_counts[int(value)] += 1
                    for value in split_array:
                        split_counts[str(value)] += 1
                    safety_counts["fall"] += int(fall[ids].sum())
                    safety_counts["slip"] += int(slip[ids].sum())
                    safety_counts["saturation"] += int(saturation[ids].sum())
                previous.copy_(action)
                if sum(len(item["obs"]) for item in buffer) >= 65536:
                    flush()
            remaining -= steps
            transition_episode += 1
        flush()
        all_parts = sorted(DATA.glob("part-*.parquet"))
        aggregate = {
            "regime": Counter(), "teacher": Counter(), "speed": Counter(), "phase": Counter(),
            "split": Counter(), "safety": Counter(), "episodes": set(), "total": 0,
        }
        for path in all_parts:
            frame = pd.read_parquet(
                path,
                columns=["regime", "teacher", "target_speed_mps", "support_phase", "split", "fall", "slip", "saturation", "episode_id"],
            )
            aggregate["total"] += len(frame)
            aggregate["regime"].update(frame["regime"].astype(str))
            aggregate["teacher"].update(frame["teacher"].astype(str))
            aggregate["speed"].update(round(float(value), 1) for value in frame["target_speed_mps"])
            aggregate["phase"].update(int(value) for value in frame["support_phase"])
            aggregate["split"].update(frame["split"].astype(str))
            aggregate["safety"]["fall"] += int(frame["fall"].sum())
            aggregate["safety"]["slip"] += int(frame["slip"].sum())
            aggregate["safety"]["saturation"] += int(frame["saturation"].sum())
            aggregate["episodes"].update(frame["episode_id"].astype(str).unique())
        if aggregate["total"] < cfg["dataset"]["minimum_steps"]:
            raise RuntimeError(f"dataset below minimum: {aggregate['total']}")
        manifest = {
            "total_steps": aggregate["total"],
            "minimum_steps": cfg["dataset"]["minimum_steps"],
            "regime_counts": dict(aggregate["regime"]),
            "teacher_counts": dict(aggregate["teacher"]),
            "speed_counts": {str(key): value for key, value in aggregate["speed"].items()},
            "support_phase_counts": {str(key): value for key, value in aggregate["phase"].items()},
            "split_counts": dict(aggregate["split"]),
            "safety_flag_counts": dict(aggregate["safety"]),
            "episodes": len(aggregate["episodes"]),
            "parts": len(all_parts),
            "run_to_walk_failure_labels_used": False,
            "ppo_training": 0,
            "reward_optimization": 0,
            "teacher_updates": 0,
        }
        (OUT / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
