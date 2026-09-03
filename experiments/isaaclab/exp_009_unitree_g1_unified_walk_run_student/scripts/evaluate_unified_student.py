"""Deterministic closed-loop retention and diagnostic evaluation in Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
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
from unified_walk_run.command_profile import minimum_jerk  # noqa: E402
from unified_walk_run.dataset import action_columns, observation_columns  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", action="append", default=[])
parser.add_argument("--collect-dagger", action="store_true")
parser.add_argument("--append", action="store_true")
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = int(cfg["evaluation"]["physical_envs"])
    task_cfg.seed = int(cfg["evaluation"]["evaluation_seed"])
    task_cfg.episode_length_s = 40.0
    task_cfg.sim.device = cfg["experiment"]["device"]
    args.device = cfg["experiment"]["device"]
    checkpoints = [Path(value) for value in args.checkpoint]
    if not checkpoints:
        manifest = json.loads((OUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        checkpoints = [REPO / item["path"] for item in manifest]
    OUT.mkdir(parents=True, exist_ok=True)
    dagger_dir = OUT / "dagger_dataset.parquet"
    dagger_buffer, dagger_part, dagger_rows = [], 0, 0

    def flush_dagger():
        nonlocal dagger_buffer, dagger_part
        if not dagger_buffer:
            return
        obs = np.concatenate([item[0] for item in dagger_buffer])
        action = np.concatenate([item[1] for item in dagger_buffer])
        episode = np.concatenate([item[2] for item in dagger_buffer])
        step = np.concatenate([item[3] for item in dagger_buffer])
        frame = pd.DataFrame({
            **{name: obs[:, index] for index, name in enumerate(observation_columns())},
            **{name: action[:, index] for index, name in enumerate(action_columns())},
            "episode_id": episode, "sequence_step": step, "split": np.full(len(obs), "train", dtype=object),
        })
        dagger_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dagger_dir / f"part-{dagger_part:04d}.parquet", compression="zstd", index=False)
        dagger_part += 1
        dagger_buffer = []

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg),
            clip_actions=agent_cfg.clip_actions,
        )
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
        wtr = WalkToRunTransitionActor152(run.actor).to(device)
        wtr.load_state_dict(
            torch.load(REPO / cfg["teachers"]["walk_to_run"]["path"], map_location=device, weights_only=False)["actor"],
            strict=True,
        )
        for module in (walk.actor, run.actor, stand.actor, stw.actor, wtr):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        robot, command_term = env.scene["robot"], env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")

        def state():
            legacy = wrapped.get_observations()["policy"]
            return canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)

        def command(speed, heading, gain=0.8):
            error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
            yaw = (gain * error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-1.5, 1.5)
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = speed, yaw
            return MotionCommand(speed, heading, target_yaw_rate_radps=yaw), error

        def safety(dones, contact_history):
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(1) > 5
            foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1) > 0.8
            effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
            saturation = (effort >= 0.95).any(1)
            landing = contacts & ~contact_history
            return contacts, dones.bool(), slip, saturation, landing

        def prepare_walk(targets):
            wrapped.reset()
            heading = robot.data.heading_w.torch.clone()
            phase = torch.zeros(n, dtype=torch.long, device=device)
            elapsed, good = torch.zeros(n, device=device), torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            previous = torch.zeros(n, 37, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            for _ in range(round(12 / dt)):
                canonical = state()
                speed = torch.where(phase < 2, torch.zeros_like(targets), torch.where(phase == 2, targets * minimum_jerk(elapsed / 1.5), targets))
                motion, error = command(speed, heading)
                with torch.no_grad():
                    a0, a1, a2 = stand(canonical, motion), stw(canonical, motion), walk(canonical, motion)
                    action = torch.where((phase < 2)[:, None], a0, torch.where((phase == 2)[:, None], a1, a2))
                    _, _, dones, _ = wrapped.step(action)
                contacts, fall, slip, sat, _ = safety(dones, torch.zeros(n, 2, dtype=torch.bool, device=device))
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                reset = torch.nonzero(dones.bool()).flatten()
                if len(reset):
                    phase[reset], elapsed[reset], good[reset], switches[reset], previous_support[reset] = 0, 0, 0, 0, 0
                    heading[reset] = robot.data.heading_w.torch[reset]
                settled = (robot.data.root_lin_vel_b.torch[:, 0].abs() < 0.1) & contacts.all(1) & ~fall
                good = torch.where((phase == 0) & settled, good + dt, torch.where(phase == 0, 0, good))
                advance = (phase == 0) & (good >= 0.4)
                phase[advance], elapsed[advance], good[advance] = 1, 0, 0
                advance = (phase == 1) & (elapsed >= 0.8)
                phase[advance], elapsed[advance] = 2, 0
                changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquire = (phase == 2) & ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.2) & (error.abs() <= 0.12) & (switches >= 2) & ~fall
                good = torch.where(acquire, good + dt, torch.where(phase == 2, 0, good))
                advance = (phase == 2) & (good >= 0.4)
                phase[advance], elapsed[advance], good[advance] = 3, 0, 0
                valid = (phase == 3) & ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.2) & (error.abs() <= 0.12) & ~fall & ~slip & ~sat
                good = torch.where(valid, good + dt, torch.where(phase == 3, 0, good))
                ready_ever |= good >= 0.10
                previous_support = support
                previous.copy_(action)
                if int(ready_ever.sum()) >= math.ceil(0.9 * n):
                    launch_ready = (
                        (phase == 3)
                        & ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.40)
                        & (error.abs() <= 0.12) & ~fall
                    )
                    if int(launch_ready.sum()) >= len(checkpoints) * 50:
                        return heading, previous, launch_ready
                elapsed += dt
            raise RuntimeError("WALK preparation below 90%")

        def prepare_run(targets):
            heading, previous, ready = prepare_walk(torch.full((n,), 1.2, device=device))
            good = torch.zeros(n, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            contacts_last = torch.zeros(n, 2, dtype=torch.bool, device=device)
            landing_count = torch.zeros(n, device=device)
            for step in range(500):
                canonical = state()
                speed = 1.2 + (targets - 1.2) * minimum_jerk(torch.full((n,), step * dt / 1.4, device=device))
                motion, error = command(speed, heading, 1.5)
                with torch.no_grad():
                    action = run(canonical, motion)
                    _, _, dones, _ = wrapped.step(action)
                contacts, fall, slip, sat, landing = safety(dones, contacts_last)
                landing_count += landing.any(1)
                contacts_last = contacts
                valid = (
                    ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.2)
                    & (error.abs() <= 0.12) & (landing_count >= 4) & ~fall
                )
                good = torch.where(valid, good + dt, torch.zeros_like(good))
                ready_ever |= good >= 0.10
                previous.copy_(action)
            ready = ready_ever
            launch_ready = (
                ((robot.data.root_lin_vel_b.torch[:, 0] - targets).abs() <= 0.30)
                & (error.abs() <= 0.12) & (landing_count >= 4) & ~fall
            )
            ready = launch_ready
            if int(ready.sum()) < len(checkpoints) * 50:
                raise RuntimeError("RUN launch-ready population below checkpoint allocation")
            return heading, previous, ready

        def assigned_targets(values):
            tensor = torch.tensor(values, device=device)
            return tensor.repeat((n + len(values) - 1) // len(values))[:n]

        def rollout(students, assignment, heading, source_ready, target, start, kind, duration=5.0):
            nonlocal dagger_rows
            steps = round(duration / dt)
            previous_contact = torch.zeros(n, 2, dtype=torch.bool, device=device)
            alive = source_ready.clone()
            ever_speed, ever_walk, ever_periodic = torch.zeros_like(alive), torch.zeros_like(alive), torch.zeros_like(alive)
            fall_any, slip_any, sat_any = torch.zeros_like(alive), torch.zeros_like(alive), torch.zeros_like(alive)
            streak = torch.zeros(n, dtype=torch.long, device=device)
            max_streak = torch.zeros_like(streak)
            landings = torch.zeros(n, device=device)
            action_jumps, headings, speed_errors = [], [], []
            first_action = True
            for step in range(steps):
                alpha = minimum_jerk(torch.full((n,), step * dt / 1.4, device=device)) if kind in {"wtr", "reverse", "intermediate"} else torch.ones(n, device=device)
                speed_command = start + (target - start) * alpha
                canonical = state()
                motion, error = command(speed_command, heading, 1.0)
                obs = to_walk_observation(canonical, motion)
                with torch.no_grad():
                    if kind == "walk":
                        teacher_action = walk(canonical, motion)
                    elif kind == "run":
                        teacher_action = run(canonical, motion)
                    elif kind == "wtr":
                        teacher_action = wtr(to_run_observation(canonical, motion, route="RUN"))
                    else:
                        teacher_action = None
                    action = students[0](obs)
                    for student_index, student in enumerate(students):
                        ids = assignment == student_index
                        action[ids] = student(obs[ids])
                    _, _, dones, _ = wrapped.step(action)
                if args.collect_dagger and teacher_action is not None:
                    ids = torch.nonzero(source_ready).flatten()
                    dagger_buffer.append((
                        obs[ids].detach().cpu().numpy().astype(np.float32),
                        teacher_action[ids].detach().cpu().numpy().astype(np.float32),
                        np.asarray([f"{kind}_{float(target[index]):.1f}_env{int(index)}" for index in ids.cpu()], dtype=object),
                        np.full(len(ids), step, dtype=np.int16),
                    ))
                    dagger_rows += len(ids)
                    if dagger_rows % 65536 < len(ids):
                        flush_dagger()
                contacts, fall, slip, sat, landing = safety(dones, previous_contact)
                previous_contact = contacts
                landings += landing.any(1)
                actual = robot.data.root_lin_vel_b.torch[:, 0]
                speed_ok = (actual - target).abs() <= 0.2
                walk_valid = speed_ok & (error.abs() <= 0.12) & contacts.any(1) & ~fall & ~slip & ~sat
                streak = torch.where(walk_valid, streak + 1, torch.zeros_like(streak))
                max_streak = torch.maximum(max_streak, streak)
                ever_speed |= speed_ok
                ever_walk |= streak >= 20
                ever_periodic |= landings >= 4
                fall_any |= fall & alive
                slip_any |= slip & alive
                sat_any |= sat & alive
                alive &= ~fall
                headings.append(error.abs())
                speed_errors.append((actual - target).abs())
                if first_action:
                    previous_action = obs[:, -37:]
                    action_jumps.append((action - previous_action).norm(dim=1))
                    first_action = False
            heading_values = torch.stack(headings)
            speed_values = torch.stack(speed_errors)
            if kind == "walk":
                success = ever_walk & alive
            elif kind == "run":
                success = ever_periodic & ever_speed & alive
            else:
                success = ever_walk & alive
            return {
                "source_ready": source_ready,
                "success": success, "speed_acquisition": ever_speed, "walk_contract": ever_walk,
                "periodic": ever_periodic, "fall": fall_any, "slip": slip_any, "saturation": sat_any,
                "timeout": ~success, "max_walk_valid_streak": max_streak,
                "heading_abs": heading_values, "speed_error": speed_values,
                "entry_action_jump": action_jumps[0],
            }

        def summarize(result, targets, values, episodes_per):
            rows = {}
            for value in values:
                ids = torch.nonzero(torch.isclose(targets, torch.tensor(value, device=device)) & result["source_ready"]).flatten()[:episodes_per]
                if len(ids) < episodes_per:
                    raise RuntimeError(f"valid source allocation shortfall for {value}: {len(ids)}/{episodes_per}")
                heading = result["heading_abs"][:, ids].flatten()
                rows[str(value)] = {
                    "episodes": len(ids),
                    "success_rate": float(result["success"][ids].float().mean()),
                    "speed_acquisition_rate": float(result["speed_acquisition"][ids].float().mean()),
                    "walk_contract_rate": float(result["walk_contract"][ids].float().mean()),
                    "periodic_rate": float(result["periodic"][ids].float().mean()),
                    "fall_rate": float(result["fall"][ids].float().mean()),
                    "slip_rate": float(result["slip"][ids].float().mean()),
                    "saturation_rate": float(result["saturation"][ids].float().mean()),
                    "timeout_rate": float(result["timeout"][ids].float().mean()),
                    "heading_p95_rad": float(torch.quantile(heading, 0.95)),
                    "walk_valid_streak_mean": float(result["max_walk_valid_streak"][ids].float().mean()),
                    "walk_valid_streak_max": int(result["max_walk_valid_streak"][ids].max()),
                    "entry_action_jump_mean": float(result["entry_action_jump"][ids].mean()),
                }
            return rows

        students = []
        all_results = {}
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            student = UnifiedWalkRunStudent123().to(device)
            student.load_state_dict(payload["student"], strict=True)
            student.eval()
            label = checkpoint.stem
            students.append(student)
            all_results[label] = {"checkpoint": str(checkpoint), "sha256": sha(checkpoint), "iteration": payload.get("epoch", 0),
                                  "walk": {}, "run": {}, "walk_to_run": {}, "run_to_walk": {}, "intermediate": {}}

        def assign_ready(ready, count):
            ids = torch.nonzero(ready).flatten()
            needed = len(students) * count
            if len(ids) < needed:
                raise RuntimeError(f"valid source allocation shortfall: {len(ids)}/{needed}")
            assignment = torch.full((n,), -1, dtype=torch.long, device=device)
            for index in range(len(students)):
                assignment[ids[index * count : (index + 1) * count]] = index
            return assignment, assignment >= 0

        def run_condition(section, source_kind, source_speed, target_speed, rollout_kind, episodes):
            source_tensor = torch.full((n,), source_speed, device=device)
            heading, _, ready = prepare_walk(source_tensor) if source_kind == "walk" else prepare_run(source_tensor)
            assignment, selected_ready = assign_ready(ready, episodes)
            target = torch.full((n,), target_speed, device=device)
            result = rollout(students, assignment, heading, selected_ready, target, source_tensor, rollout_kind)
            for index, checkpoint in enumerate(checkpoints):
                ids = torch.nonzero(assignment == index).flatten()
                headings = result["heading_abs"][:, ids].flatten()
                row = {
                    "episodes": len(ids), "success_rate": float(result["success"][ids].float().mean()),
                    "speed_acquisition_rate": float(result["speed_acquisition"][ids].float().mean()),
                    "walk_contract_rate": float(result["walk_contract"][ids].float().mean()),
                    "periodic_rate": float(result["periodic"][ids].float().mean()),
                    "fall_rate": float(result["fall"][ids].float().mean()),
                    "slip_rate": float(result["slip"][ids].float().mean()),
                    "saturation_rate": float(result["saturation"][ids].float().mean()),
                    "timeout_rate": float(result["timeout"][ids].float().mean()),
                    "heading_p95_rad": float(torch.quantile(headings, 0.95)),
                    "walk_valid_streak_mean": float(result["max_walk_valid_streak"][ids].float().mean()),
                    "walk_valid_streak_max": int(result["max_walk_valid_streak"][ids].max()),
                    "entry_action_jump_mean": float(result["entry_action_jump"][ids].mean()),
                }
                all_results[checkpoint.stem][section][str(target_speed if section != "run_to_walk" else source_speed)] = row

        for speed in [0.6, 0.8, 1.0, 1.2]:
            run_condition("walk", "walk", speed, speed, "walk", 50)
        for speed in [2.4, 2.6, 2.8]:
            run_condition("run", "run", speed, speed, "run", 50)
        for target in [2.6, 2.8]:
            run_condition("walk_to_run", "walk", 1.2, target, "wtr", 50)
        if not args.collect_dagger:
            for source in [2.6, 2.8]:
                run_condition("run_to_walk", "run", source, 1.2, "reverse", 50)
            for target in [1.4, 1.6, 1.8, 2.0, 2.2]:
                run_condition("intermediate", "walk", 1.2, target, "intermediate", 20)

        canonical = state()
        manifold_student = students[-2] if len(students) >= 2 else students[0]
        command_values = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
        outputs = []
        heading = robot.data.heading_w.torch.clone()
        for value in command_values:
            speeds = torch.full((n,), value, device=device)
            motion, _ = command(speeds, heading)
            with torch.no_grad():
                outputs.append(manifold_student(to_walk_observation(canonical, motion))[:64])
        actions = torch.stack(outputs)
        adjacent = (actions[1:] - actions[:-1]).norm(dim=2)
        derivative = (actions[1:] - actions[:-1]) / 0.2
        manifold_saved = {
            "checkpoint": str(checkpoints[-2] if len(checkpoints) >= 2 else checkpoints[0]),
            "commands_mps": command_values,
            "adjacent_action_l2_mean": adjacent.mean(1).cpu().tolist(),
            "adjacent_action_l2_p95": torch.quantile(adjacent, 0.95, dim=1).cpu().tolist(),
            "max_adjacent_action_l2": float(adjacent.max()),
            "action_derivative_abs_max": float(derivative.abs().max()),
            "ankle_knee_diagnostic": {joint_names[index]: float(derivative[:, :, index].abs().max()) for index, name in enumerate(joint_names) if "ankle" in name or "knee" in name},
            "non_finite": int((~torch.isfinite(actions)).sum()), "discrete_skill_switch": False,
        }

        if not args.collect_dagger:
            if args.append and (OUT / "checkpoint_sweep.json").exists():
                previous = json.loads((OUT / "checkpoint_sweep.json").read_text(encoding="utf-8"))
                previous.update(all_results)
                all_results = previous
            (OUT / "checkpoint_sweep.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
            (OUT / "action_manifold_audit.json").write_text(json.dumps(manifold_saved, indent=2) + "\n", encoding="utf-8")
        if args.collect_dagger:
            flush_dagger()
            (OUT / "dagger_collection.json").write_text(json.dumps({
                "round": 1, "rows": dagger_rows, "parts": dagger_part,
                "parent_checkpoints": [str(path) for path in checkpoints],
                "teacher_routing": {"walk": "WALK", "run": "RUN_LOW", "wtr": "WALK_TO_RUN"},
                "run_to_walk_excluded": True, "ppo": 0, "reward": False,
            }, indent=2) + "\n", encoding="utf-8")
        wrapped.close()


if __name__ == "__main__":
    main()
