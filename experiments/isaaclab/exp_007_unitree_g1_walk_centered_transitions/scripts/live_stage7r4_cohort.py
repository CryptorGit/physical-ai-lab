"""Live Isaac Sim R0 for in-place WALK_TO_RUN cohort activation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn
import warp as wp

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]

import g1_command_skills.tasks  # noqa: E402
import g1_flat_run.tasks  # noqa: E402
import g1_walk_centered.tasks  # noqa: E402
import isaaclab_tasks  # noqa: E402
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert, load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    canonical_state_from_legacy_observation,
    to_run_observation,
)
from g1_walk_centered.in_place_cohort import InPlaceEnvIdCohort  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import (  # noqa: E402
    WalkToRunTransitionAction,
    WalkToRunTransitionActor152,
)
from g1_walk_centered.transition_only_runner import (  # noqa: E402
    SegmentStep,
    TransitionOnlyOnPolicyRunner,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

EXPECTED = {
    "stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "stw": "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mj(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def checksum(value: torch.Tensor) -> float:
    weights = torch.arange(1, value.numel() + 1, device=value.device, dtype=torch.float64)
    return float((value.flatten().double() * weights).sum())


def tensor_json(value: torch.Tensor) -> str:
    return json.dumps(value.detach().cpu().tolist(), separators=(",", ":"))


parser = argparse.ArgumentParser()
parser.add_argument("--num-envs", type=int, required=True)
parser.add_argument("--cohort-size", type=int, required=True)
parser.add_argument("--cohorts", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--stand", required=True)
parser.add_argument("--stand-to-walk", required=True)
parser.add_argument("--walk", required=True)
parser.add_argument("--run", required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main() -> None:
    output = (REPO / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "stand": Path(args.stand).resolve(strict=True),
        "stw": Path(args.stand_to_walk).resolve(strict=True),
        "walk": Path(args.walk).resolve(strict=True),
        "run": Path(args.run).resolve(strict=True),
    }
    hashes = {key: sha(path) for key, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"protected checkpoint mismatch: {hashes}")

    cfg, agent_cfg = resolve_task_config(
        "Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = 20.0
    if args.device:
        cfg.sim.device = args.device

    timeline_rows: list[dict] = []
    map_rows: list[dict] = []
    handoff_rows: list[dict] = []
    route_rows: list[dict] = []
    segment_replay: dict | None = None
    cohort_results: list[dict] = []

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        device = env.device
        dt = float(env.step_dt)
        stand = load_walk_expert(paths["stand"], device=device)
        stw = load_walk_expert(paths["stw"], device=device)
        walk = load_walk_expert(paths["walk"], device=device)
        run = load_run_expert(paths["run"], device=device)
        transition_actor = WalkToRunTransitionActor152(run.actor).to(device)
        transition_term = WalkToRunTransitionAction(transition_actor)
        critic = nn.Sequential(
            nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
        ).to(device)
        optimizer = torch.optim.Adam(
            [p for p in transition_actor.parameters() if p.requires_grad]
            + list(critic.parameters()),
            lr=1e-4,
        )
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        joint_ids, _ = robot.find_joints(".*")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        source_trainable = list(stand.actor.parameters()) + list(stw.actor.parameters()) + list(walk.actor.parameters())
        source_trainable += list(run.actor.parameters())

        for generation in range(args.cohorts):
            wrapped.reset()
            n = args.num_envs
            phase = torch.zeros(n, dtype=torch.long, device=device)
            phase_time = torch.zeros(n, device=device)
            good_time = torch.zeros(n, device=device)
            walk_hold = torch.zeros(n, device=device)
            support_switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            previous_action = torch.zeros(n, 37, device=device)
            target_heading = robot.data.heading_w.torch.clone()
            reset_count = torch.zeros(n, dtype=torch.long, device=device)
            ready_time = torch.full((n,), math.nan, device=device)
            slip_dwell = torch.zeros(n, device=device)
            flight_dwell = torch.zeros(n, device=device)
            saturation_dwell = torch.zeros(n, device=device)
            manager = InPlaceEnvIdCohort(n, args.cohort_size, args.seed + generation, device=device)
            launch_step = -1
            selected_ids: torch.Tensor | None = None
            pre_snapshot: dict[str, torch.Tensor] = {}
            source_attempts = n
            max_source_steps = round(15.0 / dt)

            for step in range(max_source_steps):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(
                    legacy, heading_w_rad=robot.data.heading_w.torch
                )
                speed_command = torch.zeros(n, device=device)
                speed_command[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                speed_command[phase == 3] = 1.2
                heading_error = torch.atan2(
                    torch.sin(target_heading - robot.data.heading_w.torch),
                    torch.cos(target_heading - robot.data.heading_w.torch),
                )
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0] = speed_command
                command_term.vel_command_b[:, 2] = yaw
                command = MotionCommand(speed_command, target_heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    stand_action = stand(canonical, command)
                    stw_action = stw(canonical, command)
                    walk_action = walk(canonical, command)
                stand_mask = (phase == 0) | (phase == 1)
                stw_mask = phase == 2
                walk_mask = phase == 3
                full_action = torch.zeros(n, 37, device=device)
                full_action[stand_mask] = stand_action[stand_mask]
                full_action[stw_mask] = stw_action[stw_mask]
                full_action[walk_mask] = walk_action[walk_mask]
                assignment = stand_mask.long() + stw_mask.long() + walk_mask.long()
                if not bool((assignment == 1).all()):
                    raise RuntimeError("source controller assignment is not exactly one")
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                previous_action.copy_(full_action)
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                foot_speed = robot.data.body_lin_vel_w.torch[:, foot_ids, :2].norm(dim=-1)
                maximum_slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort_ratio = (
                    robot.data.applied_torque.torch[:, joint_ids].abs()
                    / robot.data.joint_effort_limits.torch[:, joint_ids].abs().clamp_min(1e-6)
                ).amax(dim=1)
                slip_dwell = torch.where(maximum_slip > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                flight_dwell = torch.where(~contacts.any(dim=1), flight_dwell + dt, torch.zeros_like(flight_dwell))
                saturation_dwell = torch.where(
                    effort_ratio >= 0.95, saturation_dwell + dt, torch.zeros_like(saturation_dwell)
                )
                root_speed = robot.data.root_lin_vel_b.torch[:, 0]
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
                timeout = info.get("time_outs", torch.zeros_like(dones)).bool()
                physical_done = dones.bool() & ~timeout

                reset_ids = torch.nonzero(dones.bool()).flatten()
                if len(reset_ids):
                    phase[reset_ids] = 0
                    phase_time[reset_ids] = 0
                    good_time[reset_ids] = 0
                    walk_hold[reset_ids] = 0
                    support_switches[reset_ids] = 0
                    previous_support[reset_ids] = 0
                    target_heading[reset_ids] = robot.data.heading_w.torch[reset_ids]
                    slip_dwell[reset_ids] = 0
                    flight_dwell[reset_ids] = 0
                    saturation_dwell[reset_ids] = 0
                    reset_count[reset_ids] += 1

                settle_good = (
                    (root_speed.abs() <= 0.08)
                    & (robot.data.root_lin_vel_b.torch[:, 2].abs() <= 0.05)
                    & (roll <= 0.10)
                    & (pitch <= 0.10)
                    & contacts.all(dim=1)
                    & (~physical_done)
                )
                mask = phase == 0
                good_time[mask] = torch.where(settle_good[mask], good_time[mask] + dt, 0)
                advance = mask & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance] = 1, 0, 0
                advance = (phase == 1) & (phase_time >= 0.8)
                phase[advance], phase_time[advance] = 2, 0
                changed = (support != previous_support) & ((support == 1) | (support == 2)) & (phase == 2)
                support_switches[changed] += 1
                acquire_good = (
                    ((root_speed - 1.2).abs() <= 0.20)
                    & (heading_error.abs() <= 0.12)
                    & (support_switches >= 2)
                    & (~physical_done)
                )
                mask = phase == 2
                good_time[mask] = torch.where(acquire_good[mask], good_time[mask] + dt, 0)
                advance = mask & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance], walk_hold[advance] = 3, 0, 0, 0
                source_good = (
                    (phase == 3)
                    & ((root_speed - 1.2).abs() <= 0.20)
                    & (heading_error.abs() <= 0.12)
                    & (~physical_done)
                    & (slip_dwell < 0.2)
                    & (flight_dwell <= 0.16)
                    & (saturation_dwell < 0.2)
                    & torch.isfinite(legacy).all(dim=1)
                    & torch.isfinite(full_action).all(dim=1)
                )
                walk_hold = torch.where(source_good, walk_hold + dt, torch.zeros_like(walk_hold))
                contract_valid = source_good & (walk_hold >= 1.0)
                manager.update_ready(contract_valid, step)
                newly_ready = contract_valid & torch.isnan(ready_time)
                ready_time[newly_ready] = step * dt
                if step % 10 == 0 or int(manager.source_ready.sum()) >= args.cohort_size:
                    timeline_rows.append({
                        "label": args.label, "generation": generation, "step": step,
                        "time_s": step * dt, "ready_count": int(manager.source_ready.sum()),
                    })
                previous_support.copy_(support)
                phase_time += dt

                cumulative_ready = int(torch.isfinite(ready_time).sum())
                required_cumulative = math.ceil(0.90 * n)
                if (
                    int(manager.source_ready.sum()) >= args.cohort_size
                    and cumulative_ready >= required_cumulative
                ):
                    launch = manager.activate(contract_valid, previous_action)
                    selected_ids = launch["physical_env_ids"]
                    if not bool(contract_valid[selected_ids].all()):
                        raise RuntimeError("source contract invalid at launch")
                    launch_step = step
                    sensor_timestamp = wp.to_torch(sensor._timestamp).clone()
                    pre_snapshot = {
                        "root_pos": robot.data.root_pos_w.torch[selected_ids].clone(),
                        "root_quat": robot.data.root_quat_w.torch[selected_ids].clone(),
                        "root_lin": robot.data.root_lin_vel_w.torch[selected_ids].clone(),
                        "root_ang": robot.data.root_ang_vel_w.torch[selected_ids].clone(),
                        "joint_pos": robot.data.joint_pos.torch[selected_ids].clone(),
                        "joint_vel": robot.data.joint_vel.torch[selected_ids].clone(),
                        "forces": forces[selected_ids].clone(),
                        "contacts": contacts[selected_ids].clone(),
                        "air": sensor.data.current_air_time.torch[selected_ids][:, sensor_feet].clone(),
                        "last_contact": sensor.data.last_contact_time.torch[selected_ids][:, sensor_feet].clone(),
                        "timestamp": sensor_timestamp[selected_ids].clone(),
                        "heading": target_heading[selected_ids].clone(),
                        "previous_action": previous_action[selected_ids].clone(),
                        "support": support[selected_ids].clone(),
                    }
                    for local, physical in enumerate(selected_ids.tolist()):
                        map_rows.append({
                            "label": args.label, "generation": generation,
                            "cohort_local_index": local, "physical_env_id": physical,
                        })
                    break

            if selected_ids is None:
                cohort_results.append({
                    "generation": generation, "formed": False,
                    "ready_count": int(manager.source_ready.sum()), "source_attempts": source_attempts,
                })
                continue

            runner = TransitionOnlyOnPolicyRunner(args.cohort_size)
            runner.start_transition(torch.ones(args.cohort_size, dtype=torch.bool, device=device))
            target = torch.empty(args.cohort_size, device=device)
            cut1, cut2 = int(0.5 * args.cohort_size), int(0.8 * args.cohort_size)
            target[:cut1], target[cut1:cut2], target[cut2:] = 2.4, 2.6, 2.8
            horizon = 16
            previous_match_count = 0
            routing_mismatch = 0
            nan_count = 0
            post_terminal_stored = 0
            actor_obs: list[torch.Tensor] = []
            actor_actions: list[torch.Tensor] = []
            actor_rewards: list[torch.Tensor] = []
            for transition_step in range(horizon):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(
                    legacy, heading_w_rad=robot.data.heading_w.torch
                )
                selected_canonical = canonical_state_from_legacy_observation(
                    legacy[selected_ids], heading_w_rad=robot.data.heading_w.torch[selected_ids]
                )
                heading_error = torch.atan2(
                    torch.sin(target_heading - robot.data.heading_w.torch),
                    torch.cos(target_heading - robot.data.heading_w.torch),
                )
                full_target = torch.zeros(n, device=device)
                full_target[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                full_target[phase == 3] = 1.2
                full_target[selected_ids] = 1.2 + (target - 1.2) * mj(
                    torch.full_like(target, transition_step * dt / 1.4)
                )
                yaw = (1.5 * heading_error).clamp(-1.5, 1.5)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0] = full_target
                command_term.vel_command_b[:, 2] = yaw
                full_command = MotionCommand(full_target, target_heading, target_yaw_rate_radps=yaw)
                selected_command = MotionCommand(
                    full_target[selected_ids], target_heading[selected_ids],
                    target_yaw_rate_radps=yaw[selected_ids],
                )
                with torch.no_grad():
                    source_stand_action = stand(canonical, full_command)
                    source_stw_action = stw(canonical, full_command)
                    source_walk_action = walk(canonical, full_command)
                    source_action = torch.zeros(n, 37, device=device)
                    source_action[(phase == 0) | (phase == 1)] = source_stand_action[(phase == 0) | (phase == 1)]
                    source_action[phase == 2] = source_stw_action[phase == 2]
                    source_action[phase == 3] = source_walk_action[phase == 3]
                observation_152 = to_run_observation(selected_canonical, selected_command, route="RUN")
                if transition_step == 0:
                    previous_match_count = int(
                        (observation_152[:, 86:123] == pre_snapshot["previous_action"]).all(dim=1).sum()
                    )
                transition_action = transition_term.apply(
                    observation_152, observation_152[:, 86:123]
                )
                source_mask = torch.ones(n, dtype=torch.bool, device=device)
                source_mask[selected_ids] = False
                transition_mask = ~source_mask
                post_mask = torch.zeros(n, dtype=torch.bool, device=device)
                assignment = source_mask.long() + transition_mask.long() + post_mask.long()
                if not bool((assignment == 1).all()):
                    raise RuntimeError("controller overlap or unassigned environment")
                full_action = torch.zeros(n, 37, device=device)
                full_action[source_mask] = source_action[source_mask]
                full_action[selected_ids] = transition_action.detach()
                applied = full_action[selected_ids].clone()
                for local in range(min(args.cohort_size, 32 if args.num_envs == 64 else 8)):
                    actor_sum = checksum(transition_action[local].detach())
                    applied_sum = checksum(applied[local])
                    mismatch = abs(actor_sum - applied_sum) > 1e-9
                    routing_mismatch += int(mismatch)
                    route_rows.append({
                        "label": args.label, "generation": generation,
                        "transition_step": transition_step, "cohort_local_index": local,
                        "physical_env_id": int(selected_ids[local]), "observation_checksum": checksum(observation_152[local]),
                        "actor_output_checksum": actor_sum, "applied_action_checksum": applied_sum,
                        "mismatch": mismatch,
                    })
                value = critic(observation_152).squeeze(-1)
                with torch.no_grad():
                    _, _, dones, _ = wrapped.step(full_action)
                speed_after = robot.data.root_lin_vel_b.torch[selected_ids, 0]
                reward = -(speed_after - target).abs()
                terminal = torch.zeros(args.cohort_size, dtype=torch.bool, device=device)
                truncated = torch.zeros_like(terminal)
                if transition_step == horizon - 1:
                    truncated[:] = True
                terminal |= dones.bool()[selected_ids]
                nan_count += int((~torch.isfinite(applied)).any(dim=1).sum())
                runner.transition_step(SegmentStep(
                    observation_152.detach(), applied.detach(), reward.detach(), value,
                    terminal, truncated, torch.zeros_like(reward),
                ))
                actor_obs.append(observation_152)
                actor_actions.append(transition_action)
                actor_rewards.append(reward)
                previous_action.copy_(full_action)

                if transition_step == 0:
                    post_timestamp = wp.to_torch(sensor._timestamp).clone()[selected_ids]
                    post_forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected_ids].clone()
                    post_contacts = post_forces.norm(dim=-1).amax(dim=1) > 5.0
                    post_air = sensor.data.current_air_time.torch[selected_ids][:, sensor_feet].clone()
                    post_last_contact = sensor.data.last_contact_time.torch[selected_ids][:, sensor_feet].clone()
                    for local, physical in enumerate(selected_ids.tolist()):
                        history_reset = bool(
                            (post_timestamp[local] <= pre_snapshot["timestamp"][local])
                            or (
                                (pre_snapshot["air"][local].abs().sum() + pre_snapshot["last_contact"][local].abs().sum() > 0)
                                and (post_air[local].abs().sum() + post_last_contact[local].abs().sum() == 0)
                            )
                        )
                        handoff_rows.append({
                            "label": args.label, "generation": generation,
                            "physical_env_id": physical, "pre_step": launch_step,
                            "post_step": launch_step + 1, "same_env_id": True,
                            "pre_root_position": tensor_json(pre_snapshot["root_pos"][local]),
                            "post_root_position": tensor_json(robot.data.root_pos_w.torch[physical]),
                            "pre_root_orientation": tensor_json(pre_snapshot["root_quat"][local]),
                            "post_root_orientation": tensor_json(robot.data.root_quat_w.torch[physical]),
                            "pre_root_linear_velocity": tensor_json(pre_snapshot["root_lin"][local]),
                            "post_root_linear_velocity": tensor_json(robot.data.root_lin_vel_w.torch[physical]),
                            "pre_root_angular_velocity": tensor_json(pre_snapshot["root_ang"][local]),
                            "post_root_angular_velocity": tensor_json(robot.data.root_ang_vel_w.torch[physical]),
                            "pre_joint_position": tensor_json(pre_snapshot["joint_pos"][local]),
                            "post_joint_position": tensor_json(robot.data.joint_pos.torch[physical]),
                            "pre_joint_velocity": tensor_json(pre_snapshot["joint_vel"][local]),
                            "post_joint_velocity": tensor_json(robot.data.joint_vel.torch[physical]),
                            "pre_contact_state": tensor_json(pre_snapshot["contacts"][local]),
                            "post_contact_state": tensor_json(post_contacts[local]),
                            "pre_contact_force": tensor_json(pre_snapshot["forces"][local]),
                            "post_contact_force": tensor_json(post_forces[local]),
                            "pre_foot_air_time": tensor_json(pre_snapshot["air"][local]),
                            "post_foot_air_time": tensor_json(post_air[local]),
                            "pre_support_phase": int(pre_snapshot["support"][local]),
                            "post_support_phase": int(post_contacts[local, 0]) + 2 * int(post_contacts[local, 1]),
                            "pre_sensor_timestamp": float(pre_snapshot["timestamp"][local]),
                            "post_sensor_timestamp": float(post_timestamp[local]),
                            "target_heading_match": bool(target_heading[physical] == pre_snapshot["heading"][local]),
                            "previous_action_match": bool(
                                torch.equal(observation_152[local, 86:123], pre_snapshot["previous_action"][local])
                            ),
                            "contact_history_reset": history_reset,
                            "state_setter_calls": 0, "teleport_calls": 0,
                        })

            returns, advantages = runner.storage.finish(torch.zeros(args.cohort_size, device=device))
            normalized = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
            optimizer.zero_grad()
            replay_action = transition_actor(torch.cat([s.observation for s in runner.storage.steps]))
            replay_value = critic(torch.cat([s.observation for s in runner.storage.steps])).squeeze(-1)
            loss = replay_action.square().mean() + replay_value.square().mean() + 0.0 * normalized.mean()
            loss.backward()
            actor_grad = all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in transition_actor.parameters() if p.requires_grad
            )
            critic_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
            frozen_grad = all(p.grad is None for p in source_trainable)
            with tempfile.TemporaryDirectory() as tmp:
                checkpoint = Path(tmp) / "r0.pt"
                torch.save({
                    "actor": transition_actor.state_dict(), "critic": critic.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }, checkpoint)
                payload = torch.load(checkpoint, map_location=device, weights_only=False)
                transition_actor.load_state_dict(payload["actor"], strict=True)
                critic.load_state_dict(payload["critic"], strict=True)
                optimizer.load_state_dict(payload["optimizer"])
                reload_ok = True
            if segment_replay is None:
                segment_replay = {
                    "observation": runner.storage.steps[0].observation[0].detach().cpu().tolist(),
                    "reward": [float(step.reward[0]) for step in runner.storage.steps],
                    "value": [float(step.value[0].detach()) for step in runner.storage.steps],
                    "terminated": [bool(step.terminated[0]) for step in runner.storage.steps],
                    "truncated": [bool(step.truncated[0]) for step in runner.storage.steps],
                    "return": returns[:, 0].detach().cpu().tolist(),
                    "advantage": advantages[:, 0].detach().cpu().tolist(),
                }
            selected_ready = ready_time[selected_ids]
            cohort_results.append({
                "generation": generation, "formed": True,
                "ready_count_at_launch": int(manager.source_ready.sum()),
                "source_contract_at_launch": int(manager.source_ready[selected_ids].sum()),
                "source_success_rate": float(torch.isfinite(ready_time).sum() / n),
                "first_ready_time_s": float(ready_time[torch.isfinite(ready_time)].min()),
                "cohort_completion_time_s": launch_step * dt,
                "ready_wait_mean_s": float((launch_step * dt - selected_ready).mean()),
                "previous_action_match": previous_match_count,
                "routing_mismatch": routing_mismatch, "nan_count": nan_count,
                "stored_steps": len(runner.storage.steps) * args.cohort_size,
                "source_prefix_stored_steps": 0, "non_selected_stored_steps": 0,
                "invalid_stored_steps": 0, "post_terminal_stored_steps": post_terminal_stored,
                "actor_gradient": actor_grad, "critic_gradient": critic_grad,
                "frozen_gradient_zero": frozen_grad, "save_reload": reload_ok,
            })

        def write_csv(name: str, rows: list[dict]) -> None:
            if not rows:
                return
            with (output / name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        write_csv(f"{args.label}_ready_timeline.csv", timeline_rows)
        write_csv(f"{args.label}_cohort_map.csv", map_rows)
        write_csv(f"{args.label}_handoff.csv", handoff_rows)
        write_csv(f"{args.label}_action_routing.csv", route_rows)
        summary = {
            "label": args.label, "actual_isaac_sim": True, "num_envs": args.num_envs,
            "cohort_size": args.cohort_size, "requested_cohorts": args.cohorts,
            "completed_cohorts": sum(result["formed"] for result in cohort_results),
            "cohorts": cohort_results, "protected_hashes": hashes,
            "state_copy": False, "state_setter_calls": 0, "teleport_calls": 0,
        }
        (output / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        if segment_replay is not None:
            (output / f"{args.label}_segment_replay.json").write_text(
                json.dumps(segment_replay, indent=2) + "\n"
            )
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
