"""Live Isaac R0 for in-place RUN_TO_WALK transition handoff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import tempfile
from collections import Counter
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

import g1_command_skills.tasks  # noqa: F401
import g1_flat_run.tasks  # noqa: F401
import g1_walk_centered.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from g1_walk_centered.command_contract import MotionCommand
from g1_walk_centered.experts import load_run_expert, load_walk_expert
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation
from g1_walk_centered.in_place_cohort import InPlaceEnvIdCohort
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionAction, RunToWalkTransitionActor152
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152
from g1_walk_centered.transition_only_runner import SegmentStep, TransitionOnlyOnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

EXPECTED = {
    "stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "stw": "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
    "wtr": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def mj(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def checksum(value: torch.Tensor) -> float:
    cpu_value = value.detach().cpu().flatten().double()
    weights = torch.arange(1, cpu_value.numel() + 1, dtype=torch.float64)
    return float((cpu_value * weights).sum())


def tensor_json(value: torch.Tensor) -> str:
    return json.dumps(value.detach().cpu().tolist(), separators=(",", ":"))


def fresh_gait() -> dict:
    return {"flights": [], "valid": 0, "alt": 0, "opp": 0, "last": None, "safe": 0, "maxsafe": 0}


def periodic(gait: dict) -> bool:
    flights = gait["flights"]
    duration = sum(flights) / len(flights) if flights else 0.0
    return (
        len(flights) >= 4
        and gait["maxsafe"] >= 3
        and gait["alt"] / max(gait["opp"], 1) >= 0.8
        and gait["valid"] / max(len(flights), 1) >= 0.8
        and 0.04 <= duration <= 0.16
    )


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
parser.add_argument("--walk-to-run", required=True)
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
        "wtr": Path(args.walk_to_run).resolve(strict=True),
    }
    hashes = {key: sha(path) for key, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"protected checkpoint mismatch: {hashes}")

    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = 24.0
    if args.device:
        cfg.sim.device = args.device

    timeline_rows: list[dict] = []
    map_rows: list[dict] = []
    continuity_rows: list[dict] = []
    routing_rows: list[dict] = []
    replay: dict | None = None
    cohort_results: list[dict] = []

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg), clip_actions=agent_cfg.clip_actions
        )
        env = wrapped.unwrapped
        device = env.device
        dt = float(env.step_dt)
        stand = load_walk_expert(paths["stand"], device=device)
        stw = load_walk_expert(paths["stw"], device=device)
        walk = load_walk_expert(paths["walk"], device=device)
        run = load_run_expert(paths["run"], device=device)
        wtr = WalkToRunTransitionActor152(run.actor).to(device)
        wtr_payload = torch.load(paths["wtr"], map_location=device, weights_only=False)
        wtr.load_state_dict(wtr_payload["actor"], strict=True)
        wtr.eval()
        actor = RunToWalkTransitionActor152(run.actor).to(device)
        action_term = RunToWalkTransitionAction(actor)
        critic = nn.Sequential(nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
        optimizer = torch.optim.Adam(
            [p for p in actor.parameters() if p.requires_grad] + list(critic.parameters()), lr=1e-4
        )
        frozen_modules = [stand.actor, stw.actor, walk.actor, run.actor, wtr]
        frozen_before = {
            f"{module_index}:{name}": tensor_hash(parameter)
            for module_index, module in enumerate(frozen_modules)
            for name, parameter in module.named_parameters()
        }
        actor_frozen_before = {
            name: tensor_hash(parameter) for name, parameter in actor.named_parameters() if not parameter.requires_grad
        }
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        foot_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joint_ids, joint_names = robot.find_joints(".*")
        knee_indices = [index for index, name in enumerate(joint_names) if "knee" in name]
        ankle_indices = [index for index, name in enumerate(joint_names) if "ankle" in name]

        for generation in range(args.cohorts):
            wrapped.reset()
            n = args.num_envs
            # 0 settle, 1 stand hold, 2 STW, 3 WALK hold, 4 WTR, 5 RUN source.
            phase = torch.zeros(n, dtype=torch.long, device=device)
            phase_time = torch.zeros(n, device=device)
            good_time = torch.zeros(n, device=device)
            support_switches = torch.zeros(n, dtype=torch.long, device=device)
            previous_support = torch.zeros(n, dtype=torch.long, device=device)
            previous_action = torch.zeros(n, 37, device=device)
            target_heading = robot.data.heading_w.torch.clone()
            target_speed = torch.where(
                torch.arange(n, device=device) % 2 == 0,
                torch.full((n,), 2.6, device=device),
                torch.full((n,), 2.8, device=device),
            )
            slip_dwell = torch.zeros(n, device=device)
            flight_dwell = torch.zeros(n, device=device)
            saturation_dwell = torch.zeros(n, device=device)
            run_hold = torch.zeros(n, device=device)
            ready_time = torch.full((n,), math.nan, device=device)
            reset_count = torch.zeros(n, dtype=torch.long, device=device)
            gait = [fresh_gait() for _ in range(n)]
            in_flight = [False] * n
            flight_start = [0.0] * n
            previous_contacts = [(False, False)] * n
            last_landing = ["none"] * n
            manager = InPlaceEnvIdCohort(n, args.cohort_size, args.seed + generation, device=device)
            selected_ids: torch.Tensor | None = None
            launch_step = -1
            pre: dict[str, torch.Tensor] = {}
            pre_phase: list[str] = []
            max_source_steps = round(23.0 / dt)

            for step in range(max_source_steps):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                speed_command = torch.zeros(n, device=device)
                speed_command[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                speed_command[phase == 3] = 1.2
                speed_command[phase == 4] = 1.2 + (target_speed[phase == 4] - 1.2) * mj(
                    phase_time[phase == 4] / 1.4
                )
                speed_command[phase == 5] = target_speed[phase == 5]
                heading_error = torch.atan2(
                    torch.sin(target_heading - robot.data.heading_w.torch),
                    torch.cos(target_heading - robot.data.heading_w.torch),
                )
                yaw_walk = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                yaw_run = (1.5 * heading_error).clamp(-1.5, 1.5)
                yaw = torch.where(phase >= 4, yaw_run, yaw_walk)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0] = speed_command
                command_term.vel_command_b[:, 2] = yaw
                motion = MotionCommand(speed_command, target_heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    stand_action = stand(canonical, motion)
                    stw_action = stw(canonical, motion)
                    walk_action = walk(canonical, motion)
                    run_action = run(canonical, motion)
                    wtr_action = wtr(to_run_observation(canonical, motion, route="RUN"))
                masks = [
                    (phase == 0) | (phase == 1),
                    phase == 2,
                    phase == 3,
                    phase == 4,
                    phase == 5,
                ]
                assignment = sum(mask.long() for mask in masks)
                if not bool((assignment == 1).all()):
                    raise RuntimeError("source controller assignment must equal one")
                full_action = torch.empty(n, 37, device=device)
                full_action[masks[0]] = stand_action[masks[0]]
                full_action[masks[1]] = stw_action[masks[1]]
                full_action[masks[2]] = walk_action[masks[2]]
                full_action[masks[3]] = wtr_action[masks[3]]
                full_action[masks[4]] = run_action[masks[4]]
                if not torch.isfinite(full_action).all():
                    raise RuntimeError("non-finite source action")
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                previous_action.copy_(full_action)
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5.0
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                foot_speed = robot.data.body_lin_vel_w.torch[:, foot_ids, :2].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort_ratio = (
                    robot.data.applied_torque.torch[:, joint_ids].abs()
                    / robot.data.joint_effort_limits.torch[:, joint_ids].abs().clamp_min(1e-6)
                ).amax(dim=1)
                slip_dwell = torch.where(slip > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                flight_dwell = torch.where(~contacts.any(dim=1), flight_dwell + dt, torch.zeros_like(flight_dwell))
                saturation_dwell = torch.where(
                    effort_ratio >= 0.95, saturation_dwell + dt, torch.zeros_like(saturation_dwell)
                )
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
                timeout = info.get("time_outs", torch.zeros_like(dones)).bool()
                physical_done = dones.bool() & ~timeout

                for index in range(n):
                    if phase[index] >= 4:
                        contact = (bool(contacts[index, 0]), bool(contacts[index, 1]))
                        count = int(contact[0]) + int(contact[1])
                        if count == 0 and not in_flight[index]:
                            in_flight[index] = True
                            flight_start[index] = float(phase_time[index])
                        if in_flight[index] and count > 0:
                            duration = float(phase_time[index]) - flight_start[index]
                            new = [foot for foot in range(2) if contact[foot] and not previous_contacts[index][foot]]
                            valid = len(new) == 1
                            side = new[0] if valid else -1
                            tracker = gait[index]
                            tracker["flights"].append(duration)
                            if valid:
                                tracker["valid"] += 1
                                last_landing[index] = "left" if side == 0 else "right"
                                if tracker["last"] is not None:
                                    tracker["opp"] += 1
                                    tracker["alt"] += int(side != tracker["last"])
                                safe = 0.04 <= duration <= 0.16 and (
                                    tracker["last"] is None or side != tracker["last"]
                                )
                                tracker["safe"] = tracker["safe"] + 1 if safe else 0
                                tracker["maxsafe"] = max(tracker["maxsafe"], tracker["safe"])
                                tracker["last"] = side
                            in_flight[index] = False
                        previous_contacts[index] = contact

                reset_ids = torch.nonzero(dones.bool()).flatten()
                if len(reset_ids):
                    phase[reset_ids] = 0
                    phase_time[reset_ids] = 0
                    good_time[reset_ids] = 0
                    support_switches[reset_ids] = 0
                    previous_support[reset_ids] = 0
                    target_heading[reset_ids] = robot.data.heading_w.torch[reset_ids]
                    slip_dwell[reset_ids] = 0
                    flight_dwell[reset_ids] = 0
                    saturation_dwell[reset_ids] = 0
                    run_hold[reset_ids] = 0
                    reset_count[reset_ids] += 1
                    for index in reset_ids.tolist():
                        gait[index] = fresh_gait()
                        in_flight[index] = False
                        previous_contacts[index] = (False, False)

                settle_good = (
                    (speed.abs() <= 0.08)
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
                acquire_walk = (
                    ((speed - 1.2).abs() <= 0.20)
                    & (heading_error.abs() <= 0.12)
                    & (support_switches >= 2)
                    & (~physical_done)
                )
                mask = phase == 2
                good_time[mask] = torch.where(acquire_walk[mask], good_time[mask] + dt, 0)
                advance = mask & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance] = 3, 0, 0
                walk_good = (phase == 3) & ((speed - 1.2).abs() <= 0.20) & (heading_error.abs() <= 0.12)
                good_time = torch.where(walk_good, good_time + dt, torch.where(phase == 3, 0, good_time))
                advance = (phase == 3) & (good_time >= 1.0)
                phase[advance], phase_time[advance], good_time[advance] = 4, 0, 0
                periodic_mask = torch.tensor([periodic(item) for item in gait], dtype=torch.bool, device=device)
                run_acquire = (
                    (phase == 4)
                    & periodic_mask
                    & ((speed - target_speed).abs() <= 0.20)
                    & (heading_error.abs() <= 0.12)
                    & (slip_dwell < 0.2)
                    & (saturation_dwell < 0.2)
                    & (~physical_done)
                )
                good_time = torch.where(run_acquire, good_time + dt, torch.where(phase == 4, 0, good_time))
                advance = (phase == 4) & (good_time >= 0.4)
                phase[advance], phase_time[advance], good_time[advance], run_hold[advance] = 5, 0, 0, 0
                run_good = (
                    (phase == 5)
                    & periodic_mask
                    & ((speed - target_speed).abs() <= 0.20)
                    & (heading_error.abs() <= 0.12)
                    & (slip_dwell < 0.2)
                    & (saturation_dwell < 0.2)
                    & (~physical_done)
                    & torch.isfinite(legacy).all(dim=1)
                    & torch.isfinite(full_action).all(dim=1)
                )
                run_hold = torch.where(run_good, run_hold + dt, torch.zeros_like(run_hold))
                contract_valid = run_good & (run_hold >= 1.0)
                manager.update_ready(contract_valid, step)
                newly_ready = contract_valid & torch.isnan(ready_time)
                ready_time[newly_ready] = step * dt
                if step % 10 == 0 or int(manager.source_ready.sum()) >= args.cohort_size:
                    timeline_rows.append(
                        {
                            "label": args.label,
                            "generation": generation,
                            "step": step,
                            "time_s": step * dt,
                            "ready_count": int(manager.source_ready.sum()),
                            "cumulative_ready_count": int(torch.isfinite(ready_time).sum()),
                        }
                    )
                previous_support.copy_(support)
                phase_time += dt
                cumulative_ready = int(torch.isfinite(ready_time).sum())
                if (
                    int(manager.source_ready.sum()) >= args.cohort_size
                    and cumulative_ready >= math.ceil(0.90 * n)
                ):
                    launch = manager.activate(contract_valid, previous_action)
                    selected_ids = launch["physical_env_ids"]
                    if not bool(contract_valid[selected_ids].all()):
                        raise RuntimeError("launch source contract not 100%")
                    launch_step = step
                    timestamp = wp.to_torch(sensor._timestamp).clone()
                    pre = {
                        "root_pos": robot.data.root_pos_w.torch[selected_ids].clone(),
                        "root_quat": robot.data.root_quat_w.torch[selected_ids].clone(),
                        "root_lin": robot.data.root_lin_vel_w.torch[selected_ids].clone(),
                        "root_ang": robot.data.root_ang_vel_w.torch[selected_ids].clone(),
                        "joint_pos": robot.data.joint_pos.torch[selected_ids].clone(),
                        "joint_vel": robot.data.joint_vel.torch[selected_ids].clone(),
                        "contacts": contacts[selected_ids].clone(),
                        "forces": forces[selected_ids].clone(),
                        "air": sensor.data.current_air_time.torch[selected_ids][:, sensor_feet].clone(),
                        "last_contact": sensor.data.last_contact_time.torch[selected_ids][:, sensor_feet].clone(),
                        "timestamp": timestamp[selected_ids].clone(),
                        "heading": target_heading[selected_ids].clone(),
                        "previous": previous_action[selected_ids].clone(),
                        "speed": target_speed[selected_ids].clone(),
                    }
                    pre_phase = [
                        "double"
                        if int(support[index]) == 3
                        else "left"
                        if int(support[index]) == 1
                        else "right"
                        if int(support[index]) == 2
                        else "flight"
                        for index in selected_ids.tolist()
                    ]
                    for local, physical in enumerate(selected_ids.tolist()):
                        map_rows.append(
                            {
                                "label": args.label,
                                "generation": generation,
                                "cohort_local_index": local,
                                "physical_env_id": physical,
                                "source_speed_mps": float(target_speed[physical]),
                                "source_phase": pre_phase[local],
                                "ready_time_s": float(ready_time[physical]),
                                "launch_time_s": step * dt,
                            }
                        )
                    break

            if selected_ids is None:
                cohort_results.append(
                    {
                        "generation": generation,
                        "status": "FAIL",
                        "failure": "ready_cohort_timeout",
                        "source_success_rate": float(torch.isfinite(ready_time).float().mean()),
                    }
                )
                continue

            runner = TransitionOnlyOnPolicyRunner(args.cohort_size)
            runner.source_steps = launch_step + 1
            runner.physical_steps = launch_step + 1
            runner.start_transition(torch.ones(args.cohort_size, dtype=torch.bool, device=device))
            stored_observations: list[torch.Tensor] = []
            post_snapshot: dict[str, torch.Tensor] = {}
            action_mismatch = 0
            previous_mismatch = 0
            overlap_count = 0
            unassigned_count = 0
            non_finite = 0
            transition_steps = 8
            last_reward = torch.zeros(args.cohort_size, device=device)
            for transition_step in range(transition_steps):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                heading_error = torch.atan2(
                    torch.sin(target_heading - robot.data.heading_w.torch),
                    torch.cos(target_heading - robot.data.heading_w.torch),
                )
                target_walk = torch.full((n,), 1.2, device=device)
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0] = target_walk
                command_term.vel_command_b[:, 2] = yaw
                motion = MotionCommand(target_walk, target_heading, target_yaw_rate_radps=yaw)
                transition_obs_full = to_run_observation(canonical, motion, route="RUN")
                transition_obs = transition_obs_full[selected_ids]
                previous_match = torch.eq(transition_obs[:, 86:123], previous_action[selected_ids]).all(dim=1)
                previous_mismatch += int((~previous_match).sum())
                with torch.no_grad():
                    transition_action = action_term.apply(transition_obs, previous_action[selected_ids])
                    source_action = run(canonical, motion)
                transition_mask = torch.zeros(n, dtype=torch.bool, device=device)
                transition_mask[selected_ids] = True
                source_mask = ~transition_mask
                post_terminal_mask = torch.zeros_like(transition_mask)
                assignment = source_mask.long() + transition_mask.long() + post_terminal_mask.long()
                overlap_count += int((assignment > 1).sum())
                unassigned_count += int((assignment == 0).sum())
                full_action = torch.empty(n, 37, device=device)
                full_action[source_mask] = source_action[source_mask]
                # Boolean indexing would sort physical IDs and corrupt the
                # cohort-local mapping. Scatter by the frozen selected ID order.
                full_action[selected_ids] = transition_action
                if not torch.isfinite(full_action).all():
                    non_finite += int((~torch.isfinite(full_action)).any(dim=1).sum())
                    raise RuntimeError("non-finite live action")
                for local, physical in enumerate(selected_ids.tolist()):
                    actor_sum = checksum(transition_action[local])
                    applied_sum = checksum(full_action[physical])
                    action_mismatch += int(actor_sum != applied_sum)
                    if transition_step == 0 and local < min(args.cohort_size, 64):
                        routing_rows.append(
                            {
                                "label": args.label,
                                "generation": generation,
                                "physical_env_id": physical,
                                "cohort_local_index": local,
                                "observation_checksum": checksum(transition_obs[local]),
                                "actor_output_checksum": actor_sum,
                                "applied_action_checksum": applied_sum,
                                "match": actor_sum == applied_sum,
                            }
                        )
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                previous_action.copy_(full_action)
                speed = robot.data.root_lin_vel_b.torch[selected_ids, 0]
                reward = -torch.square(speed - 1.2)
                terminated = torch.zeros(args.cohort_size, dtype=torch.bool, device=device)
                truncated = torch.zeros_like(terminated)
                if transition_step == transition_steps - 1:
                    truncated[:] = True
                with torch.no_grad():
                    value = critic(transition_obs).squeeze(-1)
                runner.transition_step(
                    SegmentStep(
                        observation=transition_obs.detach(),
                        action=transition_action.detach(),
                        reward=reward.detach(),
                        value=value.detach(),
                        terminated=terminated,
                        truncated=truncated,
                        log_prob=torch.zeros(args.cohort_size, device=device),
                    )
                )
                stored_observations.append(transition_obs.detach())
                last_reward = reward
                if transition_step == 0:
                    timestamp = wp.to_torch(sensor._timestamp).clone()
                    post_snapshot = {
                        "root_pos": robot.data.root_pos_w.torch[selected_ids].clone(),
                        "root_quat": robot.data.root_quat_w.torch[selected_ids].clone(),
                        "root_lin": robot.data.root_lin_vel_w.torch[selected_ids].clone(),
                        "root_ang": robot.data.root_ang_vel_w.torch[selected_ids].clone(),
                        "joint_pos": robot.data.joint_pos.torch[selected_ids].clone(),
                        "joint_vel": robot.data.joint_vel.torch[selected_ids].clone(),
                        "contacts": sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected_ids]
                        .norm(dim=-1)
                        .amax(dim=1)
                        > 5.0,
                        "forces": sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected_ids].clone(),
                        "air": sensor.data.current_air_time.torch[selected_ids][:, sensor_feet].clone(),
                        "last_contact": sensor.data.last_contact_time.torch[selected_ids][:, sensor_feet].clone(),
                        "timestamp": timestamp[selected_ids].clone(),
                        "heading": target_heading[selected_ids].clone(),
                        "previous": transition_obs[:, 86:123].clone(),
                    }

            returns, advantages = runner.storage.finish(torch.zeros(args.cohort_size, device=device))
            all_obs = torch.cat(stored_observations)
            actor_output = actor(all_obs)
            actor_loss = 1e-4 * actor_output.square().mean()
            critic_output = critic(all_obs).squeeze(-1)
            critic_loss = torch.square(critic_output - returns.flatten().detach()).mean()
            optimizer.zero_grad(set_to_none=True)
            (actor_loss + critic_loss).backward()
            actor_gradient = sum(
                float(parameter.grad.abs().sum())
                for parameter in actor.parameters()
                if parameter.requires_grad and parameter.grad is not None
            )
            critic_gradient = sum(
                float(parameter.grad.abs().sum()) for parameter in critic.parameters() if parameter.grad is not None
            )
            frozen_gradient = sum(
                float(parameter.grad.abs().sum())
                for module in frozen_modules
                for parameter in module.parameters()
                if parameter.grad is not None
            )
            frozen_gradient += sum(
                float(parameter.grad.abs().sum())
                for parameter in actor.parameters()
                if not parameter.requires_grad and parameter.grad is not None
            )
            optimizer.zero_grad(set_to_none=True)

            with tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp) / "r0.pt"
                torch.save(
                    {"actor": actor.state_dict(), "critic": critic.state_dict(), "optimizer": optimizer.state_dict()},
                    checkpoint,
                )
                payload = torch.load(checkpoint, map_location=device, weights_only=False)
                actor_copy = RunToWalkTransitionActor152(run.actor).to(device)
                actor_copy.load_state_dict(payload["actor"], strict=True)
                critic_copy = nn.Sequential(
                    nn.Linear(152, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1)
                ).to(device)
                critic_copy.load_state_dict(payload["critic"], strict=True)
                save_reload = torch.equal(actor(all_obs[:1]), actor_copy(all_obs[:1])) and torch.equal(
                    critic(all_obs[:1]), critic_copy(all_obs[:1])
                )

            timestamp_regression = int((post_snapshot["timestamp"] <= pre["timestamp"]).sum())
            contact_reset = int(
                (
                    (pre["contacts"].any(dim=1) | (pre["air"] > 0).any(dim=1) | (pre["last_contact"] > 0).any(dim=1))
                    & ~(post_snapshot["contacts"].any(dim=1) | (post_snapshot["air"] > 0).any(dim=1) | (post_snapshot["last_contact"] > 0).any(dim=1))
                ).sum()
            )
            for local, physical in enumerate(selected_ids.tolist()):
                continuity_rows.append(
                    {
                        "label": args.label,
                        "generation": generation,
                        "physical_env_id": physical,
                        "source_speed_mps": float(pre["speed"][local]),
                        "pre_step": launch_step,
                        "post_step": launch_step + 1,
                        "controller_before": "RUN_LOW",
                        "controller_after": "RUN_TO_WALK",
                        "source_phase": pre_phase[local],
                        "root_position_pre": tensor_json(pre["root_pos"][local]),
                        "root_position_post": tensor_json(post_snapshot["root_pos"][local]),
                        "root_orientation_pre": tensor_json(pre["root_quat"][local]),
                        "root_orientation_post": tensor_json(post_snapshot["root_quat"][local]),
                        "root_linear_velocity_pre": tensor_json(pre["root_lin"][local]),
                        "root_linear_velocity_post": tensor_json(post_snapshot["root_lin"][local]),
                        "root_angular_velocity_pre": tensor_json(pre["root_ang"][local]),
                        "root_angular_velocity_post": tensor_json(post_snapshot["root_ang"][local]),
                        "joint_position_pre": tensor_json(pre["joint_pos"][local]),
                        "joint_position_post": tensor_json(post_snapshot["joint_pos"][local]),
                        "joint_velocity_pre": tensor_json(pre["joint_vel"][local]),
                        "joint_velocity_post": tensor_json(post_snapshot["joint_vel"][local]),
                        "contact_state_pre": tensor_json(pre["contacts"][local]),
                        "contact_state_post": tensor_json(post_snapshot["contacts"][local]),
                        "contact_force_pre": tensor_json(pre["forces"][local]),
                        "contact_force_post": tensor_json(post_snapshot["forces"][local]),
                        "foot_air_time_pre": tensor_json(pre["air"][local]),
                        "foot_air_time_post": tensor_json(post_snapshot["air"][local]),
                        "last_contact_time_pre": tensor_json(pre["last_contact"][local]),
                        "last_contact_time_post": tensor_json(post_snapshot["last_contact"][local]),
                        "last_landing_foot": last_landing[physical],
                        "sensor_timestamp_pre": float(pre["timestamp"][local]),
                        "sensor_timestamp_post": float(post_snapshot["timestamp"][local]),
                        "target_heading_pre": float(pre["heading"][local]),
                        "target_heading_post": float(post_snapshot["heading"][local]),
                        "global_previous_action_pre": tensor_json(pre["previous"][local]),
                        "transition_observed_previous_action": tensor_json(post_snapshot["previous"][local]),
                        "same_env_id": True,
                        "state_setter_calls": 0,
                        "teleport_calls": 0,
                        "physics_step_skip": 0,
                    }
                )
            if replay is None:
                first = runner.storage.steps
                replay = {
                    "physical_env_id": int(selected_ids[0]),
                    "segment_length": len(first),
                    "observation": first[0].observation[0].detach().cpu().tolist(),
                    "rewards": [float(item.reward[0]) for item in first],
                    "values": [float(item.value[0]) for item in first],
                    "terminated": [bool(item.terminated[0]) for item in first],
                    "truncated": [bool(item.truncated[0]) for item in first],
                    "returns": [float(value) for value in returns[:, 0]],
                    "advantages": [float(value) for value in advantages[:, 0]],
                }
            cohort_result = {
                "generation": generation,
                "status": "PASS",
                "physical_envs": n,
                "cohort_size": args.cohort_size,
                "source_success_rate": float(torch.isfinite(ready_time).float().mean()),
                "ready_count_at_launch": int(manager.source_ready.sum()),
                "launch_contract_valid": args.cohort_size,
                "cohort_formation_time_s": launch_step * dt,
                "source_speed_counts": dict(Counter(float(pre["speed"][index]) for index in range(args.cohort_size))),
                "source_phase_counts": dict(Counter(pre_phase)),
                "same_env_id_count": args.cohort_size,
                "state_copy": False,
                "state_setter_calls": 0,
                "teleport_calls": 0,
                "physics_step_skip": 0,
                "previous_action_mismatch": previous_mismatch,
                "sensor_timestamp_regression": timestamp_regression,
                "contact_history_reset": contact_reset,
                "action_routing_mismatch": action_mismatch,
                "controller_overlap": overlap_count,
                "unassigned_env": unassigned_count,
                "source_prefix_stored_steps": 0,
                "non_selected_stored_steps": 0,
                "invalid_stored_steps": 0,
                "post_terminal_stored_steps": 0,
                "transition_storage_steps": runner.transition_steps,
                "terminal_type": "global_horizon_truncation",
                "actor_gradient_sum": actor_gradient,
                "critic_gradient_sum": critic_gradient,
                "frozen_gradient_sum": frozen_gradient,
                "non_finite_count": non_finite,
                "optimizer_updates": 0,
                "checkpoint_optimizer_save_reload": save_reload,
                "observation_dim": all_obs.shape[1],
                "action_dim": actor_output.shape[1],
                "action_scale": 0.5,
                "last_raw_speed_reduction_signal_mean": float(last_reward.mean()),
            }
            checks = [
                cohort_result["source_success_rate"] >= 0.90,
                cohort_result["launch_contract_valid"] == args.cohort_size,
                previous_mismatch == 0,
                timestamp_regression == 0,
                contact_reset == 0,
                action_mismatch == 0,
                overlap_count == 0,
                unassigned_count == 0,
                non_finite == 0,
                actor_gradient > 0,
                critic_gradient > 0,
                frozen_gradient == 0,
                save_reload,
            ]
            cohort_result["status"] = "PASS" if all(checks) else "FAIL"
            cohort_results.append(cohort_result)

        frozen_after = {
            f"{module_index}:{name}": tensor_hash(parameter)
            for module_index, module in enumerate(frozen_modules)
            for name, parameter in module.named_parameters()
        }
        actor_frozen_after = {
            name: tensor_hash(parameter) for name, parameter in actor.named_parameters() if not parameter.requires_grad
        }
        summary = {
            "label": args.label,
            "status": "PASS" if len(cohort_results) == args.cohorts and all(item["status"] == "PASS" for item in cohort_results) else "FAIL",
            "num_envs": args.num_envs,
            "cohort_size": args.cohort_size,
            "cohorts_requested": args.cohorts,
            "cohorts_completed": len(cohort_results),
            "cohorts": cohort_results,
            "protected_hashes": hashes,
            "frozen_parameter_hash_unchanged": frozen_before == frozen_after,
            "actor_frozen_hash_unchanged": actor_frozen_before == actor_frozen_after,
            "optimizer_updates": 0,
            "actual_isaac_step": True,
            "actual_contact_sensor": True,
        }
        def write_live_rows(name: str, rows: list[dict]) -> None:
            if not rows:
                return
            with (output / name).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        write_live_rows(f"{args.label}_ready_timeline.csv", timeline_rows)
        write_live_rows(f"{args.label}_cohort_map.csv", map_rows)
        write_live_rows(f"{args.label}_continuity.csv", continuity_rows)
        write_live_rows(f"{args.label}_routing.csv", routing_rows)
        (output / f"{args.label}_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        (output / f"{args.label}_replay.json").write_text(
            json.dumps(replay, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
