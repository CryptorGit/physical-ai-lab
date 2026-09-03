"""Build the exp_008 Stage 0 dataset by frozen exp_007 diagnostic replay."""
from __future__ import annotations

import argparse
import csv
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
from torch.distributions import Normal
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"
CFG_PATH = EXP / "configs/stage0_observability_probe.yaml"
CHECKPOINT_SHA = "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263"
EXPECTED = {
    "stand": "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "stw": "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt",
    "walk": "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
    "run": "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
    "wtr": "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
}

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
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation  # noqa: E402
from g1_walk_centered.tasks.stage7r_action import RunToWalkTransitionActor152, WalkToRunTransitionActor152  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402
from phase_transition_analysis.dataset import action_columns, assign_grouped_splits, observation_columns  # noqa: E402
from phase_transition_analysis.feature_layout import feature_layout_document  # noqa: E402
from phase_transition_analysis.labels import add_labels  # noqa: E402


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def mj(value):
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


class SourceState:
    pass


parser = argparse.ArgumentParser()
parser.add_argument("--counterfactual-only", action="store_true")
parser.add_argument(
    "--counterfactual-candidate",
    choices=["baseline", "walk_expert", "run_expert", "bounded_joint_group", "target_walk_alignment"],
)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    checkpoint = (REPO / cfg["checkpoint"]).resolve()
    if file_sha(checkpoint) != CHECKPOINT_SHA:
        raise RuntimeError("frozen checkpoint SHA mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    dump("feature_layout.json", feature_layout_document())
    task_cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    task_cfg.scene.num_envs = cfg["dataset"]["physical_envs"]
    task_cfg.seed = cfg["seed"]
    task_cfg.episode_length_s = 24.0
    task_cfg.sim.device = cfg["device"]
    args.device = cfg["device"]
    rows, episode_index = [], []
    total_rows = 0
    source_attempts, source_success = 0, 0
    phase_counter, speed_counter, reason_counter = Counter(), Counter(), Counter()
    success_count = 0

    with launch_simulation(task_cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=task_cfg), clip_actions=agent_cfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        dt = float(env.step_dt)
        paths = {name: (REPO / path).resolve() for name, path in EXPECTED.items()}
        stand = load_walk_expert(paths["stand"], device=device)
        stw = load_walk_expert(paths["stw"], device=device)
        walk = load_walk_expert(paths["walk"], device=device)
        run = load_run_expert(paths["run"], device=device)
        wtr = WalkToRunTransitionActor152(run.actor).to(device)
        wtr.load_state_dict(torch.load(paths["wtr"], map_location=device, weights_only=False)["actor"], strict=True)
        wtr.eval()
        actor = RunToWalkTransitionActor152(run.actor).to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(payload["actor"], strict=True)
        actor.eval()
        std = payload["log_std"].to(device).exp()
        robot = env.scene["robot"]
        command_term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")

        def prepare_source(seed, cohort_size):
            nonlocal source_attempts, source_success
            source_attempts += wrapped.num_envs
            wrapped.reset()
            n = wrapped.num_envs
            phase = torch.zeros(n, dtype=torch.long, device=device)
            phase_time = torch.zeros(n, device=device)
            good = torch.zeros(n, device=device)
            switches = torch.zeros(n, dtype=torch.long, device=device)
            prev_support = torch.zeros(n, dtype=torch.long, device=device)
            previous_action = torch.zeros(n, 37, device=device)
            heading = robot.data.heading_w.torch.clone()
            generator = torch.Generator(device=device).manual_seed(seed)
            source_speed = torch.where(torch.arange(n, device=device) % 2 == 0, 2.6, 2.8)
            slip_dwell = torch.zeros(n, device=device)
            flight_dwell = torch.zeros(n, device=device)
            sat_dwell = torch.zeros(n, device=device)
            run_hold = torch.zeros(n, device=device)
            ready_ever = torch.zeros(n, dtype=torch.bool, device=device)
            in_flight = torch.zeros(n, dtype=torch.bool, device=device)
            flight_start = torch.zeros(n, device=device)
            flights = torch.zeros(n, device=device)
            valid_landings = torch.zeros(n, device=device)
            alt_opp = torch.zeros(n, device=device)
            alternating = torch.zeros(n, device=device)
            last_side = torch.full((n,), -1, dtype=torch.long, device=device)
            consecutive = torch.zeros(n, device=device)
            max_consecutive = torch.zeros(n, device=device)
            flight_sum = torch.zeros(n, device=device)
            previous_contacts = torch.zeros(n, 2, dtype=torch.bool, device=device)
            for step in range(round(23.0 / dt)):
                legacy = wrapped.get_observations()["policy"]
                canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
                command_speed = torch.zeros(n, device=device)
                command_speed[phase == 2] = 1.2 * mj(phase_time[phase == 2] / 1.5)
                command_speed[phase == 3] = 1.2
                command_speed[phase == 4] = 1.2 + (source_speed[phase == 4] - 1.2) * mj(phase_time[phase == 4] / 1.4)
                command_speed[phase == 5] = source_speed[phase == 5]
                heading_error = torch.atan2(torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch))
                yaw = torch.where(phase >= 4, (1.5 * heading_error).clamp(-1.5, 1.5), (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3))
                command_term.vel_command_b.zero_()
                command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
                command = MotionCommand(command_speed, heading, target_yaw_rate_radps=yaw)
                with torch.no_grad():
                    actions = [
                        stand(canonical, command), stw(canonical, command), walk(canonical, command),
                        wtr(to_run_observation(canonical, command, route="RUN")), run(canonical, command),
                    ]
                    masks = [(phase == 0) | (phase == 1), phase == 2, phase == 3, phase == 4, phase == 5]
                    full_action = torch.empty(n, 37, device=device)
                    for mask, action in zip(masks, actions):
                        full_action[mask] = action[mask]
                    _, _, dones, info = wrapped.step(full_action)
                previous_action.copy_(full_action)
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
                contacts = forces.norm(dim=-1).amax(dim=1) > 5
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
                slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
                effort = robot.data.applied_torque.torch[:, joints].abs() / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
                slip_dwell = torch.where(slip > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                flight_dwell = torch.where(~contacts.any(1), flight_dwell + dt, torch.zeros_like(flight_dwell))
                sat_dwell = torch.where((effort >= 0.95).any(1), sat_dwell + dt, torch.zeros_like(sat_dwell))
                speed = robot.data.root_lin_vel_b.torch[:, 0]
                gravity = robot.data.projected_gravity_b.torch
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
                timeout = info.get("time_outs", torch.zeros_like(dones)).bool()
                physical_done = dones.bool() & ~timeout
                no_contact = ~contacts.any(1)
                liftoff = (~in_flight) & no_contact & (phase >= 4)
                flight_start[liftoff] = phase_time[liftoff]
                in_flight |= liftoff
                landing = in_flight & contacts.any(1) & (phase >= 4)
                duration = phase_time - flight_start
                new_contact = contacts & ~previous_contacts
                valid = landing & (new_contact.sum(1) == 1)
                side = new_contact.long().argmax(1)
                has_last = last_side >= 0
                alternate = valid & has_last & (side != last_side)
                alt_opp += (valid & has_last).float()
                alternating += alternate.float()
                safe_cycle = valid & (duration >= 0.04) & (duration <= 0.16) & ((~has_last) | alternate)
                consecutive = torch.where(safe_cycle, consecutive + 1, torch.where(landing, torch.zeros_like(consecutive), consecutive))
                max_consecutive = torch.maximum(max_consecutive, consecutive)
                flights += landing.float()
                valid_landings += valid.float()
                flight_sum += torch.where(landing, duration, torch.zeros_like(duration))
                last_side = torch.where(valid, side, last_side)
                in_flight &= ~landing
                previous_contacts = contacts
                periodic = (flights >= 4) & (max_consecutive >= 3) & (alternating / alt_opp.clamp_min(1) >= 0.8) & (valid_landings / flights.clamp_min(1) >= 0.8) & (flight_sum / flights.clamp_min(1) >= 0.04) & (flight_sum / flights.clamp_min(1) <= 0.16)
                reset = torch.nonzero(dones.bool()).flatten()
                if len(reset):
                    for tensor in (phase_time, good, slip_dwell, flight_dwell, sat_dwell, run_hold, flights, valid_landings, alt_opp, alternating, consecutive, max_consecutive, flight_sum):
                        tensor[reset] = 0
                    phase[reset], switches[reset], prev_support[reset], last_side[reset] = 0, 0, 0, -1
                    in_flight[reset], previous_contacts[reset] = False, False
                    heading[reset] = robot.data.heading_w.torch[reset]
                settled = (speed.abs() <= 0.08) & (roll <= 0.10) & (pitch <= 0.10) & contacts.all(1) & ~physical_done
                mask = phase == 0
                good[mask] = torch.where(settled[mask], good[mask] + dt, 0)
                advance = mask & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance] = 1, 0, 0
                advance = (phase == 1) & (phase_time >= 0.8)
                phase[advance], phase_time[advance] = 2, 0
                changed = (support != prev_support) & ((support == 1) | (support == 2)) & (phase == 2)
                switches[changed] += 1
                acquire_walk = ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (switches >= 2) & ~physical_done
                mask = phase == 2
                good[mask] = torch.where(acquire_walk[mask], good[mask] + dt, 0)
                advance = mask & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance] = 3, 0, 0
                walk_good = (phase == 3) & ((speed - 1.2).abs() <= 0.2) & (heading_error.abs() <= 0.12)
                good = torch.where(walk_good, good + dt, torch.where(phase == 3, 0, good))
                advance = (phase == 3) & (good >= 1.0)
                phase[advance], phase_time[advance], good[advance] = 4, 0, 0
                run_acquire = (phase == 4) & periodic & ((speed - source_speed).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (slip_dwell < 0.2) & (sat_dwell < 0.2) & ~physical_done
                good = torch.where(run_acquire, good + dt, torch.where(phase == 4, 0, good))
                advance = (phase == 4) & (good >= 0.4)
                phase[advance], phase_time[advance], good[advance], run_hold[advance] = 5, 0, 0, 0
                contract = (phase == 5) & periodic & ((speed - source_speed).abs() <= 0.2) & (heading_error.abs() <= 0.12) & (slip_dwell < 0.2) & (sat_dwell < 0.2) & ~physical_done & torch.isfinite(legacy).all(1) & torch.isfinite(full_action).all(1)
                run_hold = torch.where(contract, run_hold + dt, torch.zeros_like(run_hold))
                ready = torch.nonzero(contract & (run_hold >= 1.0)).flatten()
                if len(ready) >= cohort_size:
                    left = ready[source_speed[ready] == 2.6][: cohort_size // 2]
                    right = ready[source_speed[ready] == 2.8][: cohort_size // 2]
                    if len(left) == cohort_size // 2 and len(right) == cohort_size // 2:
                        selected = torch.cat((left, right))
                        source_success += len(selected)
                        state = SourceState()
                        state.selected, state.phase, state.phase_time = selected, phase, phase_time
                        state.previous_action, state.heading, state.source_speed = previous_action, heading, source_speed
                        state.launch_phase = torch.where(support[selected] == 1, 1, torch.where(support[selected] == 2, 2, torch.where(support[selected] == 3, 3, 0)))
                        return state
                prev_support.copy_(support)
                phase_time += dt
            return None

        def graph_background(state):
            legacy = wrapped.get_observations()["policy"]
            canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            n = wrapped.num_envs
            heading_error = torch.atan2(torch.sin(state.heading - robot.data.heading_w.torch), torch.cos(state.heading - robot.data.heading_w.torch))
            command_speed = torch.zeros(n, device=device)
            command_speed[state.phase == 2] = 1.2 * mj(state.phase_time[state.phase == 2] / 1.5)
            command_speed[state.phase == 3] = 1.2
            command_speed[state.phase == 4] = 1.2 + (state.source_speed[state.phase == 4] - 1.2) * mj(state.phase_time[state.phase == 4] / 1.4)
            command_speed[state.phase == 5] = state.source_speed[state.phase == 5]
            yaw = torch.where(state.phase >= 4, (1.5 * heading_error).clamp(-1.5, 1.5), (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3))
            command_term.vel_command_b.zero_()
            command_term.vel_command_b[:, 0], command_term.vel_command_b[:, 2] = command_speed, yaw
            command = MotionCommand(command_speed, state.heading, target_yaw_rate_radps=yaw)
            with torch.no_grad():
                actions = [stand(canonical, command), stw(canonical, command), walk(canonical, command), wtr(to_run_observation(canonical, command, route="RUN")), run(canonical, command)]
                masks = [(state.phase == 0) | (state.phase == 1), state.phase == 2, state.phase == 3, state.phase == 4, state.phase == 5]
                full = torch.empty(n, 37, device=device)
                for mask, action in zip(masks, actions):
                    full[mask] = action[mask]
            return legacy, full

        parquet_dir = OUT / "episodes.parquet"
        if not args.counterfactual_only:
            parquet_dir.mkdir(parents=True, exist_ok=True)
            dummy = pd.DataFrame(
                [
                    {
                        "episode_id": f"group_{seed}_{speed}",
                        "evaluation_seed": seed,
                        "source_speed_mps": speed,
                        "checkpoint": "stage8c_model_10",
                    }
                    for seed in range(cfg["seed"], cfg["seed"] + cfg["dataset"]["diagnostic_cohorts"])
                    for speed in cfg["dataset"]["source_speeds_mps"]
                ]
            )
            _, dummy_mapping = assign_grouped_splits(
                dummy,
                cfg["seed"],
                (cfg["split"]["train_fraction"], cfg["split"]["validation_fraction"], cfg["split"]["test_fraction"]),
            )
            group_split = {
                (int(row.evaluation_seed), round(float(row.source_speed_mps), 1)): dummy_mapping[row.episode_id]
                for row in dummy.itertuples(index=False)
            }

        for cohort_index in range(0 if args.counterfactual_only else cfg["dataset"]["diagnostic_cohorts"]):
            evaluation_seed = cfg["seed"] + cohort_index
            torch.manual_seed(evaluation_seed)
            state = prepare_source(evaluation_seed, cfg["dataset"]["cohort_size"])
            if state is None:
                raise RuntimeError(f"source cohort failed for seed {evaluation_seed}")
            selected, cohort = state.selected, len(state.selected)
            previous_contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected].norm(dim=-1).amax(1) > 5
            air_time = torch.zeros(cohort, 2, device=device)
            last_contact_time = torch.zeros(cohort, 2, device=device)
            last_landing = torch.full((cohort,), -1, dtype=torch.long, device=device)
            stable_contact = torch.zeros(cohort, device=device)
            flight_dwell = torch.zeros(cohort, device=device)
            walk_streak = torch.zeros(cohort, dtype=torch.long, device=device)
            max_streak = torch.zeros(cohort, dtype=torch.long, device=device)
            cycle_terminated = torch.zeros(cohort, dtype=torch.bool, device=device)
            active = torch.ones(cohort, dtype=torch.bool, device=device)
            first_contact = torch.full((cohort,), -1, dtype=torch.long, device=device)
            first_break = torch.full((cohort,), -1, dtype=torch.long, device=device)
            break_reason = ["none"] * cohort
            records = []
            slip_dwell = torch.zeros(cohort, device=device)
            ankle_dwell = torch.zeros(cohort, device=device)
            knee_dwell = torch.zeros(cohort, device=device)
            knee = torch.tensor([i for i, name in enumerate(joint_names) if "knee" in name], device=device)
            ankle = torch.tensor([i for i, name in enumerate(joint_names) if "ankle" in name], device=device)
            for step in range(250):
                legacy, background = graph_background(state)
                canonical = canonical_state_from_legacy_observation(legacy[selected], heading_w_rad=robot.data.heading_w.torch[selected])
                progress = mj(torch.full((cohort,), step * dt / 1.4, device=device))
                target_command = state.source_speed[selected] + (1.2 - state.source_speed[selected]) * progress
                heading_error = torch.atan2(torch.sin(state.heading[selected] - robot.data.heading_w.torch[selected]), torch.cos(state.heading[selected] - robot.data.heading_w.torch[selected]))
                yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[selected, 2]).clamp(-0.3, 0.3)
                command = MotionCommand(target_command, state.heading[selected], target_yaw_rate_radps=yaw)
                obs = to_run_observation(canonical, command, route="RUN")
                with torch.no_grad():
                    mean = actor(obs)
                    action = (
                        mean
                        if cohort_index < cfg["dataset"]["deterministic_cohorts"]
                        else Normal(mean, std.expand_as(mean)).sample()
                    )
                full_action = background
                full_action[selected] = action
                with torch.no_grad():
                    _, _, dones, info = wrapped.step(full_action)
                speed = robot.data.root_lin_vel_b.torch[selected, 0]
                lateral = robot.data.root_lin_vel_b.torch[selected, 1]
                vertical = robot.data.root_lin_vel_b.torch[selected, 2]
                gravity = robot.data.projected_gravity_b.torch[selected]
                roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
                pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2))
                forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                contacts = forces.norm(dim=-1).amax(1) > 5
                contact_force = forces.norm(dim=-1).amax(1)
                new_contact = contacts & ~previous_contacts
                landed = new_contact.any(1)
                landing_side = new_contact.long().argmax(1)
                last_landing = torch.where(landed, landing_side, last_landing)
                air_time = torch.where(contacts, torch.zeros_like(air_time), air_time + dt)
                last_contact_time = torch.where(contacts, torch.full_like(last_contact_time, step * dt), last_contact_time)
                stable_contact = torch.where(contacts.any(1), stable_contact + dt, torch.zeros_like(stable_contact))
                no_contact = ~contacts.any(1)
                flight_dwell = torch.where(no_contact, flight_dwell + dt, torch.zeros_like(flight_dwell))
                walk_contact = contacts.any(1) & (flight_dwell <= 0.16)
                cycle_terminated |= (stable_contact >= 0.12) & (step * dt >= 0.20)
                foot_speed = robot.data.body_lin_vel_w.torch[selected][:, feet, :2].norm(dim=-1)
                slip_speed = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1)
                effort = robot.data.applied_torque.torch[selected][:, joints].abs() / robot.data.joint_effort_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                joint_velocity = robot.data.joint_vel.torch[selected][:, joints]
                velocity_ratio = joint_velocity.abs() / robot.data.joint_vel_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                slip_dwell = torch.where(slip_speed > 0.8, slip_dwell + dt, torch.zeros_like(slip_dwell))
                ankle_dwell = torch.where((effort[:, ankle] >= 0.95).any(1), ankle_dwell + dt, torch.zeros_like(ankle_dwell))
                knee_dwell = torch.where((velocity_ratio[:, knee] >= 0.95).any(1), knee_dwell + dt, torch.zeros_like(knee_dwell))
                impact = forces[:, :, :, 2].abs().mean(1).amax(1) > 3500
                timeout = info.get("time_outs", torch.zeros_like(dones)).bool()[selected]
                fall = dones.bool()[selected] & ~timeout
                speed_ok = (speed - 1.2).abs() <= 0.2
                heading_ok = heading_error.abs() <= 0.12
                flight_ok = flight_dwell <= 0.16
                safety_ok = ~fall & (speed >= -0.1) & (slip_dwell < 0.2) & (ankle_dwell < 0.2) & (knee_dwell < 0.2) & ~impact
                valid = speed_ok & heading_ok & cycle_terminated & walk_contact & flight_ok & safety_ok
                previous_streak = walk_streak.clone()
                walk_streak = torch.where(valid & active, walk_streak + 1, torch.zeros_like(walk_streak))
                max_streak = torch.maximum(max_streak, walk_streak)
                first_contact = torch.where((first_contact < 0) & walk_contact, torch.full_like(first_contact, step), first_contact)
                broken = (previous_streak > 0) & ~valid & active & (first_break < 0)
                first_break = torch.where(broken, torch.full_like(first_break, step), first_break)
                for local in torch.nonzero(broken).flatten().cpu().tolist():
                    break_reason[local] = (
                        "safety" if not bool(safety_ok[local]) else
                        "flight" if not bool(flight_ok[local]) else
                        "contact" if not bool(walk_contact[local]) else
                        "heading" if not bool(heading_ok[local]) else
                        "speed" if not bool(speed_ok[local]) else "none"
                    )
                support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
                records.append({
                    "obs": obs.detach().cpu().numpy().astype(np.float32),
                    "action": action.detach().cpu().numpy().astype(np.float32),
                    "previous_action": state.previous_action[selected].detach().cpu().numpy().astype(np.float32),
                    "contacts": contacts.cpu().numpy(),
                    "contact_force": contact_force.cpu().numpy().astype(np.float32),
                    "air_time": air_time.cpu().numpy().astype(np.float32),
                    "last_contact_time": last_contact_time.cpu().numpy().astype(np.float32),
                    "last_landing": last_landing.cpu().numpy(),
                    "support": support.cpu().numpy(),
                    "vertical": vertical.cpu().numpy().astype(np.float32),
                    "roll": roll.cpu().numpy().astype(np.float32),
                    "pitch": pitch.cpu().numpy().astype(np.float32),
                    "effort": effort.cpu().numpy().astype(np.float32),
                    "joint_velocity": joint_velocity.cpu().numpy().astype(np.float32),
                    "heading": state.heading[selected].cpu().numpy().astype(np.float32),
                    "streak": walk_streak.cpu().numpy(),
                    "valid": valid.cpu().numpy(),
                    "speed_ok": speed_ok.cpu().numpy(),
                    "heading_ok": heading_ok.cpu().numpy(),
                    "flight_ok": flight_ok.cpu().numpy(),
                    "safety_ok": safety_ok.cpu().numpy(),
                })
                success = walk_streak >= 20
                active &= ~success
                previous_contacts = contacts
                state.previous_action.copy_(full_action)
            for local in range(cohort):
                contact_step = int(first_contact[local])
                break_step = int(first_break[local])
                success = int(max_streak[local]) >= 20
                if contact_step < 0:
                    contact_step = 0
                if break_step < 0:
                    break_step = 249
                begin = max(0, contact_step - cfg["dataset"]["pre_window_steps"])
                end = min(249, break_step + cfg["dataset"]["post_window_steps"])
                episode_id = f"seed{evaluation_seed}_cohort{cohort_index}_env{int(selected[local])}"
                success_count += int(success)
                speed_value = float(state.source_speed[selected][local])
                phase_value = int(state.launch_phase[local])
                speed_counter[speed_value] += 1
                phase_counter[{0: "flight", 1: "left", 2: "right", 3: "double"}[phase_value]] += 1
                reason_counter[break_reason[local]] += 1
                episode_index.append({
                    "episode_id": episode_id,
                    "evaluation_seed": evaluation_seed,
                    "source_speed_mps": speed_value,
                    "source_phase": {0: "flight", 1: "left", 2: "right", 3: "double"}[phase_value],
                    "first_walk_contact_step": contact_step,
                    "break_step": break_step,
                    "break_reason": break_reason[local],
                    "maximum_walk_valid_streak": int(max_streak[local]),
                    "episode_success": bool(success),
                    "window_start": begin,
                    "window_end": end,
                })
                for step in range(begin, end + 1):
                    record = records[step]
                    row = {
                        "episode_id": episode_id,
                        "evaluation_seed": evaluation_seed,
                        "checkpoint": "stage8c_model_10",
                        "policy_sampling": (
                            "deterministic_mean"
                            if cohort_index < cfg["dataset"]["deterministic_cohorts"]
                            else "stochastic_frozen_std"
                        ),
                        "source_speed_mps": speed_value,
                        "source_phase": {0: "flight", 1: "left", 2: "right", 3: "double"}[phase_value],
                        "transition_step": step,
                        "relative_to_first_contact": step - contact_step,
                        "relative_to_break": step - break_step,
                        "steps_until_break": max(break_step - step, 0),
                        "break_reason": break_reason[local],
                        "episode_success": bool(success),
                        "walk_valid_streak_age": int(record["streak"][local]),
                        "walk_valid": bool(record["valid"][local]),
                        "speed_condition": bool(record["speed_ok"][local]),
                        "heading_condition": bool(record["heading_ok"][local]),
                        "contact_condition": bool(record["contacts"][local].any()),
                        "flight_condition": bool(record["flight_ok"][local]),
                        "safety_condition": bool(record["safety_ok"][local]),
                        "left_contact": int(record["contacts"][local, 0]),
                        "right_contact": int(record["contacts"][local, 1]),
                        "left_foot_air_time": float(record["air_time"][local, 0]),
                        "right_foot_air_time": float(record["air_time"][local, 1]),
                        "left_last_contact_time": float(record["last_contact_time"][local, 0]),
                        "right_last_contact_time": float(record["last_contact_time"][local, 1]),
                        "last_landing_foot": int(record["last_landing"][local]),
                        "support_phase": int(record["support"][local]),
                        "left_contact_force": float(record["contact_force"][local, 0]),
                        "right_contact_force": float(record["contact_force"][local, 1]),
                        "vertical_velocity": float(record["vertical"][local]),
                        "base_roll": float(record["roll"][local]),
                        "base_pitch": float(record["pitch"][local]),
                        "target_heading": float(record["heading"][local]),
                    }
                    row.update({name: float(value) for name, value in zip(observation_columns(), record["obs"][local])})
                    row.update({name: float(value) for name, value in zip(action_columns(), record["action"][local])})
                    row.update({f"previous_action_{index:03d}": float(value) for index, value in enumerate(record["previous_action"][local])})
                    row.update({f"joint_effort_{index:03d}": float(value) for index, value in enumerate(record["effort"][local])})
                    row.update({f"joint_velocity_{index:03d}": float(value) for index, value in enumerate(record["joint_velocity"][local])})
                    rows.append(row)
            part = add_labels(pd.DataFrame(rows))
            part["split"] = [
                group_split[(int(seed), round(float(speed), 1))]
                for seed, speed in zip(part["evaluation_seed"], part["source_speed_mps"])
            ]
            part.to_parquet(parquet_dir / f"part-{cohort_index:03d}.parquet", index=False, compression="zstd")
            total_rows += len(part)
            rows.clear()
            print(
                f"[exp008 dataset] cohort={cohort_index + 1}/{cfg['dataset']['diagnostic_cohorts']} "
                f"episodes={len(episode_index)} rows={total_rows} successes={success_count}",
                flush=True,
            )
        if args.counterfactual_only:
            branch_ages = torch.tensor(cfg["counterfactual"]["branch_streak_ages"], device=device)
            candidates = (
                [args.counterfactual_candidate]
                if args.counterfactual_candidate
                else ["baseline", "walk_expert", "run_expert", "bounded_joint_group", "target_walk_alignment"]
            )
            candidate_results = []
            prebranch_reference = None
            selected_reference = None
            matching = []
            perturb_groups = {}
            for group_name in cfg["counterfactual"]["joint_groups"]:
                perturb_groups[group_name] = torch.tensor(
                    [index for index, name in enumerate(joint_names) if group_name.replace("_", ".*") in name],
                    dtype=torch.long,
                    device=device,
                )
            # Joint names contain side prefixes and the literal group token.
            perturb_groups = {
                group_name: torch.tensor(
                    [index for index, name in enumerate(joint_names) if group_name in name],
                    dtype=torch.long,
                    device=device,
                )
                for group_name in cfg["counterfactual"]["joint_groups"]
            }

            for candidate_index, candidate in enumerate(candidates):
                replay_seed = cfg["seed"] + 5000
                wrapped.seed(replay_seed)
                torch.manual_seed(replay_seed)
                state = prepare_source(replay_seed, cfg["dataset"]["cohort_size"])
                if state is None:
                    raise RuntimeError(f"counterfactual source failed: {candidate}")
                selected, cohort = state.selected, len(state.selected)
                if selected_reference is None:
                    selected_reference = selected.detach().cpu().tolist()
                elif selected.detach().cpu().tolist() != selected_reference:
                    raise RuntimeError("counterfactual selected env IDs changed across seeded replay")
                previous_contacts = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected].norm(dim=-1).amax(1) > 5
                stable_contact = torch.zeros(cohort, device=device)
                flight_dwell = torch.zeros(cohort, device=device)
                walk_streak = torch.zeros(cohort, dtype=torch.long, device=device)
                maximum_streak = torch.zeros_like(walk_streak)
                cycle_terminated = torch.zeros(cohort, dtype=torch.bool, device=device)
                branched = torch.zeros(cohort, dtype=torch.bool, device=device)
                branch_step = torch.full((cohort,), -1, dtype=torch.long, device=device)
                branch_age = branch_ages[torch.arange(cohort, device=device) % len(branch_ages)]
                post_steps = torch.zeros(cohort, dtype=torch.long, device=device)
                unsafe = torch.zeros(cohort, dtype=torch.bool, device=device)
                branch_root, branch_joint, branch_velocity = {}, {}, {}
                assignment = []
                for local in range(cohort):
                    group = cfg["counterfactual"]["joint_groups"][local % len(cfg["counterfactual"]["joint_groups"])]
                    side = "left" if (local // len(cfg["counterfactual"]["joint_groups"])) % 2 == 0 else "right"
                    sign = -1.0 if (local // (2 * len(cfg["counterfactual"]["joint_groups"]))) % 2 == 0 else 1.0
                    assignment.append((group, side, sign))

                for step in range(250):
                    legacy, background = graph_background(state)
                    canonical = canonical_state_from_legacy_observation(legacy[selected], heading_w_rad=robot.data.heading_w.torch[selected])
                    progress = mj(torch.full((cohort,), step * dt / 1.4, device=device))
                    target_command = state.source_speed[selected] + (1.2 - state.source_speed[selected]) * progress
                    heading_error = torch.atan2(
                        torch.sin(state.heading[selected] - robot.data.heading_w.torch[selected]),
                        torch.cos(state.heading[selected] - robot.data.heading_w.torch[selected]),
                    )
                    yaw = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[selected, 2]).clamp(-0.3, 0.3)
                    command = MotionCommand(target_command, state.heading[selected], target_yaw_rate_radps=yaw)
                    obs = to_run_observation(canonical, command, route="RUN")
                    walk_command = MotionCommand(torch.full_like(target_command, 1.2), state.heading[selected], target_yaw_rate_radps=yaw)
                    run_command = MotionCommand(state.source_speed[selected], state.heading[selected], target_yaw_rate_radps=yaw)
                    with torch.no_grad():
                        baseline_action = actor(obs)
                        walk_action = walk(canonical, walk_command)
                        run_action = run(canonical, run_command)
                    just_branch = (~branched) & (walk_streak >= branch_age)
                    if just_branch.any():
                        for local in torch.nonzero(just_branch).flatten().cpu().tolist():
                            env_id = int(selected[local])
                            branch_root[env_id] = (robot.data.root_pos_w.torch[env_id] - env.scene.env_origins[env_id]).detach().cpu().numpy()
                            branch_joint[env_id] = robot.data.joint_pos.torch[env_id].detach().cpu().numpy()
                            branch_velocity[env_id] = torch.cat(
                                (robot.data.root_lin_vel_b.torch[env_id], robot.data.joint_vel.torch[env_id])
                            ).detach().cpu().numpy()
                        branch_step[just_branch] = step
                        branched |= just_branch
                    correction_active = branched & (post_steps < cfg["counterfactual"]["rollout_steps_after_branch"])
                    action = baseline_action.clone()
                    if candidate == "walk_expert":
                        action[correction_active] = walk_action[correction_active]
                    elif candidate == "run_expert":
                        action[correction_active] = run_action[correction_active]
                    elif candidate == "target_walk_alignment":
                        delta = (walk_action - baseline_action).clamp(
                            -cfg["counterfactual"]["perturbation_delta"], cfg["counterfactual"]["perturbation_delta"]
                        )
                        action[correction_active] += delta[correction_active]
                    elif candidate == "bounded_joint_group":
                        for local in torch.nonzero(correction_active).flatten().cpu().tolist():
                            group, side, sign = assignment[local]
                            indices = [
                                index
                                for index, name in enumerate(joint_names)
                                if group in name and side in name
                            ]
                            if indices:
                                action[local, indices] += sign * cfg["counterfactual"]["perturbation_delta"]
                    action.clamp_(-1.0, 1.0)
                    full_action = background
                    full_action[selected] = action
                    with torch.no_grad():
                        _, _, dones, info = wrapped.step(full_action)
                    speed = robot.data.root_lin_vel_b.torch[selected, 0]
                    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :][selected]
                    contacts = forces.norm(dim=-1).amax(1) > 5
                    stable_contact = torch.where(contacts.any(1), stable_contact + dt, torch.zeros_like(stable_contact))
                    flight_dwell = torch.where(~contacts.any(1), flight_dwell + dt, torch.zeros_like(flight_dwell))
                    cycle_terminated |= (stable_contact >= 0.12) & (step * dt >= 0.20)
                    timeout = info.get("time_outs", torch.zeros_like(dones)).bool()[selected]
                    fall = dones.bool()[selected] & ~timeout
                    foot_speed = robot.data.body_lin_vel_w.torch[selected][:, feet, :2].norm(dim=-1)
                    slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(1) > 0.8
                    effort = robot.data.applied_torque.torch[selected][:, joints].abs() / robot.data.joint_effort_limits.torch[selected][:, joints].abs().clamp_min(1e-6)
                    saturation = (effort >= 0.95).any(1)
                    impact = forces[:, :, :, 2].abs().mean(1).amax(1) > 3500
                    heading_ok = heading_error.abs() <= 0.12
                    valid = (
                        ((speed - 1.2).abs() <= 0.2)
                        & heading_ok
                        & cycle_terminated
                        & contacts.any(1)
                        & (flight_dwell <= 0.16)
                        & ~fall
                        & (speed >= -0.1)
                        & ~slip
                        & ~saturation
                        & ~impact
                    )
                    walk_streak = torch.where(valid, walk_streak + 1, torch.zeros_like(walk_streak))
                    maximum_streak = torch.maximum(maximum_streak, walk_streak)
                    unsafe |= correction_active & (fall | slip | saturation | impact | ~heading_ok)
                    post_steps += correction_active.long()
                    previous_contacts = contacts
                    state.previous_action.copy_(full_action)

                current_state = {"root": branch_root, "joint": branch_joint, "velocity": branch_velocity}
                if prebranch_reference is None:
                    prebranch_reference = current_state
                else:
                    common = sorted(set(prebranch_reference["root"]) & set(branch_root))
                    root_error = max(
                        (float(np.max(np.abs(prebranch_reference["root"][key] - branch_root[key]))) for key in common),
                        default=float("inf"),
                    )
                    joint_error = max(
                        (float(np.max(np.abs(prebranch_reference["joint"][key] - branch_joint[key]))) for key in common),
                        default=float("inf"),
                    )
                    velocity_error = max(
                        (float(np.max(np.abs(prebranch_reference["velocity"][key] - branch_velocity[key]))) for key in common),
                        default=float("inf"),
                    )
                    matching.append({
                        "candidate": candidate,
                        "matched_envs": len(common),
                        "root_position_max_error_m": root_error,
                        "joint_position_max_error_rad": joint_error,
                        "velocity_max_error": velocity_error,
                    })
                successful = (maximum_streak >= 20) & ~unsafe
                for local in range(cohort):
                    group, side, sign = assignment[local]
                    candidate_results.append({
                        "candidate": candidate,
                        "physical_env_id": int(selected[local]),
                        "source_speed_mps": float(state.source_speed[selected][local]),
                        "launch_phase": {0: "flight", 1: "left_support", 2: "right_support", 3: "double_support"}[int(state.launch_phase[local])],
                        "branch_age": int(branch_age[local]),
                        "branch_step": int(branch_step[local]),
                        "joint_group": group if candidate == "bounded_joint_group" else "not_applicable",
                        "side": side if candidate == "bounded_joint_group" else "not_applicable",
                        "perturbation_sign": sign if candidate == "bounded_joint_group" else 0.0,
                        "maximum_walk_valid_streak": int(maximum_streak[local]),
                        "contract_20_step_success": bool(successful[local]),
                        "unsafe": bool(unsafe[local]),
                    })
                if args.counterfactual_candidate:
                    keys = sorted(branch_root)
                    np.savez_compressed(
                        OUT / f"prebranch_{candidate}.npz",
                        physical_env_id=np.asarray(keys, dtype=np.int64),
                        root_position=np.asarray([branch_root[key] for key in keys], dtype=np.float32),
                        joint_position=np.asarray([branch_joint[key] for key in keys], dtype=np.float32),
                        velocity=np.asarray([branch_velocity[key] for key in keys], dtype=np.float32),
                    )
                print(f"[exp008 counterfactual] candidate={candidate} safe_success={int(successful.sum())}/{cohort}", flush=True)

            result_frame = pd.DataFrame(candidate_results)
            result_frame.to_csv(
                OUT
                / (
                    f"counterfactual_raw_{args.counterfactual_candidate}.csv"
                    if args.counterfactual_candidate
                    else "counterfactual_episode_results.csv"
                ),
                index=False,
            )
            summary = {}
            for candidate, group in result_frame.groupby("candidate"):
                summary[candidate] = {
                    "branch_states": int(len(group)),
                    "safe_contract_successes": int(group["contract_20_step_success"].sum()),
                    "safe_contract_success_rate": float(group["contract_20_step_success"].mean()),
                    "unsafe_rate": float(group["unsafe"].mean()),
                    "maximum_streak_mean": float(group["maximum_walk_valid_streak"].mean()),
                    "maximum_streak_p95": float(group["maximum_walk_valid_streak"].quantile(0.95)),
                }
            dump("prebranch_state_matching.json", {
                "method": "same physical env IDs, full seeded reset/source-route/action replay; no state copy",
                "state_copy": False,
                "comparisons": matching,
                "tolerances": {
                    "root_position_m": cfg["counterfactual"]["state_match_root_position_tolerance_m"],
                    "joint_position_rad": cfg["counterfactual"]["state_match_joint_position_tolerance_rad"],
                    "velocity": cfg["counterfactual"]["state_match_velocity_tolerance"],
                },
                "all_within_tolerance": all(
                    item["root_position_max_error_m"] <= cfg["counterfactual"]["state_match_root_position_tolerance_m"]
                    and item["joint_position_max_error_rad"] <= cfg["counterfactual"]["state_match_joint_position_tolerance_rad"]
                    and item["velocity_max_error"] <= cfg["counterfactual"]["state_match_velocity_tolerance"]
                    for item in matching
                ),
            })
            dump("counterfactual_results.json", {
                "summary": summary,
                "rollout_steps_after_branch": cfg["counterfactual"]["rollout_steps_after_branch"],
                "state_copy": False,
                "production_capability_claim": False,
            })
            per_phase = {}
            for (candidate, phase), group in result_frame.groupby(["candidate", "launch_phase"]):
                per_phase[f"{candidate}:{phase}"] = {
                    "count": int(len(group)),
                    "safe_success_rate": float(group["contract_20_step_success"].mean()),
                    "unsafe_rate": float(group["unsafe"].mean()),
                }
            dump("per_phase_corrective_results.json", per_phase)
            dump("action_candidates.json", {
                "baseline": "RunToWalk model_10 deterministic mean action",
                "walk_expert": "frozen WALK expert action",
                "run_expert": "frozen RUN_LOW expert action",
                "bounded_joint_group": {
                    "groups": cfg["counterfactual"]["joint_groups"],
                    "sides": ["left", "right"],
                    "delta": cfg["counterfactual"]["perturbation_delta"],
                    "signs": [-1, 1],
                },
                "target_walk_alignment": "baseline plus per-joint delta toward WALK action, bounded by configured delta",
                "action_limit": [-1, 1],
            })
            dump("counterfactual_protocol.json", {
                "branch_streak_ages": cfg["counterfactual"]["branch_streak_ages"],
                "same_reset_seed": replay_seed,
                "same_source_route": True,
                "same_actions_before_branch": True,
                "same_physical_env_ids": True,
                "state_copy": False,
                "rollout_steps_after_branch": cfg["counterfactual"]["rollout_steps_after_branch"],
            })
        if not args.counterfactual_only:
            # Isaac Sim teardown can terminate the Windows launcher before code
            # following the context manager runs.  Seal compact metadata while
            # the application is still alive and all parquet parts are durable.
            from finalize_dataset_metadata import main as finalize_dataset_metadata

            finalize_dataset_metadata()
        wrapped.close()

    if args.counterfactual_only:
        return
    with (OUT / "sequence_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_index[0]))
        writer.writeheader()
        writer.writerows(episode_index)
    split_mapping = {
        row["episode_id"]: group_split[(int(row["evaluation_seed"]), round(float(row["source_speed_mps"]), 1))]
        for row in episode_index
    }
    split_counts = Counter(split_mapping.values())
    dump("split_manifest.json", {
        "unit": "episode grouped by evaluation_seed/source_speed/checkpoint",
        "fractions": cfg["split"],
        "episode_counts": dict(split_counts),
        "episode_mapping": split_mapping,
        "step_random_split": False,
        "same_reset_seed_cross_split": False,
    })
    dump("dataset_manifest.json", {
        "source": "diagnostic replay required because exp_007 saved trajectories lacked complete 152D/action sequences",
        "checkpoint": cfg["checkpoint"],
        "checkpoint_sha256": CHECKPOINT_SHA,
        "episodes": len(episode_index),
        "rows": total_rows,
        "successful_20_step_segments": success_count,
        "failed_segments": len(episode_index) - success_count,
        "success_target": cfg["dataset"]["successful_segment_target"],
        "failure_target": cfg["dataset"]["failed_segment_target"],
        "success_target_met": success_count >= cfg["dataset"]["successful_segment_target"],
        "failure_target_met": len(episode_index) - success_count >= cfg["dataset"]["failed_segment_target"],
        "source_speed_counts": {str(key): value for key, value in speed_counter.items()},
        "source_phase_counts": dict(phase_counter),
        "break_reason_counts": dict(reason_counter),
        "source_preparation_attempts": source_attempts,
        "source_ready_selected": source_success,
        "pre_window_steps": cfg["dataset"]["pre_window_steps"],
        "post_window_steps": cfg["dataset"]["post_window_steps"],
        "ppo_training": 0,
        "ppo_optimizer_updates": 0,
        "transition_actor_optimizer_updates": 0,
    })


if __name__ == "__main__":
    main()
