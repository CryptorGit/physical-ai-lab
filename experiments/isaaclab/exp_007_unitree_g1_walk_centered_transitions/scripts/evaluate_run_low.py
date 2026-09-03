"""Stage 6 frozen RUN_LOW steady-state preflight, audit, and formal evaluation."""

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

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [
    str(EXP / "src"),
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_006_unitree_g1_command_skills/src"),
]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_command_skills.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_run_expert  # noqa: E402
from g1_walk_centered.experts.adapters import (  # noqa: E402
    canonical_state_from_legacy_observation,
    to_run_observation,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

CHECKPOINT_REL = (
    "logs/rsl_rl/physical_ai_g1_command_skills/"
    "2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt"
)
EXPECTED_SHA = "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"
SPEEDS = (2.4, 2.6, 2.8, 3.0)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("preflight", "audit", "formal"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--episodes-per-speed", type=int, default=4)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    p = (len(ordered) - 1) * q / 100.0
    lo, hi = int(p), min(int(p) + 1, len(ordered) - 1)
    return ordered[lo] * (1.0 - (p - lo)) + ordered[hi] * (p - lo)


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def periodic(stats: dict) -> bool:
    durations = stats["flight_durations"]
    return (
        len(durations) >= 4
        and stats["max_consecutive_safe_cycles"] >= 3
        and stats["alternating_landings"] / max(stats["alternation_opportunities"], 1) >= 0.80
        and stats["valid_landings"] / max(len(durations), 1) >= 0.80
        and 0.04 <= mean(durations) <= 0.16
    )


def fresh_stats() -> dict:
    return {
        "flight_durations": [], "landing_sides": [], "landing_impacts": [],
        "valid_landings": 0, "alternating_landings": 0, "alternation_opportunities": 0,
        "consecutive_safe_cycles": 0, "max_consecutive_safe_cycles": 0,
        "flight_steps": 0, "single_steps": 0, "double_steps": 0,
        "speed": [], "speed_error": [], "lateral_speed": [], "heading": [], "yaw_rate": [],
        "path_drift": [], "slip_left": [], "slip_right": [], "pelvis_height": [],
        "roll": [], "pitch": [], "action_magnitude": [], "action_rate": [],
        "ankle_pitch_effort": [], "ankle_roll_effort": [], "knee_velocity": [],
        "previous_action_mismatches": 0, "run_contribution": [],
        "turn_contribution": [], "stop_contribution": [], "transition_contribution": [],
        "scripted_offset": [],
    }


def main() -> None:
    output = (REPO / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = (REPO / CHECKPOINT_REL).resolve(strict=True)
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash != EXPECTED_SHA:
        raise RuntimeError(f"RUN checkpoint SHA mismatch: {checkpoint_hash}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor_state = payload["actor_state_dict"]
    protected = {
        key: tensor_sha(value)
        for key, value in actor_state.items()
        if key.startswith((
            "base_mlp.", "skill_command_encoders.0.", "skill_state_adapters.0.",
            "residual_heads.0.",
        ))
    }
    expert = load_run_expert(checkpoint)
    zero = torch.zeros(1, 123)
    zero[:, 8] = -1.0
    canonical = canonical_state_from_legacy_observation(zero, heading_w_rad=torch.zeros(1))
    command = MotionCommand(torch.tensor([2.6]), torch.zeros(1), target_yaw_rate_radps=torch.zeros(1))
    obs152 = to_run_observation(canonical, command, route="RUN")
    components = expert.action_components(canonical, command, route="RUN")
    preflight = {
        "checkpoint": CHECKPOINT_REL, "checkpoint_sha256": checkpoint_hash,
        "checkpoint_exists": True, "strict_actor_load": True,
        "legacy_observation_dimension": 123, "command_observation_dimension": 29,
        "observation_dimension": int(obs152.shape[-1]), "action_dimension": int(components["action_mean"].shape[-1]),
        "action_scale": 0.5,
        "action_semantics": "default_joint_position + 0.5 * normalized_position_action",
        "observation_finite": bool(torch.isfinite(obs152).all()),
        "action_finite": bool(torch.isfinite(components["action_mean"]).all()),
        "run_route_one_hot": float(obs152[0, 123]) == 1.0 and float(obs152[0, 129]) == 1.0,
        "turn_command_zero": float(obs152[0, 125]) == 0.0,
        "stop_route_zero": float(obs152[0, 124]) == 0.0,
        "transition_route_zero": True, "scripted_offset_zero": True,
        "protected_tensor_hashes": protected, "checkpoint_weight_changed": False,
        "candidate_b_production_loaded": False,
    }
    (output / "run_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if args.mode == "preflight":
        print(json.dumps(preflight, indent=2))
        return

    count = args.episodes_per_speed
    speeds = [speed for speed in SPEEDS for _ in range(count)]
    n = len(speeds)
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 18.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent.clip_actions,
        )
        env = wrapped.unwrapped
        device = env.device
        expert = load_run_expert(checkpoint, device=device)
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        foot_body_ids, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        ankles_pitch, _ = robot.find_joints(".*_ankle_pitch_joint")
        ankles_roll, _ = robot.find_joints(".*_ankle_roll_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        all_joints, joint_names = robot.find_joints(".*")
        target = torch.tensor(speeds, device=device)
        wrapped.reset()
        heading = robot.data.heading_w.torch.clone()
        previous_action = torch.zeros(n, 37, device=device)
        phase = torch.zeros(n, dtype=torch.long, device=device)  # 0 acquisition, 1 hold, 2 done
        elapsed = torch.zeros(n, device=device)
        ramp_duration = 1.2 + 0.6 * torch.rand(n, generator=torch.Generator(device=device).manual_seed(args.seed + 19), device=device)
        contract_streak = torch.zeros(n, device=device)
        hold_elapsed = torch.zeros(n, device=device)
        formal_start = torch.zeros(n, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        acquisition_success = torch.zeros(n, dtype=torch.bool, device=device)
        acquisition_fall = torch.zeros(n, dtype=torch.bool, device=device)
        hold_fall = torch.zeros(n, dtype=torch.bool, device=device)
        torso_seen = torch.zeros(n, dtype=torch.bool, device=device)
        long_sat = torch.zeros(n, dtype=torch.bool, device=device)
        dangerous_slip = torch.zeros(n, dtype=torch.bool, device=device)
        action_discontinuity = torch.zeros(n, dtype=torch.bool, device=device)
        velocity_dwell = torch.zeros(n, len(all_joints), device=device)
        effort_dwell = torch.zeros(n, len(all_joints), device=device)
        max_velocity_dwell = torch.zeros_like(velocity_dwell)
        max_effort_dwell = torch.zeros_like(effort_dwell)
        slip_dwell = torch.zeros(n, device=device)
        stats = [fresh_stats() for _ in range(n)]
        acquisition_stats = [fresh_stats() for _ in range(n)]
        previous_contacts = [(False, False) for _ in range(n)]
        in_flight = [False] * n
        flight_start = [0.0] * n
        origin = robot.data.root_pos_w.torch[:, :2].clone()
        dt = float(env.step_dt)
        cycle_rows: list[dict] = []
        flight_rows: list[dict] = []

        def update_gait(i: int, contacts: tuple[bool, bool], robust_impact: list[float], collection: dict) -> None:
            support = int(contacts[0]) + int(contacts[1])
            if support == 0:
                collection["flight_steps"] += 1
                if not in_flight[i]:
                    in_flight[i], flight_start[i] = True, float(elapsed[i])
            elif support == 1:
                collection["single_steps"] += 1
            else:
                collection["double_steps"] += 1
            if in_flight[i] and support > 0:
                duration = float(elapsed[i]) - flight_start[i]
                new = [side for side in range(2) if contacts[side] and not previous_contacts[i][side]]
                valid = len(new) == 1
                side = new[0] if valid else -1
                collection["flight_durations"].append(duration)
                collection["landing_impacts"].append(max(robust_impact))
                if valid:
                    collection["valid_landings"] += 1
                    if collection["landing_sides"]:
                        collection["alternation_opportunities"] += 1
                        if side != collection["landing_sides"][-1]:
                            collection["alternating_landings"] += 1
                    safe = 0.04 <= duration <= 0.16 and (
                        not collection["landing_sides"] or side != collection["landing_sides"][-1]
                    )
                    collection["consecutive_safe_cycles"] = collection["consecutive_safe_cycles"] + 1 if safe else 0
                    collection["max_consecutive_safe_cycles"] = max(
                        collection["max_consecutive_safe_cycles"], collection["consecutive_safe_cycles"]
                    )
                    collection["landing_sides"].append(side)
                else:
                    collection["consecutive_safe_cycles"] = 0
                flight_rows.append({
                    "seed": args.seed, "candidate_episode": i, "phase": "RUN_LOW" if int(phase[i]) == 1 else "ACQUISITION",
                    "target_speed_mps": speeds[i], "duration_s": duration,
                    "valid_landing": valid, "landing_side": "L" if side == 0 else "R" if side == 1 else "INVALID",
                    "impact_n": max(robust_impact),
                })
                in_flight[i] = False
            previous_contacts[i] = contacts

        for _step in range(round(17.0 / dt)):
            current_phase = phase.clone()
            progress = (elapsed / ramp_duration).clamp(0.0, 1.0)
            s = 10 * progress**3 - 15 * progress**4 + 6 * progress**5
            command_vx = torch.where(current_phase == 0, target * s, target)
            heading_error_signed = torch.atan2(
                torch.sin(heading - robot.data.heading_w.torch),
                torch.cos(heading - robot.data.heading_w.torch),
            )
            # Preserve the formal exp_006 RUN command path exactly.  RUN uses
            # the environment command term's 1.5 * heading-error mapping and
            # [-1.5, 1.5] legacy yaw-rate range; WALK's lower-bandwidth
            # controller is a different expert contract.
            yaw_command = (1.5 * heading_error_signed).clamp(-1.5, 1.5)
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = command_vx
            term.vel_command_b[:, 2] = yaw_command
            legacy = wrapped.get_observations()["policy"]
            previous_match = (legacy[:, 86:123] == previous_action).all(dim=1)
            canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            motion = MotionCommand(command_vx, heading, target_yaw_rate_radps=yaw_command)
            with torch.inference_mode():
                comp = expert.action_components(canonical, motion, route="RUN")
                action = comp["action_mean"]
                action[finished] = previous_action[finished]
                _, _, dones, infos = wrapped.step(action)
            action_rate = torch.linalg.vector_norm(action - previous_action, dim=1) / dt
            action_jump = torch.linalg.vector_norm(action - previous_action, dim=1)
            previous_action[:] = action
            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts_t = forces.norm(dim=-1).amax(dim=1) > 5.0
            robust_vertical = forces[:, :, :, 2].abs().mean(dim=1)
            foot_speed = robot.data.body_lin_vel_w.torch[:, foot_body_ids, :2].norm(dim=-1)
            slip = torch.where(contacts_t, foot_speed, torch.zeros_like(foot_speed))
            vel_ratio = robot.data.joint_vel.torch[:, all_joints].abs() / robot.data.joint_vel_limits.torch[:, all_joints].abs().clamp_min(1e-6)
            effort_ratio = robot.data.applied_torque.torch[:, all_joints].abs() / robot.data.joint_effort_limits.torch[:, all_joints].abs().clamp_min(1e-6)
            velocity_dwell = torch.where((vel_ratio >= .95) & (current_phase == 1).unsqueeze(1), velocity_dwell + dt, torch.zeros_like(velocity_dwell))
            effort_dwell = torch.where((effort_ratio >= .95) & (current_phase == 1).unsqueeze(1), effort_dwell + dt, torch.zeros_like(effort_dwell))
            max_velocity_dwell = torch.maximum(max_velocity_dwell, velocity_dwell)
            max_effort_dwell = torch.maximum(max_effort_dwell, effort_dwell)
            long_sat |= (max_velocity_dwell >= .05).any(dim=1) | (max_effort_dwell >= .20).any(dim=1)
            slip_dwell = torch.where((slip.amax(dim=1) > .8) & (current_phase == 1), slip_dwell + dt, torch.zeros_like(slip_dwell))
            dangerous_slip |= slip_dwell >= .20
            torso = env.termination_manager.get_term("base_contact").bool()
            speed = robot.data.root_lin_vel_b.torch[:, 0]
            lateral = robot.data.root_lin_vel_b.torch[:, 1].abs()
            heading_abs = heading_error_signed.abs()
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2]).abs()
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2)).abs()
            timed_out = infos.get("time_outs", torch.zeros_like(dones)).bool() if isinstance(infos, dict) else torch.zeros_like(dones).bool()
            physical_fall = dones.bool() & (~timed_out)
            for i in range(n):
                if finished[i]:
                    continue
                collection = stats[i] if current_phase[i] == 1 else acquisition_stats[i]
                update_gait(i, (bool(contacts_t[i, 0]), bool(contacts_t[i, 1])), robust_vertical[i].tolist(), collection)
                collection["speed"].append(float(speed[i]))
                collection["speed_error"].append(abs(float(speed[i] - target[i])))
                collection["lateral_speed"].append(float(lateral[i]))
                collection["heading"].append(float(heading_abs[i]))
                collection["yaw_rate"].append(abs(float(robot.data.root_ang_vel_b.torch[i, 2])))
                delta = robot.data.root_pos_w.torch[i, :2] - origin[i]
                lateral_axis = torch.stack((-torch.sin(heading[i]), torch.cos(heading[i])))
                collection["path_drift"].append(abs(float((delta * lateral_axis).sum())))
                collection["slip_left"].append(float(slip[i, 0]))
                collection["slip_right"].append(float(slip[i, 1]))
                collection["pelvis_height"].append(float(robot.data.root_pos_w.torch[i, 2]))
                collection["roll"].append(float(roll[i]))
                collection["pitch"].append(float(pitch[i]))
                collection["action_magnitude"].append(float(torch.linalg.vector_norm(action[i])))
                collection["action_rate"].append(float(action_rate[i]))
                collection["ankle_pitch_effort"].append(float(effort_ratio[i, ankles_pitch].max()))
                collection["ankle_roll_effort"].append(float(effort_ratio[i, ankles_roll].max()))
                collection["knee_velocity"].append(float(vel_ratio[i, knees].max()))
                collection["previous_action_mismatches"] += int(not previous_match[i])
                collection["run_contribution"].append(float(torch.linalg.vector_norm(comp["selected_residual"][i])))
                collection["turn_contribution"].append(0.0)
                collection["stop_contribution"].append(0.0)
                collection["transition_contribution"].append(0.0)
                collection["scripted_offset"].append(0.0)
                if current_phase[i] == 0:
                    if physical_fall[i] or torso[i]:
                        acquisition_fall[i], finished[i], phase[i] = True, True, 2
                    else:
                        good = (
                            periodic(acquisition_stats[i])
                            and abs(float(speed[i] - target[i])) <= .20
                            and float(heading_abs[i]) <= .12
                            and float(roll[i]) <= .20 and float(pitch[i]) <= .20
                        )
                        contract_streak[i] = contract_streak[i] + dt if good else 0.0
                        if contract_streak[i] >= .4:
                            acquisition_success[i] = True
                            phase[i], hold_elapsed[i], formal_start[i] = 1, 0.0, elapsed[i]
                            stats[i] = fresh_stats()
                            origin[i] = robot.data.root_pos_w.torch[i, :2]
                            velocity_dwell[i].zero_(); effort_dwell[i].zero_()
                            max_velocity_dwell[i].zero_(); max_effort_dwell[i].zero_()
                            previous_contacts[i] = (bool(contacts_t[i, 0]), bool(contacts_t[i, 1]))
                            in_flight[i] = False
                        elif elapsed[i] >= 6.0:
                            finished[i], phase[i] = True, 2
                elif current_phase[i] == 1:
                    hold_elapsed[i] += dt
                    hold_fall[i] |= physical_fall[i]
                    torso_seen[i] |= torso[i]
                    action_discontinuity[i] |= action_jump[i] > 6.0
                    if physical_fall[i] or torso[i] or hold_elapsed[i] >= 8.0:
                        finished[i], phase[i] = True, 2
            elapsed += dt
            if bool(finished.all()):
                break

        rows: list[dict] = []
        for i in range(n):
            s = stats[i]
            impacts = s["landing_impacts"]
            impact_p95 = percentile(impacts, 95)
            impact_p99 = percentile(impacts, 99)
            impact_over = mean(value > 3500.0 for value in impacts)
            periodic_ok = periodic(s)
            heading_p95 = percentile(s["heading"], 95)
            speed_error_mean = mean(s["speed_error"])
            impact_failure = impact_p95 > 3500.0 or impact_over > .05
            hold_success = (
                acquisition_success[i] and hold_elapsed[i] >= 7.98 and periodic_ok
                and not hold_fall[i] and not torso_seen[i] and heading_p95 <= .12
                and speed_error_mean <= .20 and not long_sat[i] and not dangerous_slip[i]
                and not impact_failure and not action_discontinuity[i]
            )
            support_steps = max(s["flight_steps"] + s["single_steps"] + s["double_steps"], 1)
            failure = (
                "" if hold_success else
                "run_acquisition_failure" if not acquisition_success[i] else
                "fall" if hold_fall[i] else
                "periodic_running_failure" if not periodic_ok else
                "heading_failure" if heading_p95 > .12 else
                "speed_tracking_failure" if speed_error_mean > .20 else
                "saturation_failure" if long_sat[i] else
                "dangerous_slip" if dangerous_slip[i] else
                "impact_failure" if impact_failure else
                "action_discontinuity" if action_discontinuity[i] else "run_hold_failure"
            )
            row = {
                "seed": args.seed, "episode": i, "target_speed_mps": speeds[i],
                "system_state_at_reset": "UNINITIALIZED_FOR_RUN",
                "formal_run_started": bool(acquisition_success[i]),
                "acquisition_success": bool(acquisition_success[i]),
                "acquisition_time_s": float(formal_start[i]) if acquisition_success[i] else None,
                "acquisition_fall": bool(acquisition_fall[i]), "contract_rejection": not bool(acquisition_success[i]),
                "formal_hold_start_s": float(formal_start[i]) if acquisition_success[i] else None,
                "formal_hold_end_s": float(formal_start[i] + hold_elapsed[i]) if acquisition_success[i] else None,
                "run_hold_success": bool(hold_success), "periodic_running_success": bool(periodic_ok),
                "actual_speed_mean_mps": mean(s["speed"]), "actual_speed_p95_mps": percentile(s["speed"], 95),
                "forward_speed_error_mean_mps": speed_error_mean,
                "forward_speed_error_p95_mps": percentile(s["speed_error"], 95),
                "lateral_speed_mean_mps": mean(s["lateral_speed"]), "lateral_speed_p95_mps": percentile(s["lateral_speed"], 95),
                "heading_error_mean_rad": mean(s["heading"]), "heading_error_p95_rad": heading_p95,
                "yaw_rate_mean_rps": mean(s["yaw_rate"]), "yaw_rate_p95_rps": percentile(s["yaw_rate"], 95),
                "path_drift_max_m": max(s["path_drift"], default=0.0),
                "flight_fraction": s["flight_steps"] / support_steps,
                "flight_events": len(s["flight_durations"]), "safe_cycle_count": s["max_consecutive_safe_cycles"],
                "maximum_consecutive_safe_cycles": s["max_consecutive_safe_cycles"],
                "alternating_landing_ratio": s["alternating_landings"] / max(s["alternation_opportunities"], 1),
                "valid_landing_ratio": s["valid_landings"] / max(len(s["flight_durations"]), 1),
                "mean_flight_duration_s": mean(s["flight_durations"]),
                "single_support_fraction": s["single_steps"] / support_steps,
                "double_support_fraction": s["double_steps"] / support_steps,
                "left_slip_mean_mps": mean(s["slip_left"]), "right_slip_mean_mps": mean(s["slip_right"]),
                "dangerous_slip_failure": bool(dangerous_slip[i]),
                "impact_mean_n": mean(impacts), "impact_p95_n": impact_p95, "impact_p99_n": impact_p99,
                "impact_max_n": max(impacts, default=0.0), "impact_over_3500_rate": impact_over,
                "impact_failure": impact_failure,
                "pelvis_height_range_m": max(s["pelvis_height"], default=0.0) - min(s["pelvis_height"], default=0.0),
                "roll_p95_rad": percentile(s["roll"], 95), "pitch_p95_rad": percentile(s["pitch"], 95),
                "fall": bool(hold_fall[i]), "torso_contact": bool(torso_seen[i]),
                "action_magnitude_mean": mean(s["action_magnitude"]), "action_rate_p95": percentile(s["action_rate"], 95),
                "action_discontinuity_failure": bool(action_discontinuity[i]),
                "long_dwell_saturation_failure": bool(long_sat[i]),
                "max_joint_velocity_dwell_s": float(max_velocity_dwell[i].max()),
                "max_joint_effort_dwell_s": float(max_effort_dwell[i].max()),
                "knee_velocity_utilization_p95": percentile(s["knee_velocity"], 95),
                "ankle_pitch_effort_utilization_p95": percentile(s["ankle_pitch_effort"], 95),
                "ankle_roll_effort_utilization_p95": percentile(s["ankle_roll_effort"], 95),
                "run_contribution_norm_mean": mean(s["run_contribution"]),
                "turn_contribution_norm": 0.0, "stop_contribution_norm": 0.0,
                "transition_contribution_norm": 0.0, "scripted_offset_norm": 0.0,
                "simultaneous_controller_count": 0,
                "previous_action_mismatch_steps": s["previous_action_mismatches"],
                "failure_class": failure,
            }
            rows.append(row)
            cycle_rows.append({
                "seed": args.seed, "episode": i, "target_speed_mps": speeds[i],
                "flight_events": row["flight_events"], "safe_cycles": row["safe_cycle_count"],
                "alternating_landing_ratio": row["alternating_landing_ratio"],
                "valid_landing_ratio": row["valid_landing_ratio"],
                "mean_flight_duration_s": row["mean_flight_duration_s"],
            })
        summary = {
            "mode": args.mode, "seed": args.seed, "episodes": n,
            "episodes_per_speed": count,
            "acquisition_success_rate": mean(r["acquisition_success"] for r in rows),
            "run_hold_success_rate": mean(r["run_hold_success"] for r in rows),
            "periodic_running_success_rate": mean(r["periodic_running_success"] for r in rows),
            "fall_rate": mean(r["fall"] for r in rows),
            "heading_error_p95_rad": percentile([r["heading_error_p95_rad"] for r in rows], 95),
            "forward_speed_error_mean_mps": mean(r["forward_speed_error_mean_mps"] for r in rows),
            "long_dwell_saturation_failure_rate": mean(r["long_dwell_saturation_failure"] for r in rows),
            "dangerous_slip_failure_rate": mean(r["dangerous_slip_failure"] for r in rows),
            "impact_failure_rate": mean(r["impact_failure"] for r in rows),
            "action_discontinuity_failure_rate": mean(r["action_discontinuity_failure"] for r in rows),
            "previous_action_mismatch_count": sum(r["previous_action_mismatch_steps"] for r in rows),
            "checkpoint_sha256": checkpoint_hash,
        }
        write_csv(output / f"{args.label}_episodes.csv", rows)
        write_csv(output / f"{args.label}_cycle_metrics.csv", cycle_rows)
        write_csv(output / f"{args.label}_flights.csv", flight_rows)
        (output / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
