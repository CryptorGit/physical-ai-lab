"""Evaluate the three-controller Stage 3 STAND_TO_WALK hard-switch edge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import gymnasium as gym
import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src")]
import isaaclab_tasks  # noqa: E402,F401
import g1_flat_run.tasks  # noqa: E402,F401
import g1_walk_centered.tasks  # noqa: E402,F401
from g1_walk_centered.command_contract import MotionCommand  # noqa: E402
from g1_walk_centered.experts import load_walk_expert  # noqa: E402
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

SPEEDS = (0.6, 0.8, 1.0, 1.2)
FAILURES = (
    "source_stand_settle_failure", "transition_start_failure", "walk_liftoff_failure",
    "walk_cycle_acquisition_failure", "target_speed_failure", "heading_failure",
    "path_drift_failure", "action_discontinuity_at_entry", "action_discontinuity_at_exit",
    "ankle_saturation_failure", "knee_saturation_failure", "dangerous_slip",
    "excessive_flight", "transition_timeout", "false_completion", "walk_takeover_failure",
    "fall", "torso_contact", "routing_failure",
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stand-checkpoint", required=True)
parser.add_argument("--walk-checkpoint", required=True)
parser.add_argument("--transition-checkpoint")
parser.add_argument("--mode", choices=("baseline", "pilot", "formal", "smoke"), required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def avg(values):
    return sum(values) / len(values) if values else 0.0


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def runs(values, dt):
    result, current = [], 0
    for active in values:
        if active:
            current += 1
        elif current:
            result.append(current * dt)
            current = 0
    if current:
        result.append(current * dt)
    return result


def minimum_jerk(u: torch.Tensor) -> torch.Tensor:
    u = u.clamp(0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def assignments():
    if args.mode == "smoke":
        return list(SPEEDS)
    if args.mode in ("baseline", "pilot"):
        return [speed for speed in SPEEDS for _ in range(8)]
    return [0.6] * 13 + [0.8] * 13 + [1.0] * 12 + [1.2] * 12


def main() -> None:
    stand_path = Path(args.stand_checkpoint).resolve(strict=True)
    walk_path = Path(args.walk_checkpoint).resolve(strict=True)
    transition_path = Path(args.transition_checkpoint).resolve(strict=True) if args.transition_checkpoint else None
    if args.mode != "baseline" and transition_path is None:
        raise ValueError("transition checkpoint is required outside baseline mode")
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = output / "hard_switch_baseline.json"
    speeds = assignments()
    n = len(speeds)
    rng = random.Random(args.seed)
    stand_holds = [rng.uniform(0.8, 1.8) for _ in speeds]
    ramp_durations = [rng.uniform(1.3, 1.7) for _ in speeds]
    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 14.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg), clip_actions=agent_cfg.clip_actions)
        env = wrapped.unwrapped
        stand = load_walk_expert(stand_path, device=env.device)
        walk = load_walk_expert(walk_path, device=env.device)
        transition = load_walk_expert(transition_path, device=env.device) if transition_path else walk
        robot = env.scene["robot"]
        term = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        torso_ids, _ = robot.find_bodies("torso_link")
        sensor_torso = [sensor.body_names.index(robot.body_names[index]) for index in torso_ids]
        ankles, _ = robot.find_joints(".*_ankle_pitch_joint")
        knees, _ = robot.find_joints(".*_knee_joint")
        wrapped.reset()
        device = env.device
        dt = float(env.step_dt)
        target_speed = torch.tensor(speeds, device=device)
        target_heading = robot.data.heading_w.torch.clone()
        path_origin = robot.data.root_pos_w.torch[:, :2].clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)  # 0 settle, 1 hold, 2 edge, 3 takeover, 4 terminal
        phase_elapsed = torch.zeros(n, device=device)
        settle_streak = torch.zeros(n, device=device)
        completion_streak = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        filtered_yaw = torch.zeros(n, device=device)
        failed = torch.zeros(n, dtype=torch.bool, device=device)
        fallen = torch.zeros(n, dtype=torch.bool, device=device)
        source_ok = torch.zeros(n, dtype=torch.bool, device=device)
        completed = torch.zeros(n, dtype=torch.bool, device=device)
        takeover_ok = torch.zeros(n, dtype=torch.bool, device=device)
        previous_action = torch.zeros(n, 37, device=device)
        last_stand_action = torch.zeros_like(previous_action)
        last_transition_action = torch.zeros_like(previous_action)
        entry_jump = torch.zeros(n, device=device)
        exit_jump = torch.zeros(n, device=device)
        entry_joint_jump = torch.zeros(n, device=device)
        exit_joint_jump = torch.zeros(n, device=device)
        traces = [
            {key: [] for key in (
                "phase", "vx", "heading", "cross", "roll", "pitch", "support", "slip",
                "ankle", "knee", "flight", "action_rate", "torque_spike",
            )}
            for _ in speeds
        ]
        timeline_rows = []
        for step in range(round(13.5 / dt)):
            active = ~failed
            command_vx = torch.zeros(n, device=device)
            edge = phase == 2
            takeover = phase == 3
            ramp = torch.tensor(ramp_durations, device=device)
            command_vx[edge] = target_speed[edge] * minimum_jerk(phase_elapsed[edge] / ramp[edge])
            if args.mode == "baseline":
                command_vx[edge] = target_speed[edge]
            command_vx[takeover] = target_speed[takeover]
            error_signed = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw_yaw = (0.8 * error_signed - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw_yaw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            filtered_yaw[phase < 2] = 0.0
            term.vel_command_b.zero_()
            term.vel_command_b[:, 0] = command_vx
            term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            zero = torch.zeros(n, device=device)
            with torch.inference_mode():
                stand_action = stand(state, MotionCommand(zero, target_heading, target_yaw_rate_radps=zero))
                transition_action = transition(
                    state, MotionCommand(command_vx, target_heading, target_yaw_rate_radps=filtered_yaw)
                )
                walk_action = walk(
                    state, MotionCommand(target_speed, target_heading, target_yaw_rate_radps=filtered_yaw)
                )
                action = torch.where(
                    (phase < 2).unsqueeze(1),
                    stand_action,
                    torch.where(edge.unsqueeze(1), transition_action, walk_action),
                )
                _, _, dones, _ = wrapped.step(action)
            forces = sensor.data.net_forces_w_history.torch
            contacts = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            torso_contact = forces[:, :, sensor_torso, :].norm(dim=-1).amax(dim=(1, 2)) > 5.0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            slip = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            ankle_ratio = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            knee_ratio = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            torque_ratio = (
                robot.data.applied_torque.torch.abs()
                / robot.data.joint_effort_limits.torch.abs().clamp_min(1.0e-6)
            ).amax(dim=1)
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2])
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            heading = error_signed.abs()
            displacement = robot.data.root_pos_w.torch[:, :2] - path_origin
            normal = torch.stack((-torch.sin(target_heading), torch.cos(target_heading)), dim=1)
            cross = (displacement * normal).sum(dim=1).abs()
            contact_slip = torch.where(contacts, slip, torch.zeros_like(slip)).amax(dim=1)
            action_delta = action - previous_action
            action_rate = torch.linalg.vector_norm(action_delta, dim=1) / dt
            action_joint = action_delta.abs().amax(dim=1)
            started_entry = (phase == 2) & (phase_elapsed <= dt * 1.5)
            entry_jump[started_entry] = torch.linalg.vector_norm(action[started_entry] - last_stand_action[started_entry], dim=1)
            entry_joint_jump[started_entry] = (action[started_entry] - last_stand_action[started_entry]).abs().amax(dim=1)
            started_exit = (phase == 3) & (phase_elapsed <= dt * 1.5)
            exit_jump[started_exit] = torch.linalg.vector_norm(action[started_exit] - last_transition_action[started_exit], dim=1)
            exit_joint_jump[started_exit] = (action[started_exit] - last_transition_action[started_exit]).abs().amax(dim=1)
            last_stand_action[:] = torch.where((phase < 2).unsqueeze(1), action, last_stand_action)
            last_transition_action[:] = torch.where(edge.unsqueeze(1), action, last_transition_action)
            previous_action[:] = action

            safe_stand = (
                (horizontal <= 0.08) & (vertical <= 0.05) & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10) & contacts.all(dim=1)
            )
            settle_streak[:] = torch.where((phase == 0) & safe_stand, settle_streak + dt, torch.zeros_like(settle_streak))
            to_hold = (phase == 0) & (settle_streak >= 0.4)
            phase[to_hold] = 1
            phase_elapsed[to_hold] = 0.0
            source_ok[to_hold] = True
            hold_done = (phase == 1) & (phase_elapsed >= torch.tensor(stand_holds, device=device))
            phase[hold_done] = 2
            phase_elapsed[hold_done] = 0.0
            target_heading[hold_done] = robot.data.heading_w.torch[hold_done]
            path_origin[hold_done] = robot.data.root_pos_w.torch[hold_done, :2]
            previous_support[hold_done] = support[hold_done]
            switched = (phase == 2) & (support != previous_support) & (support != 0)
            support_switches += switched.long()
            previous_support[:] = torch.where(phase == 2, support, previous_support)
            good = (
                (phase == 2) & (vx >= 0.75 * target_speed) & ((vx - target_speed).abs() <= 0.20)
                & (heading <= 0.12) & (roll.abs() <= 0.20) & (pitch.abs() <= 0.20)
                & (support_switches >= 2)
            )
            completion_streak[:] = torch.where(good, completion_streak + dt, torch.zeros_like(completion_streak))
            to_walk = (phase == 2) & (completion_streak >= 0.4)
            phase[to_walk] = 3
            phase_elapsed[to_walk] = 0.0
            completed[to_walk] = True
            takeover_done = (phase == 3) & (phase_elapsed >= 3.0)
            takeover_ok[takeover_done] = True
            phase[takeover_done] = 4
            failed[takeover_done] = True
            source_timeout = (phase == 0) & (env.episode_length_buf.float() * dt >= 2.0)
            transition_timeout = (phase == 2) & (phase_elapsed >= 4.0)
            unsafe_done = dones.bool() | torso_contact
            fallen |= dones.bool()
            failed |= source_timeout | transition_timeout | unsafe_done
            phase[failed & (phase != 4)] = 4
            for i, trace in enumerate(traces):
                if not active[i]:
                    continue
                for key, value in (
                    ("phase", int(phase[i])), ("vx", float(vx[i])), ("heading", float(heading[i])),
                    ("cross", float(cross[i])), ("roll", float(roll[i])), ("pitch", float(pitch[i])),
                    ("support", int(support[i])), ("slip", float(contact_slip[i])),
                    ("ankle", float(ankle_ratio[i])), ("knee", float(knee_ratio[i])),
                    ("flight", not bool(contacts[i].any())), ("action_rate", float(action_rate[i])),
                    ("torque_spike", float(torque_ratio[i])),
                ):
                    trace[key].append(value)
                timeline_rows.append({
                    "episode": i, "time_s": step * dt, "controller_state": int(phase[i]),
                    "target_speed_mps": speeds[i], "command_vx_mps": float(command_vx[i]),
                    "actual_vx_mps": float(vx[i]), "heading_error_rad": float(heading[i]),
                    "support_state": int(support[i]), "completion_streak_s": float(completion_streak[i]),
                    "support_switches": int(support_switches[i]), "action_rate_l2_per_s": float(action_rate[i]),
                })
            phase_elapsed += dt
            if bool(failed.all()):
                break

        if args.mode == "baseline":
            direct = [float(value) for value in entry_jump]
            steady_stand = [value for trace in traces for value, p in zip(trace["action_rate"], trace["phase"]) if p == 1]
            steady_walk = [value for trace in traces for value, p in zip(trace["action_rate"], trace["phase"]) if p == 3]
            steady_step = max(pct(steady_stand, 99), pct(steady_walk, 99)) * dt
            threshold = 1.5 * max(pct(direct, 99), steady_step)
        else:
            baseline = json.loads(baseline_path.read_text())
            threshold = float(baseline["frozen_action_jump_l2_threshold"])

        records = []
        for i, trace in enumerate(traces):
            edge_ids = [k for k, p in enumerate(trace["phase"]) if p == 2]
            take_ids = [k for k, p in enumerate(trace["phase"]) if p == 3]
            edge_values = lambda key: [trace[key][k] for k in edge_ids]
            take_values = lambda key: [trace[key][k] for k in take_ids]
            ankle_dwell = max(runs([value >= 0.95 for value in edge_values("ankle")], dt), default=0.0)
            knee_dwell = max(runs([value >= 0.95 for value in edge_values("knee")], dt), default=0.0)
            slip_failure = avg(edge_values("slip")) > 0.55
            flight_failure = avg(edge_values("flight")) > 0.20
            takeover_heading = pct(take_values("heading"), 95)
            takeover_speed_error = avg([abs(value - speeds[i]) for value in take_values("vx")])
            takeover_hold = bool(takeover_ok[i]) and takeover_heading <= 0.12 and takeover_speed_error <= 0.20
            flags = {name: False for name in FAILURES}
            flags.update({
                "source_stand_settle_failure": not bool(source_ok[i]),
                "transition_start_failure": not bool(source_ok[i]),
                "walk_liftoff_failure": support_switches[i].item() == 0,
                "walk_cycle_acquisition_failure": not bool(completed[i]),
                "target_speed_failure": not bool(completed[i]),
                "heading_failure": pct(edge_values("heading"), 95) > 0.12,
                "path_drift_failure": max(edge_values("cross"), default=0.0) > 0.30,
                "action_discontinuity_at_entry": float(entry_jump[i]) > threshold,
                "action_discontinuity_at_exit": bool(completed[i]) and float(exit_jump[i]) > threshold,
                "ankle_saturation_failure": ankle_dwell >= 0.20,
                "knee_saturation_failure": knee_dwell >= 0.05,
                "dangerous_slip": slip_failure,
                "excessive_flight": flight_failure,
                "transition_timeout": bool(source_ok[i]) and not bool(completed[i]),
                "false_completion": bool(completed[i]) and not takeover_hold,
                "walk_takeover_failure": not takeover_hold,
                "fall": bool(fallen[i]),
            })
            full_success = bool(source_ok[i] and completed[i] and takeover_hold and not any(flags.values()))
            primary = next((name for name in FAILURES if flags[name]), "")
            records.append({
                "episode": i, "target_speed_mps": speeds[i], "source_stand_settle": bool(source_ok[i]),
                "transition_completion": bool(completed[i]), "walk_takeover_success": takeover_hold,
                "full_edge_success": full_success, "transition_duration_s": len(edge_ids) * dt,
                "transition_heading_p95_rad": pct(edge_values("heading"), 95),
                "takeover_heading_p95_rad": takeover_heading,
                "takeover_speed_error_mean_mps": takeover_speed_error,
                "path_drift_max_m": max(edge_values("cross") + take_values("cross"), default=0.0),
                "fall": bool(fallen[i]), "dangerous_slip": slip_failure,
                "excessive_flight": flight_failure, "ankle_saturation_max_dwell_s": ankle_dwell,
                "knee_saturation_max_dwell_s": knee_dwell, "entry_action_jump_l2": float(entry_jump[i]),
                "exit_action_jump_l2": float(exit_jump[i]), "entry_joint_max_jump": float(entry_joint_jump[i]),
                "exit_joint_max_jump": float(exit_joint_jump[i]),
                "transition_action_rate_p99": pct(edge_values("action_rate"), 99),
                "transition_action_rate_max": max(edge_values("action_rate"), default=0.0),
                "torque_spike_max_ratio": max(edge_values("torque_spike"), default=0.0),
                "primary_failure": primary, "failure_flags": json.dumps(flags, sort_keys=True),
            })

        def summarize(rows):
            return {
                "episodes": len(rows),
                "source_stand_settle_rate": avg([r["source_stand_settle"] for r in rows]),
                "transition_completion_rate": avg([r["transition_completion"] for r in rows]),
                "walk_takeover_success_rate": avg([r["walk_takeover_success"] for r in rows]),
                "full_edge_success_rate": avg([r["full_edge_success"] for r in rows]),
                "transition_duration_mean_s": avg([r["transition_duration_s"] for r in rows if r["transition_completion"]]),
                "heading_error_p95_rad": pct([r["transition_heading_p95_rad"] for r in rows], 95),
                "takeover_heading_p95_rad": pct([r["takeover_heading_p95_rad"] for r in rows], 95),
                "fall_rate": avg([r["fall"] for r in rows]),
                "slip_failure_rate": avg([r["dangerous_slip"] for r in rows]),
                "saturation_failure_rate": avg([
                    r["ankle_saturation_max_dwell_s"] >= 0.20 or r["knee_saturation_max_dwell_s"] >= 0.05
                    for r in rows
                ]),
                "entry_discontinuity_failure_rate": avg([r["entry_action_jump_l2"] > threshold for r in rows]),
                "exit_discontinuity_failure_rate": avg([
                    r["transition_completion"] and r["exit_action_jump_l2"] > threshold for r in rows
                ]),
                "entry_action_jump_l2_p95": pct([r["entry_action_jump_l2"] for r in rows], 95),
                "exit_action_jump_l2_p95": pct([r["exit_action_jump_l2"] for r in rows if r["transition_completion"]], 95),
            }

        overall = summarize(records)
        per_speed = {
            str(speed): summarize([record for record in records if record["target_speed_mps"] == speed])
            for speed in SPEEDS
        }
        checks = {
            "source_stand_settle_ge_0_95": overall["source_stand_settle_rate"] >= 0.95,
            "transition_completion_ge_0_95": overall["transition_completion_rate"] >= 0.95,
            "walk_takeover_ge_0_95": overall["walk_takeover_success_rate"] >= 0.95,
            "full_edge_ge_0_90": overall["full_edge_success_rate"] >= 0.90,
            "each_speed_ge_0_90": all(item["full_edge_success_rate"] >= 0.90 for item in per_speed.values()),
            "fall_le_0_02": overall["fall_rate"] <= 0.02,
            "heading_le_0_12": overall["heading_error_p95_rad"] <= 0.12,
            "takeover_heading_le_0_12": overall["takeover_heading_p95_rad"] <= 0.12,
            "saturation_le_0_05": overall["saturation_failure_rate"] <= 0.05,
            "slip_le_0_05": overall["slip_failure_rate"] <= 0.05,
            "entry_discontinuity_le_0_05": overall["entry_discontinuity_failure_rate"] <= 0.05,
            "exit_discontinuity_le_0_05": overall["exit_discontinuity_failure_rate"] <= 0.05,
        }
        summary = {
            "stage": "Stage 3", "mode": args.mode, "label": args.label, "seed": args.seed,
            "stand_checkpoint": str(stand_path.relative_to(REPO)), "stand_sha256": sha(stand_path),
            "walk_checkpoint": str(walk_path.relative_to(REPO)), "walk_sha256": sha(walk_path),
            "transition_checkpoint": str(transition_path.relative_to(REPO)) if transition_path else None,
            "transition_sha256": sha(transition_path) if transition_path else None,
            "controller_routing": "STAND frozen -> STAND_TO_WALK independent -> WALK frozen",
            "runtime_action_blend": False, "completion_hold_s": 0.4, "support_switches_required": 2,
            "transition_timeout_s": 4.0, "walk_takeover_hold_s": 3.0,
            "frozen_action_jump_l2_threshold": threshold,
            "overall": overall, "per_speed": per_speed, "checks": checks,
            "gate_pass": all(checks.values()), "failure_counts": dict(Counter(r["primary_failure"] or "none" for r in records)),
        }
        if args.mode == "baseline":
            summary["diagnostic_only"] = True
            baseline_path.write_text(json.dumps(summary, indent=2) + "\n")
        else:
            (output / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output / f"{args.label}_episodes.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        with (output / f"{args.label}_timelines.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timeline_rows[0]))
            writer.writeheader()
            writer.writerows(timeline_rows)
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
