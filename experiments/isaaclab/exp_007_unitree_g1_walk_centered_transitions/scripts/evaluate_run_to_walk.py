"""Stage 8A RUN_TO_WALK overlap and direct hard-switch baseline."""
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

import isaaclab_tasks  # noqa: F401
import g1_flat_run.tasks  # noqa: F401
import g1_command_skills.tasks  # noqa: F401
import g1_walk_centered.tasks  # noqa: F401
from g1_walk_centered.command_contract import MotionCommand
from g1_walk_centered.experts import load_run_expert, load_walk_expert
from g1_walk_centered.experts.adapters import canonical_state_from_legacy_observation, to_run_observation
from g1_walk_centered.tasks.stage7r_action import WalkToRunTransitionActor152
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

SOURCES = (2.6, 2.8)
EXPECTED = {
    "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
    "stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "stw": "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
    "wtr": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
}

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--attempts-per-source", type=int, default=30)
p.add_argument("--output", required=True)
p.add_argument("--label", required=True)
p.add_argument("--stand", required=True)
p.add_argument("--stand-to-walk", required=True)
p.add_argument("--walk", required=True)
p.add_argument("--run", required=True)
p.add_argument("--walk-to-run", required=True)
p.add_argument("--source-speeds", type=float, nargs="+", default=list(SOURCES))
add_launcher_args(p)
args, hydra = setup_preset_cli(p)
sys.argv = [sys.argv[0], *hydra]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mj(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(0, 1)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    x = (len(values) - 1) * q / 100
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] * (hi - x) + values[hi] * (x - lo)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fresh_gait() -> dict:
    return {"flights": [], "valid": 0, "alt": 0, "altopp": 0, "last": None, "safe": 0, "maxsafe": 0}


def periodic(gait: dict) -> bool:
    return (
        len(gait["flights"]) >= 4
        and gait["maxsafe"] >= 3
        and gait["alt"] / max(gait["altopp"], 1) >= 0.8
        and gait["valid"] / max(len(gait["flights"]), 1) >= 0.8
        and 0.04 <= mean(gait["flights"]) <= 0.16
    )


def main() -> None:
    out = (REPO / args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "stand": Path(args.stand).resolve(strict=True),
        "stw": Path(args.stand_to_walk).resolve(strict=True),
        "walk": Path(args.walk).resolve(strict=True),
        "run": Path(args.run).resolve(strict=True),
        "wtr": Path(args.walk_to_run).resolve(strict=True),
    }
    hashes = {name: sha(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise RuntimeError(f"protected hash mismatch: {hashes}")

    formal_sources = tuple(args.source_speeds)
    if any(speed not in SOURCES for speed in formal_sources):
        raise RuntimeError(f"unsupported RUN_TO_WALK source: {formal_sources}")
    speeds = [speed for speed in formal_sources for _ in range(args.attempts_per_source)]
    n = len(speeds)
    cfg, agent = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 28.0
    if args.device:
        cfg.sim.device = args.device

    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg), clip_actions=agent.clip_actions
        )
        env = wrapped.unwrapped
        device = env.device
        stand = load_walk_expert(paths["stand"], device=device)
        stw = load_walk_expert(paths["stw"], device=device)
        walk = load_walk_expert(paths["walk"], device=device)
        run = load_run_expert(paths["run"], device=device)
        wtr = WalkToRunTransitionActor152(run.actor).to(device)
        payload = torch.load(paths["wtr"], map_location=device, weights_only=False)
        wtr.load_state_dict(payload["actor"], strict=True)
        wtr.eval()

        robot = env.scene["robot"]
        command = env.command_manager.get_term("base_velocity")
        sensor = env.scene.sensors["contact_forces"]
        feet, foot_names = robot.find_bodies(".*_ankle_roll_link")
        sensor_feet = [sensor.body_names.index(name) for name in foot_names]
        joints, joint_names = robot.find_joints(".*")
        wrapped.reset()
        dt = float(env.step_dt)
        target = torch.tensor(speeds, device=device)
        heading = robot.data.heading_w.torch.clone()
        origin = robot.data.root_pos_w.torch[:, :2].clone()

        # 0 startup, 1 stand hold, 2 STW, 3 walk hold, 4 WTR, 5 RUN source hold, 6 WALK acquire, 7 WALK hold.
        phase = torch.zeros(n, dtype=torch.long, device=device)
        phase_time = torch.zeros(n, device=device)
        streak = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        previous_action = torch.zeros(n, 37, device=device)
        done = torch.zeros(n, dtype=torch.bool, device=device)
        source_valid = torch.zeros(n, dtype=torch.bool, device=device)
        walk_acquired = torch.zeros(n, dtype=torch.bool, device=device)
        takeover = torch.zeros(n, dtype=torch.bool, device=device)
        hold = torch.zeros(n, dtype=torch.bool, device=device)
        fall = torch.zeros(n, dtype=torch.bool, device=device)
        saturation = torch.zeros(n, dtype=torch.bool, device=device)
        slip_failure = torch.zeros(n, dtype=torch.bool, device=device)
        impact_failure = torch.zeros(n, dtype=torch.bool, device=device)
        excessive_flight = torch.zeros(n, dtype=torch.bool, device=device)
        reverse_failure = torch.zeros(n, dtype=torch.bool, device=device)
        slip_dwell = torch.zeros(n, device=device)
        saturation_dwell = torch.zeros(n, len(joints), device=device)
        completion_time = torch.zeros(n, device=device)
        hold_time = torch.zeros(n, device=device)
        entry_jump = torch.zeros(n, device=device)
        previous_mismatch = torch.zeros(n, dtype=torch.long, device=device)
        source_phase = [""] * n
        last_landing = [""] * n
        gait = [fresh_gait() for _ in range(n)]
        in_flight = [False] * n
        flight_start = [0.0] * n
        previous_contacts = [(False, False)] * n
        air_time = torch.zeros(n, 2, device=device)
        stance_time = torch.zeros(n, 2, device=device)
        source_air_time = torch.zeros(n, 2, device=device)
        source_stance_time = torch.zeros(n, 2, device=device)
        traces = [
            {"heading": [], "speed": [], "drift": [], "impact": [], "slip": [], "action_rate": [], "flight": []}
            for _ in range(n)
        ]
        difference_rows: list[dict] = []

        for step in range(round(27.5 / dt)):
            current = phase.clone()
            speed_command = torch.zeros(n, device=device)
            speed_command[current == 2] = 1.2 * mj(phase_time[current == 2] / 1.5)
            speed_command[current == 3] = 1.2
            speed_command[current == 4] = 1.2 + (target[current == 4] - 1.2) * mj(phase_time[current == 4] / 1.4)
            speed_command[current == 5] = target[current == 5]
            speed_command[current >= 6] = 1.2
            heading_error = torch.atan2(
                torch.sin(heading - robot.data.heading_w.torch), torch.cos(heading - robot.data.heading_w.torch)
            )
            yaw_walk = (0.8 * heading_error - 0.1 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            yaw_run = (1.5 * heading_error).clamp(-1.5, 1.5)
            yaw = torch.where((current == 4) | (current == 5), yaw_run, yaw_walk)
            command.vel_command_b.zero_()
            command.vel_command_b[:, 0] = speed_command
            command.vel_command_b[:, 2] = yaw
            legacy = wrapped.get_observations()["policy"]
            canonical = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            motion = MotionCommand(speed_command, heading, target_yaw_rate_radps=yaw)
            with torch.inference_mode():
                stand_action = stand(canonical, motion)
                stw_action = stw(canonical, motion)
                walk_action = walk(canonical, motion)
                run_action = run(canonical, motion)
                wtr_action = wtr(to_run_observation(canonical, motion, route="RUN"))
                action = torch.where(
                    (current == 0).unsqueeze(1),
                    stand_action,
                    torch.where(
                        ((current == 1) | (current == 2)).unsqueeze(1),
                        stw_action,
                        torch.where(
                            (current == 3).unsqueeze(1),
                            walk_action,
                            torch.where(
                                (current == 4).unsqueeze(1),
                                wtr_action,
                                torch.where((current == 5).unsqueeze(1), run_action, walk_action),
                            ),
                        ),
                    ),
                )
                action[done] = previous_action[done]
                applied_previous = previous_action.clone()
                _, _, dones, info = wrapped.step(action)
            action_rate = torch.linalg.vector_norm(action - applied_previous, dim=1) / dt
            previous_action[:] = action

            forces = sensor.data.net_forces_w_history.torch[:, :, sensor_feet, :]
            contacts = forces.norm(dim=-1).amax(dim=1) > 5
            impact = forces[:, :, :, 2].abs().mean(dim=1).amax(dim=1)
            foot_speed = robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1)
            slip = torch.where(contacts, foot_speed, torch.zeros_like(foot_speed)).amax(dim=1)
            effort_ratio = (
                robot.data.applied_torque.torch[:, joints].abs()
                / robot.data.joint_effort_limits.torch[:, joints].abs().clamp_min(1e-6)
            )
            saturation_dwell = torch.where(
                (effort_ratio >= 0.95) & (current >= 5).unsqueeze(1),
                saturation_dwell + dt,
                torch.zeros_like(saturation_dwell),
            )
            saturation |= (saturation_dwell >= 0.2).any(dim=1)
            slip_dwell = torch.where((slip > 0.8) & (current >= 5), slip_dwell + dt, torch.zeros_like(slip_dwell))
            slip_failure |= slip_dwell >= 0.2
            impact_failure |= (impact > 3500) & (current >= 5)
            torso = env.termination_manager.get_term("base_contact").bool()
            timeouts = info.get("time_outs", torch.zeros_like(dones)).bool()
            physical_done = dones.bool() & ~timeouts
            forward_speed = robot.data.root_lin_vel_b.torch[:, 0]
            gravity = robot.data.projected_gravity_b.torch
            roll = torch.atan2(gravity[:, 1], -gravity[:, 2]).abs()
            pitch = torch.atan2(-gravity[:, 0], torch.sqrt(gravity[:, 1] ** 2 + gravity[:, 2] ** 2)).abs()
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            air_time = torch.where(contacts, torch.zeros_like(air_time), air_time + dt)
            stance_time = torch.where(contacts, stance_time + dt, torch.zeros_like(stance_time))

            for index in range(n):
                if done[index]:
                    continue
                if current[index] >= 4:
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
                        traces[index]["flight"].append(duration)
                        if valid:
                            tracker["valid"] += 1
                            last_landing[index] = "left" if side == 0 else "right"
                            if tracker["last"] is not None:
                                tracker["altopp"] += 1
                                tracker["alt"] += int(side != tracker["last"])
                            safe = 0.04 <= duration <= 0.16 and (tracker["last"] is None or side != tracker["last"])
                            tracker["safe"] = tracker["safe"] + 1 if safe else 0
                            tracker["maxsafe"] = max(tracker["maxsafe"], tracker["safe"])
                            tracker["last"] = side
                        in_flight[index] = False
                    previous_contacts[index] = contact

                if current[index] >= 5:
                    trace = traces[index]
                    trace["heading"].append(abs(float(heading_error[index])))
                    trace["speed"].append(float(forward_speed[index]))
                    trace["impact"].append(float(impact[index]))
                    trace["slip"].append(float(slip[index]))
                    trace["action_rate"].append(float(action_rate[index]))
                    lateral = torch.stack((-torch.sin(heading[index]), torch.cos(heading[index])))
                    trace["drift"].append(
                        abs(float(((robot.data.root_pos_w.torch[index, :2] - origin[index]) * lateral).sum()))
                    )
                    excessive_flight[index] |= in_flight[index] and (
                        float(phase_time[index]) - flight_start[index] > 0.16
                    )
                    reverse_failure[index] |= forward_speed[index] < -0.1

                if physical_done[index] or torso[index]:
                    fall[index] = True
                    done[index] = True
                    continue
                if current[index] == 0:
                    good = (
                        abs(float(forward_speed[index])) <= 0.08
                        and roll[index] <= 0.1
                        and pitch[index] <= 0.1
                        and int(support[index]) == 3
                    )
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 0.4:
                        phase[index], phase_time[index], streak[index] = 1, 0, 0
                elif current[index] == 1:
                    if phase_time[index] >= 0.8:
                        phase[index], phase_time[index], streak[index], support_switches[index] = 2, 0, 0, 0
                elif current[index] == 2:
                    if int(support[index]) != int(previous_support[index]) and int(support[index]) in (1, 2):
                        support_switches[index] += 1
                    good = (
                        abs(float(forward_speed[index] - 1.2)) <= 0.2
                        and abs(float(heading_error[index])) <= 0.12
                        and support_switches[index] >= 2
                    )
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 0.4:
                        phase[index], phase_time[index], streak[index] = 3, 0, 0
                    elif phase_time[index] >= 5:
                        done[index] = True
                elif current[index] == 3:
                    good = abs(float(forward_speed[index] - 1.2)) <= 0.2 and abs(float(heading_error[index])) <= 0.12
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 1.0:
                        phase[index], phase_time[index], streak[index] = 4, 0, 0
                        origin[index] = robot.data.root_pos_w.torch[index, :2]
                    elif phase_time[index] >= 3:
                        done[index] = True
                elif current[index] == 4:
                    good = (
                        periodic(gait[index])
                        and abs(float(forward_speed[index] - target[index])) <= 0.2
                        and abs(float(heading_error[index])) <= 0.12
                        and not saturation[index]
                        and not slip_failure[index]
                        and not impact_failure[index]
                    )
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 0.4:
                        phase[index], phase_time[index], streak[index] = 5, 0, 0
                    elif phase_time[index] >= 5:
                        done[index] = True
                elif current[index] == 5:
                    good = (
                        periodic(gait[index])
                        and abs(float(forward_speed[index] - target[index])) <= 0.2
                        and abs(float(heading_error[index])) <= 0.12
                        and not saturation[index]
                        and not slip_failure[index]
                        and not impact_failure[index]
                    )
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 1.0:
                        source_valid[index] = True
                        source_phase[index] = (
                            "double"
                            if int(support[index]) == 3
                            else "left"
                            if int(support[index]) == 1
                            else "right"
                            if int(support[index]) == 2
                            else "flight"
                        )
                        source_air_time[index] = air_time[index]
                        source_stance_time[index] = stance_time[index]
                        entry_jump[index] = torch.linalg.vector_norm(run_action[index] - walk_action[index])
                        for joint_index, joint_name in enumerate(joint_names):
                            difference_rows.append(
                                {
                                    "seed": args.seed,
                                    "episode": index,
                                    "source_speed_mps": speeds[index],
                                    "contact_phase": source_phase[index],
                                    "joint_name": joint_name,
                                    "run_action": float(run_action[index, joint_index]),
                                    "walk_action": float(walk_action[index, joint_index]),
                                    "absolute_difference": abs(
                                        float(run_action[index, joint_index] - walk_action[index, joint_index])
                                    ),
                                    "action_l2_difference": float(entry_jump[index]),
                                    "pelvis_height_m": float(robot.data.root_pos_w.torch[index, 2]),
                                    "roll_rad": float(roll[index]),
                                    "pitch_rad": float(pitch[index]),
                                    "forward_velocity_mps": float(forward_speed[index]),
                                    "vertical_velocity_mps": float(robot.data.root_lin_vel_b.torch[index, 2]),
                                    "foot_air_time_left_s": float(air_time[index, 0]),
                                    "foot_air_time_right_s": float(air_time[index, 1]),
                                    "ankle_effort_max_utilization": float(effort_ratio[index].max()),
                                }
                            )
                        phase[index], phase_time[index], streak[index] = 6, 0, 0
                        origin[index] = robot.data.root_pos_w.torch[index, :2]
                    elif phase_time[index] >= 3:
                        done[index] = True
                elif current[index] == 6:
                    walk_pattern = not periodic(gait[index])
                    good = (
                        abs(float(forward_speed[index] - 1.2)) <= 0.2
                        and abs(float(heading_error[index])) <= 0.12
                        and walk_pattern
                        and not excessive_flight[index]
                        and not saturation[index]
                        and not slip_failure[index]
                    )
                    streak[index] = streak[index] + dt if good else 0
                    if streak[index] >= 0.4:
                        walk_acquired[index] = True
                        completion_time[index] = phase_time[index]
                        phase[index], phase_time[index], streak[index] = 7, 0, 0
                    elif phase_time[index] >= 5:
                        done[index] = True
                elif current[index] == 7:
                    hold_time[index] += dt
                    good = (
                        abs(float(forward_speed[index] - 1.2)) <= 0.2
                        and abs(float(heading_error[index])) <= 0.12
                        and not periodic(gait[index])
                        and not excessive_flight[index]
                        and not saturation[index]
                        and not slip_failure[index]
                        and not impact_failure[index]
                    )
                    if not good:
                        streak[index] = 0
                    else:
                        streak[index] += dt
                    if hold_time[index] >= 5:
                        takeover[index] = streak[index] >= 0.4
                        hold[index] = takeover[index]
                        done[index] = True
                previous_support[index] = support[index]
            phase_time += dt
            if bool(done.all()):
                break

        rows = []
        for index in range(n):
            trace = traces[index]
            heading_p95 = percentile(trace["heading"], 95)
            action_discontinuity = entry_jump[index] > 6.0
            success = bool(
                source_valid[index]
                and walk_acquired[index]
                and takeover[index]
                and hold[index]
                and not fall[index]
                and not saturation[index]
                and not slip_failure[index]
                and not impact_failure[index]
                and not excessive_flight[index]
                and heading_p95 <= 0.12
                and not action_discontinuity
            )
            failure = (
                ""
                if success
                else "source_preparation_failure"
                if not source_valid[index]
                else "walk_contract_acquisition_failure"
                if not walk_acquired[index]
                else "walk_hold_failure"
                if not hold[index]
                else "safety_failure"
            )
            rows.append(
                {
                    "seed": args.seed,
                    "episode": index,
                    "source_run_speed_mps": speeds[index],
                    "valid_run_source": bool(source_valid[index]),
                    "contact_phase": source_phase[index],
                    "stance_duration_left_s": float(source_stance_time[index, 0]),
                    "stance_duration_right_s": float(source_stance_time[index, 1]),
                    "foot_air_time_left_s": float(source_air_time[index, 0]),
                    "foot_air_time_right_s": float(source_air_time[index, 1]),
                    "last_landing_foot": last_landing[index],
                    "walk_target_speed_acquisition": bool(walk_acquired[index]),
                    "periodic_run_termination": bool(walk_acquired[index]),
                    "walk_contract_acquisition": bool(walk_acquired[index]),
                    "walk_takeover": bool(takeover[index]),
                    "walk_hold": bool(hold[index]),
                    "full_edge_success": success,
                    "transition_duration_s": float(completion_time[index]),
                    "last_flight_duration_s": traces[index]["flight"][-1] if traces[index]["flight"] else 0,
                    "heading_p95_rad": heading_p95,
                    "reverse_velocity_failure": bool(reverse_failure[index]),
                    "path_drift_max_m": max(trace["drift"], default=0),
                    "fall": bool(fall[index]),
                    "torso_contact": bool(fall[index]),
                    "dangerous_slip": bool(slip_failure[index]),
                    "impact_failure": bool(impact_failure[index]),
                    "long_dwell_saturation": bool(saturation[index]),
                    "excessive_flight": bool(excessive_flight[index]),
                    "entry_action_jump_l2": float(entry_jump[index]),
                    "action_discontinuity_failure": bool(action_discontinuity),
                    "previous_action_mismatch": int(previous_mismatch[index]),
                    "controller_overlap": 0,
                    "routing_error": 0,
                    "failure_class": failure,
                }
            )
        write_csv(out / f"{args.label}_episodes.csv", rows)
        if difference_rows:
            write_csv(out / f"{args.label}_action_difference.csv", difference_rows)
        valid = [row for row in rows if row["valid_run_source"]]
        summary = {
            "seed": args.seed,
            "attempts": n,
            "valid_sources": len(valid),
            "source_generation_rate": len(valid) / n,
            "per_source": {},
        }
        for speed in formal_sources:
            group = [row for row in valid if row["source_run_speed_mps"] == speed]
            summary["per_source"][str(speed)] = {
                "valid_n": len(group),
                "walk_contract_acquisition_rate": mean(row["walk_contract_acquisition"] for row in group),
                "walk_takeover_rate": mean(row["walk_takeover"] for row in group),
                "walk_hold_rate": mean(row["walk_hold"] for row in group),
                "full_edge_success_rate": mean(row["full_edge_success"] for row in group),
                "transition_duration_mean_s": mean(row["transition_duration_s"] for row in group),
                "heading_p95_rad": percentile([row["heading_p95_rad"] for row in group], 95),
                "fall_rate": mean(row["fall"] for row in group),
                "saturation_rate": mean(row["long_dwell_saturation"] for row in group),
                "slip_rate": mean(row["dangerous_slip"] for row in group),
                "impact_failure_rate": mean(row["impact_failure"] for row in group),
                "excessive_flight_rate": mean(row["excessive_flight"] for row in group),
                "action_discontinuity_failure_rate": mean(
                    row["action_discontinuity_failure"] for row in group
                ),
                "entry_action_jump_p95": percentile([row["entry_action_jump_l2"] for row in group], 95),
            }
        (out / f"{args.label}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
