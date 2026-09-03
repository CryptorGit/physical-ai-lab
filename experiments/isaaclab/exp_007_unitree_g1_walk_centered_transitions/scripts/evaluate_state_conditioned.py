"""Stage 5E state-contract-conditioned integration and startup diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
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

CONTROLLERS = (
    "stage2_model_4246", "stand_to_walk_transition_v1",
    "walk_steady_state_expert_v1", "walk_to_stand_transition_v1",
)
PHASES = ("UNINITIALIZED", "STAND_TO_WALK", "WALK", "WALK_TO_STAND", "FINAL_STAND", "DONE")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("startup", "main", "zero_point_six"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--manifest", default=str(EXP / "integration_manifest.json"))
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(values):
    return sum(values) / len(values) if values else 0.0


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * q / 100), len(ordered) - 1)]


def minimum_jerk(value):
    value = value.clamp(0.0, 1.0)
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if not rows:
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def longest(values, dt):
    best = current = 0.0
    for value in values:
        current = current + dt if value else 0.0
        best = max(best, current)
    return best


def main():
    output = Path(args.output)
    if not output.is_absolute():
        output = REPO / output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    paths = {}
    for name in CONTROLLERS:
        spec = manifest["controllers"][name]
        path = (REPO / spec["checkpoint"]).resolve(strict=True)
        if sha(path) != spec["sha256"]:
            raise RuntimeError(f"hash mismatch: {name}")
        paths[name] = path
    if args.mode == "startup":
        speeds = [0.0] * 50
        desired = {}
    elif args.mode == "main":
        speeds = [0.8] * 24 + [1.0] * 24 + [1.2] * 24
        desired = {0.8: 20, 1.0: 20, 1.2: 20}
    else:
        speeds = [0.6] * 60
        desired = {0.6: 50}
    n = len(speeds)
    rng = random.Random(args.seed)
    post_contract_holds = [rng.uniform(0.4, 1.0) for _ in speeds]
    walk_holds = [rng.uniform(2.5, 3.5) for _ in speeds]
    ramp_up = [rng.uniform(1.3, 1.7) for _ in speeds]
    ramp_down = [rng.uniform(1.4, 1.8) for _ in speeds]
    yaw_offsets = [rng.uniform(-0.03, 0.03) for _ in speeds]
    cfg, agent_cfg = resolve_task_config("Isaac-Velocity-Flat-G1-Run-Eval-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = n
    cfg.seed = args.seed
    cfg.episode_length_s = 3.0 if args.mode == "startup" else 22.0
    if args.device:
        cfg.sim.device = args.device
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(
            gym.make("Isaac-Velocity-Flat-G1-Run-Eval-v0", cfg=cfg),
            clip_actions=agent_cfg.clip_actions,
        )
        env = wrapped.unwrapped
        models = {name: load_walk_expert(path, device=env.device) for name, path in paths.items()}
        robot = env.scene["robot"]
        velocity_term = env.command_manager.get_term("base_velocity")
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
        reset_horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1).clone()
        reset_vertical = robot.data.root_lin_vel_w.torch[:, 2].clone()
        reset_heading = robot.data.heading_w.torch.clone()
        target_speed = torch.tensor(speeds, device=device)
        target_heading = reset_heading.clone()
        phase = torch.zeros(n, dtype=torch.long, device=device)
        phase_elapsed = torch.zeros(n, device=device)
        contract_elapsed = torch.zeros(n, device=device)
        settle_streak = torch.zeros(n, device=device)
        acquire_streak = torch.zeros(n, device=device)
        walk_streak = torch.zeros(n, device=device)
        stop_streak = torch.zeros(n, device=device)
        no_switch = torch.zeros(n, device=device)
        support_switches = torch.zeros(n, dtype=torch.long, device=device)
        previous_support = torch.zeros(n, dtype=torch.long, device=device)
        filtered_yaw = torch.zeros(n, device=device)
        previous_action = torch.zeros(n, 37, device=device)
        contract_valid = torch.zeros(n, dtype=torch.bool, device=device)
        startup_rejected = torch.zeros(n, dtype=torch.bool, device=device)
        startup_fall = torch.zeros(n, dtype=torch.bool, device=device)
        startup_torso = torch.zeros(n, dtype=torch.bool, device=device)
        startup_saturation = torch.zeros(n, dtype=torch.bool, device=device)
        ankle_run = torch.zeros(n, device=device)
        knee_run = torch.zeros(n, device=device)
        graph_ankle_run = torch.zeros(n, device=device)
        graph_knee_run = torch.zeros(n, device=device)
        graph_saturation = torch.zeros(n, dtype=torch.bool, device=device)
        graph_fall = torch.zeros(n, dtype=torch.bool, device=device)
        graph_torso = torch.zeros(n, dtype=torch.bool, device=device)
        up_ok = torch.zeros(n, dtype=torch.bool, device=device)
        walk_ok = torch.zeros(n, dtype=torch.bool, device=device)
        down_ok = torch.zeros(n, dtype=torch.bool, device=device)
        final_ok = torch.zeros(n, dtype=torch.bool, device=device)
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        previous_mismatch = torch.zeros(n, dtype=torch.long, device=device)
        traces = [defaultdict(list) for _ in speeds]
        boundary_rows = []
        previous_controller = ["stage2_model_4246"] * n
        max_steps = round((2.5 if args.mode == "startup" else 21.5) / dt)

        for step in range(max_steps):
            current_phase = phase.clone()
            active = ~finished
            command_vx = torch.zeros(n, device=device)
            up, walk, down = current_phase == 1, current_phase == 2, current_phase == 3
            up_d = torch.tensor(ramp_up, device=device)
            down_d = torch.tensor(ramp_down, device=device)
            command_vx[up] = target_speed[up] * minimum_jerk(phase_elapsed[up] / up_d[up])
            command_vx[walk] = target_speed[walk]
            command_vx[down] = target_speed[down] * (1.0 - minimum_jerk(phase_elapsed[down] / down_d[down]))
            heading_signed = torch.atan2(
                torch.sin(target_heading - robot.data.heading_w.torch),
                torch.cos(target_heading - robot.data.heading_w.torch),
            )
            raw_yaw = (0.8 * heading_signed - 0.10 * robot.data.root_ang_vel_b.torch[:, 2]).clamp(-0.3, 0.3)
            low = filtered_yaw + 0.15 * (raw_yaw - filtered_yaw)
            filtered_yaw += (low - filtered_yaw).clamp(-0.01, 0.01)
            filtered_yaw[(current_phase == 0) | (current_phase == 4)] = 0.0
            velocity_term.vel_command_b.zero_()
            velocity_term.vel_command_b[:, 0] = command_vx
            velocity_term.vel_command_b[:, 2] = filtered_yaw
            legacy = wrapped.get_observations()["policy"]
            previous_match = (legacy[:, 86:123] == previous_action).all(dim=1)
            previous_mismatch += (~previous_match & active).long()
            state = canonical_state_from_legacy_observation(legacy, heading_w_rad=robot.data.heading_w.torch)
            command = MotionCommand(command_vx, target_heading, target_yaw_rate_radps=filtered_yaw)
            action = torch.zeros_like(previous_action)
            controllers = [
                "stage2_model_4246" if current_phase[i] in (0, 4, 5)
                else "stand_to_walk_transition_v1" if current_phase[i] == 1
                else "walk_steady_state_expert_v1" if current_phase[i] == 2
                else "walk_to_stand_transition_v1"
                for i in range(n)
            ]
            with torch.inference_mode():
                for name, model in models.items():
                    mask = torch.tensor([active[i] and controllers[i] == name for i in range(n)], device=device)
                    if bool(mask.any()):
                        candidate = model(state, command)
                        action[mask] = candidate[mask]
                action[finished] = previous_action[finished]
                for i in range(n):
                    if finished[i] or controllers[i] == previous_controller[i]:
                        continue
                    delta = action[i] - previous_action[i]
                    boundary_rows.append({
                        "candidate_episode": i, "speed_mps": speeds[i],
                        "boundary": f"{previous_controller[i]}->{controllers[i]}",
                        "action_l2_jump": float(torch.linalg.vector_norm(delta)),
                        "joint_max_jump": float(delta.abs().amax()),
                        "previous_action_match": bool(previous_match[i]),
                    })
                    previous_controller[i] = controllers[i]
                _, _, dones, _ = wrapped.step(action)

            forces = sensor.data.net_forces_w_history.torch
            contacts = forces[:, :, sensor_feet, :].norm(dim=-1).amax(dim=1) > 5.0
            torso = forces[:, :, sensor_torso, :].norm(dim=-1).amax(dim=(1, 2)) > 5.0
            support = contacts[:, 0].long() + 2 * contacts[:, 1].long()
            slip = torch.where(
                contacts, robot.data.body_lin_vel_w.torch[:, feet, :2].norm(dim=-1),
                torch.zeros_like(contacts, dtype=torch.float32)
            ).amax(dim=1)
            ankle = (
                robot.data.applied_torque.torch[:, ankles].abs()
                / robot.data.joint_effort_limits.torch[:, ankles].abs().clamp_min(1e-6)
            ).amax(dim=1)
            knee = (
                robot.data.joint_vel.torch[:, knees].abs()
                / robot.data.joint_vel_limits.torch[:, knees].abs().clamp_min(1e-6)
            ).amax(dim=1)
            g = robot.data.projected_gravity_b.torch
            roll = torch.atan2(g[:, 1], -g[:, 2])
            pitch = torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
            vx = robot.data.root_lin_vel_b.torch[:, 0]
            horizontal = robot.data.root_lin_vel_b.torch[:, :2].norm(dim=1)
            vertical = robot.data.root_lin_vel_w.torch[:, 2].abs()
            heading = heading_signed.abs()
            action_jump = torch.linalg.vector_norm(action - previous_action, dim=1)
            previous_action[:] = action
            uninit = current_phase == 0
            ankle_run = torch.where(uninit & (ankle >= 0.95), ankle_run + dt, torch.zeros_like(ankle_run))
            knee_run = torch.where(uninit & (knee >= 0.95), knee_run + dt, torch.zeros_like(knee_run))
            startup_saturation |= (ankle_run >= 0.20) | (knee_run >= 0.05)
            startup_fall |= dones.bool() & uninit
            startup_torso |= torso & uninit
            # Only physical graph phases are part of the evaluation window.
            # Phase 5 is DONE; parallel environments may remain alive until the
            # slowest peer finishes and must not accrue post-evaluation failures.
            graph_active = contract_valid & (current_phase >= 1) & (current_phase <= 4)
            graph_ankle_run = torch.where(graph_active & (ankle >= 0.95), graph_ankle_run + dt, torch.zeros_like(graph_ankle_run))
            graph_knee_run = torch.where(graph_active & (knee >= 0.95), graph_knee_run + dt, torch.zeros_like(graph_knee_run))
            graph_saturation |= (graph_ankle_run >= 0.20) | (graph_knee_run >= 0.05)
            graph_fall |= dones.bool() & graph_active
            graph_torso |= torso & graph_active
            stand_safe = (
                (horizontal <= 0.08) & (vertical <= 0.05) & (roll.abs() <= 0.10)
                & (pitch.abs() <= 0.10) & contacts.all(dim=1) & (~dones.bool())
                & (~torso) & torch.isfinite(legacy).all(dim=1) & torch.isfinite(action).all(dim=1)
            )
            settle_streak = torch.where(
                ((current_phase == 0) | (current_phase == 4)) & stand_safe,
                settle_streak + dt, torch.zeros_like(settle_streak)
            )
            for i in range(n):
                if finished[i]:
                    continue
                trace = traces[i]
                trace["phase"].append(int(current_phase[i]))
                trace["vx"].append(float(vx[i]))
                trace["horizontal"].append(float(horizontal[i]))
                trace["heading"].append(float(heading[i]))
                trace["support"].append(int(support[i]))
                trace["slip"].append(float(slip[i]))
                trace["flight"].append(int(support[i] == 0))
                trace["ankle"].append(float(ankle[i]))
                trace["knee"].append(float(knee[i]))
                trace["action_jump"].append(float(action_jump[i]))
                if current_phase[i] == 0:
                    fatal_startup = startup_fall[i] or startup_torso[i] or startup_saturation[i]
                    if fatal_startup:
                        startup_rejected[i] = True
                        finished[i] = True
                    elif settle_streak[i] >= 0.4 and not contract_valid[i]:
                        contract_valid[i] = True
                        contract_elapsed[i] = phase_elapsed[i]
                        if args.mode == "startup":
                            finished[i] = True
                    elif phase_elapsed[i] >= 2.0 and not contract_valid[i]:
                        startup_rejected[i] = True
                        finished[i] = True
                    elif contract_valid[i] and not stand_safe[i]:
                        # The graph may start only from a currently valid source state.
                        # A transient loss during the preregistered post-contract hold
                        # returns the episode to UNINITIALIZED rather than being hidden.
                        contract_valid[i] = False
                        contract_elapsed[i] = 0.0
                    elif contract_valid[i] and phase_elapsed[i] >= contract_elapsed[i] + post_contract_holds[i]:
                        phase[i], phase_elapsed[i] = 1, 0.0
                        target_heading[i] = robot.data.heading_w.torch[i] + yaw_offsets[i]
                        support_switches[i] = 0
                        previous_support[i] = support[i]
                elif current_phase[i] == 1:
                    if support[i] and support[i] != previous_support[i]:
                        support_switches[i] += 1
                    previous_support[i] = support[i]
                    good = (
                        vx[i] >= 0.75 * target_speed[i] and abs(vx[i] - target_speed[i]) <= 0.20
                        and heading[i] <= 0.12 and abs(roll[i]) <= 0.20 and abs(pitch[i]) <= 0.20
                        and support_switches[i] >= 2 and not dones[i] and not torso[i]
                    )
                    acquire_streak[i] = acquire_streak[i] + dt if good else 0.0
                    if acquire_streak[i] >= 0.4:
                        up_ok[i] = True
                        phase[i], phase_elapsed[i], walk_streak[i] = 2, 0.0, 0.0
                elif current_phase[i] == 2:
                    good = abs(vx[i] - target_speed[i]) <= 0.20 and heading[i] <= 0.12 and not dones[i]
                    walk_streak[i] = walk_streak[i] + dt if good else 0.0
                    required = 3.0 if args.mode == "zero_point_six" else walk_holds[i]
                    if walk_streak[i] >= required:
                        walk_ok[i] = True
                        if args.mode == "zero_point_six":
                            finished[i] = True
                            phase[i] = 5
                        else:
                            phase[i], phase_elapsed[i], no_switch[i] = 3, 0.0, 0.0
                            previous_support[i] = support[i]
                elif current_phase[i] == 3:
                    switched = support[i] and support[i] != previous_support[i]
                    no_switch[i] = 0.0 if switched else no_switch[i] + dt
                    previous_support[i] = support[i]
                    good = (
                        horizontal[i] <= 0.08 and vertical[i] <= 0.05 and heading[i] <= 0.12
                        and abs(roll[i]) <= 0.10 and abs(pitch[i]) <= 0.10
                        and support[i] == 3 and no_switch[i] >= 0.4 and not dones[i]
                    )
                    stop_streak[i] = stop_streak[i] + dt if good else 0.0
                    if stop_streak[i] >= 0.4:
                        down_ok[i] = True
                        phase[i], phase_elapsed[i], settle_streak[i] = 4, 0.0, 0.0
                elif current_phase[i] == 4 and settle_streak[i] >= 5.0:
                    final_ok[i] = True
                    finished[i] = True
                    phase[i] = 5
                if current_phase[i] in (1, 3) and phase_elapsed[i] > 5.0:
                    finished[i] = True
                if graph_active[i] and (dones[i] or torso[i]):
                    finished[i] = True
            phase_elapsed += dt
            if bool(finished.all()):
                break

        startup_rows, candidate_rows = [], []
        for i, trace in enumerate(traces):
            startup_rows.append({
                "candidate_episode": i, "seed": args.seed, "assigned_speed_mps": speeds[i],
                "state_contract_valid": bool(contract_valid[i]),
                "settle_time_s": float(contract_elapsed[i]) if contract_valid[i] else None,
                "fall": bool(startup_fall[i]), "torso_contact": bool(startup_torso[i]),
                "ankle_saturation": bool(startup_saturation[i]),
                "no_contact_reset": bool(trace["support"] and trace["support"][0] == 0),
                "reset_horizontal_velocity_mps": float(reset_horizontal[i]),
                "reset_vertical_velocity_mps": float(reset_vertical[i]),
                "initial_double_support": bool(trace["support"] and trace["support"][0] == 3),
                "state_contract_rejected": bool(startup_rejected[i]),
                "graph_route_started": any(value == 1 for value in trace["phase"]),
                "system_state_at_start": "UNINITIALIZED",
            })
            graph_ids = [k for k, value in enumerate(trace["phase"]) if value in (1, 2, 3, 4)]
            walk_ids = [k for k, value in enumerate(trace["phase"]) if value == 2]
            final_ids = [k for k, value in enumerate(trace["phase"]) if value == 4]
            sat = bool(graph_saturation[i])
            dangerous_slip = mean([trace["slip"][k] for k in graph_ids]) > 0.55
            excessive_flight = mean([trace["flight"][k] for k in walk_ids]) > 0.20
            reverse = longest([trace["vx"][k] < -0.10 for k in graph_ids], dt) >= 0.20
            discontinuity = any(
                row["candidate_episode"] == i and row["action_l2_jump"] >
                (5.8614 if "stand_to_walk" in row["boundary"] or "walk_steady" in row["boundary"] else 7.7062)
                for row in boundary_rows
            )
            expected_final = args.mode == "main"
            full = bool(
                contract_valid[i] and up_ok[i] and walk_ok[i]
                and (not expected_final or (down_ok[i] and final_ok[i]))
                and not graph_fall[i] and not graph_torso[i] and not sat
                and not dangerous_slip and not excessive_flight and not reverse and not discontinuity
            )
            candidate_rows.append({
                "candidate_episode": i, "seed": args.seed, "target_speed_mps": speeds[i],
                "state_contract_valid": bool(contract_valid[i]),
                "stand_to_walk_completion": bool(up_ok[i]), "walk_takeover": bool(up_ok[i]),
                "walk_hold": bool(walk_ok[i]), "walk_to_stand_completion": bool(down_ok[i]),
                "stand_takeover": bool(down_ok[i]), "final_stand_hold": bool(final_ok[i]),
                "full_success": full, "fall": bool(graph_fall[i]),
                "saturation_failure": sat, "dangerous_slip": dangerous_slip,
                "excessive_flight": excessive_flight, "reverse_motion": reverse,
                "action_discontinuity": discontinuity,
                "walk_heading_p95_rad": pct([trace["heading"][k] for k in walk_ids], 95),
                "final_speed_p95_mps": pct([trace["horizontal"][k] for k in final_ids], 95),
                "final_double_support": bool(final_ids and trace["support"][final_ids[-1]] == 3),
                "final_stand_flight": bool(any(trace["flight"][k] for k in final_ids)),
                "transition_completion_time_s": len([k for k, value in enumerate(trace["phase"]) if value == 1]) * dt,
                "previous_action_mismatch_steps": int(previous_mismatch[i]),
            })

        if args.mode == "startup":
            selected = []
        else:
            selected = []
            for speed, count in desired.items():
                valid = [row for row in candidate_rows if row["target_speed_mps"] == speed and row["state_contract_valid"]]
                if len(valid) < count:
                    raise RuntimeError(f"insufficient preregistered valid source states for {speed}: {len(valid)}/{count}")
                selected.extend(valid[:count])
            for index, row in enumerate(selected):
                row["formal_episode"] = index
        def summary(rows):
            return {
                "episodes": len(rows),
                "stand_to_walk_completion_rate": mean([r["stand_to_walk_completion"] for r in rows]),
                "walk_takeover_rate": mean([r["walk_takeover"] for r in rows]),
                "walk_hold_rate": mean([r["walk_hold"] for r in rows]),
                "walk_to_stand_completion_rate": mean([r["walk_to_stand_completion"] for r in rows]),
                "stand_takeover_rate": mean([r["stand_takeover"] for r in rows]),
                "final_stand_hold_rate": mean([r["final_stand_hold"] for r in rows]),
                "full_success_rate": mean([r["full_success"] for r in rows]),
                "fall_rate": mean([r["fall"] for r in rows]),
                "saturation_failure_rate": mean([r["saturation_failure"] for r in rows]),
                "dangerous_slip_rate": mean([r["dangerous_slip"] for r in rows]),
                "excessive_flight_rate": mean([r["excessive_flight"] for r in rows]),
                "reverse_motion_rate": mean([r["reverse_motion"] for r in rows]),
                "action_discontinuity_rate": mean([r["action_discontinuity"] for r in rows]),
                "walk_heading_p95_rad": pct([r["walk_heading_p95_rad"] for r in rows], 95),
                "final_speed_p95_mps": pct([r["final_speed_p95_mps"] for r in rows], 95),
                "final_double_support_rate": mean([r["final_double_support"] for r in rows]),
                "final_stand_flight_rate": mean([r["final_stand_flight"] for r in rows]),
                "previous_action_mismatch_rate": mean([r["previous_action_mismatch_steps"] > 0 for r in rows]),
            }
        startup_summary = {
            "seed": args.seed, "episodes": n,
            "valid_stand_rate": mean([r["state_contract_valid"] for r in startup_rows]),
            "settle_time_mean_s": mean([r["settle_time_s"] for r in startup_rows if r["settle_time_s"] is not None]),
            "settle_time_p95_s": pct([r["settle_time_s"] for r in startup_rows if r["settle_time_s"] is not None], 95),
            "fall_rate": mean([r["fall"] for r in startup_rows]),
            "ankle_saturation_rate": mean([r["ankle_saturation"] for r in startup_rows]),
            "no_contact_reset_rate": mean([r["no_contact_reset"] for r in startup_rows]),
            "reset_horizontal_velocity_mean_mps": mean([r["reset_horizontal_velocity_mps"] for r in startup_rows]),
            "reset_horizontal_velocity_p95_mps": pct([r["reset_horizontal_velocity_mps"] for r in startup_rows], 95),
            "reset_vertical_velocity_mean_mps": mean([r["reset_vertical_velocity_mps"] for r in startup_rows]),
            "initial_double_support_rate": mean([r["initial_double_support"] for r in startup_rows]),
            "state_contract_rejection_rate": mean([r["state_contract_rejected"] for r in startup_rows]),
            "graph_formal_denominator": 0,
            "startup_recovery_capability": "NOT_IMPLEMENTED",
        }
        report = {
            "mode": args.mode, "label": args.label, "seed": args.seed,
            "diagnostic_only_startup": args.mode == "startup",
            "startup": startup_summary,
            "requested_candidates": n, "valid_source_candidates": sum(r["state_contract_valid"] for r in startup_rows),
            "formal_selection_rule": "first preregistered valid source states per speed before graph outcome",
            "formal": summary(selected), "per_speed": {
                str(speed): summary([row for row in selected if row["target_speed_mps"] == speed])
                for speed in desired
            },
            "protected_hashes": {name: sha(path) for name, path in paths.items()},
        }
        write_json(output / f"{args.label}_summary.json", report)
        write_csv(output / f"{args.label}_startup_candidates.csv", startup_rows)
        write_csv(output / f"{args.label}_episodes.csv", selected)
        selected_ids = {row["candidate_episode"] for row in selected}
        write_csv(output / f"{args.label}_boundaries.csv", [
            row for row in boundary_rows if row["candidate_episode"] in selected_ids
        ])
        print(json.dumps(report, indent=2))
        wrapped.close()


if __name__ == "__main__":
    main()
